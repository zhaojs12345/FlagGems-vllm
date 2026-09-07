#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
生成 baseline 采集脚本。

读取 tools/baseline_shape.yaml，为每个算子生成对应的采集代码，
输出到 tools/collect_baseline_nvidia.py。

用法:
    python tools/generate_baseline_collector.py

生成后运行:
    python tools/collect_baseline_nvidia.py --output op_perf_baseline.json
"""

import yaml
from pathlib import Path

SCRIPT_TEMPLATE = '''#!/usr/bin/env python3
"""
NVIDIA baseline 数据采集脚本（自动生成）。

此脚本在 NVIDIA 硬件上执行，调用原生 CUDA kernel 并通过 NCU 获取：
- latency_ms: kernel 执行时间（毫秒）
- cuda_flops: CUDA Core 计算量（浮点运算数）
- sm_utilization: SM 硬件利用率（0-1）

生成数据用于后续 benchmark 的归一化对比。
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import triton


# NCU 指标：FLOP 计数（fp32 的 fadd/fmul/ffma + fp16/bf16 的 hadd/hmul/hfma）
# 加上 SM 吞吐利用率。ffma/hfma 每条指令计 2 次浮点运算。
FLOP_METRICS = {{
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 2,
    "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum": 2,
}}
# 与 ncu-ui 的 "Compute (SM) Throughput" 一致，用 elapsed（除以总耗时）
# 而非 active（除以活跃 cycle）。这样脚本抓到的数能和 GUI 对上。
SM_UTIL_METRIC = "sm__throughput.avg.pct_of_peak_sustained_elapsed"


def _build_profile_script(module_path, import_name, shape, dtype_str,
                          warmup=3):
    """生成用于 NCU 分析的独立临时脚本内容。

    脚本先 warmup 若干次，再执行 1 次被分析的 kernel。NCU 通过
    --launch-skip / --launch-count 精确锁定最后一次 launch。
    """
    return f"""import torch
from {{module_path}} import {{import_name}}

dtype = {{dtype_str}}
shape = {{shape!r}}
x = torch.randn(shape, dtype=dtype, device='cuda')
gate = torch.randn(shape, dtype=dtype, device='cuda')

# warmup（这些 launch 会被 --launch-skip 跳过）
for _ in range({{warmup}}):
    {{import_name}}(x, gate)
torch.cuda.synchronize()

# 被分析的 launch
{{import_name}}(x, gate)
torch.cuda.synchronize()
"""


def _parse_ncu_csv(csv_text, wanted_metrics):
    """解析 ``ncu --import --page raw --csv`` 的输出。

    raw 页是**宽表**：每个 kernel launch 一行，每个 metric 是独立的一列，
    列名即 metric 名（如 "sm__throughput.avg.pct_of_peak_sustained_elapsed"）。
    表头之后通常还有一行**单位行**（如 ""、"%"），需要跳过。

    按 metric 名在所有 kernel 行上求和（一次调用可能触发多个 kernel）。
    对利用率类指标（百分比）求和意义不大，调用方需自行按需处理；这里
    统一返回“各列的数值之和”，FLOP 求和正确，利用率取和后由调用方判断。

    返回 {{metric_name: {{"sum": float, "count": int}}}}。
    """
    import csv
    import io

    reader = csv.reader(io.StringIO(csv_text))
    rows = [r for r in reader if r]
    if not rows:
        return {{}}

    header = rows[0]
    # 定位每个目标 metric 的列索引
    col_idx = {{}}
    for m in wanted_metrics:
        if m in header:
            col_idx[m] = header.index(m)

    agg = {{m: {{"sum": 0.0, "count": 0}} for m in col_idx}}

    for row in rows[1:]:
        # 跳过单位行：数据行的 metric 列应能解析为数字，单位行不能
        for m, idx in col_idx.items():
            if idx >= len(row):
                continue
            cell = row[idx].replace(",", "").strip()
            if cell == "" or cell.lower() in ("%", "n/a", "nan"):
                continue
            try:
                num = float(cell)
            except ValueError:
                continue
            agg[m]["sum"] += num
            agg[m]["count"] += 1
    return agg


def profile_with_ncu(module_path, import_name, shape, dtype_str,
                     warmup=3, report_dir=None):
    """调用 NCU 获取 kernel 的计算量（FLOP）和 SM 利用率。

    分两步：
    1. ``ncu --export`` 把 profiling 结果存成 .ncu-rep（可用 ncu-ui 打开，
       方便你在 GUI 里核对 Speed Of Light 等区块）。
    2. ``ncu --import ... --csv`` 非交互地重新解析同一份报告，提取数值。

    通过生成临时脚本 + ncu 子进程实现，因为 NCU 需要分析一个独立的
    python 进程。返回 {{"cuda_flops": float, "sm_utilization": float,
    "report_path": str}}；NCU 不可用或解析失败时 flops/util 返回 0。
    """
    metrics = list(FLOP_METRICS.keys()) + [SM_UTIL_METRIC]

    script = _build_profile_script(module_path, import_name, shape, dtype_str,
                                   warmup=warmup)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(script)
        script_path = f.name

    # 报告文件路径（.ncu-rep）。ncu 会自动补 .ncu-rep 后缀。
    if report_dir is None:
        report_dir = Path(tempfile.gettempdir())
    else:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
    dtype_tag = dtype_str.replace("torch.", "").replace(".", "_")
    shape_tag = "x".join(str(s) for s in shape)
    report_base = report_dir / f"{{import_name}}_{{shape_tag}}_{{dtype_tag}}"
    report_path = str(report_base) + ".ncu-rep"

    empty = {{"cuda_flops": 0.0, "sm_utilization": 0.0, "report_path": None}}

    try:
        # --- 第 1 步：profiling 并导出报告 ---
        export_cmd = [
            "ncu",
            "--metrics", ",".join(metrics),
            "--launch-skip", str(warmup),
            "--launch-count", "1",
            "--force-overwrite",
            "--export", str(report_base),
            sys.executable, script_path,
        ]
        try:
            proc = subprocess.run(
                export_cmd, capture_output=True, text=True, timeout=600
            )
        except FileNotFoundError:
            print("  ✗ 未找到 ncu 命令，跳过 NCU profiling")
            return empty
        except subprocess.TimeoutExpired:
            print(f"  ✗ NCU 超时: {{shape}} {{dtype_str}}")
            return empty

        if proc.returncode != 0:
            print(f"  ✗ NCU profiling 失败 (code={{proc.returncode}}): "
                  f"{{proc.stderr.strip()[:200]}}")
            return empty

        # --- 第 2 步：从报告解析 CSV ---
        import_cmd = [
            "ncu",
            "--import", report_path,
            "--csv",
            "--page", "raw",
        ]
        imp = subprocess.run(
            import_cmd, capture_output=True, text=True, timeout=120
        )
        if imp.returncode != 0:
            print(f"  ✗ NCU import 失败 (code={{imp.returncode}}): "
                  f"{{imp.stderr.strip()[:200]}}")
            return {{**empty, "report_path": report_path}}

        agg = _parse_ncu_csv(imp.stdout, metrics)

        # FLOP：各指令列跨所有 kernel 求和，ffma/hfma 计 2 次
        cuda_flops = 0.0
        for metric_name, weight in FLOP_METRICS.items():
            cuda_flops += weight * agg.get(metric_name, {{}}).get("sum", 0.0)

        # SM 吞吐是百分比（0-100），多 kernel 取均值后转成 0-1
        sm_entry = agg.get(SM_UTIL_METRIC, {{"sum": 0.0, "count": 0}})
        if sm_entry["count"] > 0:
            sm_util = (sm_entry["sum"] / sm_entry["count"]) / 100.0
        else:
            sm_util = 0.0

        return {{
            "cuda_flops": cuda_flops,
            "sm_utilization": sm_util,
            "report_path": report_path,
        }}
    finally:
        try:
            Path(script_path).unlink()
        except OSError:
            pass


def benchmark_latency(kernel_fn, warmup=25, rep=100):
    """用 triton.testing.do_bench 测量 kernel 延迟（毫秒）。"""
    return triton.testing.do_bench(kernel_fn, warmup=warmup, rep=rep)


def collect_baseline(output_path, ncu_enabled=True, report_dir=None):
    """采集所有算子的 baseline 数据。

    report_dir: .ncu-rep 报告的存放目录（可用 ncu-ui 打开核对）。
    """

    # 检查是否在 NVIDIA 硬件
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 设备")

    device_name = torch.cuda.get_device_name()
    if "NVIDIA" not in device_name.upper():
        print(f"警告: 当前设备 {{device_name}} 可能不是 NVIDIA 硬件")

    print(f"采集设备: {{device_name}}")
    if ncu_enabled and report_dir:
        print(f"NCU 报告目录: {{report_dir}}")

    results = {{}}

{op_collectors}

    # 写入输出文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\\n✓ Baseline 数据已写入: {{output_path}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="采集 NVIDIA 原生 kernel 的 baseline 数据"
    )
    parser.add_argument(
        "--output",
        default="op_perf_baseline.json",
        help="输出文件路径（默认: op_perf_baseline.json）"
    )
    parser.add_argument(
        "--no-ncu",
        action="store_true",
        help="跳过 NCU profiling（只测 latency）"
    )
    parser.add_argument(
        "--report-dir",
        default="ncu_reports",
        help="NCU .ncu-rep 报告存放目录（可用 ncu-ui 打开，默认: ncu_reports）"
    )
    args = parser.parse_args()

    collect_baseline(
        args.output,
        ncu_enabled=not args.no_ncu,
        report_dir=args.report_dir,
    )
'''

OP_COLLECTOR_TEMPLATE = '''
    # ========== {op_name} ==========
    print("\\n采集算子: {op_name}")

    try:
        from {module_path} import {import_name}
    except ImportError as e:
        print(f"  ✗ 跳过 {op_name}: {{e}}")
    else:
        results["{op_name}"] = {{
            "native_api": "{native_api}",
            "shapes": {{}}
        }}

        for shape in {shapes}:
            shape_key = str(shape)
            results["{op_name}"]["shapes"][shape_key] = {{}}

            for dtype_str in {dtypes_str}:
                dtype = eval(dtype_str)

                # 生成输入张量
                x = torch.randn(shape, dtype=dtype, device='cuda')
                gate = torch.randn(shape, dtype=dtype, device='cuda')

                # 测量延迟
                kernel_fn = lambda: {import_name}(x, gate)
                latency_ms = benchmark_latency(kernel_fn)

                # NCU profiling（在独立子进程中分析，同时导出 .ncu-rep）
                if ncu_enabled:
                    ncu_result = profile_with_ncu(
                        "{module_path}", "{import_name}", shape, dtype_str,
                        report_dir=report_dir
                    )
                else:
                    ncu_result = {{
                        "cuda_flops": 0.0,
                        "sm_utilization": 0.0,
                        "report_path": None,
                    }}

                results["{op_name}"]["shapes"][shape_key][dtype_str] = {{
                    "latency_ms": latency_ms,
                    "cuda_flops": ncu_result["cuda_flops"],
                    "sm_utilization": ncu_result["sm_utilization"],
                    "report_path": ncu_result.get("report_path"),
                }}

                print(f"  {{shape}} {{dtype_str}}: {{latency_ms:.4f}} ms")
'''


def generate_collector():
    """读取 baseline_shape.yaml，生成采集脚本。"""
    yaml_path = Path(__file__).parent / "baseline_shape.yaml"

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    # 为每个算子生成采集代码
    op_collectors = []
    for op_name, op_config in config.items():
        native_api = op_config["native_api"]
        shapes = op_config["shapes"]
        dtypes = op_config["dtypes"]

        # 解析模块路径（如 "vllm._C" -> from vllm import _C）
        parts = native_api.rsplit(".", 1)
        if len(parts) == 2:
            module_path, import_name = parts
        else:
            raise ValueError(f"native_api 格式错误: {native_api}")

        op_collector = OP_COLLECTOR_TEMPLATE.format(
            op_name=op_name,
            native_api=native_api,
            module_path=module_path,
            import_name=import_name,
            shapes=shapes,
            dtypes_str=[str(dt) for dt in dtypes]
        )
        op_collectors.append(op_collector)

    # 生成完整脚本
    script_content = SCRIPT_TEMPLATE.format(
        op_collectors="".join(op_collectors)
    )

    output_path = Path(__file__).parent / "collect_baseline_nvidia.py"
    with open(output_path, "w") as f:
        f.write(script_content)

    output_path.chmod(0o755)
    print(f"✓ 已生成采集脚本: {output_path}")


if __name__ == "__main__":
    generate_collector()
