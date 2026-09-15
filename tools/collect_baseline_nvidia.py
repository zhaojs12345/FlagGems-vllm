#!/usr/bin/env python3
"""NVIDIA baseline 采集脚本。

在 NVIDIA 卡上调用原生 CUDA kernel，用 NCU 抓《普通算子后端缺失性能基准
方案》所需的三维实测数据：CUDA Core / Tensor Core / Global Memory 各自的
实际工作量与硬件利用率，取利用率最高者为瓶颈单元，供跨平台按 80% 达标线
对比。每条记录的字段见 profile_with_ncu 的返回值。
"""

import argparse
import csv
import importlib
import io
import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import triton
import yaml

# ---- NCU metric 名（H800/Hopper 已核对，换架构前用 `ncu --query-metrics` 复核）----

# CUDA Core 计算量 F_c：FP 指令计数，ffma/hfma 每条计 2 FLOP。
FLOP_CUDA_CORE = {
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 2,
    "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum": 2,
}

# Tensor Core 计算量 F_t：counter 名随源 dtype 而变，按 dtype 取（未知则记 0）。
FLOP_TENSOR_CORE = {
    "torch.float16": "sm__ops_path_tensor_src_fp16.sum",
    "torch.bfloat16": "sm__ops_path_tensor_src_bf16_dst_fp32.sum",
    "torch.float8_e4m3fn": "sm__ops_path_tensor_src_fp8.sum",
    "torch.float8_e5m2": "sm__ops_path_tensor_src_fp8.sum",
    "torch.float64": "sm__ops_path_tensor_src_fp64.sum",
}

# 访存量 M、三维利用率 U_c/U_t/U_m、整体 SM 吞吐（均为 % of peak，elapsed 口径）。
MEMORY_BYTES = "dram__bytes.sum"
UTIL = {
    "cuda_core": "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "tensor_core": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "memory_bw": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
}
SM_UTIL = "sm__throughput.avg.pct_of_peak_sustained_elapsed"

# 采集哪些算子、每个算子的 native 调用坐标 / shape 网格 / 输入构造，全部由
# 这份 yaml 驱动（方案B），逐步添加算子只改 yaml、不改本脚本。
CONFIG_PATH = Path(__file__).with_name("baseline_shape.yaml")

def resolve_native_op(module, symbol):
    """解析一个 NV 原生 callable；解析不到返回 None（该算子无基准，跳过）。

    benchmark/ 里真调 NV 原生的算子分三族，差别只在符号从哪来，本函数统一处理：
      族A  torch.ops._C.<symbol> / torch.ops._moe_C.<symbol>
           —— 需先 import vllm._custom_ops 触发 torch.ops 命名空间注册
      族B  vllm._custom_ops.<symbol>          （python wrapper）
      族C  vllm.v1.attention.ops.* / vllm.third_party.* 等内部模块.<symbol>

    三族都归约成「按 module 路径 import，再 getattr(symbol)」。族A 只是
    module 恰好是 torch.ops._C 这个已注册命名空间的特例。

    module: 模块路径字符串，如 "torch.ops._C" / "vllm._custom_ops" /
            "vllm.v1.attention.ops.deepseek_v4_ops"
    symbol: 该模块下的算子名。
    """
    # 触发 torch.ops._C / _moe_C 等命名空间注册（族A 依赖，其他族无害）。
    try:
        importlib.import_module("vllm._custom_ops")
    except ImportError:
        pass

    # 先按真实模块 import（族B/C）；torch.ops._C 这类是 torch 动态生成的命名空间
    # 对象、并非真模块，import_module 会失败，退回从 torch 起逐段 getattr。
    try:
        mod = importlib.import_module(module)
    except (ImportError, ModuleNotFoundError):
        mod = _resolve_dotted_attr(module)
        if mod is None:
            return None
    op = getattr(mod, symbol, None)
    return op if callable(op) else None


def _resolve_dotted_attr(dotted):
    """把 "torch.ops._C" 这类点号路径按「import 顶层包 + 逐段 getattr」解析成对象。"""
    head, *rest = dotted.split(".")
    try:
        obj = importlib.import_module(head)
    except ImportError:
        return None
    for part in rest:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _expand_grid(grid):
    """把 {M:[...], N:[...]} 笛卡尔积展开成 [{M:m, N:n}, ...] 绑定列表。"""
    dims = list(grid)
    return [dict(zip(dims, combo))
            for combo in itertools.product(*(grid[d] for d in dims))]


def _resolve_dim(token, binding):
    """把一个 shape 维度 token 解析成具体 int：int 直接用；字符串支持
    维度名（如 "N"）与「系数*维度名」（如 "2*N"）。"""
    if isinstance(token, int):
        return token
    token = str(token).strip()
    if "*" in token:
        coeff, name = token.split("*", 1)
        return int(coeff) * binding[name.strip()]
    return binding[token] if token in binding else int(token)


def _resolve_inputs(input_specs, binding, dtype_str):
    """把 yaml 的 inputs 规范 + 一组维度绑定，解析成可实例化的具体描述：
    张量 -> {"tensor": [具体int形状], "dtype": dtype_str}；标量 -> {"scalar": v}。"""
    resolved = []
    for spec in input_specs:
        if "tensor" in spec:
            shape = [_resolve_dim(t, binding) for t in spec["tensor"]]
            resolved.append({"tensor": shape, "dtype": dtype_str})
        elif "scalar" in spec:
            resolved.append({"scalar": spec["scalar"]})
        else:
            raise ValueError(f"输入项须含 tensor 或 scalar: {spec}")
    return resolved


def _instantiate_args(resolved):
    """按具体描述在 CUDA 上实例化实参列表（张量用 randn，标量原样）。"""
    args = []
    for item in resolved:
        if "tensor" in item:
            dtype = getattr(torch, item["dtype"].split(".")[-1])
            args.append(torch.randn(item["tensor"], dtype=dtype, device="cuda"))
        else:
            args.append(item["scalar"])
    return args


def _key_shape(op_cfg, binding):
    """算子在 JSON 里的主键 shape：优先 key_dims，否则取第一个 tensor 输入形状。"""
    if "key_dims" in op_cfg:
        return [_resolve_dim(d, binding) for d in op_cfg["key_dims"]]
    for spec in op_cfg["inputs"]:
        if "tensor" in spec:
            return [_resolve_dim(t, binding) for t in spec["tensor"]]
    raise ValueError("算子缺少 tensor 输入且未指定 key_dims，无法确定主键 shape")


def _empty_result(report_path=None):
    """NCU 不可用/失败时的占位结果，字段与成功时一致。"""
    return {
        "flops_cuda_core": 0.0, "flops_tensor_core": 0.0, "memory_bytes": 0.0,
        "util_cuda_core": 0.0, "util_tensor_core": 0.0, "util_memory_bw": 0.0,
        "bottleneck": None, "bottleneck_util": 0.0,
        # 旧字段：兼容 benchmark/base.py 现有归一化 speedup 公式。
        "cuda_flops": 0.0, "sm_utilization": 0.0,
        "report_path": report_path,
    }


def _build_profile_script(module, symbol, resolved_inputs, warmup):
    """临时脚本：解析 native callable，warmup 若干次后跑 1 次目标 op，供 NCU
    以 --launch-skip=warmup --launch-count=1 锁定最后一次 launch。

    脚本自包含（子进程独立运行），把本模块的 resolver 逻辑内联进去；输入按
    resolved_inputs 的具体描述用 randn/标量重建。对原地算子，被采样的那次调用
    前重新构造实参，避免 warmup 的原地累积污染数值。
    """
    return f"""import importlib
import torch


def _resolve(module, symbol):
    try:
        importlib.import_module("vllm._custom_ops")
    except ImportError:
        pass
    try:
        mod = importlib.import_module(module)
    except (ImportError, ModuleNotFoundError):
        head, *rest = module.split(".")
        mod = importlib.import_module(head)
        for part in rest:
            mod = getattr(mod, part)
    return getattr(mod, symbol)


def _make_args():
    specs = {resolved_inputs!r}
    args = []
    for it in specs:
        if "tensor" in it:
            dt = getattr(torch, it["dtype"].split(".")[-1])
            args.append(torch.randn(it["tensor"], dtype=dt, device="cuda"))
        else:
            args.append(it["scalar"])
    return args


op = _resolve({module!r}, {symbol!r})
args = _make_args()
for _ in range({warmup}):
    op(*args)
torch.cuda.synchronize()
args = _make_args()
op(*args)
torch.cuda.synchronize()
"""


def _parse_ncu_csv(csv_text, wanted):
    """解析 `ncu --import --page raw --csv`（宽表：每 launch 一行，metric 为列）。

    返回 {metric: [数值, ...]}，跳过无法解析为数字的单位行/空格。
    """
    rows = [r for r in csv.reader(io.StringIO(csv_text)) if r]
    if not rows:
        return {}
    col = {m: rows[0].index(m) for m in wanted if m in rows[0]}
    out = {m: [] for m in col}
    for row in rows[1:]:
        for m, idx in col.items():
            if idx < len(row):
                cell = row[idx].replace(",", "").strip()
                try:
                    out[m].append(float(cell))
                except ValueError:
                    pass
    return out


def profile_with_ncu(op_name, module, symbol, resolved_inputs, shape,
                     dtype_str, warmup=3, report_dir=None):
    """NCU 分析目标算子，返回三维工作量/利用率/瓶颈。

    先 `ncu --export` 存 .ncu-rep（可用 ncu-ui 核对），再 `ncu --import --csv`
    解析。返回字段见 _empty_result；NCU 不可用或失败时各值为 0。
    op_name/module/symbol/resolved_inputs 定位并重建被测算子；shape/dtype_str
    仅用于命名报告与选 Tensor Core metric。
    """
    tensor_metric = FLOP_TENSOR_CORE.get(dtype_str)
    metrics = list(FLOP_CUDA_CORE) + [MEMORY_BYTES, *UTIL.values(), SM_UTIL]
    if tensor_metric:
        metrics.append(tensor_metric)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(_build_profile_script(module, symbol, resolved_inputs, warmup))
        script_path = f.name

    report_dir = Path(report_dir) if report_dir else Path(tempfile.gettempdir())
    report_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{op_name}_{'x'.join(map(str, shape))}_{dtype_str.split('.')[-1]}"
    report_base = report_dir / tag
    report_path = f"{report_base}.ncu-rep"

    try:
        export_cmd = [
            "ncu", "--metrics", ",".join(metrics),
            "--launch-skip", str(warmup), "--launch-count", "1",
            "--force-overwrite", "--export", str(report_base),
            sys.executable, script_path,
        ]
        try:
            proc = subprocess.run(export_cmd, capture_output=True, text=True,
                                  timeout=600)
        except FileNotFoundError:
            print("  ✗ 未找到 ncu 命令，跳过 NCU profiling")
            return _empty_result()
        except subprocess.TimeoutExpired:
            print(f"  ✗ NCU 超时: {shape} {dtype_str}")
            return _empty_result()
        if proc.returncode != 0:
            print(f"  ✗ NCU profiling 失败 (code={proc.returncode}): "
                  f"{proc.stderr.strip()[:200]}")
            return _empty_result()

        imp = subprocess.run(
            ["ncu", "--import", report_path, "--csv", "--page", "raw"],
            capture_output=True, text=True, timeout=120)
        if imp.returncode != 0:
            print(f"  ✗ NCU import 失败 (code={imp.returncode}): "
                  f"{imp.stderr.strip()[:200]}")
            return _empty_result(report_path)

        agg = _parse_ncu_csv(imp.stdout, metrics)

        def total(m):
            return sum(agg.get(m, []))

        def mean_pct(m):
            vals = agg.get(m, [])
            return sum(vals) / len(vals) / 100.0 if vals else 0.0

        flops_cuda = sum(w * total(m) for m, w in FLOP_CUDA_CORE.items())
        flops_tensor = total(tensor_metric) if tensor_metric else 0.0
        util = {dim: mean_pct(name) for dim, name in UTIL.items()}
        bottleneck = max(util, key=util.get)
        if util[bottleneck] == 0.0:  # 三维全 0（解析失败/metric 不匹配）
            bottleneck = None

        return {
            "flops_cuda_core": flops_cuda,
            "flops_tensor_core": flops_tensor,
            "memory_bytes": total(MEMORY_BYTES),
            "util_cuda_core": util["cuda_core"],
            "util_tensor_core": util["tensor_core"],
            "util_memory_bw": util["memory_bw"],
            "bottleneck": bottleneck,
            "bottleneck_util": util[bottleneck] if bottleneck else 0.0,
            "cuda_flops": flops_cuda,        # 旧字段（兼容 base.py）
            "sm_utilization": mean_pct(SM_UTIL),
            "report_path": report_path,
        }
    finally:
        try:
            Path(script_path).unlink()
        except OSError:
            pass


def _collect_one_op(op_name, op_cfg, ncu_enabled, report_dir):
    """按 yaml 配置采集单个算子；native 解析不到则返回 None（跳过）。"""
    module = op_cfg["native"]["module"]
    symbol = op_cfg["native"]["symbol"]
    op = resolve_native_op(module, symbol)
    if op is None:
        print(f"\n跳过算子 {op_name}: 无法解析 native {module}.{symbol}")
        return None

    dtypes = op_cfg["dtypes"]
    bindings = _expand_grid(op_cfg["grid"])

    # 预检：解析到的 callable 可能只注册了 schema、没有 CUDA kernel（如本环境把
    # vLLM 原生 kernel 换成了 Triton），真调才暴露 NotImplementedError。用最小的
    # 那个 shape 试调一次，没实现就整体跳过，避免逐点崩掉。
    probe = _instantiate_args(_resolve_inputs(op_cfg["inputs"], bindings[0],
                                              dtypes[0]))
    try:
        op(*probe)
        torch.cuda.synchronize()
    except NotImplementedError as e:
        print(f"\n跳过算子 {op_name}: native {module}.{symbol} 无 CUDA 实现 "
              f"（{str(e)[:80]}）")
        return None

    shapes = {}
    print(f"\n采集算子: {op_name}  (native {module}.{symbol})")
    for binding in bindings:
        shape = _key_shape(op_cfg, binding)
        per_dtype = shapes.setdefault(str(shape), {})
        for dtype_str in dtypes:
            resolved = _resolve_inputs(op_cfg["inputs"], binding, dtype_str)
            args = _instantiate_args(resolved)
            latency_ms = triton.testing.do_bench(lambda: op(*args),
                                                 warmup=25, rep=100)
            ncu = (profile_with_ncu(op_name, module, symbol, resolved, shape,
                                    dtype_str, report_dir=report_dir)
                   if ncu_enabled else _empty_result())
            per_dtype[dtype_str] = {"latency_ms": latency_ms, **ncu}

            bn = ncu["bottleneck"]
            suffix = (f"  瓶颈={bn} ({ncu['bottleneck_util']*100:.1f}%)"
                      if bn else "")
            print(f"  {shape} {dtype_str}: {latency_ms:.4f} ms{suffix}")

    return {"native_api": f"{module}.{symbol}", "shapes": shapes}


def collect_baseline(output_path, config_path=CONFIG_PATH, ncu_enabled=True,
                     report_dir=None):
    """按 yaml 配置采集其中每个算子的 baseline 数据并写入 JSON。"""
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 设备")
    device_name = torch.cuda.get_device_name()
    if "NVIDIA" not in device_name.upper():
        print(f"警告: 当前设备 {device_name} 可能不是 NVIDIA 硬件")
    print(f"采集设备: {device_name}")

    config = yaml.safe_load(Path(config_path).read_text())
    results = {}
    for op_name, op_cfg in config.items():
        entry = _collect_one_op(op_name, op_cfg, ncu_enabled, report_dir)
        if entry is not None:
            results[op_name] = entry

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Baseline 数据已写入: {output_path}  (共 {len(results)} 个算子)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="采集 NVIDIA 原生 kernel 的 baseline 数据")
    parser.add_argument("--output", default="op_perf_baseline.json",
                        help="输出文件路径")
    parser.add_argument("--config", default=str(CONFIG_PATH),
                        help="算子采集配置 yaml（默认同目录 baseline_shape.yaml）")
    parser.add_argument("--no-ncu", action="store_true",
                        help="跳过 NCU profiling（只测 latency）")
    parser.add_argument("--report-dir", default="ncu_reports",
                        help="NCU .ncu-rep 报告存放目录（可用 ncu-ui 打开）")
    args = parser.parse_args()
    collect_baseline(args.output, config_path=args.config,
                     ncu_enabled=not args.no_ncu, report_dir=args.report_dir)
