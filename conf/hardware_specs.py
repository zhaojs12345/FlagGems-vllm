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

"""各厂商芯片的基础规格(显存带宽 / 各数据类型算力)。

数据来源:《厂商带宽和算力汇总》。每颗芯片记录两套数值:

* ``nominal``  —— 标定值(厂商标称规格)。
* ``measured`` —— 实测值。

CSV 中的比例(实测/标定)可由两者推导,故不单独存储。

缺失值约定:
* ``○``(厂商出于保密未提供)-> ``None``
* ``/``(该芯片不具备对应能力)-> ``None``
* ``>=350`` 之类的下界值      -> 取下界数值(如 ``350.0``),并在注释中标注。

单位:显存容量 GiB;显存带宽 GB/s;浮点算力 TFLOPS;INT8 算力 TOPS。

用法示例::

    from benchmark import hardware_specs as hw

    hw.get_bandwidth("ascend")                     # -> 3072.0 (实测)
    hw.get_compute("mthreads", hw.ComputeDType.BF16)  # -> 449.29
    hw.get_compute("hygon", torch.float16)         # torch dtype 也可
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class HardwareVendor(Enum):
    """厂商枚举。取值为小写英文名,与运行时 ``flaggems_vllm.vendor_name`` 对齐。"""

    ASCEND = "ascend"        # 华为
    METAX = "metax"          # 沐曦
    KUNLUNXIN = "kunlunxin"  # 昆仑芯
    MTHREADS = "mthreads"    # 摩尔线程
    HYGON = "hygon"          # 海光
    ILUVATAR = "iluvatar"    # 天数
    TSINGMICRO = "tsingmicro"  # 清微智能
    CAMBRICON = "cambricon"  # 寒武纪

    @classmethod
    def from_name(cls, name: "str | HardwareVendor") -> "HardwareVendor":
        """接受字符串(大小写不敏感)或枚举本身,统一返回枚举。"""
        if isinstance(name, cls):
            return name
        try:
            return cls(str(name).lower())
        except ValueError as e:
            raise KeyError(f"未知厂商: {name!r}") from e


class ComputeDType(Enum):
    """算力对应的数据类型。INT8 单位为 TOPS,其余为 TFLOPS。"""

    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    TF32 = "tf32"
    INT8 = "int8"
    FP64 = "fp64"
    FP8 = "fp8"


@dataclass(frozen=True)
class ComputePower:
    """一颗芯片在各数据类型下的算力。``None`` 表示未提供或不具备。"""

    bf16: Optional[float] = None
    fp16: Optional[float] = None
    fp32: Optional[float] = None
    tf32: Optional[float] = None
    int8: Optional[float] = None  # 单位 TOPS
    fp64: Optional[float] = None
    fp8: Optional[float] = None

    def get(self, dtype: ComputeDType) -> Optional[float]:
        """按 :class:`ComputeDType` 取对应算力。"""
        return getattr(self, dtype.value)


@dataclass(frozen=True)
class HardwareSpec:
    """单颗芯片的完整规格,含标定值与实测值两套数据。"""

    vendor: HardwareVendor
    chip: str

    # 标定值(厂商标称)
    memory_gib_nominal: Optional[float]
    bandwidth_gbps_nominal: Optional[float]
    compute_nominal: ComputePower

    # 实测值
    memory_gib_measured: Optional[float]
    bandwidth_gbps_measured: Optional[float]
    compute_measured: ComputePower


# ---------------------------------------------------------------------------
# 数据表:key 为 HardwareVendor,value 为该厂商代表芯片的规格。
# ---------------------------------------------------------------------------
HARDWARE_SPECS: Dict[HardwareVendor, HardwareSpec] = {
    HardwareVendor.ASCEND: HardwareSpec(
        vendor=HardwareVendor.ASCEND,
        chip="Atlas 800T A3",
        memory_gib_nominal=128.0,
        bandwidth_gbps_nominal=None,  # ○ 保密
        compute_nominal=ComputePower(
            bf16=None,  # ○ 保密
            fp16=752.0,
            fp32=None,  # ○ 保密
            tf32=None,  # / 不具备
            int8=None,  # ○ 保密
            fp64=None,  # / 不具备
            fp8=None,   # / 不具备
        ),
        memory_gib_measured=128.0,
        bandwidth_gbps_measured=3072.0,
        compute_measured=ComputePower(
            bf16=730.26,
            fp16=752.46,
            fp32=199.21,
            tf32=None,
            int8=1460.53,
            fp64=None,
            fp8=None,
        ),
    ),
    HardwareVendor.METAX: HardwareSpec(
        vendor=HardwareVendor.METAX,
        chip="C550",
        memory_gib_nominal=64.0,
        bandwidth_gbps_nominal=1800.0,
        compute_nominal=ComputePower(
            bf16=320.0,
            fp16=320.0,
            fp32=40.0,
            tf32=160.0,
            int8=640.0,
            fp64=None,
            fp8=None,
        ),
        memory_gib_measured=62.95,
        bandwidth_gbps_measured=1441.58,
        compute_measured=ComputePower(
            bf16=273.99,
            fp16=267.35,
            fp32=39.68,
            tf32=132.49,
            int8=562.2,
            fp64=None,
            fp8=None,
        ),
    ),
    HardwareVendor.KUNLUNXIN: HardwareSpec(
        vendor=HardwareVendor.KUNLUNXIN,
        chip="P900",
        memory_gib_nominal=96.0,
        bandwidth_gbps_nominal=2600.0,
        compute_nominal=ComputePower(
            bf16=350.0,  # 标称 ">=350",取下界
            fp16=350.0,  # 标称 ">=350",取下界
            fp32=None,   # ○ 保密
            tf32=None,   # ○ 保密
            int8=None,   # ○ 保密
            fp64=None,
            fp8=None,
        ),
        memory_gib_measured=96.0,
        bandwidth_gbps_measured=2554.93,
        compute_measured=ComputePower(
            bf16=342.75,
            fp16=349.08,
            fp32=138.7,
            tf32=138.68,
            int8=643.46,
            fp64=None,
            fp8=None,
        ),
    ),
    HardwareVendor.MTHREADS: HardwareSpec(
        vendor=HardwareVendor.MTHREADS,
        chip="MTT-S5000",
        memory_gib_nominal=80.0,
        bandwidth_gbps_nominal=1600.0,
        compute_nominal=ComputePower(
            bf16=430.0,
            fp16=430.0,
            fp32=29.0,
            tf32=215.0,
            int8=860.0,
            fp64=14.5,
            fp8=860.0,
        ),
        memory_gib_measured=79.73,
        bandwidth_gbps_measured=1369.09,
        compute_measured=ComputePower(
            bf16=449.29,
            fp16=448.25,
            fp32=28.45,
            tf32=224.07,
            int8=895.32,
            fp64=14.21,
            fp8=897.5,
        ),
    ),
    HardwareVendor.HYGON: HardwareSpec(
        vendor=HardwareVendor.HYGON,
        chip="BW1000",
        memory_gib_nominal=64.0,
        bandwidth_gbps_nominal=1800.0,
        compute_nominal=ComputePower(
            bf16=480.0,
            fp16=480.0,
            fp32=60.0,
            tf32=240.0,
            int8=960.0,
            fp64=30.0,
            fp8=None,
        ),
        memory_gib_measured=63.72,
        bandwidth_gbps_measured=1529.47,
        compute_measured=ComputePower(
            bf16=473.88,
            fp16=472.1,
            fp32=60.93,
            tf32=236.94,
            int8=947.74,
            fp64=30.46,
            fp8=None,
        ),
    ),
    HardwareVendor.ILUVATAR: HardwareSpec(
        vendor=HardwareVendor.ILUVATAR,
        chip="BI-V150",
        memory_gib_nominal=64.0,
        bandwidth_gbps_nominal=1600.0,
        compute_nominal=ComputePower(
            bf16=256.0,
            fp16=256.0,
            fp32=64.0,
            tf32=None,  # / 不具备
            int8=768.0,
            fp64=None,
            fp8=None,
        ),
        memory_gib_measured=63.52,
        bandwidth_gbps_measured=1151.1,
        compute_measured=ComputePower(
            bf16=211.21,
            fp16=210.45,
            fp32=45.69,
            tf32=None,
            int8=755.5,
            fp64=None,
            fp8=None,
        ),
    ),
    HardwareVendor.TSINGMICRO: HardwareSpec(
        vendor=HardwareVendor.TSINGMICRO,
        chip="REX1032",
        memory_gib_nominal=512.0,
        bandwidth_gbps_nominal=800.0,
        compute_nominal=ComputePower(
            bf16=512.0,
            fp16=512.0,
            fp32=12.0,
            tf32=512.0,
            int8=1024.0,
            fp64=None,
            fp8=None,
        ),
        memory_gib_measured=473.0,
        bandwidth_gbps_measured=637.45,
        compute_measured=ComputePower(
            bf16=503.81,
            fp16=503.81,
            fp32=12.2,
            tf32=444.8,
            int8=1007.62,
            fp64=None,
            fp8=None,
        ),
    ),
    HardwareVendor.CAMBRICON: HardwareSpec(
        vendor=HardwareVendor.CAMBRICON,
        chip="MLU590",
        # 注:寒武纪未参与本轮评测,数据取自 2024 年 XLC 项目基础规格评测结果。
        memory_gib_nominal=80.0,
        bandwidth_gbps_nominal=2250.0,
        compute_nominal=ComputePower(
            bf16=330.0,
            fp16=330.0,
            fp32=82.5,
            tf32=165.0,
            int8=660.0,
            fp64=None,
            fp8=None,
        ),
        memory_gib_measured=78.68,
        bandwidth_gbps_measured=2020.0,
        compute_measured=ComputePower(
            bf16=303.25,
            fp16=303.34,
            fp32=78.0,
            tf32=153.12,
            int8=602.86,
            fp64=None,
            fp8=None,
        ),
    ),
}


# ---------------------------------------------------------------------------
# torch dtype -> ComputeDType 映射(延迟导入 torch,避免无谓依赖)。
# ---------------------------------------------------------------------------
def torch_dtype_to_compute(dtype) -> ComputeDType:
    """把 ``torch.dtype`` 映射为 :class:`ComputeDType`。"""
    import torch

    mapping = {
        torch.bfloat16: ComputeDType.BF16,
        torch.float16: ComputeDType.FP16,
        torch.float32: ComputeDType.FP32,
        torch.float64: ComputeDType.FP64,
        torch.int8: ComputeDType.INT8,
    }
    # FP8 类型在部分 torch 版本才有
    for attr in ("float8_e4m3fn", "float8_e5m2"):
        fp8 = getattr(torch, attr, None)
        if fp8 is not None:
            mapping.setdefault(fp8, ComputeDType.FP8)
    if dtype not in mapping:
        raise KeyError(f"无法为 torch dtype {dtype!r} 匹配算力类型")
    return mapping[dtype]


def _as_compute_dtype(dtype) -> ComputeDType:
    if isinstance(dtype, ComputeDType):
        return dtype
    if isinstance(dtype, str):
        return ComputeDType(dtype.lower())
    # 其余当作 torch.dtype 处理
    return torch_dtype_to_compute(dtype)


# ---------------------------------------------------------------------------
# 访问接口
# ---------------------------------------------------------------------------
def get_spec(vendor) -> HardwareSpec:
    """返回某厂商代表芯片的完整规格。``vendor`` 可为字符串或 :class:`HardwareVendor`。"""
    return HARDWARE_SPECS[HardwareVendor.from_name(vendor)]


def get_bandwidth(vendor, prefer: str = "measured") -> Optional[float]:
    """获取显存带宽(GB/s)。

    ``prefer`` 为 ``"measured"``(默认)时优先返回实测值,缺失则回退标定值;
    为 ``"nominal"`` 时优先返回标定值,缺失则回退实测值。
    """
    spec = get_spec(vendor)
    measured, nominal = spec.bandwidth_gbps_measured, spec.bandwidth_gbps_nominal
    return _prefer(measured, nominal, prefer)


def get_compute(vendor, dtype, prefer: str = "measured") -> Optional[float]:
    """获取指定数据类型的算力(TFLOPS,INT8 为 TOPS)。

    ``dtype`` 支持 :class:`ComputeDType`、字符串(如 ``"bf16"``)或 ``torch.dtype``。
    """
    spec = get_spec(vendor)
    cdt = _as_compute_dtype(dtype)
    measured = spec.compute_measured.get(cdt)
    nominal = spec.compute_nominal.get(cdt)
    return _prefer(measured, nominal, prefer)


def get_memory_gib(vendor, prefer: str = "measured") -> Optional[float]:
    """获取单卡显存容量(GiB)。"""
    spec = get_spec(vendor)
    return _prefer(spec.memory_gib_measured, spec.memory_gib_nominal, prefer)


def _prefer(measured, nominal, prefer: str):
    if prefer == "measured":
        return measured if measured is not None else nominal
    if prefer == "nominal":
        return nominal if nominal is not None else measured
    raise ValueError(f"prefer 只能是 'measured' 或 'nominal',收到 {prefer!r}")
