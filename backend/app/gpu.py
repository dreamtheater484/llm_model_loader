from __future__ import annotations

import csv
import io
import re
import subprocess
from dataclasses import dataclass, asdict
from typing import Any


RTX_VRAM_FALLBACK_GB: dict[str, int | tuple[int, ...]] = {
    "NVIDIA GeForce RTX 5090": 32,
    "NVIDIA GeForce RTX 5080": 16,
    "NVIDIA GeForce RTX 5070 Ti": 16,
    "NVIDIA GeForce RTX 5070": 12,
    "NVIDIA GeForce RTX 5060 Ti": (8, 16),
    "NVIDIA GeForce RTX 5060": 8,
    "NVIDIA GeForce RTX 5090 Laptop GPU": 24,
    "NVIDIA GeForce RTX 5080 Laptop GPU": 16,
    "NVIDIA GeForce RTX 5070 Ti Laptop GPU": 12,
    "NVIDIA GeForce RTX 5070 Laptop GPU": (8, 12),
    "NVIDIA GeForce RTX 5060 Laptop GPU": 8,
    "NVIDIA GeForce RTX 4070 Laptop GPU": 8,
}


@dataclass
class GpuSnapshot:
    index: int
    name: str
    uuid: str
    memory_total_mib: int
    memory_used_mib: int
    memory_free_mib: int
    power_draw_w: float | None
    power_limit_w: float | None
    utilization_gpu_percent: float | None
    vram_source: str
    supported: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", value or "")
    return float(match.group(0)) if match else None


def _int_mib(value: str) -> int:
    num = _number(value)
    return int(round(num or 0))


def fallback_vram_mib(name: str) -> int | None:
    normalized = " ".join(name.split())
    for key, gb in RTX_VRAM_FALLBACK_GB.items():
        if key.lower() in normalized.lower():
            if isinstance(gb, tuple):
                return max(gb) * 1024
            return gb * 1024
    return None


def parse_nvidia_smi_csv(text: str) -> list[GpuSnapshot]:
    reader = csv.DictReader(io.StringIO(text.strip()), skipinitialspace=True)
    gpus: list[GpuSnapshot] = []
    for row in reader:
        name = row.get("name", "").strip()
        total = _int_mib(row.get("memory.total [MiB]", "0"))
        fallback_total = fallback_vram_mib(name)
        vram_source = "nvidia-smi"
        if total <= 0 and fallback_total:
            total = fallback_total
            vram_source = "fallback-table"
        used = _int_mib(row.get("memory.used [MiB]", "0"))
        free = _int_mib(row.get("memory.free [MiB]", "0"))
        if free <= 0 and total > used:
            free = total - used
        gpus.append(
            GpuSnapshot(
                index=int(_number(row.get("index", "0")) or 0),
                name=name,
                uuid=row.get("uuid", "").strip(),
                memory_total_mib=total,
                memory_used_mib=used,
                memory_free_mib=free,
                power_draw_w=_number(row.get("power.draw [W]", "")),
                power_limit_w=_number(row.get("power.limit [W]", "")),
                utilization_gpu_percent=_number(row.get("utilization.gpu [%]", "")),
                vram_source=vram_source,
                supported="nvidia" in name.lower(),
            )
        )
    return gpus


def query_gpus() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,power.draw,power.limit,utilization.gpu",
        "--format=csv",
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except FileNotFoundError:
        return []
    except subprocess.TimeoutExpired:
        return []
    if completed.returncode != 0:
        return []
    return [gpu.to_dict() for gpu in parse_nvidia_smi_csv(completed.stdout)]
