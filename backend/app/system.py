from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from functools import lru_cache
from typing import Any

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None

from .gpu import query_gpus

_last_cpu_times: tuple[float, float] | None = None

_SMBIOS_MEMORY_TYPES = {
    20: "DDR",
    21: "DDR2",
    24: "DDR3",
    26: "DDR4",
    34: "DDR5",
}


def _fallback_cpu_percent() -> float | None:
    global _last_cpu_times
    try:
        times = os.times()
    except Exception:
        return None
    busy = times.user + times.system
    total = busy + times.children_user + times.children_system + time.perf_counter()
    if _last_cpu_times is None:
        _last_cpu_times = (busy, total)
        return None
    last_busy, last_total = _last_cpu_times
    _last_cpu_times = (busy, total)
    delta_total = max(total - last_total, 0.001)
    return max(0.0, min(100.0, ((busy - last_busy) / delta_total) * 100.0))


def _fallback_memory() -> tuple[int | None, int | None]:
    if sys.platform != "win32":
        return None, None
    try:
        import ctypes
        from ctypes import wintypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", wintypes.DWORD),
                ("memory_load", wintypes.DWORD),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None, None
        return int(status.total_physical), int(status.available_physical)
    except (AttributeError, OSError):
        return None, None


@lru_cache(maxsize=1)
def _memory_hardware() -> dict[str, Any]:
    if sys.platform != "win32":
        return {"type": None, "speed_mts": None}
    command = (
        "Get-CimInstance Win32_PhysicalMemory | "
        "Select-Object SMBIOSMemoryType,ConfiguredClockSpeed,Speed | "
        "ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=5,
        )
        completed.check_returncode()
        modules = json.loads(completed.stdout)
    except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError):
        return {"type": None, "speed_mts": None}

    if isinstance(modules, dict):
        modules = [modules]
    types = {
        _SMBIOS_MEMORY_TYPES.get(module.get("SMBIOSMemoryType"))
        for module in modules
        if isinstance(module, dict)
    }
    types.discard(None)
    speeds = {
        module.get("ConfiguredClockSpeed") or module.get("Speed")
        for module in modules
        if isinstance(module, dict)
    }
    speeds.discard(None)
    return {
        "type": " / ".join(sorted(types)) or None,
        "speed_mts": next(iter(speeds)) if len(speeds) == 1 else None,
    }


def telemetry(loaded_count: int = 0, loading_count: int = 0) -> dict[str, Any]:
    memory_hardware = _memory_hardware()
    if psutil:
        cpu_percent = float(psutil.cpu_percent(interval=None))
        memory = psutil.virtual_memory()
        memory_total_bytes = int(memory.total)
        memory_available_bytes = int(memory.available)
    else:
        cpu_percent = _fallback_cpu_percent()
        memory_total_bytes, memory_available_bytes = _fallback_memory()
    return {
        "timestamp": time.time(),
        "gpus": query_gpus(),
        "loaded_models": loaded_count,
        "loading_models": loading_count,
        "cpu_load_percent": cpu_percent,
        "cpu_power_w": None,
        "memory_total_bytes": memory_total_bytes,
        "memory_available_bytes": memory_available_bytes,
        "memory_used_bytes": memory_total_bytes - memory_available_bytes if memory_total_bytes is not None else None,
        "memory_type": memory_hardware["type"],
        "memory_speed_mts": memory_hardware["speed_mts"],
    }
