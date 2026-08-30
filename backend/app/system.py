from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from functools import lru_cache
from typing import Any

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None

from .gpu import query_gpus

_last_cpu_times: tuple[float, float] | None = None
_runtime_speed_cache: dict[str, dict[str, float]] = {}

_POWER_PLATFORM_OVERHEAD_W = 35.0
_POWER_PSU_EFFICIENCY = 0.90
_MAX_POWER_SAMPLE_INTERVAL_SECONDS = 15.0

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


def cpu_package_power_w() -> float | None:
    if sys.platform != "win32":
        return None
    command = r"""
$samples = (Get-Counter '\Energy Meter(*)\Power' -ErrorAction Stop).CounterSamples
$package = @($samples | Where-Object { $_.InstanceName -match '(?i)package\d+_(pkg|package)$' })
if (-not $package) {
    $package = @($samples | Where-Object { $_.InstanceName -match '(?i)^(cpu power|current socket power|socket power)$' })
}
if (-not $package) { exit 1 }
$milliwatts = ($package | Measure-Object -Property CookedValue -Sum).Sum
[Console]::WriteLine(($milliwatts / 1000).ToString([Globalization.CultureInfo]::InvariantCulture))
""".strip()
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=5,
        )
        completed.check_returncode()
        value = float(completed.stdout.strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None
    return value if value >= 0 else None


class PowerTracker:
    def __init__(
        self,
        total_ever_wh: float = 0.0,
        total_ever_seconds: float = 0.0,
        save_total_ever_wh: Any | None = None,
        save_total_ever_seconds: Any | None = None,
    ) -> None:
        self.session_wh = 0.0
        self.session_seconds = 0.0
        self.total_ever_wh = max(0.0, total_ever_wh)
        self.total_ever_seconds = max(0.0, total_ever_seconds)
        self._save_total_ever_wh = save_total_ever_wh
        self._save_total_ever_seconds = save_total_ever_seconds
        self._last_timestamp: float | None = None
        self._last_power_w: float | None = None
        self._lock = threading.Lock()

    def sample(self, power_w: float | None, timestamp: float) -> dict[str, float]:
        with self._lock:
            if power_w is not None and self._last_timestamp is not None and self._last_power_w is not None:
                elapsed = timestamp - self._last_timestamp
                if 0 < elapsed <= _MAX_POWER_SAMPLE_INTERVAL_SECONDS:
                    added_wh = ((self._last_power_w + power_w) / 2) * elapsed / 3600
                    self.session_wh += added_wh
                    self.session_seconds += elapsed
                    self.total_ever_wh += added_wh
                    self.total_ever_seconds += elapsed
                    if self._save_total_ever_wh:
                        self._save_total_ever_wh(self.total_ever_wh)
                    if self._save_total_ever_seconds:
                        self._save_total_ever_seconds(self.total_ever_seconds)
            self._last_timestamp = timestamp
            self._last_power_w = power_w
            return {
                "session_kwh": self.session_wh / 1000,
                "session_measured_seconds": self.session_seconds,
                "total_ever_kwh": self.total_ever_wh / 1000,
                "total_ever_measured_seconds": self.total_ever_seconds,
            }


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


def _parse_prometheus_metrics(payload: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in payload.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            name, value = line.rsplit(None, 1)
            if "{" not in name:
                metrics[name] = float(value)
        except (ValueError, TypeError):
            continue
    return metrics


def _probe_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::", "[::]", "*"} else host


def runtime_token_speed(endpoints: list[tuple[str, int]]) -> dict[str, Any] | None:
    successful = 0
    active_requests = 0
    prompt_tokens = 0.0
    prompt_seconds = 0.0
    decode_tokens = 0.0
    decode_seconds = 0.0
    current_prompt = 0.0
    current_decode = 0.0
    current_prompt_count = 0
    current_decode_count = 0

    for host, port in endpoints:
        endpoint = f"{_probe_host(host)}:{port}"
        try:
            with urllib.request.urlopen(f"http://{endpoint}/metrics", timeout=0.75) as response:
                metrics = _parse_prometheus_metrics(response.read().decode("utf-8", errors="replace"))
        except (OSError, ValueError):
            continue
        required = (
            "llamacpp:prompt_tokens_total",
            "llamacpp:prompt_seconds_total",
            "llamacpp:tokens_predicted_total",
            "llamacpp:tokens_predicted_seconds_total",
        )
        if not all(name in metrics for name in required):
            continue

        successful += 1
        endpoint_prompt_tokens = metrics[required[0]]
        endpoint_decode_tokens = metrics[required[2]]
        cached = _runtime_speed_cache.setdefault(endpoint, {})
        if (
            endpoint_prompt_tokens < cached.get("prompt_tokens_total", 0)
            or endpoint_decode_tokens < cached.get("decode_tokens_total", 0)
        ):
            cached.clear()
        cached["prompt_tokens_total"] = endpoint_prompt_tokens
        cached["decode_tokens_total"] = endpoint_decode_tokens

        observed_prompt = metrics.get("llamacpp:prompt_tokens_seconds", 0.0)
        observed_decode = metrics.get("llamacpp:predicted_tokens_seconds", 0.0)
        if observed_prompt > 0:
            cached["current_prompt_tps"] = observed_prompt
        if observed_decode > 0:
            cached["current_decode_tps"] = observed_decode
        if cached.get("current_prompt_tps") is not None:
            current_prompt += cached["current_prompt_tps"]
            current_prompt_count += 1
        if cached.get("current_decode_tps") is not None:
            current_decode += cached["current_decode_tps"]
            current_decode_count += 1

        prompt_tokens += endpoint_prompt_tokens
        prompt_seconds += metrics[required[1]]
        decode_tokens += endpoint_decode_tokens
        decode_seconds += metrics[required[3]]
        active_requests += max(0, int(metrics.get("llamacpp:requests_processing", 0)))

    if not successful:
        return None
    active_keys = {f"{_probe_host(host)}:{port}" for host, port in endpoints}
    for stale in set(_runtime_speed_cache) - active_keys:
        _runtime_speed_cache.pop(stale, None)
    return {
        "preprocessing": {
            "current_tps": current_prompt if current_prompt_count else None,
            "session_average_tps": prompt_tokens / prompt_seconds if prompt_seconds > 0 else None,
        },
        "decode": {
            "current_tps": current_decode if current_decode_count else None,
            "session_average_tps": decode_tokens / decode_seconds if decode_seconds > 0 else None,
        },
        "active_requests": active_requests,
        "servers": successful,
    }


def telemetry(
    loaded_count: int = 0,
    loading_count: int = 0,
    endpoints: list[tuple[str, int]] | None = None,
    power_tracker: PowerTracker | None = None,
) -> dict[str, Any]:
    timestamp = time.time()
    memory_hardware = _memory_hardware()
    if psutil:
        cpu_percent = float(psutil.cpu_percent(interval=None))
        memory = psutil.virtual_memory()
        memory_total_bytes = int(memory.total)
        memory_available_bytes = int(memory.available)
    else:
        cpu_percent = _fallback_cpu_percent()
        memory_total_bytes, memory_available_bytes = _fallback_memory()
    gpus = query_gpus()
    gpu_power_values = [gpu["power_draw_w"] for gpu in gpus if gpu.get("power_draw_w") is not None]
    gpu_power_w = sum(gpu_power_values) if gpu_power_values else None
    cpu_power_w = cpu_package_power_w()
    estimated_system_power_w = None
    if gpu_power_w is not None and cpu_power_w is not None:
        estimated_system_power_w = (gpu_power_w + cpu_power_w + _POWER_PLATFORM_OVERHEAD_W) / _POWER_PSU_EFFICIENCY
    energy = power_tracker.sample(estimated_system_power_w, timestamp) if power_tracker else {
        "session_kwh": 0.0,
        "session_measured_seconds": 0.0,
        "total_ever_kwh": 0.0,
        "total_ever_measured_seconds": 0.0,
    }
    return {
        "timestamp": timestamp,
        "gpus": gpus,
        "loaded_models": loaded_count,
        "loading_models": loading_count,
        "cpu_load_percent": cpu_percent,
        "cpu_power_w": cpu_power_w,
        "power": {
            "current_system_w": estimated_system_power_w,
            "gpu_w": gpu_power_w,
            "cpu_w": cpu_power_w,
            "platform_overhead_w": _POWER_PLATFORM_OVERHEAD_W,
            "psu_efficiency": _POWER_PSU_EFFICIENCY,
            "session_kwh": energy["session_kwh"],
            "session_measured_seconds": energy["session_measured_seconds"],
            "total_ever_kwh": energy["total_ever_kwh"],
            "total_ever_measured_seconds": energy["total_ever_measured_seconds"],
            "estimated": True,
        },
        "memory_total_bytes": memory_total_bytes,
        "memory_available_bytes": memory_available_bytes,
        "memory_used_bytes": memory_total_bytes - memory_available_bytes if memory_total_bytes is not None else None,
        "memory_type": memory_hardware["type"],
        "memory_speed_mts": memory_hardware["speed_mts"],
        "token_speed": runtime_token_speed(endpoints or []),
    }
