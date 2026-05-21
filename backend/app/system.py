from __future__ import annotations

import os
import time
from typing import Any

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None

from .gpu import query_gpus

_last_cpu_times: tuple[float, float] | None = None


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


def telemetry(loaded_count: int = 0, loading_count: int = 0) -> dict[str, Any]:
    if psutil:
        cpu_percent = float(psutil.cpu_percent(interval=None))
    else:
        cpu_percent = _fallback_cpu_percent()
    return {
        "timestamp": time.time(),
        "gpus": query_gpus(),
        "loaded_models": loaded_count,
        "loading_models": loading_count,
        "cpu_load_percent": cpu_percent,
        "cpu_power_w": None,
    }

