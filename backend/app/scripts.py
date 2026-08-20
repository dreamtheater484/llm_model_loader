from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


QUANT_RE = re.compile(r"(?:^|[-_:])((?:UD-)?(?:IQ|Q|F)\d(?:_[A-Z0-9]+)+|F16|BF16|Q8_0|Q4_K_M|Q5_K_M|NVFP4|MXFP4)(?:$|[-_.:])", re.I)


def _strip_shell_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text.strip('"')


@dataclass
class ScriptInfo:
    executable: str | None
    args: list[str]
    host: str
    port: int
    ctx_size: int | None
    quantization: str | None
    flash_attention: bool
    mtp: bool
    model_ref: str | None
    alias: str | None
    n_cpu_moe: int | None
    fit: bool = False
    gpu_layers: str | None = None
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    parallel: int | None = None
    runtime: str = "llama.cpp"
    wsl_distro: str | None = None
    wsl_launcher: str | None = None
    concurrency: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _powershell_to_argv(raw: str) -> list[str]:
    text = raw.replace("`", " ")
    text = text.replace("& ", "")
    text = re.sub(r"\$env:USERPROFILE", str(Path.home()).replace("\\", "\\\\"), text, flags=re.I)
    try:
        return [_strip_shell_quotes(arg) for arg in shlex.split(text, posix=False)]
    except ValueError:
        return [_strip_shell_quotes(arg) for arg in text.split()]


def _is_executable_token(value: str) -> bool:
    normalized = value.strip().strip('"').lower()
    return (
        normalized.endswith("llama-server.exe")
        or normalized.endswith("llama-server")
        or normalized.endswith("wsl.exe")
        or normalized == "wsl"
        or normalized.endswith("ninfer-serve")
        or normalized.endswith("ninfer-serve.exe")
    )


def _is_ninfer_executable(value: str) -> bool:
    normalized = value.strip('"').lower()
    return (
        normalized.endswith("wsl.exe")
        or normalized == "wsl"
        or normalized.endswith("ninfer-serve")
        or normalized.endswith("ninfer-serve.exe")
    )


def _value_after(args: list[str], *names: str) -> str | None:
    for index, arg in enumerate(args):
        if arg in names and index + 1 < len(args):
            return args[index + 1].strip('"')
        for name in names:
            prefix = f"{name}="
            if arg.startswith(prefix):
                return arg[len(prefix) :].strip('"')
    return None


def _bash_lc_payload(raw_script: str) -> str | None:
    # The ninfer payload is a single-quoted bash -lc argument in the raw
    # PowerShell line. shlex.split(posix=False) fragments it on spaces, so
    # extract it straight from the source instead. Accept the unterminated
    # form too (older generated scripts relied on the trailing quote being
    # implied by the end of the line).
    match = re.search(r"-lc\s+['\"](.*?)(['\"]\s*)?$", raw_script, re.DOTALL)
    return match.group(1) if match else None


NINFER_LAUNCHER_RE = re.compile(r"(\S*run-qwen38-nvfp4\.sh)")


def _ninfer_env(payload: str | None, name: str) -> str | None:
    if not payload:
        return None
    match = re.search(rf"\b{re.escape(name)}=(\S+)", payload)
    return match.group(1) if match else None


def _ninfer_int(payload: str | None, name: str) -> int | None:
    value = _ninfer_env(payload, name)
    return int(value) if value and value.isdigit() else None


def detect_quantization(*values: str | None) -> str | None:
    for value in values:
        if not value:
            continue
        match = QUANT_RE.search(value)
        if match:
            return match.group(1).upper()
        if ":" in value:
            tail = value.rsplit(":", 1)[-1]
            if re.match(r"^(?:UD-)?[A-Z0-9_]+$", tail, re.I):
                return tail.upper()
    return None


def parse_script(raw_script: str) -> ScriptInfo:
    argv = _powershell_to_argv(raw_script)
    executable = None
    args = argv
    if argv and _is_executable_token(argv[0]):
        executable = argv[0].strip('"')
        args = argv[1:]
    is_ninfer = bool(executable and _is_ninfer_executable(executable))
    payload = _bash_lc_payload(raw_script) if is_ninfer else None
    if payload is not None:
        # Rebuild args so -lc carries the whole payload as one token (shlex
        # posix=False would have fragmented it on spaces).
        lc_index = next((i for i, a in enumerate(args) if a == "-lc"), None)
        if lc_index is not None:
            args = args[:lc_index] + ["-lc", payload]
    host = _value_after(args, "--host") or "127.0.0.1"
    port = int(_value_after(args, "--port") or 8080)
    model_ref = _value_after(args, "-m", "--model", "-hf", "-hfr", "--hf-repo")
    alias = _value_after(args, "--alias")
    n_cpu_moe = _value_after(args, "--n-cpu-moe")
    fit_value = (_value_after(args, "--fit") or "").lower()
    gpu_layers = _value_after(args, "-ngl", "--gpu-layers", "--n-gpu-layers")
    cache_type_k = _value_after(args, "-ctk", "--cache-type-k")
    cache_type_v = _value_after(args, "-ctv", "--cache-type-v")
    parallel = _value_after(args, "-np", "--parallel")
    wsl_distro = wsl_launcher = None
    concurrency = None
    if is_ninfer:
        wsl_distro = _value_after(args, "-d", "--distribution")
        launcher_match = NINFER_LAUNCHER_RE.search(payload or "")
        wsl_launcher = launcher_match.group(1) if launcher_match else None
        host = _ninfer_env(payload, "NINFER_HOST") or host
        port = _ninfer_int(payload, "NINFER_PORT") or (8081 if is_ninfer else port)
        concurrency = _ninfer_int(payload, "NINFER_CONCURRENCY")
        max_context = _ninfer_int(payload, "NINFER_MAX_CONTEXT")
        min_context = _ninfer_int(payload, "NINFER_MIN_CONTEXT")
        model_id_match = re.search(r"--model-id\s+(\S+)", payload or "")
        model_ref = model_id_match.group(1) if model_id_match else None
    ctx = _value_after(args, "-c", "--ctx-size")
    if is_ninfer:
        # The ladder starts at NINFER_MAX_CONTEXT, but NINFER_MIN_CONTEXT is the
        # guaranteed floor the server will accept, so prefer it for display/estimates.
        if max_context and min_context:
            ctx = str(min_context)
        elif max_context:
            ctx = str(max_context)
        elif min_context:
            ctx = str(min_context)
        else:
            ctx = None
    quant = detect_quantization(model_ref, alias, _ninfer_env(payload, "NINFER_MODEL_FILE"), raw_script)
    flash_value = (_value_after(args, "-fa", "--flash-attn") or "").lower()
    flash = flash_value in {"on", "true", "1", "yes"}
    # NInfer's pinned preset hardcodes the MTP3 spec in the launcher, so any
    # ninfer script implies speculative decoding is active.
    mtp = is_ninfer or "draft-mtp" in raw_script.lower() or "--spec-type" in args or (payload is not None and "--spec mtp" in payload)
    return ScriptInfo(
        executable=executable,
        args=args,
        host=host,
        port=port,
        ctx_size=int(ctx) if ctx and ctx.isdigit() else None,
        quantization=quant,
        flash_attention=flash,
        mtp=mtp,
        model_ref=model_ref,
        alias=alias,
        n_cpu_moe=int(n_cpu_moe) if n_cpu_moe and n_cpu_moe.isdigit() else None,
        fit=fit_value in {"on", "true", "1", "yes"},
        gpu_layers=gpu_layers,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        parallel=int(parallel) if parallel and parallel.isdigit() else None,
        runtime="ninfer" if is_ninfer else "llama.cpp",
        wsl_distro=wsl_distro,
        wsl_launcher=wsl_launcher,
        concurrency=concurrency,
    )


def is_fit_managed(info: ScriptInfo | dict[str, Any]) -> bool:
    """Return whether llama.cpp is responsible for choosing GPU residency."""
    values = info if isinstance(info, dict) else info.to_dict()
    layers = values.get("gpu_layers")
    return bool(values.get("fit") and (layers is None or str(layers).lower() == "auto"))


def autosuggest_name(model_name: str, raw_script: str) -> str:
    info = parse_script(raw_script)
    parts = [model_name.strip() or info.alias or "llama.cpp model"]
    if info.ctx_size:
        parts.append(f"{info.ctx_size // 1024}kctx" if info.ctx_size >= 1024 else f"{info.ctx_size}ctx")
    if info.quantization:
        parts.append(info.quantization)
    if info.flash_attention:
        parts.append("Flash")
    if info.mtp:
        parts.append("MTP")
    return " / ".join(parts)


def estimate_vram_mib(
    model_size_bytes: int | None,
    ctx_size: int | None,
    safety_mib: int = 1024,
    n_cpu_moe: int | None = None,
) -> int | None:
    if not model_size_bytes:
        return None
    if n_cpu_moe:
        return None
    weight_mib = model_size_bytes / (1024 * 1024)
    ctx_mib = 0
    if ctx_size:
        ctx_mib = max(256, ctx_size * 0.012)
    return int(weight_mib * 1.08 + ctx_mib + safety_mib)


def can_fit_vram(
    free_mib: int,
    estimated_mib: int | None,
    manual_mib: int | None = None,
    reserve_mib: int = 1024,
    allow_unknown: bool = False,
) -> tuple[bool, str]:
    needed = manual_mib or estimated_mib
    if not needed:
        if allow_unknown:
            return True, "VRAM gate skipped because this script keeps MoE experts on CPU/RAM."
        return False, "Missing VRAM estimate. Enter a manual estimate before starting."
    if free_mib - reserve_mib < needed:
        return False, f"Needs about {needed} MiB plus {reserve_mib} MiB reserve, but only {free_mib} MiB is free."
    return True, "VRAM gate passed."
