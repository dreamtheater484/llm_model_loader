from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


QUANT_RE = re.compile(r"(?:^|[-_:])((?:UD-)?(?:IQ|Q|F)\d(?:_[A-Z0-9]+)+|F16|BF16|Q8_0|Q4_K_M|Q5_K_M)(?:$|[-_.:])", re.I)


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
    return normalized.endswith("llama-server.exe") or normalized.endswith("llama-server")


def _value_after(args: list[str], *names: str) -> str | None:
    for index, arg in enumerate(args):
        if arg in names and index + 1 < len(args):
            return args[index + 1].strip('"')
        for name in names:
            prefix = f"{name}="
            if arg.startswith(prefix):
                return arg[len(prefix) :].strip('"')
    return None


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
    host = _value_after(args, "--host") or "127.0.0.1"
    port = int(_value_after(args, "--port") or 8080)
    ctx = _value_after(args, "-c", "--ctx-size")
    model_ref = _value_after(args, "-m", "--model", "-hf", "-hfr", "--hf-repo")
    alias = _value_after(args, "--alias")
    n_cpu_moe = _value_after(args, "--n-cpu-moe")
    quant = detect_quantization(model_ref, alias, raw_script)
    flash_value = (_value_after(args, "-fa", "--flash-attn") or "").lower()
    flash = flash_value in {"on", "true", "1", "yes"}
    mtp = "draft-mtp" in raw_script.lower() or "--spec-type" in args
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
    )


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
