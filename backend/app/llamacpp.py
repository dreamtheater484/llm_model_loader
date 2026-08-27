from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import ai_root


def _candidate_score(path: Path) -> tuple[int, str]:
    text = str(path).lower()
    score = 0
    if "backup" in text:
        score += 100
    if "llama.cpp" not in text:
        score += 20
    if "ai" not in text:
        score += 10
    return score, text


def discover_llama_server() -> dict[str, Any]:
    candidates: list[Path] = []
    command = shutil.which("llama-server.exe") or shutil.which("llama-server")
    if command:
        candidates.append(Path(command))
    for root in [ai_root(), Path.cwd()]:
        if not root.exists():
            continue
        try:
            candidates.extend(root.rglob("llama-server.exe"))
        except OSError:
            continue
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        key = str(resolved).lower()
        if key not in seen and resolved.exists():
            seen.add(key)
            unique.append(resolved)
    unique.sort(key=_candidate_score)
    selected = str(unique[0]) if unique else ""
    return {
        "selected": selected,
        "candidates": [{"path": str(path), "backup": "backup" in str(path).lower()} for path in unique[:20]],
        "validated": validate_llama_server(selected) if selected else False,
    }


def resolve_llama_server_path(path: str) -> str:
    if not path:
        return ""
    file_path = Path(path).expanduser()
    if file_path.is_file():
        return str(file_path.resolve())
    if file_path.is_dir():
        candidates = [
            file_path / "llama-server.exe",
            file_path / "llama.cpp" / "llama-server.exe",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return str(candidate.resolve())
    return path


def validate_llama_server(path: str) -> bool:
    if not path:
        return False
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return False
    try:
        completed = subprocess.run([str(file_path), "--help"], capture_output=True, text=True, timeout=5)
    except Exception:
        return file_path.name.lower() == "llama-server.exe"
    output = f"{completed.stdout}\n{completed.stderr}".lower()
    return completed.returncode in {0, 1} and ("llama" in output or "usage" in output or "server" in output)
