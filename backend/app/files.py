from __future__ import annotations

import os
import string
from pathlib import Path
from typing import Any

from .config import user_home
from .gguf import inspect_model_file
from .storage import store


def windows_roots() -> list[dict[str, str]]:
    roots: list[dict[str, str]] = []
    for letter in string.ascii_uppercase:
        path = Path(f"{letter}:\\")
        if path.exists():
            roots.append({"name": f"{letter}:\\", "path": str(path)})
    return roots


def browser_shortcuts() -> list[dict[str, str]]:
    home = user_home()
    candidates = [
        ("Home", home),
        ("Managed models", Path(store.setting("model_dir"))),
        ("AI", home / "AI"),
        ("Downloads", home / "Downloads"),
        ("Documents", home / "Documents"),
    ]
    return [{"name": name, "path": str(path)} for name, path in candidates if path.exists()]


def browse_files(path: str | None = None, gguf_only: bool = True, executable_only: bool = False) -> dict[str, Any]:
    current = Path(path).expanduser() if path else Path(store.setting("model_dir"))
    if not current.exists() or not current.is_dir():
        current = user_home()
    current = current.resolve()
    entries: list[dict[str, Any]] = []
    error: str | None = None
    try:
        children = sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        for child in children:
            try:
                is_dir = child.is_dir()
                if not is_dir and executable_only and child.name.lower() not in {"llama-server.exe", "llama-server"}:
                    continue
                if not is_dir and gguf_only and not executable_only and child.suffix.lower() != ".gguf":
                    continue
                item: dict[str, Any] = {
                    "name": child.name,
                    "path": str(child),
                    "type": "directory" if is_dir else "file",
                    "size_bytes": None if is_dir else child.stat().st_size,
                    "modified_at": child.stat().st_mtime,
                }
                if not is_dir and child.suffix.lower() == ".gguf":
                    item.update(inspect_model_file(str(child)))
                entries.append(item)
            except OSError:
                continue
    except OSError as exc:
        error = str(exc)
    parent = str(current.parent) if current.parent != current else None
    return {
        "path": str(current),
        "parent": parent,
        "roots": windows_roots() if os.name == "nt" else [{"name": "/", "path": "/"}],
        "shortcuts": browser_shortcuts(),
        "entries": entries,
        "error": error,
    }
