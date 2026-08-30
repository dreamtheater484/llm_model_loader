from __future__ import annotations

import os
from pathlib import Path


APP_NAME = "llm-model-loader"
DEFAULT_HOST = os.environ.get("LLM_MODEL_LOADER_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("LLM_MODEL_LOADER_PORT", "8174"))
AI_ROOT = Path(r"D:\Documents\AI")


def user_home() -> Path:
    return Path(os.environ.get("USERPROFILE") or Path.home())


def ai_root() -> Path:
    return AI_ROOT


def app_data_dir() -> Path:
    override = os.environ.get("LLM_MODEL_LOADER_DATA_DIR")
    if override:
        root = Path(override)
        root.mkdir(parents=True, exist_ok=True)
        return root
    base = os.environ.get("LOCALAPPDATA")
    if base:
        root = Path(base) / APP_NAME
    else:
        root = user_home() / ".llm-model-loader"
    try:
        root.mkdir(parents=True, exist_ok=True)
        return root
    except PermissionError:
        fallback = Path.cwd() / ".data"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def default_model_dir() -> Path:
    return ai_root() / "models"


def default_opencode_db_path() -> Path:
    """Return the normal OpenCode database location without creating it."""
    override = os.environ.get("OPENCODE_DB_PATH")
    if override:
        return Path(override).expanduser()

    candidates = [
        user_home() / ".local" / "share" / "opencode" / "opencode.db",
    ]
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "opencode" / "opencode.db")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DB_PATH = app_data_dir() / "loader.sqlite3"
