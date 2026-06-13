from __future__ import annotations

import os
from pathlib import Path


APP_NAME = "llm-model-loader"
DEFAULT_HOST = os.environ.get("LLM_MODEL_LOADER_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("LLM_MODEL_LOADER_PORT", "8174"))


def user_home() -> Path:
    return Path(os.environ.get("USERPROFILE") or Path.home())


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
    return user_home() / "AI" / "models"


DB_PATH = app_data_dir() / "loader.sqlite3"
