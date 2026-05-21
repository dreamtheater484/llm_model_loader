from __future__ import annotations

import os
from pathlib import Path

from .scripts import detect_quantization


def inspect_model_file(path: str) -> dict[str, object]:
    file_path = Path(path)
    size = file_path.stat().st_size if file_path.exists() else 0
    return {
        "path": str(file_path),
        "filename": file_path.name,
        "size_bytes": size,
        "quantization": detect_quantization(file_path.name),
        "format": "gguf" if file_path.suffix.lower() == ".gguf" else file_path.suffix.lower().lstrip("."),
        "exists": file_path.exists(),
        "modified_at": os.path.getmtime(file_path) if file_path.exists() else None,
    }

