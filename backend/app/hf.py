from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

from .scripts import detect_quantization, estimate_vram_mib


def _get_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "llm-model-loader/0.1"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _repo_url(repo_id: str, suffix: str = "") -> str:
    return f"https://huggingface.co/api/models/{urllib.parse.quote(repo_id, safe='/')}{suffix}"


def search_models(query: str, limit: int = 20) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "search": query,
            "limit": min(max(limit, 1), 50),
            "sort": "downloads",
            "direction": "-1",
        }
    )
    data = _get_json(f"https://huggingface.co/api/models?{params}")
    results = []
    for item in data:
        tags = item.get("tags") or []
        if query and "gguf" not in " ".join(tags).lower() and "gguf" not in item.get("modelId", "").lower():
            pass
        results.append(
            {
                "repo_id": item.get("modelId"),
                "author": item.get("author"),
                "downloads": item.get("downloads", 0),
                "likes": item.get("likes", 0),
                "tags": tags[:12],
                "private": item.get("private", False),
                "gated": item.get("gated", False),
            }
        )
    return results


def model_files(repo_id: str) -> list[dict[str, Any]]:
    try:
        tree = _get_json(_repo_url(repo_id, "/tree/main?recursive=true"))
        files = [_file_payload(item.get("path", ""), item) for item in tree if item.get("type") == "file" and item.get("path", "").lower().endswith(".gguf")]
        if files:
            return files
    except Exception:
        pass
    data = _get_json(_repo_url(repo_id))
    siblings = data.get("siblings") or []
    return [_file_payload(file.get("rfilename", ""), file) for file in siblings if file.get("rfilename", "").lower().endswith(".gguf")]


def _file_payload(filename: str, metadata: dict[str, Any]) -> dict[str, Any]:
    lfs = metadata.get("lfs") or {}
    size = metadata.get("size") or lfs.get("size")
    quantization = detect_quantization(filename)
    return {
        "filename": filename,
        "size_bytes": size,
        "quantization": quantization,
        "estimated_vram_mib": estimate_vram_mib(size, None) if size else None,
        "lfs": lfs,
    }
