from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

from .scripts import detect_quantization, estimate_vram_mib
from .shards import parse_gguf_shard


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
            return _group_shards(files)
    except Exception:
        pass
    data = _get_json(_repo_url(repo_id))
    siblings = data.get("siblings") or []
    files = [_file_payload(file.get("rfilename", ""), file) for file in siblings if file.get("rfilename", "").lower().endswith(".gguf")]
    return _group_shards(files)


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


def _group_shards(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    groups: dict[tuple[str, int], list[tuple[int, dict[str, Any]]]] = {}
    group_positions: dict[tuple[str, int], int] = {}
    for file in files:
        shard = parse_gguf_shard(file["filename"])
        if not shard:
            results.append({**file, "filenames": [file["filename"]], "shard_count": 1, "complete": True})
            continue
        key = (shard.base.lower(), shard.count)
        if key not in groups:
            group_positions[key] = len(results)
            results.append({})
            groups[key] = []
        groups[key].append((shard.index, file))

    for key, shards in groups.items():
        ordered = sorted(shards, key=lambda item: item[0])
        primary = next((file for index, file in ordered if index == 1), ordered[0][1])
        primary_shard = parse_gguf_shard(primary["filename"])
        expected_count = key[1]
        sizes = [file.get("size_bytes") for _, file in ordered]
        total_size = sum(size for size in sizes if size is not None) if all(size is not None for size in sizes) else None
        results[group_positions[key]] = {
            **primary,
            "display_name": f"{primary_shard.base}.gguf" if primary_shard else primary["filename"],
            "filenames": [file["filename"] for _, file in ordered],
            "shard_count": expected_count,
            "complete": len(ordered) == expected_count and [index for index, _ in ordered] == list(range(1, expected_count + 1)),
            "size_bytes": total_size,
            "estimated_vram_mib": estimate_vram_mib(total_size, None) if total_size else None,
        }
    return results
