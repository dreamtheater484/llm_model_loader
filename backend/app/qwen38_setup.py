"""Register the pinned Qwen3.8-27B NVFP4-MTP preset after the Windows setup step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .gguf import inspect_model_file
from .scripts import parse_script
from .storage import Store, new_id, normalize_path, now, store


PRESET_NAME = "Qwen3.8-27B-NVFP4-MTP / 160kctx x2 / Full CUDA / GPU KV / MTP / Vision"
PRESET_VRAM_MIB = 30720


def preset_script(model_path: str, runtime_path: str, mmproj_path: str) -> str:
    return f'''& "{runtime_path}" `
  -m "{model_path}" `
  --alias "Qwen3.8-27B-NVFP4-MTP" `
  --mmproj "{mmproj_path}" `
  --mmproj-offload `
  --image-max-tokens 1024 `
  --host 0.0.0.0 `
  --port 8080 `
  -c 163840 `
  -n 32768 `
  -np 2 `
  -ngl 999 `
  -fa on `
  -ctk q4_0 `
  -ctv q4_0 `
  -b 1024 `
  -ub 256 `
  -t 10 `
  -tb 12 `
  --reasoning on `
  --reasoning-budget 8192 `
  --chat-template-kwargs {{"enable_thinking":true}} `
  --spec-type draft-mtp `
  --spec-draft-n-max 3 `
  --spec-draft-p-min 0.0 `
  --temp 0.6 `
  --top-p 0.95 `
  --top-k 20 `
  --min-p 0.0 `
  --metrics `
  --perf `
  --log-verbosity 1'''


def register_qwen38_model(
    model_path: str,
    runtime_path: str,
    mmproj_path: str,
    target_store: Store = store,
) -> dict[str, Any]:
    model_file = Path(model_path).resolve()
    runtime_file = Path(runtime_path).resolve()
    mmproj_file = Path(mmproj_path).resolve()
    meta = inspect_model_file(str(model_file))
    normalized = normalize_path(str(model_file))
    existing = target_store.row("select * from models where normalized_path=?", (normalized,))
    if existing:
        model_id = existing["id"]
    else:
        order_row = target_store.row("select coalesce(max(display_order), -1) + 1 as next_order from models")
        model_id = new_id("model")
        target_store.execute(
            """
            insert into models(
                id, name, repo_id, filename, path, normalized_path, size_bytes,
                quantization, source, managed, manual_vram_mib, display_order, created_at
            ) values(?, ?, ?, ?, ?, ?, ?, ?, 'huggingface', 1, null, ?, ?)
            """,
            (
                model_id,
                "Qwen3.8-27B-NVFP4-MTP",
                "felippeburk/Qwen3.8-27B-NVFP4-MTP-GGUF",
                model_file.name,
                str(model_file),
                normalized,
                meta["size_bytes"],
                meta["quantization"],
                int((order_row or {}).get("next_order") or 0),
                now(),
            ),
        )

    raw_script = preset_script(str(model_file), str(runtime_file), str(mmproj_file))
    parsed = parse_script(raw_script).to_dict()
    script = target_store.row(
        "select id, raw_script from scripts where model_id=? and name=?",
        (model_id, PRESET_NAME),
    )
    if not script:
        script_id = new_id("script")
        timestamp = now()
        target_store.execute(
            """
            insert into scripts(
                id, model_id, name, raw_script, parsed_json, estimated_vram_mib,
                created_at, updated_at
            ) values(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (script_id, model_id, PRESET_NAME, raw_script, json.dumps(parsed), PRESET_VRAM_MIB, timestamp, timestamp),
        )
    else:
        script_id = script["id"]
        if script["raw_script"] != raw_script:
            target_store.execute(
                "update scripts set raw_script=?, parsed_json=?, updated_at=? where id=?",
                (raw_script, json.dumps(parsed), now(), script_id),
            )
    return {
        "model_id": model_id,
        "script_id": script_id,
        "model_path": str(model_file),
        "runtime_path": str(runtime_file),
        "mmproj_path": str(mmproj_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Register the Qwen3.8-27B NVFP4-MTP loading preset.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--runtime-path", required=True)
    parser.add_argument("--mmproj-path", required=True)
    args = parser.parse_args()
    result = register_qwen38_model(args.model_path, args.runtime_path, args.mmproj_path)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
