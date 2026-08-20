"""Register the pinned NInfer + Qwen3.8-27B NVFP4 preset after the WSL setup step."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .config import app_data_dir
from .scripts import detect_quantization, parse_script
from .storage import Store, new_id, normalize_path, now, store


PRESET_NAME = "Qwen3.8-27B-NVFP4 (NInfer) / Vision / MTP3 / INT8 KV / C3 / 160kctx+"
PRESET_VRAM_MIB = 30720
MODEL_NAME = "Qwen3.8-27B-NVFP4"
MODEL_REPO = "neroued/Qwen3.8-27B-nvfp4-NInfer"
MODEL_FILENAME = "qwen3_8_27b_nvfp4.ninfer"
MODEL_ID = "qwen3.8-27b"
NINFER_INFO_FILE = "ninfer-wsl.json"


def ninfer_info_path() -> Path:
    return app_data_dir() / NINFER_INFO_FILE


def load_ninfer_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing NInfer facts file: {path}. Run setup_ninfer_qwen38.cmd first."
        )
    info = json.loads(path.read_text(encoding="utf-8-sig"))
    required = {"distro", "home", "launcher_path", "model_path", "model_size_bytes"}
    missing = required - set(info)
    if missing:
        raise ValueError(f"NInfer facts file is missing keys: {sorted(missing)}")
    return info


def model_unc_path(info: dict[str, Any]) -> str:
    distro = info["distro"]
    relative = info["model_path"].lstrip("/").replace("/", "\\")
    return rf"\\wsl.localhost\{distro}\{relative}"


def preset_script(info: dict[str, Any]) -> str:
    distro = info["distro"]
    launcher = info["launcher_path"]
    port = int(info.get("port") or 8080)
    concurrency = int(info.get("concurrency") or 3)
    min_context = int(info.get("min_context") or 163840)
    model_file = Path(info["model_path"]).name
    # NOTE: NINFER_MAX_CONTEXT is deliberately NOT pinned here. The launcher
    # only runs its startup ladder (262144 -> MIN_CONTEXT) when NINFER_MAX_CONTEXT
    # is unset; pinning it would collapse the ladder to a single attempt.
    env = " ".join(
        [
            f"NINFER_PORT={port}",
            f"NINFER_CONCURRENCY={concurrency}",
            f"NINFER_MIN_CONTEXT={min_context}",
            f"NINFER_MODEL_FILE={model_file}",
        ]
    )
    return f'& wsl.exe -d "{distro}" -- bash -lc \'{env} {launcher} --model-id {MODEL_ID}\''


def register_ninfer_model(info: dict[str, Any], target_store: Store = store) -> dict[str, Any]:
    unc_path = model_unc_path(info)
    normalized = normalize_path(unc_path)
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
            ) values(?, ?, ?, ?, ?, ?, ?, ?, 'ninfer', 0, ?, ?, ?)
            """,
            (
                model_id,
                MODEL_NAME,
                MODEL_REPO,
                Path(info["model_path"]).name,
                unc_path,
                normalized,
                int(info["model_size_bytes"]),
                detect_quantization(None, None, MODEL_FILENAME, MODEL_FILENAME) or "NVFP4",
                PRESET_VRAM_MIB,
                int((order_row or {}).get("next_order") or 0),
                now(),
            ),
        )

    raw_script = preset_script(info)
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
        "model_path": unc_path,
        "wsl_distro": info["distro"],
        "wsl_launcher": info["launcher_path"],
        "port": parsed.get("port"),
        "context": parsed.get("ctx_size"),
        "concurrency": parsed.get("concurrency"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Register the NInfer Qwen3.8-27B NVFP4 loading preset.")
    parser.add_argument("--info", default=None, help="Path to ninfer-wsl.json (defaults to the app data dir).")
    args = parser.parse_args()
    path = Path(args.info) if args.info else ninfer_info_path()
    info = load_ninfer_info(path)
    print(json.dumps(register_ninfer_model(info)))


if __name__ == "__main__":
    main()
