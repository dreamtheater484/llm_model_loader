from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .benchmarks import benchmark_manager
from .config import DEFAULT_HOST, DEFAULT_PORT
from .downloads import download_manager
from .events import event_hub
from .files import browse_files
from .gguf import inspect_model_file
from .gpu import query_gpus
from .hf import model_files, search_models
from .llamacpp import discover_llama_server, resolve_llama_server_path, validate_llama_server
from .runs import run_manager
from .scripts import autosuggest_name, estimate_vram_mib, is_fit_managed, parse_script
from .shards import local_model_files, parse_gguf_shard
from .storage import decode_json_field, new_id, normalize_path, now, store
from .system import telemetry


app = FastAPI(title="LLM Model Loader", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SettingsIn(BaseModel):
    llama_server_path: str | None = None
    model_dir: str | None = None


class DownloadIn(BaseModel):
    repo_id: str
    filename: str
    filenames: list[str] | None = None
    target_dir: str | None = None


class ImportModelIn(BaseModel):
    path: str
    name: str | None = None
    copy_to_managed: bool = True
    manual_vram_mib: int | None = None


class ScriptIn(BaseModel):
    name: str | None = None
    raw_script: str
    estimated_vram_mib: int | None = None


class ScriptFavoriteIn(BaseModel):
    is_favorite: bool


class StartRunIn(BaseModel):
    script_id: str
    manual_vram_mib: int | None = None


class BenchmarkIn(BaseModel):
    script_id: str
    preset_id: str
    prompt: str | None = None
    output_tokens: int | None = None


class ModelOrderIn(BaseModel):
    model_ids: list[str]


def _model_rows() -> list[dict[str, Any]]:
    models = store.rows("select * from models order by display_order asc, created_at desc")
    for model in models:
        shard = parse_gguf_shard(model.get("filename") or Path(model["path"]).name)
        model["shard_count"] = shard.count if shard else 1
        scripts = []
        for row in store.rows("select * from scripts where model_id=? order by created_at desc", (model["id"],)):
            parsed = parse_script(row["raw_script"]).to_dict()
            row["parsed_json"] = parsed
            if parsed.get("n_cpu_moe") or is_fit_managed(parsed):
                row["estimated_vram_mib"] = None
            scripts.append(row)
        model["scripts"] = scripts
    return models


@app.on_event("startup")
async def startup() -> None:
    event_hub.bind_loop()
    download_manager.resume_incomplete()


@app.get("/api/settings")
def get_settings() -> dict[str, str]:
    llama_server_path = resolve_llama_server_path(store.setting("llama_server_path"))
    if llama_server_path and validate_llama_server(llama_server_path):
        store.set_setting("llama_server_path", llama_server_path)
    else:
        discovered = discover_llama_server()
        llama_server_path = discovered["selected"]
        if llama_server_path:
            store.set_setting("llama_server_path", llama_server_path)
    return {
        "llama_server_path": llama_server_path,
        "model_dir": store.setting("model_dir"),
        "host": DEFAULT_HOST,
        "port": str(DEFAULT_PORT),
    }


@app.patch("/api/settings")
def update_settings(body: SettingsIn) -> dict[str, str]:
    if body.llama_server_path is not None:
        resolved = resolve_llama_server_path(body.llama_server_path)
        if resolved and not validate_llama_server(resolved):
            raise HTTPException(status_code=400, detail="Selected path is not a valid llama-server executable.")
        store.set_setting("llama_server_path", resolved)
    if body.model_dir is not None:
        Path(body.model_dir).mkdir(parents=True, exist_ok=True)
        store.set_setting("model_dir", body.model_dir)
    return get_settings()


@app.get("/api/system/gpus")
def gpus() -> list[dict[str, Any]]:
    return query_gpus()


@app.get("/api/system/telemetry")
def system_telemetry() -> dict[str, Any]:
    loaded, loading = run_manager.active_counts()
    return telemetry(loaded, loading)


@app.get("/api/hf/search")
def hf_search(q: str, limit: int = 20) -> list[dict[str, Any]]:
    try:
        return search_models(q, limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/hf/models/{repo_id:path}/files")
def hf_files(repo_id: str) -> list[dict[str, Any]]:
    try:
        return model_files(repo_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/api/downloads")
def create_download(body: DownloadIn) -> dict[str, Any]:
    if body.filenames and len(body.filenames) > 1:
        try:
            downloads = download_manager.start_group(body.repo_id, body.filenames, body.filename, body.target_dir)
            return {"downloads": downloads}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    return download_manager.start(body.repo_id, body.filename, body.target_dir)


@app.get("/api/downloads")
def list_downloads() -> list[dict[str, Any]]:
    return download_manager.list()


@app.post("/api/downloads/{download_id}/cancel")
def cancel_download(download_id: str) -> dict[str, Any]:
    download_manager.cancel(download_id)
    return {"ok": True}


@app.post("/api/downloads/{download_id}/resume")
def resume_download(download_id: str) -> dict[str, Any]:
    try:
        return download_manager.resume(download_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/models")
def list_models() -> list[dict[str, Any]]:
    return _model_rows()


@app.patch("/api/models/order")
def update_model_order(body: ModelOrderIn) -> dict[str, Any]:
    current_ids = [row["id"] for row in store.rows("select id from models order by display_order asc, created_at desc")]
    requested = body.model_ids
    if len(requested) != len(set(requested)):
        raise HTTPException(status_code=400, detail="Model order contains duplicate IDs.")
    if set(requested) != set(current_ids):
        raise HTTPException(status_code=400, detail="Model order must contain every current model exactly once.")
    for index, model_id in enumerate(requested):
        store.execute("update models set display_order=? where id=?", (index, model_id))
    return {"ok": True, "model_ids": requested}


@app.get("/api/files/browse")
def browse_local_files(path: str | None = None, gguf_only: bool = True, executable_only: bool = False) -> dict[str, Any]:
    return browse_files(path, gguf_only, executable_only)


@app.get("/api/llamacpp/discover")
def discover_llamacpp() -> dict[str, Any]:
    discovered = discover_llama_server()
    if discovered["selected"]:
        store.set_setting("llama_server_path", discovered["selected"])
    return discovered


@app.post("/api/models")
def import_model(body: ImportModelIn) -> dict[str, Any]:
    source = Path(body.path)
    if not source.exists() or not source.is_file():
        raise HTTPException(status_code=400, detail="Model file does not exist.")
    target = source
    managed = int(body.copy_to_managed)
    if body.copy_to_managed:
        model_dir = Path(store.setting("model_dir"))
        model_dir.mkdir(parents=True, exist_ok=True)
        target = model_dir / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    normalized_path = normalize_path(str(target))
    existing = store.row("select * from models where normalized_path = ?", (normalized_path,))
    if existing:
        existing["already_exists"] = True
        return existing
    meta = inspect_model_file(str(target))
    model_id = new_id("model")
    order_row = store.row("select coalesce(max(display_order), -1) + 1 as next_order from models")
    display_order = int((order_row or {}).get("next_order") or 0)
    store.execute(
        """
        insert into models(id, name, filename, path, normalized_path, size_bytes, quantization, source, managed, manual_vram_mib, display_order, created_at)
        values(?, ?, ?, ?, ?, ?, ?, 'import', ?, ?, ?, ?)
        """,
        (
            model_id,
            body.name or Path(str(meta["filename"])).stem,
            meta["filename"],
            meta["path"],
            normalized_path,
            meta["size_bytes"],
            meta["quantization"],
            managed,
            body.manual_vram_mib,
            display_order,
            now(),
        ),
    )
    return store.row("select * from models where id=?", (model_id,)) or {"id": model_id}


@app.delete("/api/models/{model_id}")
def delete_model(model_id: str, force: bool = False) -> dict[str, Any]:
    model = store.row("select * from models where id=?", (model_id,))
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")
    if model["managed"] or force:
        managed_root = Path(store.setting("model_dir")).resolve()
        paths = [path for path in local_model_files(model["path"]) if path.exists()]
        for path in paths:
            resolved = path.resolve()
            if not force and managed_root not in resolved.parents and resolved != managed_root:
                raise HTTPException(status_code=400, detail="Refusing to delete a file outside the managed model directory.")
        for path in paths:
            path.unlink()
    store.execute("delete from models where id=?", (model_id,))
    return {"ok": True}


@app.post("/api/models/{model_id}/scripts")
def create_script(model_id: str, body: ScriptIn) -> dict[str, Any]:
    model = store.row("select * from models where id=?", (model_id,))
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")
    parsed = parse_script(body.raw_script).to_dict()
    estimate = None if is_fit_managed(parsed) else body.estimated_vram_mib or estimate_vram_mib(model.get("size_bytes"), parsed.get("ctx_size"), n_cpu_moe=parsed.get("n_cpu_moe"))
    script_id = new_id("script")
    name = body.name or autosuggest_name(model["name"], body.raw_script)
    store.execute(
        """
        insert into scripts(id, model_id, name, raw_script, parsed_json, estimated_vram_mib, created_at, updated_at)
        values(?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (script_id, model_id, name, body.raw_script, json.dumps(parsed), estimate, now(), now()),
    )
    return decode_json_field(store.row("select * from scripts where id=?", (script_id,)) or {"id": script_id}, "parsed_json")


@app.patch("/api/models/{model_id}/scripts/{script_id}")
def update_script(model_id: str, script_id: str, body: ScriptIn) -> dict[str, Any]:
    script = store.row("select * from scripts where id=? and model_id=?", (script_id, model_id))
    if not script:
        raise HTTPException(status_code=404, detail="Script not found.")
    model = store.row("select * from models where id=?", (model_id,))
    parsed = parse_script(body.raw_script).to_dict()
    estimate = None if is_fit_managed(parsed) else body.estimated_vram_mib or estimate_vram_mib((model or {}).get("size_bytes"), parsed.get("ctx_size"), n_cpu_moe=parsed.get("n_cpu_moe"))
    store.execute(
        "update scripts set name=?, raw_script=?, parsed_json=?, estimated_vram_mib=?, updated_at=? where id=?",
        (body.name or autosuggest_name((model or {}).get("name", ""), body.raw_script), body.raw_script, json.dumps(parsed), estimate, now(), script_id),
    )
    return decode_json_field(store.row("select * from scripts where id=?", (script_id,)) or {"id": script_id}, "parsed_json")


@app.patch("/api/models/{model_id}/scripts/{script_id}/favorite")
def update_script_favorite(model_id: str, script_id: str, body: ScriptFavoriteIn) -> dict[str, Any]:
    script = store.row("select * from scripts where id=? and model_id=?", (script_id, model_id))
    if not script:
        raise HTTPException(status_code=404, detail="Script not found.")
    store.execute(
        "update scripts set is_favorite=?, updated_at=? where id=?",
        (int(body.is_favorite), now(), script_id),
    )
    return store.row("select * from scripts where id=?", (script_id,)) or {"id": script_id}


@app.delete("/api/models/{model_id}/scripts/{script_id}")
def delete_script(model_id: str, script_id: str) -> dict[str, Any]:
    store.execute("delete from scripts where id=? and model_id=?", (script_id, model_id))
    return {"ok": True}


@app.post("/api/runs/start")
def start_run(body: StartRunIn) -> dict[str, Any]:
    try:
        return run_manager.start(body.script_id, body.manual_vram_mib)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/runs/validate")
def validate_run(body: StartRunIn) -> dict[str, Any]:
    try:
        return run_manager.validate_start(body.script_id, body.manual_vram_mib)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    return run_manager.list()


@app.post("/api/runs/{run_id}/abort")
def abort_run(run_id: str) -> dict[str, Any]:
    return run_manager.stop(run_id, "aborted")


@app.post("/api/runs/{run_id}/unload")
def unload_run(run_id: str) -> dict[str, Any]:
    return run_manager.stop(run_id, "unloaded")


@app.delete("/api/runs/history")
def delete_run_history() -> dict[str, int]:
    return run_manager.delete_history()


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str) -> dict[str, bool]:
    try:
        return run_manager.delete(run_id)
    except ValueError as exc:
        detail = str(exc)
        status = 404 if detail == "Run not found." else 400
        raise HTTPException(status_code=status, detail=detail)


@app.get("/api/benchmarks")
def benchmark_presets() -> list[dict[str, Any]]:
    return benchmark_manager.presets()


@app.post("/api/benchmarks")
def start_benchmark(body: BenchmarkIn) -> dict[str, Any]:
    try:
        return benchmark_manager.start(body.script_id, body.preset_id, body.prompt, body.output_tokens)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/benchmarks/history")
def benchmark_history(
    model_id: str | None = None,
    script_id: str | None = None,
    preset_id: str | None = None,
    active_only: bool = False,
    limit: int = 7,
    offset: int = 0,
) -> dict[str, Any]:
    return benchmark_manager.history(model_id, script_id, preset_id, active_only, limit, offset)


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket) -> None:
    await event_hub.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        event_hub.disconnect(websocket)


dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if dist.exists():
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str) -> FileResponse:
        target = dist / path
        if target.exists() and target.is_file():
            return FileResponse(target)
        return FileResponse(dist / "index.html")
