from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from .config import DB_PATH, default_model_dir


class Store:
    def __init__(self, path: Path = DB_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists settings (
                    key text primary key,
                    value text not null
                );
                create table if not exists models (
                    id text primary key,
                    name text not null,
                    repo_id text,
                    filename text,
                    path text not null,
                    normalized_path text,
                    size_bytes integer,
                    quantization text,
                    source text not null,
                    managed integer not null default 1,
                    manual_vram_mib integer,
                    display_order integer,
                    created_at real not null
                );
                create table if not exists downloads (
                    id text primary key,
                    repo_id text not null,
                    filename text not null,
                    url text not null,
                    target_path text not null,
                    status text not null,
                    bytes_done integer not null default 0,
                    bytes_total integer,
                    started_at real,
                    finished_at real,
                    error text
                );
                create table if not exists scripts (
                    id text primary key,
                    model_id text not null,
                    name text not null,
                    raw_script text not null,
                    parsed_json text not null,
                    estimated_vram_mib integer,
                    is_favorite integer not null default 0,
                    created_at real not null,
                    updated_at real not null,
                    foreign key(model_id) references models(id) on delete cascade
                );
                create table if not exists runs (
                    id text primary key,
                    script_id text not null,
                    model_id text not null,
                    pid integer,
                    status text not null,
                    status_message text,
                    host text,
                    port integer,
                    started_at real not null,
                    ended_at real,
                    load_seconds real,
                    log_tail text not null default '',
                    error text,
                    last_heartbeat_at real
                );
                create table if not exists benchmark_presets (
                    id text primary key,
                    name text not null,
                    prompt text not null,
                    prompt_tokens integer not null,
                    output_tokens integer not null
                );
                create table if not exists benchmark_runs (
                    id text primary key,
                    script_id text not null,
                    model_id text not null,
                    preset_id text,
                    status text not null,
                    prompt text not null,
                    output_tokens integer not null,
                    fttt_ms real,
                    prefill_tps real,
                    generation_tps real,
                    average_tps real,
                    duration_seconds real,
                    raw_log text not null default '',
                    started_at real not null,
                    ended_at real,
                    error text
                );
                """
            )
            self._migrate(conn)
            self._seed(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        columns = {row["name"] for row in conn.execute("pragma table_info(models)").fetchall()}
        if "normalized_path" not in columns:
            conn.execute("alter table models add column normalized_path text")
        if "display_order" not in columns:
            conn.execute("alter table models add column display_order integer")
        for row in conn.execute("select id, path from models where normalized_path is null or normalized_path = ''").fetchall():
            conn.execute(
                "update models set normalized_path = ? where id = ?",
                (normalize_path(row["path"]), row["id"]),
            )
        for index, row in enumerate(
            conn.execute(
                """
                select id from models
                where display_order is null
                order by created_at desc, id asc
                """
            ).fetchall()
        ):
            conn.execute("update models set display_order = ? where id = ?", (index, row["id"]))
        self._dedupe_models(conn)
        conn.execute("create unique index if not exists idx_models_normalized_path on models(normalized_path)")
        run_columns = {row["name"] for row in conn.execute("pragma table_info(runs)").fetchall()}
        if "status_message" not in run_columns:
            conn.execute("alter table runs add column status_message text")
        if "last_heartbeat_at" not in run_columns:
            conn.execute("alter table runs add column last_heartbeat_at real")
        script_columns = {row["name"] for row in conn.execute("pragma table_info(scripts)").fetchall()}
        if "is_favorite" not in script_columns:
            conn.execute("alter table scripts add column is_favorite integer not null default 0")

    def _dedupe_models(self, conn: sqlite3.Connection) -> None:
        duplicate_groups = conn.execute(
            """
            select normalized_path from models
            where normalized_path is not null and normalized_path != ''
            group by normalized_path
            having count(*) > 1
            """
        ).fetchall()
        for group in duplicate_groups:
            rows = conn.execute(
                "select id from models where normalized_path = ? order by created_at asc, id asc",
                (group["normalized_path"],),
            ).fetchall()
            canonical = rows[0]["id"]
            duplicates = [row["id"] for row in rows[1:]]
            for duplicate in duplicates:
                conn.execute("update scripts set model_id = ? where model_id = ?", (canonical, duplicate))
                conn.execute("update runs set model_id = ? where model_id = ?", (canonical, duplicate))
                conn.execute("update benchmark_runs set model_id = ? where model_id = ?", (canonical, duplicate))
                conn.execute("delete from models where id = ?", (duplicate,))

    def _seed(self, conn: sqlite3.Connection) -> None:
        defaults = {
            "model_dir": str(default_model_dir()),
            "llama_server_path": "",
        }
        for key, value in defaults.items():
            conn.execute("insert or ignore into settings(key, value) values(?, ?)", (key, value))
        presets = [
            ("small", "Small", "Write a precise two-sentence summary of why local LLM benchmarking matters.", 20, 100),
            ("medium", "Medium", "Analyze the tradeoffs between latency, throughput, context length, and quantization for a local assistant. Use practical examples.", 500, 700),
            ("large", "Large", "You are evaluating a local language model server. Discuss startup time, prompt processing, generation speed, memory pressure, cache behavior, and operator ergonomics in detail.", 2000, 2500),
            ("extra-large", "Extra Large", "Produce a rigorous long-form technical assessment of a local LLM deployment pipeline. Cover hardware selection, VRAM planning, model download management, process lifecycle, logging, benchmarking methodology, safety checks, and operational failure modes.", 4000, 5000),
        ]
        for preset in presets:
            conn.execute(
                "insert or ignore into benchmark_presets(id, name, prompt, prompt_tokens, output_tokens) values(?, ?, ?, ?, ?)",
                preset,
            )

    def rows(self, query: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(query, tuple(params)).fetchall()]

    def row(self, query: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(query, tuple(params)).fetchone()
            return dict(row) if row else None

    def execute(self, query: str, params: Iterable[Any] = ()) -> None:
        with self.connect() as conn:
            conn.execute(query, tuple(params))

    def setting(self, key: str) -> str:
        row = self.row("select value from settings where key = ?", (key,))
        return row["value"] if row else ""

    def set_setting(self, key: str, value: str) -> None:
        self.execute("insert into settings(key, value) values(?, ?) on conflict(key) do update set value=excluded.value", (key, value))


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def normalize_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.expanduser(path)))


def now() -> float:
    return time.time()


def decode_json_field(row: dict[str, Any], key: str) -> dict[str, Any]:
    if key in row and isinstance(row[key], str):
        row[key] = json.loads(row[key])
    return row


store = Store()
