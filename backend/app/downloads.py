from __future__ import annotations

import shutil
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .events import event_hub
from .gguf import inspect_model_file
from .scripts import detect_quantization
from .shards import model_name_from_filename
from .storage import new_id, normalize_path, now, store


class DownloadManager:
    def __init__(self) -> None:
        self._cancel: set[str] = set()
        self._active: set[str] = set()
        self._lock = threading.Lock()
        self._registration_lock = threading.Lock()

    def start(self, repo_id: str, filename: str, target_dir: str | None = None) -> dict[str, Any]:
        download = self._queue(repo_id, filename, target_dir)
        self._start_thread(download["id"])
        return self.get(download["id"]) or download

    def start_group(self, repo_id: str, filenames: list[str], primary_filename: str, target_dir: str | None = None) -> list[dict[str, Any]]:
        unique_filenames = list(dict.fromkeys(filenames))
        if not unique_filenames or primary_filename not in unique_filenames:
            raise ValueError("A download group must include its primary file.")
        group_id = new_id("group")
        downloads = [
            self._queue(repo_id, filename, target_dir, group_id, filename == primary_filename)
            for filename in unique_filenames
        ]
        for download in downloads:
            self._start_thread(download["id"])
        return [self.get(download["id"]) or download for download in downloads]

    def _queue(
        self,
        repo_id: str,
        filename: str,
        target_dir: str | None = None,
        group_id: str | None = None,
        group_primary: bool = True,
    ) -> dict[str, Any]:
        model_dir = Path(target_dir or store.setting("model_dir"))
        model_dir.mkdir(parents=True, exist_ok=True)
        safe_name = filename.replace("/", "__").replace("\\", "__")
        target_path = model_dir / safe_name
        encoded = urllib.parse.quote(filename)
        url = f"https://huggingface.co/{repo_id}/resolve/main/{encoded}"
        download_id = new_id("dl")
        store.execute(
            """
            insert into downloads(id, repo_id, filename, url, target_path, status, started_at, group_id, group_primary)
            values(?, ?, ?, ?, ?, 'queued', ?, ?, ?)
            """,
            (download_id, repo_id, filename, url, str(target_path), now(), group_id, int(group_primary)),
        )
        return self.get(download_id) or {"id": download_id}

    def cancel(self, download_id: str) -> None:
        with self._lock:
            self._cancel.add(download_id)

    def get(self, download_id: str) -> dict[str, Any] | None:
        return store.row("select * from downloads where id = ?", (download_id,))

    def list(self) -> list[dict[str, Any]]:
        self.resume_incomplete()
        return store.rows("select * from downloads order by started_at desc")

    def resume(self, download_id: str) -> dict[str, Any]:
        row = self.get(download_id)
        if not row:
            raise ValueError("Download not found.")
        if row["status"] == "completed":
            return row
        with self._lock:
            self._cancel.discard(download_id)
        self._start_thread(download_id)
        return self.get(download_id) or {"id": download_id}

    def resume_incomplete(self) -> None:
        rows = store.rows("select id from downloads where status in ('queued','running','retrying')")
        for row in rows:
            self._start_thread(row["id"])

    def _start_thread(self, download_id: str) -> bool:
        with self._lock:
            if download_id in self._active:
                return False
            self._active.add(download_id)
            self._cancel.discard(download_id)
        thread = threading.Thread(target=self._download, args=(download_id,), daemon=True)
        thread.start()
        return True

    def _is_cancelled(self, download_id: str) -> bool:
        with self._lock:
            return download_id in self._cancel

    def _download(self, download_id: str) -> None:
        try:
            self._download_with_retries(download_id)
        finally:
            with self._lock:
                self._active.discard(download_id)
                self._cancel.discard(download_id)

    def _download_with_retries(self, download_id: str) -> None:
        failures = 0
        while True:
            row = self.get(download_id)
            if not row:
                return
            if row["status"] in {"completed", "cancelled"}:
                return
            if self._is_cancelled(download_id):
                store.execute("update downloads set status='cancelled', error=null where id=?", (download_id,))
                event_hub.publish_threadsafe("download", self.get(download_id) or {})
                return
            try:
                if self._download_once(download_id):
                    return
                failures = 0
            except Exception as exc:
                failures += 1
                if self._is_cancelled(download_id):
                    store.execute("update downloads set status='cancelled', error=null where id=?", (download_id,))
                    event_hub.publish_threadsafe("download", self.get(download_id) or {})
                    return
                if failures > 20:
                    store.execute("update downloads set status='paused', error=? where id=?", (f"Paused after repeated download errors: {exc}", download_id))
                    event_hub.publish_threadsafe("download", self.get(download_id) or {})
                    return
                delay = min(60, 2 ** min(failures, 5))
                store.execute(
                    "update downloads set status='retrying', error=? where id=?",
                    (f"{exc}. Retrying in {delay}s.", download_id),
                )
                event_hub.publish_threadsafe("download", self.get(download_id) or {})
                for _ in range(delay):
                    if self._is_cancelled(download_id):
                        store.execute("update downloads set status='cancelled', error=null where id=?", (download_id,))
                        event_hub.publish_threadsafe("download", self.get(download_id) or {})
                        return
                    time.sleep(1)

    def _download_once(self, download_id: str) -> bool:
        row = self.get(download_id)
        if not row:
            return True
        target = Path(row["target_path"])
        part = target.with_suffix(target.suffix + ".part")
        bytes_done = part.stat().st_size if part.exists() else 0
        if target.exists() and row.get("bytes_total") and target.stat().st_size >= int(row["bytes_total"]):
            self._finalize(download_id, target, target.stat().st_size)
            return True
        headers = {}
        if bytes_done:
            headers["Range"] = f"bytes={bytes_done}-"
        store.execute("update downloads set status='running', bytes_done=?, error=null where id=?", (bytes_done, download_id))
        event_hub.publish_threadsafe("download", self.get(download_id) or {})
        request = urllib.request.Request(row["url"], headers={**headers, "User-Agent": "llm-model-loader/0.1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) + bytes_done if total_header else row.get("bytes_total")
            if bytes_done and response.status == 200:
                bytes_done = 0
                part.unlink(missing_ok=True)
            mode = "ab" if bytes_done else "wb"
            started = time.time()
            last_event = 0.0
            with part.open(mode, buffering=0) as handle:
                while True:
                    if self._is_cancelled(download_id):
                        store.execute("update downloads set status='cancelled', error=null where id=?", (download_id,))
                        event_hub.publish_threadsafe("download", self.get(download_id) or {})
                        return True
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    bytes_done += len(chunk)
                    elapsed = max(time.time() - started, 0.001)
                    store.execute(
                        "update downloads set bytes_done=?, bytes_total=? where id=?",
                        (bytes_done, total, download_id),
                    )
                    current = time.time()
                    if current - last_event >= 0.5:
                        payload = {
                            **(self.get(download_id) or {}),
                            "bytes_done": bytes_done,
                            "bytes_total": total,
                            "speed_bps": max(0, bytes_done - (row.get("bytes_done") or 0)) / elapsed,
                        }
                        event_hub.publish_threadsafe("download", payload)
                        last_event = current
        if total and bytes_done < total:
            raise RuntimeError(f"Connection closed early at {bytes_done} of {total} bytes")
        shutil.move(str(part), str(target))
        self._finalize(download_id, target, bytes_done)
        return True

    def _finalize(self, download_id: str, target: Path, bytes_done: int) -> None:
        row = self.get(download_id)
        if not row:
            return
        store.execute(
            "update downloads set status='completed', bytes_done=?, bytes_total=?, finished_at=?, error=null where id=?",
            (bytes_done, bytes_done, now(), download_id),
        )
        if row.get("group_id"):
            self._register_completed_group(row["group_id"])
        else:
            self._register_model(row, target, bytes_done)
        event_hub.publish_threadsafe("download", self.get(download_id) or {})

    def _register_completed_group(self, group_id: str) -> None:
        with self._registration_lock:
            rows = store.rows("select * from downloads where group_id=?", (group_id,))
            if not rows or any(row["status"] != "completed" for row in rows):
                return
            primary = next((row for row in rows if row.get("group_primary")), None)
            if not primary:
                return
            total_size = sum(int(row.get("bytes_done") or 0) for row in rows)
            self._register_model(primary, Path(primary["target_path"]), total_size)

    def _register_model(self, row: dict[str, Any], target: Path, size_bytes: int) -> None:
        meta = inspect_model_file(str(target))
        normalized_path = normalize_path(str(target))
        existing = store.row("select id from models where normalized_path = ?", (normalized_path,))
        if not existing:
            model_id = new_id("model")
            store.execute(
                """
                insert into models(id, name, repo_id, filename, path, normalized_path, size_bytes, quantization, source, managed, created_at)
                values(?, ?, ?, ?, ?, ?, ?, ?, 'huggingface', 1, ?)
                """,
                (
                    model_id,
                    model_name_from_filename(row["filename"]),
                    row["repo_id"],
                    row["filename"],
                    str(target),
                    normalized_path,
                    size_bytes,
                    meta["quantization"] or detect_quantization(row["filename"]),
                    now(),
                ),
            )


download_manager = DownloadManager()
