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
from .storage import new_id, normalize_path, now, store


class DownloadManager:
    def __init__(self) -> None:
        self._cancel: set[str] = set()
        self._lock = threading.Lock()

    def start(self, repo_id: str, filename: str, target_dir: str | None = None) -> dict[str, Any]:
        model_dir = Path(target_dir or store.setting("model_dir"))
        model_dir.mkdir(parents=True, exist_ok=True)
        safe_name = filename.replace("/", "__").replace("\\", "__")
        target_path = model_dir / safe_name
        encoded = urllib.parse.quote(filename)
        url = f"https://huggingface.co/{repo_id}/resolve/main/{encoded}"
        download_id = new_id("dl")
        store.execute(
            """
            insert into downloads(id, repo_id, filename, url, target_path, status, started_at)
            values(?, ?, ?, ?, ?, 'queued', ?)
            """,
            (download_id, repo_id, filename, url, str(target_path), now()),
        )
        thread = threading.Thread(target=self._download, args=(download_id,), daemon=True)
        thread.start()
        return self.get(download_id) or {"id": download_id}

    def cancel(self, download_id: str) -> None:
        with self._lock:
            self._cancel.add(download_id)

    def get(self, download_id: str) -> dict[str, Any] | None:
        return store.row("select * from downloads where id = ?", (download_id,))

    def list(self) -> list[dict[str, Any]]:
        return store.rows("select * from downloads order by started_at desc")

    def _is_cancelled(self, download_id: str) -> bool:
        with self._lock:
            return download_id in self._cancel

    def _download(self, download_id: str) -> None:
        row = self.get(download_id)
        if not row:
            return
        target = Path(row["target_path"])
        part = target.with_suffix(target.suffix + ".part")
        bytes_done = part.stat().st_size if part.exists() else 0
        headers = {}
        if bytes_done:
            headers["Range"] = f"bytes={bytes_done}-"
        try:
            store.execute("update downloads set status = 'running', bytes_done = ? where id = ?", (bytes_done, download_id))
            request = urllib.request.Request(row["url"], headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                total_header = response.headers.get("Content-Length")
                total = int(total_header) + bytes_done if total_header else None
                mode = "ab" if bytes_done else "wb"
                started = time.time()
                with part.open(mode + "") as handle:
                    while True:
                        if self._is_cancelled(download_id):
                            store.execute("update downloads set status='cancelled', error=null where id=?", (download_id,))
                            event_hub.publish_threadsafe("download", self.get(download_id) or {})
                            return
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        bytes_done += len(chunk)
                        elapsed = max(time.time() - started, 0.001)
                        payload = {
                            **(self.get(download_id) or {}),
                            "bytes_done": bytes_done,
                            "bytes_total": total,
                            "speed_bps": bytes_done / elapsed,
                        }
                        store.execute(
                            "update downloads set bytes_done=?, bytes_total=? where id=?",
                            (bytes_done, total, download_id),
                        )
                        event_hub.publish_threadsafe("download", payload)
            shutil.move(str(part), str(target))
            meta = inspect_model_file(str(target))
            normalized_path = normalize_path(str(target))
            existing = store.row("select id from models where normalized_path = ?", (normalized_path,))
            if existing:
                store.execute("update downloads set status='completed', bytes_done=?, bytes_total=?, finished_at=? where id=?", (bytes_done, bytes_done, now(), download_id))
                event_hub.publish_threadsafe("download", self.get(download_id) or {})
                return
            model_id = new_id("model")
            store.execute(
                """
                insert into models(id, name, repo_id, filename, path, normalized_path, size_bytes, quantization, source, managed, created_at)
                values(?, ?, ?, ?, ?, ?, ?, ?, 'huggingface', 1, ?)
                """,
                (
                    model_id,
                    Path(row["filename"]).stem,
                    row["repo_id"],
                    row["filename"],
                    str(target),
                    normalized_path,
                    meta["size_bytes"],
                    meta["quantization"] or detect_quantization(row["filename"]),
                    now(),
                ),
            )
            store.execute("update downloads set status='completed', bytes_done=?, bytes_total=?, finished_at=? where id=?", (bytes_done, bytes_done, now(), download_id))
        except Exception as exc:
            store.execute("update downloads set status='failed', error=? where id=?", (str(exc), download_id))
        event_hub.publish_threadsafe("download", self.get(download_id) or {})


download_manager = DownloadManager()
