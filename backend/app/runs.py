from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import urllib.request
from typing import Any

from .events import event_hub
from .gpu import query_gpus
from .llamacpp import resolve_llama_server_path
from .scripts import can_fit_vram, estimate_vram_mib, parse_script
from .storage import decode_json_field, new_id, now, store


class RunManager:
    INACTIVE_STATUSES = {"aborted", "failed", "unloaded", "exited"}
    PROTECTED_STATUSES = {"loading", "loaded", "orphaned"}

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()
        self._log_lock = threading.Lock()

    def active_counts(self) -> tuple[int, int]:
        self.reconcile_stale_runs()
        rows = store.rows("select status, count(*) as n from runs where status in ('loading','loaded') group by status")
        loaded = sum(row["n"] for row in rows if row["status"] == "loaded")
        loading = sum(row["n"] for row in rows if row["status"] == "loading")
        return loaded, loading

    def list(self) -> list[dict[str, Any]]:
        self.reconcile_stale_runs()
        return store.rows(
            """
            select r.*, m.name as model_name
            from runs r
            left join models m on m.id = r.model_id
            order by r.started_at desc
            limit 20
            """
        )

    def validate_start(self, script_id: str, manual_vram_mib: int | None = None) -> dict[str, Any]:
        plan = self._launch_plan(script_id, manual_vram_mib)
        return {
            "ok": True,
            "reason": plan["vram_reason"],
            "llama_server": plan["llama_server"],
            "host": plan["host"],
            "port": plan["port"],
            "estimated_vram_mib": plan["estimated_vram_mib"],
            "manual_vram_mib": plan["manual_vram_mib"],
            "n_cpu_moe": plan["parsed"].get("n_cpu_moe"),
            "args": plan["args"],
        }

    def reconcile_stale_runs(self) -> None:
        rows = store.rows("select id, pid, status from runs where status in ('loading','loaded')")
        for row in rows:
            pid = row.get("pid")
            with self._lock:
                tracked = row["id"] in self._processes
            if pid and self._pid_running(int(pid)) and not tracked:
                message = (
                    "Process is still running, but this backend session is no longer attached to its output. "
                    "Abort it and start again to capture live llama.cpp logs."
                )
                store.execute(
                    "update runs set status='orphaned', status_message=?, last_heartbeat_at=? where id=?",
                    (message, now(), row["id"]),
                )
                self._append_log(row["id"], f"[loader] {message}")
            elif not pid or not self._pid_running(int(pid)):
                store.execute(
                    """
                    update runs
                    set status='failed',
                        ended_at=?,
                        error=coalesce(error, 'Process is not running.'),
                        status_message=coalesce(status_message, 'Process is not running.')
                    where id=?
                    """,
                    (now(), row["id"]),
                )

    def start(self, script_id: str, manual_vram_mib: int | None = None) -> dict[str, Any]:
        plan = self._launch_plan(script_id, manual_vram_mib)
        script = plan["script"]
        model = plan["model"]
        parsed = plan["parsed"]
        args = plan["args"]
        run_id = new_id("run")
        message = "Launching llama.cpp server."
        store.execute(
            """
            insert into runs(id, script_id, model_id, status, status_message, host, port, started_at, last_heartbeat_at)
            values(?, ?, ?, 'loading', ?, ?, ?, ?, ?)
            """,
            (run_id, script_id, model["id"], message, parsed.get("host"), parsed.get("port"), now(), now()),
        )
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        with self._lock:
            self._processes[run_id] = process
        store.execute("update runs set pid=? where id=?", (process.pid, run_id))
        self._append_log(run_id, f"[loader] Starting llama-server PID {process.pid}.")
        self._append_log(run_id, f"[loader] Executable: {plan['llama_server']}")
        self._append_log(run_id, f"[loader] Health check: http://{parsed.get('host') or '127.0.0.1'}:{parsed.get('port') or 8080}/health")
        threading.Thread(target=self._watch, args=(run_id, process), daemon=True).start()
        return store.row("select * from runs where id=?", (run_id,)) or {"id": run_id}

    def _launch_plan(self, script_id: str, manual_vram_mib: int | None = None) -> dict[str, Any]:
        script = store.row("select * from scripts where id=?", (script_id,))
        if not script:
            raise ValueError("Script not found.")
        script = decode_json_field(script, "parsed_json")
        model = store.row("select * from models where id=?", (script["model_id"],))
        if not model:
            raise ValueError("Model not found.")
        gpus = query_gpus()
        free_mib = gpus[0]["memory_free_mib"] if gpus else 0
        parsed = parse_script(script["raw_script"]).to_dict()
        manual = manual_vram_mib or model.get("manual_vram_mib")
        is_moe_cpu = bool(parsed.get("n_cpu_moe"))
        estimate = None
        if not (is_moe_cpu and not manual):
            estimate = script.get("estimated_vram_mib") or estimate_vram_mib(
                model.get("size_bytes"),
                parsed.get("ctx_size"),
                n_cpu_moe=parsed.get("n_cpu_moe"),
            )
        allow_unknown = bool(is_moe_cpu and not manual)
        ok, reason = can_fit_vram(free_mib, estimate, manual, allow_unknown=allow_unknown)
        if not ok:
            raise ValueError(reason)
        llama_server = resolve_llama_server_path(store.setting("llama_server_path") or script["parsed_json"].get("executable") or "")
        if not llama_server:
            raise ValueError("Configure llama-server.exe before starting a script.")
        if not os.path.exists(llama_server):
            raise ValueError(f"llama-server.exe was not found: {llama_server}")
        parsed_args = parsed["args"]
        args = [llama_server] + parsed_args
        return {
            "script": script,
            "model": model,
            "parsed": parsed,
            "args": args,
            "llama_server": llama_server,
            "host": parsed.get("host") or "127.0.0.1",
            "port": parsed.get("port") or 8080,
            "estimated_vram_mib": estimate,
            "manual_vram_mib": manual,
            "vram_reason": reason,
        }

    def stop(self, run_id: str, status: str = "unloaded") -> dict[str, Any]:
        with self._lock:
            process = self._processes.get(run_id)
        row = store.row("select * from runs where id=?", (run_id,))
        if process and process.poll() is None:
            self._kill_process(process)
        elif row and row.get("pid"):
            self._kill_pid(int(row["pid"]))
        message = "Load aborted." if status == "aborted" else "Model unloaded."
        store.execute("update runs set status=?, ended_at=?, status_message=? where id=?", (status, now(), message, run_id))
        self._append_log(run_id, f"[loader] {message}")
        return store.row("select * from runs where id=?", (run_id,)) or {"id": run_id}

    def delete(self, run_id: str) -> dict[str, bool]:
        row = store.row("select * from runs where id=?", (run_id,))
        if not row:
            raise ValueError("Run not found.")
        if row["status"] in self.PROTECTED_STATUSES:
            raise ValueError("Abort or unload this run before deleting its terminal history.")
        with self._lock:
            process = self._processes.get(run_id)
            if process and process.poll() is None:
                raise ValueError("Abort or unload this run before deleting its terminal history.")
            self._processes.pop(run_id, None)
        store.execute("delete from runs where id=?", (run_id,))
        event_hub.publish_threadsafe("run", {"id": run_id, "deleted": True})
        return {"ok": True}

    def delete_history(self) -> dict[str, int]:
        self.reconcile_stale_runs()
        inactive_rows = store.rows(
            f"select id from runs where status in ({','.join('?' for _ in self.INACTIVE_STATUSES)})",
            tuple(self.INACTIVE_STATUSES),
        )
        kept_row = store.row(
            f"select count(*) as n from runs where status in ({','.join('?' for _ in self.PROTECTED_STATUSES)})",
            tuple(self.PROTECTED_STATUSES),
        )
        inactive_ids = [row["id"] for row in inactive_rows]
        if inactive_ids:
            with self._lock:
                for run_id in inactive_ids:
                    self._processes.pop(run_id, None)
            store.execute(
                f"delete from runs where id in ({','.join('?' for _ in inactive_ids)})",
                tuple(inactive_ids),
            )
            event_hub.publish_threadsafe("run", {"history_deleted": inactive_ids})
        return {"deleted": len(inactive_ids), "kept_active": int((kept_row or {}).get("n") or 0)}

    def _watch(self, run_id: str, process: subprocess.Popen[str]) -> None:
        start = time.time()
        loaded = False
        row = store.row("select host, port from runs where id=?", (run_id,))
        host, port = (row or {}).get("host") or "127.0.0.1", (row or {}).get("port") or 8080
        threading.Thread(target=self._read_output, args=(run_id, process), daemon=True).start()
        self._set_status(run_id, "loading", f"Loading: waiting for llama.cpp health on {host}:{port}.")
        last_heartbeat = 0.0
        while process.poll() is None:
            if not loaded and self._health(host, int(port)):
                loaded = True
                load_seconds = time.time() - start
                message = f"Loaded: llama.cpp is healthy after {self._format_seconds(load_seconds)}."
                store.execute(
                    "update runs set status='loaded', status_message=?, load_seconds=?, last_heartbeat_at=? where id=?",
                    (message, load_seconds, now(), run_id),
                )
                self._append_log(run_id, f"[loader] {message}")
                event_hub.publish_threadsafe("run", store.row("select * from runs where id=?", (run_id,)) or {})
            current = time.time()
            if current - last_heartbeat >= 5:
                status = "loaded" if loaded else "loading"
                message = (
                    f"Loaded: server process is healthy and running for {self._format_seconds(current - start)}."
                    if loaded
                    else f"Loading: process alive for {self._format_seconds(current - start)}, waiting for /health on {host}:{port}."
                )
                store.execute(
                    "update runs set status_message=?, last_heartbeat_at=? where id=?",
                    (message, now(), run_id),
                )
                self._append_log(run_id, f"[loader] {message}")
                event_hub.publish_threadsafe("run", store.row("select * from runs where id=?", (run_id,)) or {})
                last_heartbeat = current
            time.sleep(1)
        code = process.returncode
        final_status = "exited" if loaded else "failed"
        if code == 0 and loaded:
            final_status = "unloaded"
        if final_status == "unloaded":
            message = "Unloaded: llama.cpp process exited cleanly."
        elif final_status == "exited":
            message = f"Exited: loaded server stopped with code {code}."
        else:
            message = f"Failed: llama.cpp exited with code {code} before becoming healthy."
        self._append_log(run_id, f"[loader] {message}")
        current = store.row("select status from runs where id=?", (run_id,))
        if current and current.get("status") in {"aborted", "unloaded"}:
            with self._lock:
                self._processes.pop(run_id, None)
            event_hub.publish_threadsafe("run", store.row("select * from runs where id=?", (run_id,)) or {})
            return
        store.execute(
            "update runs set status=?, ended_at=?, status_message=?, error=? where id=?",
            (final_status, now(), message, None if code == 0 else f"Process exited with code {code}", run_id),
        )
        with self._lock:
            self._processes.pop(run_id, None)
        event_hub.publish_threadsafe("run", store.row("select * from runs where id=?", (run_id,)) or {})

    def _read_output(self, run_id: str, process: subprocess.Popen[str]) -> None:
        if not process.stdout:
            return
        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            self._append_log(run_id, line.rstrip())

    def _append_log(self, run_id: str, line: str) -> None:
        if not line:
            return
        with self._log_lock:
            row = store.row("select log_tail from runs where id=?", (run_id,))
            if not row:
                return
            log_tail = (row.get("log_tail") or "").splitlines()
            if log_tail and log_tail[-1] == line:
                return
            log_tail.append(line)
            tail = "\n".join(log_tail[-300:])
            store.execute("update runs set log_tail=? where id=?", (tail, run_id))
        event_hub.publish_threadsafe("run_log", {"id": run_id, "line": line})

    @staticmethod
    def _set_status(run_id: str, status: str, message: str) -> None:
        store.execute(
            "update runs set status=?, status_message=?, last_heartbeat_at=? where id=?",
            (status, message, now(), run_id),
        )
        event_hub.publish_threadsafe("run", store.row("select * from runs where id=?", (run_id,)) or {})

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        seconds = max(0, int(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes:02d}m {seconds:02d}s"
        if minutes:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"

    @staticmethod
    def _health(host: str, port: int) -> bool:
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=0.5) as response:
                return 200 <= response.status < 300
        except Exception:
            return False

    @staticmethod
    def _kill_process(process: subprocess.Popen[str]) -> None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)

    @staticmethod
    def _kill_pid(pid: int) -> None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)

    @staticmethod
    def _pid_running(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            if os.name == "nt":
                completed = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                return str(pid) in completed.stdout
            os.kill(pid, 0)
            return True
        except Exception:
            return False


run_manager = RunManager()
