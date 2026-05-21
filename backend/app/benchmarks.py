from __future__ import annotations

import json
import threading
import time
import urllib.request
from typing import Any

from .events import event_hub
from .storage import decode_json_field, new_id, now, store


class BenchmarkManager:
    def presets(self) -> list[dict[str, Any]]:
        return store.rows("select * from benchmark_presets order by case id when 'small' then 1 when 'medium' then 2 when 'large' then 3 else 4 end")

    def history(self) -> list[dict[str, Any]]:
        return store.rows("select * from benchmark_runs order by started_at desc limit 100")

    def start(self, script_id: str, preset_id: str, prompt: str | None = None, output_tokens: int | None = None) -> dict[str, Any]:
        script = store.row("select * from scripts where id=?", (script_id,))
        if not script:
            raise ValueError("Script not found.")
        script = decode_json_field(script, "parsed_json")
        preset = store.row("select * from benchmark_presets where id=?", (preset_id,))
        if not preset:
            raise ValueError("Benchmark preset not found.")
        run = store.row("select * from runs where script_id=? and status='loaded' order by started_at desc limit 1", (script_id,))
        if not run:
            raise ValueError("Load the script before benchmarking it.")
        bench_id = new_id("bench")
        bench_prompt = prompt or preset["prompt"]
        max_tokens = output_tokens or preset["output_tokens"]
        store.execute(
            """
            insert into benchmark_runs(id, script_id, model_id, preset_id, status, prompt, output_tokens, started_at)
            values(?, ?, ?, ?, 'running', ?, ?, ?)
            """,
            (bench_id, script_id, script["model_id"], preset_id, bench_prompt, max_tokens, now()),
        )
        threading.Thread(target=self._run, args=(bench_id, script, bench_prompt, max_tokens), daemon=True).start()
        return store.row("select * from benchmark_runs where id=?", (bench_id,)) or {"id": bench_id}

    def _run(self, bench_id: str, script: dict[str, Any], prompt: str, max_tokens: int) -> None:
        host = script["parsed_json"].get("host") or "127.0.0.1"
        port = script["parsed_json"].get("port") or 8080
        url = f"http://{host}:{port}/completion"
        payload = json.dumps(
            {
                "prompt": prompt,
                "n_predict": max_tokens,
                "stream": True,
                "timings_per_token": True,
                "return_progress": True,
            }
        ).encode("utf-8")
        request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        started = time.time()
        first_token_at: float | None = None
        tokens = 0
        raw_log: list[str] = []
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                for raw in response:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    data = self._parse_stream_json(line)
                    raw_log.append(line)
                    content = data.get("content") if isinstance(data, dict) else None
                    timings = data.get("timings") if isinstance(data, dict) else None
                    if first_token_at is None and content:
                        first_token_at = time.time()
                    if content:
                        tokens += 1
                    elapsed = max(time.time() - started, 0.001)
                    generation_elapsed = max(time.time() - (first_token_at or time.time()), 0.001)
                    metrics = {
                        "id": bench_id,
                        "fttt_ms": ((first_token_at - started) * 1000) if first_token_at else None,
                        "prefill_tps": self._timing_value(timings, "prompt_per_second"),
                        "generation_tps": tokens / generation_elapsed if first_token_at else None,
                        "average_tps": tokens / elapsed,
                        "raw_log": "\n".join(raw_log[-300:]),
                    }
                    if timings:
                        metrics["generation_tps"] = self._timing_value(timings, "predicted_per_second") or metrics["generation_tps"]
                    store.execute(
                        "update benchmark_runs set fttt_ms=?, prefill_tps=?, generation_tps=?, average_tps=?, raw_log=? where id=?",
                        (metrics["fttt_ms"], metrics["prefill_tps"], metrics["generation_tps"], metrics["average_tps"], metrics["raw_log"], bench_id),
                    )
                    event_hub.publish_threadsafe("benchmark", metrics)
            duration = time.time() - started
            store.execute(
                "update benchmark_runs set status='completed', duration_seconds=?, ended_at=? where id=?",
                (duration, now(), bench_id),
            )
        except Exception as exc:
            store.execute("update benchmark_runs set status='failed', error=?, ended_at=? where id=?", (str(exc), now(), bench_id))
        event_hub.publish_threadsafe("benchmark", store.row("select * from benchmark_runs where id=?", (bench_id,)) or {})

    @staticmethod
    def _parse_stream_json(line: str) -> dict[str, Any]:
        if line.startswith("data:"):
            line = line[5:].strip()
        try:
            data = json.loads(line)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _timing_value(timings: Any, key: str) -> float | None:
        if isinstance(timings, dict) and timings.get(key) is not None:
            try:
                return float(timings[key])
            except (TypeError, ValueError):
                return None
        return None


benchmark_manager = BenchmarkManager()
