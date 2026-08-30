import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import HTTPException

from backend.app.main import ModelOrderIn, ScriptFavoriteIn, update_model_order, update_script_favorite
from backend.app.runs import RunManager, _with_loader_defaults
from backend.app.storage import Store


class ModelOrderTests(unittest.TestCase):
    def test_migration_backfills_display_order_from_current_visual_order(self):
        db_path = Path(tempfile.gettempdir()) / f"llm-loader-test-{uuid.uuid4().hex}.db"
        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    create table models (
                        id text primary key,
                        name text not null,
                        path text not null,
                        size_bytes integer,
                        quantization text,
                        source text not null,
                        managed integer not null default 1,
                        created_at real not null
                    )
                    """
                )
                conn.execute("insert into models(id, name, path, source, created_at) values('oldest', 'Oldest', 'C:/a.gguf', 'import', 1)")
                conn.execute("insert into models(id, name, path, source, created_at) values('newest', 'Newest', 'C:/b.gguf', 'import', 3)")
                conn.execute("insert into models(id, name, path, source, created_at) values('middle', 'Middle', 'C:/c.gguf', 'import', 2)")

            test_store = Store(db_path)
            rows = test_store.rows("select id, display_order from models order by display_order asc")
        finally:
            try:
                db_path.unlink()
            except OSError:
                pass

        self.assertEqual(rows, [
            {"id": "newest", "display_order": 0},
            {"id": "middle", "display_order": 1},
            {"id": "oldest", "display_order": 2},
        ])

    def test_update_model_order_accepts_complete_permutation(self):
        executed = []
        with patch("backend.app.main.store.rows", return_value=[{"id": "a"}, {"id": "b"}, {"id": "c"}]):
            with patch("backend.app.main.store.execute", side_effect=lambda query, params=(): executed.append(tuple(params))):
                result = update_model_order(ModelOrderIn(model_ids=["b", "c", "a"]))

        self.assertTrue(result["ok"])
        self.assertEqual(executed, [(0, "b"), (1, "c"), (2, "a")])

    def test_update_model_order_rejects_missing_duplicate_or_unknown_ids(self):
        with patch("backend.app.main.store.rows", return_value=[{"id": "a"}, {"id": "b"}]):
            for payload in (["a"], ["a", "a"], ["a", "x"]):
                with self.subTest(payload=payload):
                    with self.assertRaises(HTTPException):
                        update_model_order(ModelOrderIn(model_ids=payload))


class RunHistoryTests(unittest.TestCase):
    def test_llama_cpp_ui_statistics_are_enabled_by_default(self):
        self.assertEqual(
            _with_loader_defaults(["--model", "model.gguf"]),
            [
                "--model",
                "model.gguf",
                "--host",
                "0.0.0.0",
                "--perf",
                "--ui-config",
                '{"showMessageStats":true}',
            ],
        )

    def test_models_are_exposed_on_the_lan_by_default(self):
        self.assertIn("--host", _with_loader_defaults(["--model", "model.gguf"]))
        result = _with_loader_defaults(["--model", "model.gguf"])
        self.assertEqual(result[result.index("--host") + 1], "0.0.0.0")

    def test_explicit_host_is_preserved(self):
        for args in (
            ["--model", "model.gguf", "--host", "127.0.0.1"],
            ["--model", "model.gguf", "--host=192.168.1.50"],
        ):
            with self.subTest(args=args):
                result = _with_loader_defaults(args)
                self.assertEqual(result.count("--host"), args.count("--host"))
                self.assertNotIn("0.0.0.0", result)

    def test_explicit_perf_and_ui_choices_are_preserved(self):
        for flag in ("--perf", "--no-perf"):
            args = [
                "--model",
                "model.gguf",
                "--host",
                "127.0.0.1",
                flag,
                "--ui-config",
                '{"showMessageStats":false}',
            ]
            with self.subTest(flag=flag):
                self.assertIs(_with_loader_defaults(args), args)

    def test_list_includes_model_and_script_names(self):
        manager = RunManager()
        seen = []

        with patch.object(manager, "reconcile_stale_runs"):
            with patch("backend.app.runs.store.rows", side_effect=lambda query, params=(): seen.append(query) or [{"id": "run_1", "model_name": "Qwen", "script_name": "128k performance"}]):
                result = manager.list()

        self.assertEqual(result[0]["model_name"], "Qwen")
        self.assertEqual(result[0]["script_name"], "128k performance")
        self.assertIn("left join models", seen[0])
        self.assertIn("left join scripts", seen[0])

    def test_reconcile_leaves_tracked_launch_under_watcher_control(self):
        manager = RunManager()
        manager._processes = {"run_1": Mock()}

        with patch("backend.app.runs.store.rows", return_value=[{"id": "run_1", "pid": None, "status": "loading"}]):
            with patch.object(manager, "_pid_running") as pid_running:
                with patch("backend.app.runs.store.execute") as execute:
                    manager.reconcile_stale_runs()

        pid_running.assert_not_called()
        execute.assert_not_called()

    def test_start_registers_process_before_persisting_visible_run(self):
        manager = RunManager()
        process = Mock(pid=5908)
        plan = {
            "script": {"id": "script_1"},
            "model": {"id": "model_1"},
            "parsed": {"host": "127.0.0.1", "port": 8080, "runtime": "llama.cpp"},
            "args": ["llama-server.exe", "-m", "model.gguf"],
            "llama_server": "llama-server.exe",
        }
        inserts = []

        def record_insert(query, params=()):
            if "insert into runs" in query:
                self.assertIs(manager._processes.get("run_1"), process)
                inserts.append(tuple(params))

        with patch.object(manager, "_launch_plan", return_value=plan):
            with patch("backend.app.runs.new_id", return_value="run_1"):
                with patch("backend.app.runs.subprocess.Popen", return_value=process):
                    with patch("backend.app.runs.store.execute", side_effect=record_insert):
                        with patch("backend.app.runs.store.row", return_value={"id": "run_1", "pid": 5908}):
                            with patch.object(manager, "_append_log"):
                                with patch("backend.app.runs.threading.Thread"):
                                    result = manager.start("script_1")

        self.assertEqual(result["pid"], 5908)
        self.assertEqual(len(inserts), 1)
        self.assertEqual(inserts[0][3], 5908)

    def test_script_favorite_is_persisted(self):
        executed = []
        rows = [
            {"id": "script_1", "model_id": "model_1", "is_favorite": 0},
            {"id": "script_1", "model_id": "model_1", "is_favorite": 1},
        ]
        with patch("backend.app.main.store.row", side_effect=rows):
            with patch("backend.app.main.store.execute", side_effect=lambda query, params=(): executed.append(tuple(params))):
                result = update_script_favorite("model_1", "script_1", ScriptFavoriteIn(is_favorite=True))

        self.assertEqual(result["is_favorite"], 1)
        self.assertEqual(executed[0][0], 1)
        self.assertEqual(executed[0][2], "script_1")

    def test_delete_history_removes_only_inactive_runs(self):
        manager = RunManager()
        failed_process = Mock()
        failed_process.poll.return_value = 1
        manager._processes = {"failed_1": failed_process, "loaded_1": Mock()}
        executed = []

        with patch.object(manager, "reconcile_stale_runs"):
            with patch("backend.app.runs.store.rows", return_value=[{"id": "failed_1"}, {"id": "unloaded_1"}]):
                with patch("backend.app.runs.store.row", return_value={"n": 2}):
                    with patch("backend.app.runs.store.execute", side_effect=lambda query, params=(): executed.append((query, tuple(params)))):
                        with patch("backend.app.runs.event_hub.publish_threadsafe"):
                            result = manager.delete_history()

        self.assertEqual(result, {"deleted": 2, "kept_active": 2})
        self.assertNotIn("failed_1", manager._processes)
        self.assertIn("loaded_1", manager._processes)
        self.assertEqual(executed[0][1], ("failed_1", "unloaded_1"))

    def test_delete_history_preserves_inactive_row_with_live_process(self):
        manager = RunManager()
        live_process = Mock()
        live_process.poll.return_value = None
        finished_process = Mock()
        finished_process.poll.return_value = 1
        manager._processes = {"failed_live": live_process, "failed_done": finished_process}
        executed = []

        with patch.object(manager, "reconcile_stale_runs"):
            with patch("backend.app.runs.store.rows", return_value=[{"id": "failed_live"}, {"id": "failed_done"}]):
                with patch("backend.app.runs.store.row", return_value={"n": 2}):
                    with patch("backend.app.runs.store.execute", side_effect=lambda query, params=(): executed.append((query, tuple(params)))):
                        with patch("backend.app.runs.event_hub.publish_threadsafe"):
                            result = manager.delete_history()

        self.assertEqual(result, {"deleted": 1, "kept_active": 3})
        self.assertIn("failed_live", manager._processes)
        self.assertNotIn("failed_done", manager._processes)
        self.assertEqual(executed[0][1], ("failed_done",))

    def test_launch_plan_prefers_script_runtime_and_delegates_fit(self):
        manager = RunManager()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script_runtime = root / "prism-llama-server.exe"
            global_runtime = root / "stock-llama-server.exe"
            script_runtime.write_bytes(b"runtime")
            global_runtime.write_bytes(b"runtime")
            script = {
                "id": "script_1",
                "model_id": "model_1",
                "raw_script": f'& "{script_runtime}" -m "model.gguf" --fit on -ngl auto -c 131072',
                "parsed_json": {},
                "estimated_vram_mib": None,
            }
            model = {"id": "model_1", "path": "model.gguf", "size_bytes": 7 * 1024 * 1024 * 1024}
            with patch("backend.app.runs.store.row", side_effect=[script, model]):
                with patch("backend.app.runs.store.setting", return_value=str(global_runtime)):
                    with patch("backend.app.runs.query_gpus", return_value=[{"memory_free_mib": 4096}]):
                        plan = manager._launch_plan("script_1")
            self.assertEqual(plan["llama_server"], str(script_runtime.resolve()))
            self.assertIsNone(plan["estimated_vram_mib"])
            self.assertIn("delegated", plan["vram_reason"])

    def test_launch_plan_uses_global_runtime_when_script_has_none(self):
        manager = RunManager()
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "stock-llama-server.exe"
            runtime.write_bytes(b"runtime")
            script = {
                "id": "script_1",
                "model_id": "model_1",
                "raw_script": '-m "model.gguf" -ngl 99 -c 4096',
                "parsed_json": {},
                "estimated_vram_mib": None,
            }
            model = {"id": "model_1", "path": "model.gguf", "size_bytes": 1024 * 1024}
            with patch("backend.app.runs.store.row", side_effect=[script, model]):
                with patch("backend.app.runs.store.setting", return_value=str(runtime)):
                    with patch("backend.app.runs.query_gpus", return_value=[{"memory_free_mib": 4096}]):
                        plan = manager._launch_plan("script_1")
            self.assertEqual(plan["llama_server"], str(runtime.resolve()))

    def test_fixed_gpu_layers_keep_vram_gate(self):
        manager = RunManager()
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "llama-server.exe"
            runtime.write_bytes(b"runtime")
            script = {
                "id": "script_1",
                "model_id": "model_1",
                "raw_script": f'& "{runtime}" -m "model.gguf" --fit on -ngl 99 -c 131072',
                "parsed_json": {},
                "estimated_vram_mib": None,
            }
            model = {"id": "model_1", "path": "model.gguf", "size_bytes": 7 * 1024 * 1024 * 1024}
            with patch("backend.app.runs.store.row", side_effect=[script, model]):
                with patch("backend.app.runs.store.setting", return_value=str(runtime)):
                    with patch("backend.app.runs.query_gpus", return_value=[{"memory_free_mib": 4096}]):
                        with self.assertRaises(ValueError):
                            manager._launch_plan("script_1")


if __name__ == "__main__":
    unittest.main()
