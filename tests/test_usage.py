import json
import gc
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from backend.app.storage import Store
from backend.app.usage import UsageUnavailable, _open_opencode, save_model_usage_settings, usage_snapshot


class UsageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.loader = Store(root / "loader.sqlite3")
        self.opencode = root / "opencode.db"
        with sqlite3.connect(self.opencode) as connection:
            connection.executescript(
                """
                create table session (
                    id text primary key, parent_id text, title text, agent text,
                    model text, time_created integer, time_updated integer
                );
                create table message (
                    id text primary key, session_id text, time_created integer,
                    time_updated integer, data text
                );
                create table part (id text primary key, message_id text, data text);
                """
            )
        self.model_id = "model_test"
        self.loader.execute(
            """
            insert into models(id, name, filename, path, normalized_path, source, managed, created_at, display_order)
            values(?, ?, ?, ?, ?, 'import', 0, ?, 0)
            """,
            (self.model_id, "Test Qwen", "test.gguf", str(Path(self.directory.name) / "test.gguf"), "test.gguf", time.time()),
        )
        self.loader.execute(
            """
            insert into scripts(id, model_id, name, raw_script, parsed_json, created_at, updated_at)
            values('script_test', ?, 'Test', 'llama-server.exe --alias test-local --model test.gguf', '{}', ?, ?)
            """,
            (self.model_id, time.time(), time.time()),
        )

    def tearDown(self):
        self.loader = None
        gc.collect()
        self.directory.cleanup()

    def add_session(self, session_id, parent_id=None, title="Task", agent="main", model_id="test-local"):
        stamp = int(time.time() * 1000)
        with sqlite3.connect(self.opencode) as connection:
            connection.execute(
                "insert into session values(?, ?, ?, ?, ?, ?, ?)",
                (session_id, parent_id, title, agent, json.dumps({"providerID": "localllama", "id": model_id}), stamp, stamp),
            )

    def add_message(self, message_id, session_id, tokens, model_id="test-local", role="assistant", reasoning_part=False):
        stamp = int(time.time() * 1000)
        data = {
            "role": role,
            "providerID": "localllama",
            "modelID": model_id,
            "tokens": tokens,
        }
        with sqlite3.connect(self.opencode) as connection:
            connection.execute(
                "insert into message values(?, ?, ?, ?, ?)",
                (message_id, session_id, stamp, stamp, json.dumps(data)),
            )
            if reasoning_part:
                connection.execute(
                    "insert into part values(?, ?, ?)",
                    (f"part_{message_id}", message_id, json.dumps({"type": "reasoning", "text": "private"})),
                )

    def test_tree_aggregation_and_reasoning_status(self):
        self.add_session("root", title="Main task")
        self.add_session("child", parent_id="root", title="Research child", agent="researcher")
        self.add_message("m_root", "root", {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 20, "write": 2}}, reasoning_part=True)
        self.add_message("m_child", "child", {"input": 3, "output": 4, "reasoning": 1, "cache": {"read": 7, "write": 0}})
        snapshot = usage_snapshot(opencode_path=self.opencode, loader_store=self.loader)
        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["summary"]["total_tokens"], 52)
        self.assertEqual(snapshot["summary"]["input_tokens"], 13)
        self.assertEqual(snapshot["summary"]["cache_read_tokens"], 27)
        self.assertEqual(snapshot["summary"]["reasoning_tokens"], 1)
        self.assertEqual(snapshot["summary"]["reasoning_status"], "not_reported")
        task = snapshot["recent_task"]
        self.assertEqual(task["main"]["total_tokens"], 37)
        self.assertEqual(task["subagents"]["total_tokens"], 15)
        self.assertEqual(task["children"][0]["title"], "Research child")
        model = next(item for item in snapshot["models"] if item["model_id"] == self.model_id)
        self.assertEqual(model["usage"]["total_tokens"], 52)

    def test_costs_and_explicit_binding(self):
        self.add_session("root", title="Priced task", model_id="unmapped")
        self.add_message("m_root", "root", {"input": 1_000_000, "output": 2_000_000, "reasoning": 0, "cache": {"read": 0, "write": 0}}, model_id="unmapped")
        save_model_usage_settings(
            self.model_id,
            {
                "input_per_million": "1.234",
                "output_per_million": "2.506",
                "bindings": [{"provider_id": "localllama", "external_model_id": "unmapped"}],
            },
            self.loader,
        )
        snapshot = usage_snapshot(opencode_path=self.opencode, loader_store=self.loader)
        model = next(item for item in snapshot["models"] if item["model_id"] == self.model_id)
        self.assertEqual(model["usage"]["cost"], "6.25")
        self.assertEqual(model["usage"]["costs"]["input"], "1.23")
        self.assertEqual(model["usage"]["costs"]["output"], "5.01")
        self.assertEqual(model["usage"]["costs"]["cache"], "0.00")
        self.assertEqual(model["usage"]["costs"]["total"], "6.25")
        self.assertEqual(model["usage"]["cost_status"], "priced")
        self.assertFalse(snapshot["unmapped"])

    def test_non_assistant_rows_are_ignored_and_missing_source_is_contained(self):
        self.add_session("root")
        self.add_message("m_user", "root", {"input": 100, "output": 100, "cache": {}}, role="user")
        snapshot = usage_snapshot(opencode_path=self.opencode, loader_store=self.loader)
        self.assertEqual(snapshot["summary"]["requests"], 0)
        with self.assertRaises(UsageUnavailable):
            usage_snapshot(opencode_path=Path(self.directory.name) / "missing.db", loader_store=self.loader)

    def test_opencode_connection_is_read_only(self):
        connection = _open_opencode(self.opencode)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("create table should_not_exist (id integer)")
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
