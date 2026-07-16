import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from backend.app.main import ModelOrderIn, update_model_order
from backend.app.runs import RunManager
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
    def test_list_includes_model_name(self):
        manager = RunManager()
        seen = []

        with patch.object(manager, "reconcile_stale_runs"):
            with patch("backend.app.runs.store.rows", side_effect=lambda query, params=(): seen.append(query) or [{"id": "run_1", "model_name": "Qwen"}]):
                result = manager.list()

        self.assertEqual(result[0]["model_name"], "Qwen")
        self.assertIn("left join models", seen[0])

    def test_delete_history_removes_only_inactive_runs(self):
        manager = RunManager()
        manager._processes = {"failed_1": object(), "loaded_1": object()}
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


if __name__ == "__main__":
    unittest.main()
