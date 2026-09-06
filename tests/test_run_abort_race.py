import tempfile
import unittest
from contextlib import closing, contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

from backend.app.runs import RunManager
from backend.app.storage import Store


class TestStore(Store):
    @contextmanager
    def connect(self):
        with closing(super().connect()) as connection, connection:
            yield connection


class RunAbortRaceTests(unittest.TestCase):
    def test_abort_during_health_probe_keeps_terminal_status_and_message(self):
        for healthy in (False, True):
            with self.subTest(healthy=healthy), tempfile.TemporaryDirectory() as directory:
                database = TestStore(Path(directory) / "loader.sqlite3")
                database.execute(
                    "insert into runs(id, script_id, model_id, status, started_at) "
                    "values('run_test', 'script_test', 'model_test', 'loading', 1)"
                )
                process = Mock(stdout=None, returncode=1)
                process.poll.return_value = None
                manager = RunManager()
                manager._processes["run_test"] = process

                def health(host, port):
                    database.execute(
                        "update runs set status='aborted', status_message='Load aborted.', ended_at=2 where id='run_test'"
                    )
                    process.poll.return_value = 1
                    return healthy

                with patch("backend.app.runs.store", database), \
                     patch("backend.app.runs.event_hub"), \
                     patch.object(manager, "_health", side_effect=health), \
                     patch("backend.app.runs.time.sleep"):
                    manager._watch("run_test", process)

                run = database.row("select * from runs where id='run_test'")
                self.assertEqual(run["status"], "aborted")
                self.assertEqual(run["status_message"], "Load aborted.")
                self.assertIsNone(run["load_seconds"])
                self.assertNotIn("Failed:", run["log_tail"] or "")
                self.assertNotIn("waiting for /health", run["log_tail"] or "")
                self.assertNotIn("run_test", manager._processes)
