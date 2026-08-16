import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.downloads import DownloadManager


class DownloadResumeTests(unittest.TestCase):
    def test_resume_incomplete_restarts_active_states_only(self):
        manager = DownloadManager()
        rows = [{"id": "dl_running"}, {"id": "dl_retrying"}]
        started = []

        with patch("backend.app.downloads.store.rows", return_value=rows):
            with patch.object(manager, "_start_thread", side_effect=lambda download_id: started.append(download_id)):
                manager.resume_incomplete()

        self.assertEqual(started, ["dl_running", "dl_retrying"])

    def test_completed_group_registers_only_the_primary_file(self):
        manager = DownloadManager()
        rows = [
            {"status": "completed", "group_primary": 1, "target_path": "C:/models/model-00001-of-00002.gguf", "bytes_done": 5},
            {"status": "completed", "group_primary": 0, "target_path": "C:/models/model-00002-of-00002.gguf", "bytes_done": 95},
        ]

        with patch("backend.app.downloads.store.rows", return_value=rows):
            with patch.object(manager, "_register_model") as register:
                manager._register_completed_group("group_1")

        register.assert_called_once_with(rows[0], Path(rows[0]["target_path"]), 100)


if __name__ == "__main__":
    unittest.main()
