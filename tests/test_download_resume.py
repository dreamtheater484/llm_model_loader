import unittest
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


if __name__ == "__main__":
    unittest.main()
