import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.runs import RunManager

NINFER_RAW = """& wsl.exe -d "Ubuntu-24.04" -- bash -lc 'NINFER_PORT=8081 NINFER_CONCURRENCY=3 NINFER_MAX_CONTEXT=262144 NINFER_MIN_CONTEXT=163840 NINFER_MODEL_FILE=qwen3_8_27b_nvfp4.ninfer ~/ninfer-qwen38/run-qwen38-nvfp4.sh --model-id qwen3.8-27b'"""


class NInferRunTests(unittest.TestCase):
    def test_launch_plan_uses_wsl_and_skips_vram_gate(self):
        manager = RunManager()
        with tempfile.TemporaryDirectory() as directory:
            script = {
                "id": "script_1",
                "model_id": "model_1",
                "raw_script": NINFER_RAW,
                "parsed_json": {},
                "estimated_vram_mib": 30720,
            }
            model = {"id": "model_1", "path": "model.gguf", "size_bytes": 21 * 1024 * 1024 * 1024}
            with patch("backend.app.runs.store.row", side_effect=[script, model]):
                with patch("backend.app.runs.shutil.which", return_value=r"C:\Windows\System32\wsl.exe"):
                    # Only 4 GiB free: a llama.cpp preset would fail the gate.
                    with patch("backend.app.runs.query_gpus", return_value=[{"memory_free_mib": 4096}]):
                        plan = manager._launch_plan("script_1")

        self.assertEqual(plan["llama_server"], r"C:\Windows\System32\wsl.exe")
        self.assertEqual(plan["host"], "127.0.0.1")
        self.assertEqual(plan["port"], 8081)
        self.assertIn("auto-sizes", plan["vram_reason"])
        self.assertEqual(plan["args"][0], r"C:\Windows\System32\wsl.exe")
        self.assertIn("-d", plan["args"])
        self.assertIn("Ubuntu-24.04", plan["args"])
        self.assertIn("bash", plan["args"])
        self.assertIn("-lc", plan["args"])
        payload = plan["args"][plan["args"].index("-lc") + 1]
        self.assertTrue(payload.startswith("NINFER_HOST=0.0.0.0 "))

    def test_launch_plan_requires_wsl(self):
        manager = RunManager()
        script = {
            "id": "script_1",
            "model_id": "model_1",
            "raw_script": NINFER_RAW,
            "parsed_json": {},
            "estimated_vram_mib": 30720,
        }
        model = {"id": "model_1", "path": "model.gguf", "size_bytes": 21 * 1024 * 1024 * 1024}
        with patch("backend.app.runs.store.row", side_effect=[script, model]):
            with patch("backend.app.runs.shutil.which", return_value=None):
                with patch("backend.app.runs.query_gpus", return_value=[{"memory_free_mib": 999999}]):
                    with self.assertRaises(ValueError):
                        manager._launch_plan("script_1")

    def test_stop_args_use_launcher_stop_subcommand(self):
        manager = RunManager()
        with patch("backend.app.runs.store.row", side_effect=[
            {"script_id": "script_1"},
            {"raw_script": NINFER_RAW},
        ]):
            with patch("backend.app.runs.shutil.which", return_value=r"C:\Windows\System32\wsl.exe"):
                stop_args = manager._ninfer_stop_args("run_1")

        self.assertEqual(stop_args, [
            r"C:\Windows\System32\wsl.exe",
            "-d",
            "Ubuntu-24.04",
            "--",
            "bash",
            "-lc",
            "~/ninfer-qwen38/run-qwen38-nvfp4.sh stop",
        ])

    def test_stop_args_none_for_non_ninfer(self):
        manager = RunManager()
        with patch("backend.app.runs.store.row", side_effect=[
            {"script_id": "script_1"},
            {"raw_script": '& "C:\\llama-server.exe" -m "model.gguf"'},
        ]):
            with patch("backend.app.runs.shutil.which", return_value=r"C:\Windows\System32\wsl.exe"):
                self.assertIsNone(manager._ninfer_stop_args("run_1"))

    def test_stop_releases_lan_forward_for_ninfer(self):
        manager = RunManager()
        run_row = {"id": "run_1", "script_id": "script_1", "pid": 1234}
        with patch("backend.app.runs.store.row", side_effect=[
            run_row,
            {"script_id": "script_1"},
            {"raw_script": NINFER_RAW},
            {"script_id": "script_1"},
            {"raw_script": NINFER_RAW},
            {"id": "run_1"},
        ]):
            with patch("backend.app.runs.shutil.which", return_value=r"C:\Windows\System32\wsl.exe"):
                with patch.object(manager, "_kill_pid") as kill:
                    with patch.object(RunManager, "_sync_lan_forward") as sync:
                        manager.stop("run_1")
        kill.assert_called_once_with(1234)
        sync.assert_called_once()

    def test_stop_does_not_touch_lan_forward_for_llamacpp(self):
        manager = RunManager()
        run_row = {"id": "run_2", "script_id": "script_2", "pid": 5678}
        with patch("backend.app.runs.store.row", side_effect=[
            run_row,
            {"script_id": "script_2"},
            {"raw_script": '& "C:\\llama-server.exe" -m "model.gguf"'},
            {"script_id": "script_2"},
            {"raw_script": '& "C:\\llama-server.exe" -m "model.gguf"'},
            {"id": "run_2"},
        ]):
            with patch("backend.app.runs.shutil.which", return_value=r"C:\Windows\System32\wsl.exe"):
                with patch.object(manager, "_kill_pid") as kill:
                    with patch.object(RunManager, "_sync_lan_forward") as sync:
                        manager.stop("run_2")
        kill.assert_called_once_with(5678)
        sync.assert_not_called()


if __name__ == "__main__":
    unittest.main()