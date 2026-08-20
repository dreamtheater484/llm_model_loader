import tempfile
import unittest
import gc
from pathlib import Path

from backend.app.qwen38_setup import PRESET_NAME, register_qwen38_model
from backend.app.scripts import detect_quantization, parse_script
from backend.app.storage import Store


class Qwen38SetupTests(unittest.TestCase):
    def test_registration_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Store(root / "loader.sqlite3")
            model = root / "Qwen3.8-27B-NVFP4-MTP.gguf"
            model.write_bytes(b"test model")
            runtime = root / "llama-server.exe"
            runtime.write_bytes(b"test runtime")
            mmproj = root / "mmproj-F16.gguf"
            mmproj.write_bytes(b"test mmproj")

            first = register_qwen38_model(str(model), str(runtime), str(mmproj), db)
            second = register_qwen38_model(str(model), str(runtime), str(mmproj), db)

            self.assertEqual(first, second)
            self.assertEqual(db.row("select count(*) as n from models")["n"], 1)
            self.assertEqual(db.row("select count(*) as n from scripts")["n"], 1)
            model_row = db.row("select * from models")
            self.assertEqual(model_row["quantization"], "NVFP4")
            script = db.row("select * from scripts where name=?", (PRESET_NAME,))
            self.assertIsNotNone(script)
            self.assertIn("-c 163840", script["raw_script"])
            self.assertIn("-np 2", script["raw_script"])
            self.assertIn("--mmproj", script["raw_script"])
            self.assertIn("--mmproj-offload", script["raw_script"])
            self.assertIn("--spec-type draft-mtp", script["raw_script"])
            self.assertIn("--spec-draft-n-max 3", script["raw_script"])
            self.assertIn("-ctk q4_0", script["raw_script"])
            self.assertIn("-ctv q4_0", script["raw_script"])
            self.assertIn("-ngl 999", script["raw_script"])
            self.assertIn("-fa on", script["raw_script"])
            self.assertEqual(script["estimated_vram_mib"], 30720)

            parsed = parse_script(script["raw_script"])
            self.assertEqual(parsed.ctx_size, 163840)
            self.assertEqual(parsed.parallel, 2)
            self.assertTrue(parsed.mtp)
            self.assertTrue(parsed.flash_attention)
            self.assertEqual(parsed.cache_type_k, "q4_0")
            self.assertEqual(parsed.cache_type_v, "q4_0")
            self.assertEqual(parsed.quantization, "NVFP4")
            del db
            gc.collect()

    def test_registration_follows_newer_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Store(root / "loader.sqlite3")
            model = root / "Qwen3.8-27B-NVFP4-MTP.gguf"
            model.write_bytes(b"test model")
            mmproj = root / "mmproj-F16.gguf"
            mmproj.write_bytes(b"test mmproj")
            old_runtime = root / "b9804" / "llama-server.exe"
            old_runtime.parent.mkdir()
            old_runtime.write_bytes(b"test runtime")
            new_runtime = root / "b10453" / "llama-server.exe"
            new_runtime.parent.mkdir()
            new_runtime.write_bytes(b"test runtime")

            first = register_qwen38_model(str(model), str(old_runtime), str(mmproj), db)
            second = register_qwen38_model(str(model), str(new_runtime), str(mmproj), db)

            self.assertEqual(first["script_id"], second["script_id"])
            self.assertEqual(db.row("select count(*) as n from scripts")["n"], 1)
            script = db.row("select * from scripts")
            self.assertIn(str(new_runtime), script["raw_script"])
            self.assertNotIn(str(old_runtime), script["raw_script"])
            del db
            gc.collect()

    def test_nvfp4_quantization_detection(self):
        self.assertEqual(detect_quantization("Qwen3.8-27B-NVFP4-MTP.gguf"), "NVFP4")
        self.assertEqual(detect_quantization("Qwen3.6-27B-NVFP4-MTP-GGUF.gguf"), "NVFP4")
        self.assertEqual(detect_quantization("Qwen3.8-27B-Q4_K_M.gguf"), "Q4_K_M")
        self.assertIsNone(detect_quantization("Qwen3.8-27B.gguf"))


if __name__ == "__main__":
    unittest.main()
