import tempfile
import unittest
import gc
from pathlib import Path

from backend.app.bonsai_setup import PRESET_NAME, register_bonsai_model
from backend.app.storage import Store


class BonsaiSetupTests(unittest.TestCase):
    def test_registration_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = Store(root / "loader.sqlite3")
            model = root / "Ternary-Bonsai-27B-Q2_0.gguf"
            model.write_bytes(b"test model")
            runtime = root / "llama-server.exe"
            runtime.write_bytes(b"test runtime")

            first = register_bonsai_model(str(model), str(runtime), db)
            second = register_bonsai_model(str(model), str(runtime), db)

            self.assertEqual(first, second)
            self.assertEqual(db.row("select count(*) as n from models")["n"], 1)
            self.assertEqual(db.row("select count(*) as n from scripts")["n"], 1)
            script = db.row("select * from scripts where name=?", (PRESET_NAME,))
            self.assertIsNotNone(script)
            self.assertIn("-c 8192", script["raw_script"])
            self.assertIn("--fit off", script["raw_script"])
            self.assertIn("-ngl 99", script["raw_script"])
            self.assertIn("-ctk q4_0", script["raw_script"])
            self.assertIn("--kv-offload", script["raw_script"])
            self.assertEqual(script["estimated_vram_mib"], 6856)
            del db
            gc.collect()
