import gc
import json
import tempfile
import unittest
from pathlib import Path

from backend.app.ninfer_setup import (
    MODEL_NAME,
    PRESET_NAME,
    load_ninfer_info,
    model_unc_path,
    preset_script,
    register_ninfer_model,
)
from backend.app.scripts import parse_script
from backend.app.storage import Store

FACTS = {
    "distro": "Ubuntu",
    "home": "/home/roy",
    "launcher_path": "/home/roy/ninfer-qwen38/run-qwen38-nvfp4.sh",
    "model_path": "/home/roy/ninfer-qwen38/models/qwen3_8_27b_nvfp4.ninfer",
    "model_size_bytes": 21492695040,
    "model_sha256": "bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32",
    "runtime_revision": "5d2c1f5590b8f4c3d106a75f65210eb4efb8f4e1",
    "port": 8081,
    "concurrency": 3,
    "max_context": 262144,
    "min_context": 163840,
}


class NInferSetupTests(unittest.TestCase):
    def test_preset_script_shape(self):
        raw = preset_script(FACTS)
        self.assertIn('& wsl.exe -d "Ubuntu" -- bash -lc', raw)
        self.assertIn("NINFER_PORT=8081", raw)
        self.assertIn("NINFER_CONCURRENCY=3", raw)
        self.assertIn("NINFER_MIN_CONTEXT=163840", raw)
        self.assertIn("NINFER_MODEL_FILE=qwen3_8_27b_nvfp4.ninfer", raw)
        # The preset must NOT pin NINFER_MAX_CONTEXT: the launcher only runs its
        # startup ladder when that variable is unset.
        self.assertNotIn("NINFER_MAX_CONTEXT=", raw)
        self.assertIn("run-qwen38-nvfp4.sh --model-id qwen3.8-27b", raw)
        parsed = parse_script(raw)
        self.assertEqual(parsed.runtime, "ninfer")
        self.assertEqual(parsed.wsl_distro, "Ubuntu")
        self.assertEqual(parsed.port, 8081)
        self.assertEqual(parsed.ctx_size, 163840)
        self.assertEqual(parsed.concurrency, 3)
        self.assertEqual(parsed.model_ref, "qwen3.8-27b")
        self.assertEqual(parsed.quantization, "NVFP4")

    def test_unc_path(self):
        self.assertEqual(
            model_unc_path(FACTS),
            r"\\wsl.localhost\Ubuntu\home\roy\ninfer-qwen38\models\qwen3_8_27b_nvfp4.ninfer",
        )

    def test_load_ninfer_info_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "nope.json"
            with self.assertRaises(FileNotFoundError):
                load_ninfer_info(missing)
            bad = Path(directory) / "bad.json"
            bad.write_text(json.dumps({"distro": "Ubuntu"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_ninfer_info(bad)

    def test_registration_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Store(Path(directory) / "loader.sqlite3")
            first = register_ninfer_model(FACTS, db)
            second = register_ninfer_model(FACTS, db)

            self.assertEqual(first, second)
            self.assertEqual(db.row("select count(*) as n from models")["n"], 1)
            self.assertEqual(db.row("select count(*) as n from scripts")["n"], 1)

            model = db.row("select * from models")
            self.assertEqual(model["name"], MODEL_NAME)
            self.assertEqual(model["quantization"], "NVFP4")
            self.assertEqual(model["source"], "ninfer")
            self.assertEqual(model["managed"], 0)
            self.assertEqual(model["manual_vram_mib"], 30720)
            self.assertEqual(model["size_bytes"], FACTS["model_size_bytes"])
            self.assertEqual(
                model["path"],
                r"\\wsl.localhost\Ubuntu\home\roy\ninfer-qwen38\models\qwen3_8_27b_nvfp4.ninfer",
            )

            script = db.row("select * from scripts where name=?", (PRESET_NAME,))
            self.assertIsNotNone(script)
            self.assertEqual(script["estimated_vram_mib"], 30720)
            parsed = parse_script(script["raw_script"])
            self.assertEqual(parsed.runtime, "ninfer")
            self.assertEqual(parsed.wsl_distro, "Ubuntu")
            self.assertEqual(parsed.ctx_size, 163840)
            self.assertEqual(parsed.quantization, "NVFP4")
            del db
            gc.collect()

    def test_registration_updates_changed_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Store(Path(directory) / "loader.sqlite3")
            register_ninfer_model(FACTS, db)
            changed = dict(FACTS, launcher_path="/home/roy/ninfer-qwen38/run-qwen38-v2.sh")
            second = register_ninfer_model(changed, db)
            self.assertEqual(db.row("select count(*) as n from models")["n"], 1)
            self.assertEqual(db.row("select count(*) as n from scripts")["n"], 1)
            script = db.row("select * from scripts")
            self.assertEqual(script["id"], second["script_id"])
            self.assertIn("run-qwen38-v2.sh", script["raw_script"])
            del db
            gc.collect()


if __name__ == "__main__":
    unittest.main()