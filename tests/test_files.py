import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.files import browse_files


class BrowseFilesTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        (self.root / "model-Q4_K_M.gguf").write_bytes(b"gguf")
        (self.root / "qwen3_8_27b_nvfp4.ninfer").write_bytes(b"ninfer")
        (self.root / "notes.txt").write_text("hi")

    def tearDown(self):
        self._temp.cleanup()

    def _browse(self, **kwargs):
        with patch("backend.app.files.store.setting", return_value=str(self.root)):
            return browse_files(**kwargs)

    def test_browse_shows_gguf_and_ninfer_files(self):
        result = self._browse()
        names = {entry["name"] for entry in result["entries"]}
        self.assertEqual(names, {"model-Q4_K_M.gguf", "qwen3_8_27b_nvfp4.ninfer"})

    def test_ninfer_entry_is_marked_as_ninfer(self):
        result = self._browse()
        ninfer = next(entry for entry in result["entries"] if entry["name"].endswith(".ninfer"))
        self.assertEqual(ninfer["format"], "ninfer")
        self.assertEqual(ninfer["quantization"], "NVFP4")
        self.assertEqual(ninfer["size_bytes"], 6)

    def test_gguf_entry_still_inspected(self):
        result = self._browse()
        gguf = next(entry for entry in result["entries"] if entry["name"].endswith(".gguf"))
        self.assertEqual(gguf["format"], "gguf")
        self.assertEqual(gguf["quantization"], "Q4_K_M")

    def test_executable_browser_ignores_ninfer_filter(self):
        result = self._browse(executable_only=True)
        self.assertEqual(result["entries"], [])


if __name__ == "__main__":
    unittest.main()