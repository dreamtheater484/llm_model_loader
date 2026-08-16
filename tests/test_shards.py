import tempfile
import unittest
from pathlib import Path

from backend.app.shards import local_model_files, model_name_from_filename, parse_gguf_shard


class GgufShardTests(unittest.TestCase):
    def test_parse_and_model_name(self):
        filename = "Q3/Model-Q3_K_M-00001-of-00004.gguf"

        shard = parse_gguf_shard(filename)

        self.assertIsNotNone(shard)
        self.assertEqual(shard.index, 1)
        self.assertEqual(shard.count, 4)
        self.assertEqual(model_name_from_filename(filename), "Model-Q3_K_M")

    def test_local_model_files_returns_the_complete_ordered_set(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [root / f"model-0000{index}-of-00003.gguf" for index in (3, 1, 2)]
            for path in paths:
                path.touch()

            found = local_model_files(str(paths[1]))

        self.assertEqual([path.name for path in found], [
            "model-00001-of-00003.gguf",
            "model-00002-of-00003.gguf",
            "model-00003-of-00003.gguf",
        ])


if __name__ == "__main__":
    unittest.main()
