import unittest
from unittest.mock import patch

from backend.app.hf import model_files


class HuggingFaceFileTests(unittest.TestCase):
    def test_tree_endpoint_sizes_are_used_for_gguf_variants(self):
        payload = [
            {"type": "file", "path": "README.md", "size": 10},
            {
                "type": "file",
                "path": "Model-Q4_K_M.gguf",
                "size": 16_861_398_400,
                "lfs": {"size": 16_861_398_400},
            },
            {
                "type": "file",
                "path": "subdir/Model-Q5_K_S.gguf",
                "lfs": {"size": 18_990_663_040},
            },
        ]

        with patch("backend.app.hf._get_json", return_value=payload):
            files = model_files("author/repo")

        self.assertEqual([file["filename"] for file in files], ["Model-Q4_K_M.gguf", "subdir/Model-Q5_K_S.gguf"])
        self.assertEqual(files[0]["size_bytes"], 16_861_398_400)
        self.assertEqual(files[0]["quantization"], "Q4_K_M")
        self.assertIsInstance(files[0]["estimated_vram_mib"], int)
        self.assertEqual(files[1]["size_bytes"], 18_990_663_040)
        self.assertEqual(files[1]["quantization"], "Q5_K_S")

    def test_split_files_are_one_aggregated_variant(self):
        payload = [
            {"type": "file", "path": "Q3/Model-Q3_K_M-00003-of-00003.gguf", "size": 30},
            {"type": "file", "path": "Q3/Model-Q3_K_M-00001-of-00003.gguf", "size": 10},
            {"type": "file", "path": "Q3/Model-Q3_K_M-00002-of-00003.gguf", "size": 20},
        ]

        with patch("backend.app.hf._get_json", return_value=payload):
            files = model_files("author/repo")

        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["filename"], "Q3/Model-Q3_K_M-00001-of-00003.gguf")
        self.assertEqual(files[0]["display_name"], "Q3/Model-Q3_K_M.gguf")
        self.assertEqual(
            files[0]["filenames"],
            [
                "Q3/Model-Q3_K_M-00001-of-00003.gguf",
                "Q3/Model-Q3_K_M-00002-of-00003.gguf",
                "Q3/Model-Q3_K_M-00003-of-00003.gguf",
            ],
        )
        self.assertEqual(files[0]["shard_count"], 3)
        self.assertEqual(files[0]["size_bytes"], 60)
        self.assertTrue(files[0]["complete"])


if __name__ == "__main__":
    unittest.main()
