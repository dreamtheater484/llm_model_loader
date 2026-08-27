import unittest
from unittest.mock import patch

from backend.app.hf import model_files, search_models


class HuggingFaceSearchTests(unittest.TestCase):
    def test_search_only_returns_gguf_repositories(self):
        payload = [
            {
                "modelId": "unsloth/GLM-5.3-Flash",
                "author": "unsloth",
                "tags": ["transformers", "safetensors", "glm5_next"],
            },
            {
                "modelId": "unsloth/GLM-5.3-Flash-GGUF",
                "author": "unsloth",
                "tags": ["transformers", "text-generation"],
            },
            {
                "modelId": "community/GLM-5.3-Flash",
                "author": "community",
                "tags": ["gguf", "quantized"],
            },
        ]

        with patch("backend.app.hf._get_json", return_value=payload):
            results = search_models("GLM-5.3-Flash")

        self.assertEqual(
            [result["repo_id"] for result in results],
            ["unsloth/GLM-5.3-Flash-GGUF", "community/GLM-5.3-Flash"],
        )

    def test_non_gguf_matches_do_not_consume_result_limit(self):
        payload = [
            {"modelId": "author/base-model", "tags": ["safetensors"]},
            {"modelId": "author/quantized-GGUF", "tags": []},
        ]

        with patch("backend.app.hf._get_json", return_value=payload):
            results = search_models("model", limit=1)

        self.assertEqual([result["repo_id"] for result in results], ["author/quantized-GGUF"])


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
