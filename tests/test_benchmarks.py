import unittest
from unittest.mock import patch

from backend.app.benchmarks import BenchmarkManager


class BenchmarkHistoryTests(unittest.TestCase):
    def test_history_filters_and_paginates_with_model_metadata(self):
        manager = BenchmarkManager()
        seen = []

        def fake_row(query, params=()):
            seen.append((query, list(params)))
            return {"n": 12}

        def fake_rows(query, params=()):
            seen.append((query, list(params)))
            return [{"id": "bench_1", "model_name": "Model", "script_name": "Script"}]

        with patch("backend.app.benchmarks.store.row", side_effect=fake_row):
            with patch("backend.app.benchmarks.store.rows", side_effect=fake_rows):
                result = manager.history(model_id="model_1", script_id="script_1", preset_id="small", active_only=True, limit=7, offset=14)

        self.assertEqual(result["total"], 12)
        self.assertEqual(result["items"][0]["model_name"], "Model")
        self.assertIn("left join models", seen[1][0])
        self.assertIn("left join scripts", seen[1][0])
        self.assertEqual(seen[1][1], ["model_1", "script_1", "small", 7, 14])


if __name__ == "__main__":
    unittest.main()
