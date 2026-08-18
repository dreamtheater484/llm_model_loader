import tempfile
import unittest
import uuid
from pathlib import Path

from backend.app.shards import local_model_files, model_name_from_filename, parse_gguf_shard
from backend.app.storage import Store


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

    def test_storage_migration_consolidates_complete_shard_sets(self):
        db_path = Path(tempfile.gettempdir()) / f"llm-loader-shards-{uuid.uuid4().hex}.db"
        try:
            test_store = Store(db_path)
            for index, size in ((1, 5), (2, 40), (3, 45)):
                test_store.execute(
                    """
                    insert into models(
                        id, name, repo_id, filename, path, normalized_path, size_bytes,
                        quantization, source, managed, display_order, created_at
                    ) values(?, ?, 'owner/repo', ?, ?, ?, ?, 'Q2_K', 'huggingface', 1, ?, ?)
                    """,
                    (
                        f"model_{index}",
                        f"Model-0000{index}-of-00003",
                        f"quant/Model-0000{index}-of-00003.gguf",
                        str(db_path.parent / f"Model-0000{index}-of-00003.gguf"),
                        str(db_path.parent / f"model-0000{index}-of-00003.gguf"),
                        size,
                        index - 1,
                        float(index),
                    ),
                )
            test_store.execute(
                """
                insert into scripts(id, model_id, name, raw_script, parsed_json, created_at, updated_at)
                values('script_3', 'model_3', 'Shard script', '-m model.gguf', '{}', 1, 1)
                """
            )

            migrated = Store(db_path)
            models = migrated.rows("select id, name, size_bytes from models")
            script = migrated.row("select model_id from scripts where id='script_3'")
        finally:
            try:
                db_path.unlink()
            except OSError:
                pass

        self.assertEqual(models, [{"id": "model_1", "name": "Model", "size_bytes": 90}])
        self.assertEqual(script, {"model_id": "model_1"})

    def test_storage_migration_leaves_incomplete_shard_sets_alone(self):
        db_path = Path(tempfile.gettempdir()) / f"llm-loader-shards-{uuid.uuid4().hex}.db"
        try:
            test_store = Store(db_path)
            for index in (1, 3):
                path = str(db_path.parent / f"Model-0000{index}-of-00003.gguf")
                test_store.execute(
                    """
                    insert into models(
                        id, name, repo_id, filename, path, normalized_path, size_bytes,
                        source, managed, display_order, created_at
                    ) values(?, 'Model', 'owner/repo', ?, ?, ?, 1, 'huggingface', 1, ?, ?)
                    """,
                    (f"model_{index}", f"Model-0000{index}-of-00003.gguf", path, path, index, float(index)),
                )

            migrated = Store(db_path)
            count = migrated.row("select count(*) as count from models")
        finally:
            try:
                db_path.unlink()
            except OSError:
                pass

        self.assertEqual(count, {"count": 2})


if __name__ == "__main__":
    unittest.main()
