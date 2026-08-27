import json
import tempfile
import unittest
from pathlib import Path

from autotrain_bench.config import TaskConfig


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_public_manifest_hides_private_splits(self):
        config = TaskConfig.load(ROOT / "examples/toy/task.json")
        serialized = json.dumps(config.public_manifest("B"))
        self.assertNotIn(str(config.val_path), serialized)
        self.assertNotIn(str(config.test_path), serialized)
        self.assertEqual(
            config.public_manifest("B")["budget"]["official_val_queries_per_lease"],
            1,
        )


if __name__ == "__main__":
    unittest.main()

