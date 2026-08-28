import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPARE_PATH = ROOT / "dev_tasks/smoke_lm/prepare_data.py"
SPEC = importlib.util.spec_from_file_location("smoke_lm_prepare_data", PREPARE_PATH)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class SmokeLmPreparationTests(unittest.TestCase):
    def test_sequences_use_bos_bytes_and_respect_record_size(self):
        long_text = "a" * (PREPARE.SEQUENCE_TARGET_TOKENS + 1)
        sequences = PREPARE.make_sequences([long_text])
        self.assertEqual(len(sequences), 2)
        self.assertTrue(all(row[0] == PREPARE.BOS_TOKEN_ID for row in sequences))
        self.assertTrue(
            all(len(row) - 1 <= PREPARE.SEQUENCE_TARGET_TOKENS for row in sequences)
        )
        self.assertEqual(sum(len(row) - 1 for row in sequences), len(long_text))

    def test_eval_token_cap_is_exact(self):
        sequences = PREPARE.make_sequences(["abcdef"], max_target_tokens=4)
        self.assertEqual(sequences, [[PREPARE.BOS_TOKEN_ID, 97, 98, 99, 100]])

    def test_ordered_document_digest_has_unambiguous_boundaries(self):
        self.assertNotEqual(
            PREPARE._ordered_document_digest(["ab", "c"]),
            PREPARE._ordered_document_digest(["a", "bc"]),
        )


if __name__ == "__main__":
    unittest.main()
