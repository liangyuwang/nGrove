"""Challenge-local regressions for preview holdout capacity; no corpus download."""

import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch


PREPARE = Path(__file__).with_name("prepare_data.py")
SPEC = importlib.util.spec_from_file_location("lm_pretrain_holdout_regression", PREPARE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def collectors(target_examples=MODULE.PREVIEW_EVAL_EXAMPLES):
    return {
        name: MODULE.EvaluationCollector(
            target_examples=target_examples,
            context_lengths=MODULE.EVAL_CONTEXT_LENGTHS,
            boundary_token_id=MODULE.EOT_TOKEN_ID,
        )
        for name in ("analysis", "test")
    }


class PreviewHoldoutTests(unittest.TestCase):
    def test_default_hash_pool_fills_quotas_where_the_old_pool_exhausts(self):
        # Synthetic row identities and document lengths, not FineWeb statistics.
        # Both pools use the actual preview_split and EvaluationCollector code.
        old = collectors()
        current = collectors()
        tokens = [MODULE.EOT_TOKEN_ID] + [1] * 8_192
        for index in range(500_000):
            key = f"000_00000.parquet:{index}"
            split = MODULE.preview_split(key)
            if split == "train":
                continue
            current[split].add_document(key, tokens)
            with patch.object(MODULE, "SPLIT_BUCKETS", 4_096):
                old_split = MODULE.preview_split(key)
            if old_split != "train":
                old[old_split].add_document(key, tokens)

        for name in current:
            self.assertFalse(old[name].complete, name)
            self.assertTrue(current[name].complete, name)
            self.assertEqual(len(current[name].examples), 2_048)
            self.assertEqual(
                current[name].context_length_histogram(),
                {
                    "64": 342,
                    "128": 342,
                    "256": 341,
                    "512": 341,
                    "1024": 341,
                    "2048": 341,
                },
            )
        analysis_sources = {item["source_id"] for item in current["analysis"].examples}
        test_sources = {item["source_id"] for item in current["test"].examples}
        self.assertTrue(analysis_sources.isdisjoint(test_sources))
        MODULE._require_complete_holdouts(current["analysis"], current["test"])
        with self.assertRaises(MODULE.PreparationError):
            MODULE._require_complete_holdouts(old["analysis"], old["test"])

    def test_shortfall_diagnostic_identifies_the_missing_long_context(self):
        groups = collectors(target_examples=6)
        # Earlier short windows consume the only long document. Sufficient
        # document count alone does not establish capacity for every quota.
        for index, length in enumerate(reversed(MODULE.EVAL_CONTEXT_LENGTHS)):
            groups["analysis"].add_document(
                f"analysis-{index}", [MODULE.EOT_TOKEN_ID] + [1] * length
            )
        groups["test"].add_document("test-0", [MODULE.EOT_TOKEN_ID] + [1] * 8_192)

        with self.assertRaises(MODULE.PreparationError) as failure:
            MODULE._require_complete_holdouts(groups["analysis"], groups["test"])
        self.assertIn(
            "source sample did not fill preview ANALYSIS/TEST", str(failure.exception)
        )
        shortfalls = json.loads(str(failure.exception).split("shortfalls=", 1)[1])
        self.assertEqual(
            shortfalls, {"analysis": {"2048": {"actual": 0, "required": 1}}}
        )


if __name__ == "__main__":
    unittest.main()
