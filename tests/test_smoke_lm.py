import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from envcore.benchmark import BenchmarkRelease
from envcore.config import TaskConfig
from envcore.training_data import verify_training_data


ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = ROOT / "dev_tasks/smoke_lm-pretrain"
PRETRAIN_ROOT = ROOT / "research_tasks/lm-pretrain"
PREPARE_PATH = SMOKE_ROOT / "prepare_data.py"
TASK_FIXTURE = ROOT / "tests/fixtures/task/task.json"
SPEC = importlib.util.spec_from_file_location("smoke_lm_prepare_data", PREPARE_PATH)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class SmokeLmPreparationTests(unittest.TestCase):
    def test_preparation_writes_public_inventory_and_invalidates_damaged_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train/train.json"
            with patch.multiple(
                PREPARE,
                DATA_DIR=root,
                TRAIN_PATH=train,
                ANALYSIS_PATH=root / "private/analysis",
                TEST_PATH=root / "private/test",
                METADATA_PATH=root / "metadata.json",
                TRAIN_DOCUMENTS=2,
                ANALYSIS_DOCUMENTS=2,
                TEST_DOCUMENTS=2,
                EVAL_EXAMPLES=2,
                EVAL_CONTEXT_LENGTHS=(1, 2),
            ):
                with patch.object(
                    PREPARE, "download_documents", return_value=["abcdefgh"] * 6
                ), patch("builtins.print"):
                    PREPARE.prepare()
                self.assertEqual(verify_training_data(train.parent)["files"], 1)
                public = json.loads((train.parent / "manifest.json").read_bytes())
                self.assertEqual(
                    set(public), {"source", "tokenizer", "train", "integrity"}
                )
                self.assertNotIn("context_lengths", json.dumps(public))
                self.assertTrue(PREPARE._existing_data_is_valid())
                with patch.object(PREPARE, "download_documents") as download, patch(
                    "builtins.print"
                ):
                    PREPARE.prepare()
                    download.assert_not_called()
                original = train.read_bytes()
                train.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                self.assertFalse(PREPARE._existing_data_is_valid())
                train.write_bytes(original)
                (train.parent / "manifest.json").unlink()
                self.assertFalse(PREPARE._existing_data_is_valid())

    def test_release_uses_the_profile_bound_benchmark_start_contract(self):
        release = BenchmarkRelease.load(SMOKE_ROOT / "benchmark.json")
        task = replace(
            TaskConfig.load(TASK_FIXTURE),
            task_id="tinystories-byte-lm-smoke-v1",
        ).validate()
        resolved = release.resolve(
            task,
            track_config_id="smoke-b-3x75s",
            runtime_profile_id="cpu-local",
        )
        self.assertEqual(resolved.track, "B")
        self.assertEqual(resolved.task.task_kind, "development_smoke")
        self.assertEqual(resolved.task.track_b_max_leases, 3)
        self.assertEqual(resolved.task.track_b_lease_seconds, 75)
        self.assertEqual(resolved.resources.nvidia_gpu_count, 0)
        self.assertFalse(release.leaderboard_eligible)

    def test_policy_and_evaluator_match_the_benchmark(self):
        self.assertEqual(
            (SMOKE_ROOT / "POLICY.md").read_bytes(),
            (PRETRAIN_ROOT / "POLICY.md").read_bytes(),
        )
        self.assertEqual(
            (SMOKE_ROOT / "evaluator/evaluator.py").read_bytes(),
            (PRETRAIN_ROOT / "evaluator/evaluator.py").read_bytes(),
        )

    def test_no_reference_agent_is_bundled(self):
        self.assertFalse((SMOKE_ROOT / "reference_agent").exists())

    def test_sequences_use_bos_bytes_and_respect_record_size(self):
        long_text = "a" * (PREPARE.SEQUENCE_TARGET_TOKENS + 1)
        sequences = PREPARE.make_sequences([long_text])
        self.assertEqual(len(sequences), 2)
        self.assertTrue(all(row[0] == PREPARE.BOS_TOKEN_ID for row in sequences))
        self.assertTrue(
            all(len(row) - 1 <= PREPARE.SEQUENCE_TARGET_TOKENS for row in sequences)
        )
        self.assertEqual(sum(len(row) - 1 for row in sequences), len(long_text))

    def test_evaluation_examples_are_balanced_single_target_and_non_overlapping(self):
        collector = PREPARE.EvaluationCollector(
            target_examples=5,
            context_lengths=(1, 2),
            boundary_token_id=PREPARE.BOS_TOKEN_ID,
        )
        collector.add_document("doc-1", b"abcdefghijk")

        self.assertTrue(collector.complete)
        self.assertEqual(collector.context_length_histogram(), {"1": 3, "2": 2})
        self.assertEqual(
            [example["target"] for example in collector.examples],
            [ord("a"), ord("c"), ord("d"), ord("f"), ord("g")],
        )
        spans = [
            (example["source_start"], example["source_end"])
            for example in collector.examples
        ]
        self.assertEqual(spans, [(0, 1), (1, 3), (3, 4), (4, 6), (6, 7)])
        self.assertTrue(
            all(
                len(example["context"])
                == example["source_end"] - example["source_start"]
                for example in collector.examples
            )
        )
        self.assertEqual(
            collector.context_payload("smoke-test")["candidate_protocol"],
            "offline_next_token_logits_v3",
        )

    def test_evaluation_requires_at_least_one_example_per_context_length(self):
        with self.assertRaisesRegex(ValueError, "cover every"):
            PREPARE.EvaluationCollector(
                target_examples=1,
                context_lengths=(1, 2),
                boundary_token_id=PREPARE.BOS_TOKEN_ID,
            )

    def test_ordered_document_digest_has_unambiguous_boundaries(self):
        self.assertNotEqual(
            PREPARE._ordered_document_digest(["ab", "c"]),
            PREPARE._ordered_document_digest(["a", "bc"]),
        )


if __name__ == "__main__":
    unittest.main()
