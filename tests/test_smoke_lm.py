import importlib.util
import unittest
from dataclasses import replace
from pathlib import Path

from autotrain_bench.challenge import ChallengeRelease
from autotrain_bench.config import TaskConfig


ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = ROOT / "dev_tasks/smoke_lm"
PRETRAIN_ROOT = ROOT / "challenges/lm-pretrain-v1"
PREPARE_PATH = SMOKE_ROOT / "prepare_data.py"
TASK_FIXTURE = ROOT / "tests/fixtures/task/task.json"
SPEC = importlib.util.spec_from_file_location("smoke_lm_prepare_data", PREPARE_PATH)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class SmokeLmPreparationTests(unittest.TestCase):
    def test_release_uses_the_profile_bound_challenge_start_contract(self):
        release = ChallengeRelease.load(SMOKE_ROOT / "challenge.json")
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

    def test_policy_and_evaluator_match_the_challenge(self):
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
