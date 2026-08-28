import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

from autotrain_bench.candidate_protocol import (
    CandidateProtocolError,
    load_candidate,
    normalized_logits,
)


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "examples/mini_lm/evaluator/evaluator.py"
SPEC = importlib.util.spec_from_file_location("mini_lm_evaluator", EVALUATOR)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - repository invariant
    raise RuntimeError(f"cannot import evaluator at {EVALUATOR}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class IsolatedEvaluatorTests(unittest.TestCase):
    def test_prefixes_are_batched_without_exposing_future_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data.json"
            data.write_text(
                json.dumps({"sequences": [[0, 1, 2, 0], [2, 1, 0, 2]]}),
                encoding="utf-8",
            )
            calls = []

            def predict(prefixes):
                calls.append([list(prefix) for prefix in prefixes])
                return [[0.0, 0.0, 0.0] for _ in prefixes]

            loss = MODULE.evaluate(data, 3, predict, batch_size=3)

            self.assertAlmostEqual(loss, math.log(3), places=7)
            self.assertEqual([len(call) for call in calls], [3, 3])
            self.assertEqual(calls[0], [[0], [0, 1], [0, 1, 2]])

    def test_candidate_abi_is_batch_by_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory)
            (candidate / "candidate.py").write_text(
                "def load(candidate_dir, device): return {'device': device}\n"
                "def next_token_logits(model, input_ids):\n"
                "    return [[float(token == prefix[-1]) for token in range(3)] "
                "for prefix in input_ids]\n",
                encoding="utf-8",
            )
            module, model = load_candidate(candidate, "cpu")
            logits = normalized_logits(
                module, model, [[0], [1, 2]], vocab_size=3
            )
            self.assertEqual(logits, [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    def test_legacy_sequence_logits_are_rejected(self):
        class Candidate:
            @staticmethod
            def next_token_logits(model, input_ids):
                del model
                return [[[0.0, 0.0, 0.0] for _ in prefix] for prefix in input_ids]

        with self.assertRaises(CandidateProtocolError):
            normalized_logits(Candidate, None, [[0], [1, 2]], vocab_size=3)


if __name__ == "__main__":
    unittest.main()
