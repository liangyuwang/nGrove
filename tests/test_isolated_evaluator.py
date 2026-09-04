import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autotrain_bench.candidate_protocol import (
    CandidateProtocolError,
    load_candidate,
    normalized_logits,
)


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "dev_tasks/smoke_lm/evaluator/evaluator.py"
SPEC = importlib.util.spec_from_file_location("smoke_lm_evaluator", EVALUATOR)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - repository invariant
    raise RuntimeError(f"cannot import evaluator at {EVALUATOR}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ADAPTER_PATH = ROOT / "autotrain_bench/docker_evaluator_adapter.py"
ADAPTER_SPEC = importlib.util.spec_from_file_location(
    "docker_evaluator_adapter", ADAPTER_PATH
)
if ADAPTER_SPEC is None or ADAPTER_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import adapter at {ADAPTER_PATH}")
ADAPTER = importlib.util.module_from_spec(ADAPTER_SPEC)
ADAPTER_SPEC.loader.exec_module(ADAPTER)


class IsolatedEvaluatorTests(unittest.TestCase):
    @staticmethod
    def _write_examples(path, examples):
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "candidate_protocol": "next_token_logits_v2",
                    "examples": examples,
                }
            ),
            encoding="utf-8",
        )

    def test_each_context_is_scored_once_and_targets_are_never_sent(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data.json"
            self._write_examples(
                data,
                [
                    {
                        "context": [0],
                        "target": 9,
                        "source_id": "doc-a",
                        "source_start": 0,
                        "source_end": 1,
                    },
                    {
                        "context": [2, 3],
                        "target": 8,
                        "source_id": "doc-b",
                        "source_start": 0,
                        "source_end": 2,
                    },
                    {
                        "context": [4, 5, 6],
                        "target": 7,
                        "source_id": "doc-c",
                        "source_start": 0,
                        "source_end": 3,
                    },
                    {
                        "context": [1],
                        "target": 9,
                        "source_id": "doc-d",
                        "source_start": 0,
                        "source_end": 1,
                    },
                ],
            )
            calls = []

            def predict(contexts):
                calls.append([list(context) for context in contexts])
                return [[0.0] * 10 for _ in contexts]

            loss = MODULE.evaluate(
                data,
                10,
                predict,
                batch_size=2,
                context_lengths=(1, 2, 3),
            )

            self.assertAlmostEqual(loss, math.log(10), places=7)
            self.assertEqual(calls, [[[0], [1]], [[2, 3]], [[4, 5, 6]]])
            for call in calls:
                self.assertEqual(len({len(context) for context in call}), 1)
            delivered_tokens = {
                token for call in calls for context in call for token in context
            }
            self.assertTrue({7, 8, 9}.isdisjoint(delivered_tokens))

    def test_legacy_progressive_sequence_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data.json"
            data.write_text(
                json.dumps({"sequences": [[0, 1, 2, 0], [2, 1, 0, 2]]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(MODULE.EvaluationError, "schema"):
                MODULE.evaluate(
                    data,
                    3,
                    lambda contexts: [[0.0] * 3 for _ in contexts],
                    batch_size=16,
                    context_lengths=(1, 2, 3),
                )

    def test_overlapping_source_spans_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data.json"
            self._write_examples(
                data,
                [
                    {
                        "context": [0, 1],
                        "target": 2,
                        "source_id": "same-document",
                        "source_start": 0,
                        "source_end": 2,
                    },
                    {
                        "context": [1, 2],
                        "target": 3,
                        "source_id": "same-document",
                        "source_start": 1,
                        "source_end": 3,
                    },
                ],
            )

            with self.assertRaisesRegex(MODULE.EvaluationError, "overlap"):
                MODULE.evaluate(
                    data,
                    4,
                    lambda contexts: [[0.0] * 4 for _ in contexts],
                    batch_size=2,
                    context_lengths=(2,),
                )

    def test_every_release_declared_context_length_must_be_present(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data.json"
            self._write_examples(
                data,
                [
                    {
                        "context": [0],
                        "target": 1,
                        "source_id": "doc",
                        "source_start": 0,
                        "source_end": 1,
                    }
                ],
            )

            with self.assertRaisesRegex(MODULE.EvaluationError, "cover"):
                MODULE.evaluate(
                    data,
                    3,
                    lambda contexts: [[0.0] * 3 for _ in contexts],
                    batch_size=1,
                    context_lengths=(1, 2),
                )

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
            logits = normalized_logits(module, model, [[0], [1, 2]], vocab_size=3)
            self.assertEqual(logits, [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    def test_legacy_sequence_logits_are_rejected(self):
        class Candidate:
            @staticmethod
            def next_token_logits(model, input_ids):
                del model
                return [[[0.0, 0.0, 0.0] for _ in prefix] for prefix in input_ids]

        with self.assertRaises(CandidateProtocolError):
            normalized_logits(Candidate, None, [[0], [1, 2]], vocab_size=3)

    def test_candidate_abi_rejects_boolean_and_string_logits(self):
        for raw_logit in (True, "1.0"):
            with self.subTest(raw_logit=raw_logit):

                class Candidate:
                    @staticmethod
                    def next_token_logits(model, input_ids):
                        del model
                        return [[raw_logit, 0.0, 0.0] for _ in input_ids]

                with self.assertRaisesRegex(
                    CandidateProtocolError, "logits must be numeric"
                ):
                    normalized_logits(Candidate, None, [[0]], vocab_size=3)

    def test_evaluator_client_rejects_boolean_and_string_json_logits(self):
        client = ADAPTER.HttpCandidateClient.__new__(ADAPTER.HttpCandidateClient)
        client.base_url = "http://candidate"
        client.vocab_size = 2
        client.timeout_seconds = 1

        class Response:
            def __init__(self, encoded):
                self.encoded = encoded

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                del exc_type, exc, traceback

            def read(self, maximum_bytes):
                return self.encoded[:maximum_bytes]

        for raw_logit in (True, "1.0"):
            with self.subTest(raw_logit=raw_logit):
                encoded = json.dumps({"logits": [[raw_logit, 0.0]]}).encode("utf-8")
                with patch.object(
                    ADAPTER.urllib.request,
                    "urlopen",
                    return_value=Response(encoded),
                ):
                    with self.assertRaisesRegex(
                        ADAPTER.AdapterError, "logits must be numeric"
                    ):
                        client.predict([[0]])


if __name__ == "__main__":
    unittest.main()
