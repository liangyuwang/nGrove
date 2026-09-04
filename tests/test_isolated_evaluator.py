import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

from autotrain_bench.candidate_protocol import (
    CANDIDATE_PROTOCOL,
    CandidateProtocolError,
    load_candidate,
    load_context_request,
    validate_prediction_artifact,
    write_prediction_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "dev_tasks/smoke_lm/evaluator/evaluator.py"
SPEC = importlib.util.spec_from_file_location("smoke_lm_evaluator", EVALUATOR)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - repository invariant
    raise RuntimeError(f"cannot import evaluator at {EVALUATOR}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class IsolatedEvaluatorTests(unittest.TestCase):
    @staticmethod
    def _write_split(root, contexts, targets, *, lengths=(1, 2, 3), vocab=10):
        root.mkdir()
        context_path = root / "contexts.json"
        target_path = root / "targets.json"
        context_payload = {
            "schema_version": 2,
            "candidate_protocol": CANDIDATE_PROTOCOL,
            "dataset_id": "test-split",
            "vocab_size": vocab,
            "context_lengths": list(lengths),
            "examples": contexts,
        }
        context_path.write_text(
            json.dumps(context_payload, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        target_payload = {
            "schema_version": 2,
            "candidate_protocol": CANDIDATE_PROTOCOL,
            "dataset_id": "test-split",
            "contexts_sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
            "examples": targets,
        }
        target_path.write_text(
            json.dumps(target_payload, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return context_path, target_path

    @staticmethod
    def _examples():
        contexts = [
            {"example_id": "one", "input_ids": [0]},
            {"example_id": "two", "input_ids": [2, 3]},
            {"example_id": "three", "input_ids": [4, 5, 6]},
        ]
        targets = [
            {
                "example_id": "one",
                "target": 9,
                "source_id": "doc-a",
                "source_start": 0,
                "source_end": 1,
            },
            {
                "example_id": "two",
                "target": 8,
                "source_id": "doc-b",
                "source_start": 0,
                "source_end": 2,
            },
            {
                "example_id": "three",
                "target": 7,
                "source_id": "doc-c",
                "source_start": 0,
                "source_end": 3,
            },
        ]
        return contexts, targets

    def test_target_free_contexts_are_scored_once_from_offline_logits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contexts, targets = self._examples()
            context_path, target_path = self._write_split(
                root / "split", contexts, targets
            )
            predictions = root / "predictions"
            write_prediction_artifact(
                predictions,
                dataset_id="test-split",
                logits=([0.0] * 10 for _ in contexts),
                examples=len(contexts),
                vocab_size=10,
            )

            visible = load_context_request(context_path)
            result = MODULE.evaluate(
                context_path,
                target_path,
                predictions,
                10,
                context_lengths=(1, 2, 3),
            )

            self.assertNotIn("target", json.dumps(visible))
            self.assertAlmostEqual(result["nll"], math.log(10), places=7)
            self.assertEqual(result["examples"], 3)
            self.assertEqual(result["input_tokens"], 6)
            self.assertGreater(result["prediction_bytes"], 3 * 10 * 4)

    def test_legacy_combined_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contexts = root / "contexts.json"
            targets = root / "targets.json"
            contexts.write_text(
                json.dumps({"sequences": [[0, 1, 2, 0]]}), encoding="utf-8"
            )
            targets.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(MODULE.EvaluationError, "schema"):
                MODULE.validate_inputs(contexts, targets, 3, context_lengths=(1, 2, 3))

    def test_overlapping_source_spans_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            contexts = [
                {"example_id": "a", "input_ids": [0, 1]},
                {"example_id": "b", "input_ids": [1, 2]},
            ]
            targets = [
                {
                    "example_id": "a",
                    "target": 2,
                    "source_id": "same",
                    "source_start": 0,
                    "source_end": 2,
                },
                {
                    "example_id": "b",
                    "target": 3,
                    "source_id": "same",
                    "source_start": 1,
                    "source_end": 3,
                },
            ]
            context_path, target_path = self._write_split(
                Path(directory) / "split",
                contexts,
                targets,
                lengths=(2,),
                vocab=4,
            )
            with self.assertRaisesRegex(MODULE.EvaluationError, "overlap"):
                MODULE.validate_inputs(
                    context_path, target_path, 4, context_lengths=(2,)
                )

    def test_every_release_context_length_must_be_present(self):
        with tempfile.TemporaryDirectory() as directory:
            context_path, target_path = self._write_split(
                Path(directory) / "split",
                [{"example_id": "a", "input_ids": [0]}],
                [
                    {
                        "example_id": "a",
                        "target": 1,
                        "source_id": "doc",
                        "source_start": 0,
                        "source_end": 1,
                    }
                ],
                lengths=(1, 2),
                vocab=3,
            )
            with self.assertRaisesRegex(MODULE.EvaluationError, "cover"):
                MODULE.validate_inputs(
                    context_path, target_path, 3, context_lengths=(1, 2)
                )

    def test_targets_are_bound_to_exact_context_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            contexts, targets = self._examples()
            context_path, target_path = self._write_split(
                Path(directory) / "split", contexts, targets
            )
            context_path.write_text(
                context_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )
            with self.assertRaisesRegex(MODULE.EvaluationError, "bind"):
                MODULE.validate_inputs(
                    context_path, target_path, 10, context_lengths=(1, 2, 3)
                )

    def test_candidate_abi_owns_the_complete_offline_run(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory)
            (candidate / "candidate.py").write_text(
                "def run_offline(candidate_dir, contexts_path, output_dir, device):\n"
                "    return (candidate_dir, contexts_path, output_dir, device)\n",
                encoding="utf-8",
            )
            module = load_candidate(candidate)
            self.assertTrue(callable(module.run_offline))

            (candidate / "candidate.py").write_text(
                "def load(candidate_dir, device): return None\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CandidateProtocolError, "run_offline"):
                load_candidate(candidate)

    def test_prediction_artifact_rejects_bad_shapes_and_values(self):
        for row, message in (([0.0, 0.0], "vocabulary"), ([True] * 3, "finite")):
            with self.subTest(row=row), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(CandidateProtocolError, message):
                    write_prediction_artifact(
                        Path(directory) / "predictions",
                        dataset_id="test-split",
                        logits=[row],
                        examples=1,
                        vocab_size=3,
                    )

    def test_target_free_validator_checks_raw_logit_size(self):
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions"
            write_prediction_artifact(
                predictions,
                dataset_id="test-split",
                logits=[[0.0, 0.0, 0.0]],
                examples=1,
                vocab_size=3,
            )
            self.assertGreater(
                validate_prediction_artifact(
                    predictions,
                    dataset_id="test-split",
                    examples=1,
                    vocab_size=3,
                ),
                12,
            )
            with (predictions / "logits.f32").open("ab") as stream:
                stream.write(b"x")
            with self.assertRaisesRegex(CandidateProtocolError, "byte length"):
                validate_prediction_artifact(
                    predictions,
                    dataset_id="test-split",
                    examples=1,
                    vocab_size=3,
                )

    def test_prediction_manifest_rejects_boolean_integer_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions"
            write_prediction_artifact(
                predictions,
                dataset_id="test-split",
                logits=[[0.0, 0.0, 0.0]],
                examples=1,
                vocab_size=3,
            )
            manifest_path = predictions / "predictions.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["examples"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(CandidateProtocolError, "manifest"):
                validate_prediction_artifact(
                    predictions,
                    dataset_id="test-split",
                    examples=1,
                    vocab_size=3,
                )


if __name__ == "__main__":
    unittest.main()
