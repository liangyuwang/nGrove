import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchcore.candidate_protocol import (
    CANDIDATE_PROTOCOL,
    PREDICTION_MANIFEST_MAX_BYTES,
    CandidateProtocolError,
    load_candidate,
    load_context_request,
    validate_prediction_artifact,
    write_prediction_artifact,
)
from benchcore.worker_protocol import PerformanceScoreContract
from benchcore.config import TaskConfig


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "dev_tasks/smoke_lm/evaluator/evaluator.py"
SPEC = importlib.util.spec_from_file_location("smoke_lm_evaluator", EVALUATOR)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - repository invariant
    raise RuntimeError(f"cannot import evaluator at {EVALUATOR}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class IsolatedEvaluatorTests(unittest.TestCase):
    def test_context_fields_and_reader_return_match_the_public_declaration(self):
        boundary = TaskConfig.load(
            ROOT / "tests/fixtures/task/task.json"
        ).public_manifest("B")["evaluation"]["candidate_boundary"]
        contract = boundary["context_manifest"]
        with tempfile.TemporaryDirectory() as temporary:
            contexts, targets = self._examples()
            path, target_path = self._write_split(
                Path(temporary) / "split", contexts, targets
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(load_context_request(path), payload)
            self.assertEqual(set(payload), set(contract["exact_fields"]))
            for name, spec in contract["fields"].items():
                self.assertIs(
                    type(payload[name]),
                    {"integer": int, "string": str, "array": list}[spec["json_type"]],
                )
                if "constant" in spec:
                    self.assertEqual(payload[name], spec["constant"])
            example_contract = contract["fields"]["examples"]["items"]
            for example in payload["examples"]:
                self.assertEqual(set(example), set(example_contract["exact_fields"]))
                self.assertIsInstance(example["example_id"], str)
                self.assertIsInstance(example["input_ids"], list)
                self.assertIn(len(example["input_ids"]), payload["context_lengths"])
            self.assertEqual(
                MODULE.validate_inputs(
                    path, target_path, 10, context_lengths=(1, 2, 3)
                )["examples"],
                len(payload["examples"]),
            )

    def test_reader_and_evaluator_reject_undeclared_context_keys_and_values(self):
        invalid_examples = [
            {"example_id": "one", "context": [0]},
            {"example_id": "one", "tokens": [0]},
            {"example_id": "one", "input_ids": None},
            {"example_id": "one", "input_ids": [True]},
            {"example_id": "one", "input_ids": [-1]},
            {"example_id": "one", "input_ids": [10]},
            {"example_id": "one", "input_ids": [0], "target": 1},
        ]
        for invalid in invalid_examples:
            with self.subTest(
                example=invalid
            ), tempfile.TemporaryDirectory() as temporary:
                contexts, targets = self._examples()
                contexts[0] = invalid
                path, target_path = self._write_split(
                    Path(temporary) / "split", contexts, targets
                )
                with self.assertRaises(CandidateProtocolError):
                    load_context_request(path)
                with self.assertRaises(MODULE.EvaluationError):
                    MODULE.validate_inputs(
                        path, target_path, 10, context_lengths=(1, 2, 3)
                    )

    def test_reader_and_evaluator_require_an_integer_context_schema_version(self):
        for invalid in (2.0, True, "2"):
            with self.subTest(
                version=invalid
            ), tempfile.TemporaryDirectory() as temporary:
                contexts, targets = self._examples()
                path, target_path = self._write_split(
                    Path(temporary) / "split", contexts, targets
                )
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["schema_version"] = invalid
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(CandidateProtocolError, "schema"):
                    load_context_request(path)
                with self.assertRaisesRegex(MODULE.EvaluationError, "schema"):
                    MODULE.validate_inputs(
                        path, target_path, 10, context_lengths=(1, 2, 3)
                    )

    def test_public_contract_files_round_trip_through_runner_and_both_scorers(self):
        boundary = TaskConfig.load(
            ROOT / "tests/fixtures/task/task.json"
        ).public_manifest("B")["evaluation"]["candidate_boundary"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contexts, targets = self._examples()
            for context, target in zip(contexts, targets):
                target["target"] = context["input_ids"][-1]
            context_path, target_path = self._write_split(
                root / "split", contexts, targets
            )
            candidate = root / "candidate"
            candidate.mkdir()
            (candidate / "contract.json").write_text(
                json.dumps(boundary), encoding="utf-8"
            )
            (candidate / boundary["entrypoint"]["module_file"]).write_text(
                """
import json, struct
def run_offline(candidate_dir, contexts_path, output_dir, device):
    contract = json.loads((candidate_dir / "contract.json").read_text())
    request = json.loads(contexts_path.read_text())
    fields = contract["context_manifest"]["fields"]
    examples_key = next(name for name, spec in fields.items()
                        if spec["json_type"] == "array" and spec["items"]["json_type"] == "object")
    example_fields = fields[examples_key]["items"]["fields"]
    input_key = next(name for name, spec in example_fields.items() if spec["json_type"] == "array")
    rows = request[examples_key]
    vocab = request["vocab_size"]
    prediction = contract["prediction_manifest"]
    sources = {
        "contexts.json.dataset_id": request["dataset_id"],
        "len(contexts.json.examples)": len(rows),
        "contexts.json.vocab_size": vocab,
    }
    manifest = {name: spec["constant"] if "constant" in spec else sources[spec["source"]]
                for name, spec in prediction["fields"].items()}
    output_dir.mkdir(exist_ok=True)
    (output_dir / prediction["filename"]).write_text(json.dumps(manifest))
    layout = prediction["logits_artifact"]
    assert layout["layout"] == "row_major" and layout["header_bytes"] == 0
    encoding = {"ieee754_float32_little_endian": "<f"}[layout["encoding"]]
    with (output_dir / layout["filename"]).open("wb") as stream:
        for row in rows:
            for token in range(vocab):
                stream.write(struct.pack(encoding, 2.0 if token == row[input_key][-1] else 0.0))
""",
                encoding="utf-8",
            )
            output = root / "predictions"
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchcore.candidate_offline_runner",
                    "--candidate",
                    str(candidate),
                    "--contexts",
                    str(context_path),
                    "--output",
                    str(output),
                    "--device",
                    "cpu",
                ],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for task in ("dev_tasks/smoke_lm", "challenges/lm-pretrain-v1"):
                with self.subTest(task=task):
                    spec = importlib.util.spec_from_file_location(
                        "contract_scorer", ROOT / task / "evaluator/evaluator.py"
                    )
                    evaluator = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(evaluator)
                    score = evaluator.evaluate(
                        context_path, target_path, output, 10, context_lengths=(1, 2, 3)
                    )
                    self.assertAlmostEqual(
                        score["raw_metrics"]["mean_nll"], math.log(math.exp(2) + 9) - 2
                    )

    def test_uniform_scores_are_invariant_to_large_finite_logit_offsets(self):
        for task in ("dev_tasks/smoke_lm", "challenges/lm-pretrain-v1"):
            spec = importlib.util.spec_from_file_location(
                "shifted_logit_evaluator", ROOT / task / "evaluator/evaluator.py"
            )
            evaluator = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(evaluator)
            for vocab in (257, 50257):
                for offset in (0.0, 1e20, -1e20, 3e38, -3e38):
                    with self.subTest(task=task, vocab=vocab, offset=offset):
                        with tempfile.TemporaryDirectory() as temporary:
                            root = Path(temporary)
                            contexts, targets = self._examples()
                            c, t = self._write_split(
                                root / "split", contexts, targets, vocab=vocab
                            )
                            write_prediction_artifact(
                                root / "predictions",
                                dataset_id="test-split",
                                logits=([offset] * vocab for _ in contexts),
                                examples=len(contexts),
                                vocab_size=vocab,
                            )
                            result = evaluator.evaluate(
                                c,
                                t,
                                root / "predictions",
                                vocab,
                                context_lengths=(1, 2, 3),
                            )
                            self.assertAlmostEqual(
                                result["raw_metrics"]["mean_nll"], math.log(vocab)
                            )
                            self.assertAlmostEqual(result["performance_score"], 0.0)

    def test_public_path_helpers_accept_strings_and_path_objects(self):
        for path_type in (str, Path):
            with self.subTest(
                path_type=path_type
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                contexts, targets = self._examples()
                c, _ = self._write_split(root / "split", contexts, targets)
                request = load_context_request(path_type(c))
                self.assertEqual(len(request["examples"]), 3)
                write_prediction_artifact(
                    path_type(root / "predictions"),
                    dataset_id="test-split",
                    logits=([0.0] * 10 for _ in contexts),
                    examples=3,
                    vocab_size=10,
                )
                self.assertGreater(
                    validate_prediction_artifact(
                        path_type(root / "predictions"),
                        dataset_id="test-split",
                        examples=3,
                        vocab_size=10,
                    ),
                    0,
                )
                with self.assertRaisesRegex(
                    CandidateProtocolError, "context request is invalid"
                ):
                    load_context_request(path_type(root / "missing.json"))

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
            self.assertAlmostEqual(
                result["raw_metrics"]["mean_nll"], math.log(10), places=7
            )
            self.assertEqual(set(result["raw_metrics"]), {"mean_nll"})
            self.assertAlmostEqual(result["performance_score"], 0.0, places=7)
            self.assertEqual(result["examples"], 3)
            self.assertEqual(result["input_tokens"], 6)
            self.assertGreater(result["prediction_bytes"], 3 * 10 * 4)

    def test_performance_score_is_normalized_against_the_uniform_baseline(self):
        uniform_nll = math.log(10)
        contract = PerformanceScoreContract(
            "uniform_normalized_log_score_v1", ("mean_nll",)
        )

        perfect = MODULE._performance_score(0.0, 10)
        halfway = MODULE._performance_score(uniform_nll / 2.0, 10)
        uniform = MODULE._performance_score(uniform_nll, 10)
        below_uniform = MODULE._performance_score(uniform_nll * 2.0, 10)

        self.assertEqual(perfect, 100.0)
        self.assertEqual(halfway, 50.0)
        self.assertEqual(uniform, 0.0)
        self.assertEqual(below_uniform, 0.0)
        self.assertEqual(MODULE._performance_score(1e300, 10), 0.0)
        for mean_nll in (0.0, uniform_nll, uniform_nll * 2.0, 1e300):
            contract.validate_measurement(
                MODULE._performance_score(mean_nll, 10),
                {"mean_nll": mean_nll},
            )
        for invalid_nll in (-0.1, float("nan"), float("inf")):
            with self.subTest(invalid_nll=invalid_nll):
                with self.assertRaisesRegex(
                    MODULE.EvaluationError, "finite and non-negative"
                ):
                    MODULE._performance_score(invalid_nll, 10)

    def test_large_finite_nll_is_not_rejected_by_perplexity_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contexts, targets = self._examples()
            context_path, target_path = self._write_split(
                root / "split", contexts, targets
            )
            rows = []
            for target in targets:
                logits = [0.0] * 10
                logits[target["target"]] = -1e30
                rows.append(logits)
            predictions = root / "predictions"
            write_prediction_artifact(
                predictions,
                dataset_id="test-split",
                logits=rows,
                examples=len(rows),
                vocab_size=10,
            )

            result = MODULE.evaluate(
                context_path,
                target_path,
                predictions,
                10,
                context_lengths=(1, 2, 3),
            )

            self.assertTrue(math.isfinite(result["raw_metrics"]["mean_nll"]))
            self.assertEqual(result["performance_score"], 0.0)

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
            with self.assertRaisesRegex(CandidateProtocolError, "cover"):
                load_context_request(context_path)
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

    def test_prediction_manifest_errors_identify_the_invalid_contract_part(self):
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
            valid = json.loads(manifest_path.read_text(encoding="utf-8"))
            cases = [
                (
                    "missing field",
                    {key: value for key, value in valid.items() if key != "dtype"},
                    r'missing fields \["dtype"\]',
                ),
                (
                    "unexpected field",
                    {**valid, "format": "row-major"},
                    r'unexpected fields \["format"\]',
                ),
                (
                    "schema version",
                    {**valid, "schema_version": True},
                    r'field "schema_version".*integer constant 1',
                ),
                (
                    "protocol",
                    {**valid, "candidate_protocol": "offline_logits_v2"},
                    r'field "candidate_protocol".*offline_next_token_logits_v3',
                ),
                (
                    "dataset",
                    {**valid, "dataset_id": "guessed-split"},
                    r'field "dataset_id".*contexts\.json\.dataset_id',
                ),
                (
                    "example count",
                    {**valid, "examples": 2},
                    r'field "examples".*len\(contexts\.json\.examples\)',
                ),
                (
                    "vocabulary",
                    {**valid, "vocab_size": 4},
                    r'field "vocab_size".*contexts\.json\.vocab_size',
                ),
                (
                    "dtype",
                    {**valid, "dtype": "float32"},
                    r'field "dtype".*float32-le',
                ),
                (
                    "logit filename",
                    {**valid, "logits_file": "predictions.f32"},
                    r'field "logits_file".*logits\.f32',
                ),
            ]

            for name, manifest, message in cases:
                with self.subTest(name=name):
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    with self.assertRaisesRegex(CandidateProtocolError, message):
                        validate_prediction_artifact(
                            predictions,
                            dataset_id="test-split",
                            examples=1,
                            vocab_size=3,
                        )

    def test_prediction_manifest_is_bounded_before_json_loading(self):
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
            (predictions / "predictions.json").write_bytes(
                b" " * (PREDICTION_MANIFEST_MAX_BYTES + 1)
            )

            with self.assertRaisesRegex(CandidateProtocolError, "byte limit"):
                validate_prediction_artifact(
                    predictions,
                    dataset_id="test-split",
                    examples=len(contexts),
                    vocab_size=10,
                )
            with self.assertRaisesRegex(MODULE.EvaluationError, "byte limit"):
                MODULE.evaluate(
                    context_path,
                    target_path,
                    predictions,
                    10,
                    context_lengths=(1, 2, 3),
                )


if __name__ == "__main__":
    unittest.main()
