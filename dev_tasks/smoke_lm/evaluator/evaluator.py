"""Trusted scorer for the offline AutoTrain-Bench Candidate ABI."""

from __future__ import annotations

import hashlib
import json
import math
import stat
import sys
from array import array
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DATASET_SCHEMA_VERSION = 2
PREDICTION_SCHEMA_VERSION = 1
CANDIDATE_PROTOCOL = "offline_next_token_logits_v3"
LOGITS_FILENAME = "logits.f32"
PREDICTIONS_FILENAME = "predictions.json"
PREDICTION_MANIFEST_MAX_BYTES = 64 * 1024


class EvaluationError(RuntimeError):
    pass


def _performance_score(mean_nll: float, vocab_size: int) -> float:
    if not math.isfinite(mean_nll) or mean_nll < 0:
        raise EvaluationError("Candidate mean NLL must be finite and non-negative")
    if vocab_size == 1:
        return 100.0
    return 100.0 / (1.0 + mean_nll / math.log(vocab_size))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EvaluationError("evaluation artifact could not be read") from exc
    return digest.hexdigest()


def _context_lengths(value: Sequence[int]) -> Tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EvaluationError("context lengths must be a sequence")
    result = tuple(value)
    if (
        not result
        or any(
            isinstance(length, bool) or not isinstance(length, int) for length in result
        )
        or any(length <= 0 for length in result)
        or tuple(sorted(set(result))) != result
    ):
        raise EvaluationError(
            "context lengths must be positive, unique, and strictly increasing"
        )
    return result


def _json_object(path: Path, description: str) -> Dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise EvaluationError(f"{description} is invalid") from exc
    if not isinstance(payload, dict):
        raise EvaluationError(f"{description} must be an object")
    return payload


def _identifier(value: object, description: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512 or "\0" in value:
        raise EvaluationError(f"{description} is invalid")
    return value


def _load_dataset(
    contexts_path: Path,
    targets_path: Path,
    vocab_size: int,
    context_lengths: Sequence[int],
) -> Tuple[Dict[str, object], List[int]]:
    allowed_lengths = _context_lengths(context_lengths)
    contexts = _json_object(contexts_path, "evaluation contexts")
    targets = _json_object(targets_path, "evaluation targets")
    if set(contexts) != {
        "schema_version",
        "candidate_protocol",
        "dataset_id",
        "vocab_size",
        "context_lengths",
        "examples",
    }:
        raise EvaluationError("evaluation contexts have an invalid schema")
    if set(targets) != {
        "schema_version",
        "candidate_protocol",
        "dataset_id",
        "contexts_sha256",
        "examples",
    }:
        raise EvaluationError("evaluation targets have an invalid schema")
    if (
        isinstance(contexts["schema_version"], bool)
        or not isinstance(contexts["schema_version"], int)
        or contexts["schema_version"] != DATASET_SCHEMA_VERSION
        or isinstance(targets["schema_version"], bool)
        or not isinstance(targets["schema_version"], int)
        or targets["schema_version"] != DATASET_SCHEMA_VERSION
    ):
        raise EvaluationError("evaluation data has an unsupported schema version")
    if (
        contexts["candidate_protocol"] != CANDIDATE_PROTOCOL
        or targets["candidate_protocol"] != CANDIDATE_PROTOCOL
    ):
        raise EvaluationError("evaluation data uses a different Candidate protocol")
    dataset_id = _identifier(contexts["dataset_id"], "evaluation dataset_id")
    if targets["dataset_id"] != dataset_id:
        raise EvaluationError("evaluation context and target identities differ")
    if (
        isinstance(contexts["vocab_size"], bool)
        or not isinstance(contexts["vocab_size"], int)
        or contexts["vocab_size"] != vocab_size
    ):
        raise EvaluationError("evaluation vocabulary does not match the task")
    if _context_lengths(contexts["context_lengths"]) != allowed_lengths:
        raise EvaluationError("evaluation context lengths do not match the task")
    contexts_sha256 = _sha256_file(contexts_path)
    if targets["contexts_sha256"] != contexts_sha256:
        raise EvaluationError("evaluation targets do not bind the context artifact")

    raw_contexts = contexts["examples"]
    raw_targets = targets["examples"]
    if (
        not isinstance(raw_contexts, list)
        or not raw_contexts
        or not isinstance(raw_targets, list)
        or len(raw_targets) != len(raw_contexts)
    ):
        raise EvaluationError("evaluation examples are missing or misaligned")

    seen_ids = set()
    observed_lengths = set()
    histogram = {length: 0 for length in allowed_lengths}
    protected_spans: Dict[str, List[Tuple[int, int]]] = {}
    ordered_targets: List[int] = []
    input_tokens = 0
    for context_example, target_example in zip(raw_contexts, raw_targets):
        if not isinstance(context_example, dict) or set(context_example) != {
            "example_id",
            "input_ids",
        }:
            raise EvaluationError("evaluation context example has an invalid schema")
        if not isinstance(target_example, dict) or set(target_example) != {
            "example_id",
            "target",
            "source_id",
            "source_start",
            "source_end",
        }:
            raise EvaluationError("evaluation target example has an invalid schema")
        example_id = _identifier(context_example["example_id"], "example_id")
        if target_example["example_id"] != example_id or example_id in seen_ids:
            raise EvaluationError("evaluation example IDs are duplicated or misaligned")
        seen_ids.add(example_id)

        input_ids = context_example["input_ids"]
        if not isinstance(input_ids, list) or len(input_ids) not in allowed_lengths:
            raise EvaluationError("evaluation context has an undeclared length")
        if not all(
            isinstance(token, int)
            and not isinstance(token, bool)
            and 0 <= token < vocab_size
            for token in input_ids
        ):
            raise EvaluationError("evaluation context token is outside the vocabulary")
        target = target_example["target"]
        if (
            isinstance(target, bool)
            or not isinstance(target, int)
            or target < 0
            or target >= vocab_size
        ):
            raise EvaluationError("evaluation target is outside the fixed vocabulary")

        source_id = _identifier(target_example["source_id"], "source_id")
        source_start = target_example["source_start"]
        source_end = target_example["source_end"]
        if (
            isinstance(source_start, bool)
            or not isinstance(source_start, int)
            or source_start < 0
            or isinstance(source_end, bool)
            or not isinstance(source_end, int)
            or source_end - source_start != len(input_ids)
        ):
            raise EvaluationError("evaluation protected source span is invalid")
        protected_spans.setdefault(source_id, []).append((source_start, source_end))
        observed_lengths.add(len(input_ids))
        histogram[len(input_ids)] += 1
        input_tokens += len(input_ids)
        ordered_targets.append(target)

    if observed_lengths != set(allowed_lengths):
        raise EvaluationError("evaluation data does not cover every context length")
    for source_spans in protected_spans.values():
        previous_end = -1
        for start, end in sorted(source_spans):
            if start < previous_end:
                raise EvaluationError("evaluation protected source spans overlap")
            previous_end = end

    return (
        {
            "dataset_id": dataset_id,
            "contexts_sha256": contexts_sha256,
            "examples": len(raw_contexts),
            "input_tokens": input_tokens,
            "context_length_histogram": {
                str(length): histogram[length] for length in allowed_lengths
            },
        },
        ordered_targets,
    )


def validate_inputs(
    contexts_path: Path,
    targets_path: Path,
    vocab_size: int,
    *,
    context_lengths: Sequence[int],
) -> Dict[str, object]:
    """Validate both artifacts before the target-free part reaches Candidate code."""

    statistics, _targets = _load_dataset(
        contexts_path, targets_path, vocab_size, context_lengths
    )
    return statistics


def _prediction_paths(predictions_dir: Path) -> Tuple[Path, Path]:
    try:
        entries = {entry.name: entry for entry in predictions_dir.iterdir()}
    except OSError as exc:
        raise EvaluationError("Candidate prediction artifact is unavailable") from exc
    if set(entries) != {PREDICTIONS_FILENAME, LOGITS_FILENAME}:
        raise EvaluationError("Candidate prediction artifact has unexpected entries")
    for entry in entries.values():
        try:
            metadata = entry.lstat()
        except OSError as exc:
            raise EvaluationError(
                "Candidate prediction artifact is unavailable"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise EvaluationError("Candidate prediction entries must be regular files")
    if entries[PREDICTIONS_FILENAME].stat().st_size > PREDICTION_MANIFEST_MAX_BYTES:
        raise EvaluationError("Candidate prediction manifest exceeds its byte limit")
    return entries[PREDICTIONS_FILENAME], entries[LOGITS_FILENAME]


def evaluate(
    contexts_path: Path,
    targets_path: Path,
    predictions_dir: Path,
    vocab_size: int,
    *,
    context_lengths: Sequence[int],
) -> Dict[str, object]:
    """Score one permanently hidden target per context from an offline logit file."""

    statistics, targets = _load_dataset(
        contexts_path, targets_path, vocab_size, context_lengths
    )
    manifest_path, logits_path = _prediction_paths(predictions_dir)
    manifest = _json_object(manifest_path, "Candidate prediction manifest")
    expected_manifest = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "candidate_protocol": CANDIDATE_PROTOCOL,
        "dataset_id": statistics["dataset_id"],
        "examples": statistics["examples"],
        "vocab_size": vocab_size,
        "dtype": "float32-le",
        "logits_file": LOGITS_FILENAME,
    }
    if (
        set(manifest) != set(expected_manifest)
        or isinstance(manifest.get("schema_version"), bool)
        or not isinstance(manifest.get("schema_version"), int)
        or isinstance(manifest.get("examples"), bool)
        or not isinstance(manifest.get("examples"), int)
        or isinstance(manifest.get("vocab_size"), bool)
        or not isinstance(manifest.get("vocab_size"), int)
        or manifest != expected_manifest
    ):
        raise EvaluationError(
            "Candidate prediction manifest does not match the request"
        )

    row_bytes = vocab_size * 4
    expected_bytes = len(targets) * row_bytes
    try:
        actual_bytes = logits_path.stat().st_size
    except OSError as exc:
        raise EvaluationError("Candidate logits are unavailable") from exc
    if actual_bytes != expected_bytes:
        raise EvaluationError("Candidate logits have the wrong byte length")
    if array("f").itemsize != 4:
        raise EvaluationError("Evaluator platform lacks IEEE-754 float32 arrays")

    losses: List[float] = []
    try:
        with logits_path.open("rb") as stream:
            for target in targets:
                encoded = stream.read(row_bytes)
                if len(encoded) != row_bytes:
                    raise EvaluationError("Candidate logits ended unexpectedly")
                logits = array("f")
                logits.frombytes(encoded)
                if sys.byteorder != "little":
                    logits.byteswap()
                if any(not math.isfinite(value) for value in logits):
                    raise EvaluationError("Candidate logits must be finite float32")
                maximum = max(logits)
                log_normalizer = maximum + math.log(
                    math.fsum(math.exp(value - maximum) for value in logits)
                )
                losses.append(log_normalizer - logits[target])
            if stream.read(1):
                raise EvaluationError("Candidate logits contain trailing bytes")
    except OSError as exc:
        raise EvaluationError("Candidate logits could not be read") from exc

    mean_nll = math.fsum(losses) / len(losses)
    performance_score = _performance_score(mean_nll, vocab_size)

    return {
        "performance_score": performance_score,
        "raw_metrics": {
            "mean_nll": mean_nll,
        },
        "examples": statistics["examples"],
        "input_tokens": statistics["input_tokens"],
        "prediction_bytes": manifest_path.stat().st_size + actual_bytes,
    }
