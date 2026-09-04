"""Trusted single-target NLL evaluator for the AutoTrain-Bench Candidate ABI."""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple


DATASET_SCHEMA_VERSION = 1
CANDIDATE_PROTOCOL = "next_token_logits_v2"


class EvaluationError(RuntimeError):
    pass


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


def _examples(
    data_path: Path,
    vocab_size: int,
    context_lengths: Sequence[int],
) -> List[Tuple[List[int], int]]:
    allowed_lengths = _context_lengths(context_lengths)
    try:
        payload = json.loads(data_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise EvaluationError("evaluation data is invalid") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "candidate_protocol",
        "examples",
    }:
        raise EvaluationError("evaluation data has an invalid schema")
    if payload["schema_version"] != DATASET_SCHEMA_VERSION:
        raise EvaluationError("evaluation data has an unsupported schema version")
    if payload["candidate_protocol"] != CANDIDATE_PROTOCOL:
        raise EvaluationError("evaluation data uses a different Candidate protocol")
    raw_examples = payload["examples"]
    if not isinstance(raw_examples, list) or not raw_examples:
        raise EvaluationError("evaluation data must contain non-empty examples")

    result: List[Tuple[List[int], int]] = []
    spans: Dict[str, List[Tuple[int, int]]] = {}
    observed_lengths = set()
    expected_fields = {
        "context",
        "target",
        "source_id",
        "source_start",
        "source_end",
    }
    for example in raw_examples:
        if not isinstance(example, dict) or set(example) != expected_fields:
            raise EvaluationError("evaluation example has an invalid schema")
        context = example["context"]
        target = example["target"]
        if not isinstance(context, list) or len(context) not in allowed_lengths:
            raise EvaluationError("evaluation context has an undeclared length")
        if not all(
            isinstance(token, int)
            and not isinstance(token, bool)
            and 0 <= token < vocab_size
            for token in context
        ):
            raise EvaluationError("evaluation context token is outside the vocabulary")
        if (
            isinstance(target, bool)
            or not isinstance(target, int)
            or target < 0
            or target >= vocab_size
        ):
            raise EvaluationError("evaluation target is outside the fixed vocabulary")

        source_id = example["source_id"]
        source_start = example["source_start"]
        source_end = example["source_end"]
        if (
            not isinstance(source_id, str)
            or not source_id
            or len(source_id) > 512
            or "\0" in source_id
        ):
            raise EvaluationError("evaluation source_id is invalid")
        if (
            isinstance(source_start, bool)
            or not isinstance(source_start, int)
            or source_start < 0
            or isinstance(source_end, bool)
            or not isinstance(source_end, int)
            or source_end - source_start != len(context)
        ):
            raise EvaluationError("evaluation source span is invalid")
        spans.setdefault(source_id, []).append((source_start, source_end))
        observed_lengths.add(len(context))
        result.append((list(context), target))

    if observed_lengths != set(allowed_lengths):
        raise EvaluationError("evaluation data does not cover every context length")
    for source_spans in spans.values():
        previous_end = -1
        for start, end in sorted(source_spans):
            if start < previous_end:
                raise EvaluationError("evaluation source spans overlap")
            previous_end = end
    return result


def evaluate(
    data_path: Path,
    vocab_size: int,
    predict: Callable[[Sequence[Sequence[int]]], Sequence[Sequence[float]]],
    *,
    batch_size: int,
    context_lengths: Sequence[int],
) -> float:
    """Score exactly one permanently hidden target for each context."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise EvaluationError("batch size must be positive")
    normalized_lengths = _context_lengths(context_lengths)
    examples = _examples(data_path, vocab_size, normalized_lengths)
    buckets: Dict[int, List[Tuple[List[int], int]]] = {
        length: [] for length in normalized_lengths
    }
    for context, target in examples:
        buckets[len(context)].append((context, target))

    losses: List[float] = []
    for length in buckets:
        bucket = buckets[length]
        for start in range(0, len(bucket), batch_size):
            batch = bucket[start : start + batch_size]
            contexts = [context for context, _target in batch]
            targets = [target for _context, target in batch]
            output = predict(contexts)
            if len(output) != len(contexts):
                raise EvaluationError("Candidate returned the wrong batch dimension")
            for logits, target in zip(output, targets):
                if len(logits) != vocab_size:
                    raise EvaluationError(
                        "Candidate returned the wrong vocabulary dimension"
                    )
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(float(value))
                    for value in logits
                ):
                    raise EvaluationError("Candidate logits must be finite numbers")
                normalized = [float(value) for value in logits]
                maximum = max(normalized)
                log_normalizer = maximum + math.log(
                    math.fsum(math.exp(value - maximum) for value in normalized)
                )
                losses.append(log_normalizer - normalized[target])
    if not losses:
        raise EvaluationError("evaluation data contains no prediction targets")
    return math.fsum(losses) / len(losses)
