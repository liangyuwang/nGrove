"""Trusted NLL evaluator for the AutoTrain-Bench Candidate ABI."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, List, Sequence


class EvaluationError(RuntimeError):
    pass


def _sequences(data_path: Path, vocab_size: int) -> List[List[int]]:
    try:
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        sequences = payload["sequences"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError, TypeError) as exc:
        raise EvaluationError("evaluation data is invalid") from exc
    if not isinstance(sequences, list) or not sequences:
        raise EvaluationError("evaluation data must contain non-empty sequences")
    result: List[List[int]] = []
    for sequence in sequences:
        if not isinstance(sequence, list) or len(sequence) < 2:
            raise EvaluationError("every evaluation sequence needs at least two tokens")
        if not all(
            isinstance(token, int)
            and not isinstance(token, bool)
            and 0 <= token < vocab_size
            for token in sequence
        ):
            raise EvaluationError("evaluation token is outside the fixed vocabulary")
        result.append(sequence)
    return result


def evaluate(
    data_path: Path,
    vocab_size: int,
    predict: Callable[[Sequence[Sequence[int]]], Sequence[Sequence[float]]],
    *,
    batch_size: int,
) -> float:
    if batch_size <= 0:
        raise EvaluationError("batch size must be positive")
    total_nll = 0.0
    token_count = 0
    prefixes: List[List[int]] = []
    targets: List[int] = []

    def evaluate_batch() -> None:
        nonlocal total_nll, token_count
        if not prefixes:
            return
        output = predict(prefixes)
        if len(output) != len(prefixes):
            raise EvaluationError("Candidate returned the wrong batch dimension")
        for logits, target in zip(output, targets):
            if len(logits) != vocab_size:
                raise EvaluationError(
                    "Candidate returned the wrong vocabulary dimension"
                )
            maximum = max(logits)
            log_normalizer = maximum + math.log(
                sum(math.exp(value - maximum) for value in logits)
            )
            total_nll += log_normalizer - logits[target]
            token_count += 1
        prefixes.clear()
        targets.clear()

    sequences = _sequences(data_path, vocab_size)
    maximum_length = max(len(sequence) for sequence in sequences)
    # Never mix prefix lengths in one request. Score a position before sending
    # any longer prefix, so the untrusted Candidate cannot recover a target
    # from another row or from a later request already in flight.
    for position in range(1, maximum_length):
        for sequence in sequences:
            if position >= len(sequence):
                continue
            prefixes.append(sequence[:position])
            targets.append(sequence[position])
            if len(prefixes) >= batch_size:
                evaluate_batch()
        evaluate_batch()
    if token_count == 0:
        raise EvaluationError("evaluation data contains no prediction targets")
    return total_nll / token_count
