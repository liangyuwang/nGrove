"""Provisional, dependency-free candidate evaluator used by the toy task.

This is intentionally not the final high-performance Candidate ABI. It calls
the candidate once per prefix so future tokens are never supplied to
participant code.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, List, Sequence


class EvaluationError(RuntimeError):
    pass


def _load_candidate(candidate_dir: Path) -> Any:
    module_path = candidate_dir / "candidate.py"
    if not module_path.is_file():
        raise EvaluationError("candidate.py is missing")
    sys.path.insert(0, str(candidate_dir))
    spec = importlib.util.spec_from_file_location("autotrain_candidate", module_path)
    if spec is None or spec.loader is None:
        raise EvaluationError("candidate.py could not be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "load", None)):
        raise EvaluationError("candidate.py must define load(candidate_dir, device)")
    if not callable(getattr(module, "next_token_logits", None)):
        raise EvaluationError(
            "candidate.py must define next_token_logits(model, input_ids)"
        )
    model = module.load(candidate_dir, "cpu")
    return module, model


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


def _last_logits(value: Any, prefix_length: int, vocab_size: int) -> Sequence[float]:
    try:
        row = value[0][prefix_length - 1]
    except (IndexError, KeyError, TypeError) as exc:
        raise EvaluationError(
            "next_token_logits must return [batch, sequence, vocabulary]"
        ) from exc
    if not isinstance(row, (list, tuple)) or len(row) != vocab_size:
        raise EvaluationError("candidate returned the wrong vocabulary dimension")
    try:
        logits = [float(item) for item in row]
    except (TypeError, ValueError) as exc:
        raise EvaluationError("candidate logits must be numeric") from exc
    if not all(math.isfinite(item) for item in logits):
        raise EvaluationError("candidate logits must be finite")
    return logits


def evaluate(candidate_dir: Path, data_path: Path, vocab_size: int) -> float:
    module, model = _load_candidate(candidate_dir)
    total_nll = 0.0
    token_count = 0
    for sequence in _sequences(data_path, vocab_size):
        for position in range(1, len(sequence)):
            prefix = sequence[:position]
            output = module.next_token_logits(model, [prefix])
            logits = _last_logits(output, len(prefix), vocab_size)
            maximum = max(logits)
            log_normalizer = maximum + math.log(
                sum(math.exp(value - maximum) for value in logits)
            )
            total_nll += log_normalizer - logits[sequence[position]]
            token_count += 1
    if token_count == 0:
        raise EvaluationError("evaluation data contains no prediction targets")
    return total_nll / token_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--vocab-size", required=True, type=int)
    args = parser.parse_args()
    loss = evaluate(args.candidate, args.data, args.vocab_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"metric": "autoregressive_nll", "nll": loss}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

