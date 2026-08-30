"""Trusted NLL evaluator for the AutoTrain-Bench Candidate ABI."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Sequence


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


class CandidateClient:
    def __init__(
        self,
        base_url: str,
        *,
        vocab_size: int,
        timeout_seconds: float,
        readiness_seconds: float = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.vocab_size = vocab_size
        self.timeout_seconds = timeout_seconds
        self._wait_until_ready(readiness_seconds)

    def _wait_until_ready(self, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_error: Optional[BaseException] = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"{self.base_url}/health", timeout=1
                ) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError) as exc:
                last_error = exc
                time.sleep(0.05)
        raise EvaluationError(f"Candidate server did not become ready: {last_error}")

    def predict(self, prefixes: Sequence[Sequence[int]]) -> List[List[float]]:
        body = json.dumps(
            {"input_ids": prefixes}, separators=(",", ":")
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/next-token-logits",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        maximum_response_bytes = max(
            1024 * 1024,
            len(prefixes) * self.vocab_size * 32,
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                encoded = response.read(maximum_response_bytes + 1)
        except (OSError, urllib.error.URLError) as exc:
            raise EvaluationError("Candidate inference request failed") from exc
        if len(encoded) > maximum_response_bytes:
            raise EvaluationError("Candidate response exceeds the protocol limit")
        try:
            payload = json.loads(encoded)
            value = payload["logits"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise EvaluationError("Candidate returned malformed JSON") from exc
        if not isinstance(value, list) or len(value) != len(prefixes):
            raise EvaluationError("Candidate must return [batch, vocabulary] logits")
        result: List[List[float]] = []
        for row in value:
            if not isinstance(row, list) or len(row) != self.vocab_size:
                raise EvaluationError("Candidate returned the wrong vocabulary dimension")
            if any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in row
            ):
                raise EvaluationError("Candidate logits must be numeric")
            try:
                normalized = [float(item) for item in row]
            except (OverflowError, TypeError, ValueError) as exc:
                raise EvaluationError("Candidate logits must be numeric") from exc
            if not all(math.isfinite(item) for item in normalized):
                raise EvaluationError("Candidate logits must be finite")
            result.append(normalized)
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
                raise EvaluationError("Candidate returned the wrong vocabulary dimension")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--request-timeout-seconds", type=float, default=120)
    args = parser.parse_args()
    client = CandidateClient(
        args.candidate_url,
        vocab_size=args.vocab_size,
        timeout_seconds=args.request_timeout_seconds,
    )
    loss = evaluate(
        args.data,
        args.vocab_size,
        client.predict,
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"metric": "autoregressive_nll", "nll": loss}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
