"""HTTP/TCP Candidate-channel adapter for the bundled Docker Worker.

The Candidate application protocol is release-owned.  This transport is not:
an external Worker may provide a Unix socket, proxy, shared-memory RPC, or any
other channel that preserves the same ``predict(prefixes) -> logits`` contract
and the isolation requirements in Worker Protocol v5.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, List, Sequence


class AdapterError(RuntimeError):
    pass


class HttpCandidateClient:
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
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"{self.base_url}/health", timeout=1
                ) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError):
                time.sleep(0.05)
        raise AdapterError("Candidate server did not become ready")

    def predict(self, prefixes: Sequence[Sequence[int]]) -> List[List[float]]:
        body = json.dumps({"input_ids": prefixes}, separators=(",", ":")).encode(
            "utf-8"
        )
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
            raise AdapterError("Candidate inference request failed") from exc
        if len(encoded) > maximum_response_bytes:
            raise AdapterError("Candidate response exceeds the protocol limit")
        try:
            value = json.loads(encoded)["logits"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise AdapterError("Candidate returned malformed JSON") from exc
        if not isinstance(value, list) or len(value) != len(prefixes):
            raise AdapterError("Candidate must return [batch, vocabulary] logits")
        result: List[List[float]] = []
        for row in value:
            if not isinstance(row, list) or len(row) != self.vocab_size:
                raise AdapterError("Candidate returned the wrong vocabulary dimension")
            if any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in row
            ):
                raise AdapterError("Candidate logits must be numeric")
            normalized = [float(item) for item in row]
            if not all(math.isfinite(item) for item in normalized):
                raise AdapterError("Candidate logits must be finite")
            result.append(normalized)
        return result


def _load_evaluate(path: Path) -> Any:
    module_name = f"autotrain_evaluator_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AdapterError("Evaluator module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluate = getattr(module, "evaluate", None)
    if not callable(evaluate):
        raise AdapterError("Evaluator module must define evaluate(..., predict, ...)")
    return evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--request-timeout-seconds", type=float, default=120)
    args = parser.parse_args()
    client = HttpCandidateClient(
        args.candidate_url,
        vocab_size=args.vocab_size,
        timeout_seconds=args.request_timeout_seconds,
    )
    loss = _load_evaluate(args.evaluator)(
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
