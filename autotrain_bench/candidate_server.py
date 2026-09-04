"""Serve an untrusted Candidate inside its isolated evaluation container."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict

from .candidate_protocol import (
    CandidateProtocolError,
    load_candidate,
    normalized_logits,
    validate_contexts,
)


class CandidateApplication:
    def __init__(
        self,
        *,
        candidate_dir: Path,
        device: str,
        vocab_size: int,
        context_length: int,
        maximum_batch_size: int,
    ) -> None:
        self.module, self.model = load_candidate(candidate_dir, device)
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.maximum_batch_size = maximum_batch_size
        self.maximum_request_bytes = max(
            1024 * 1024,
            context_length * maximum_batch_size * 16,
        )

    def predict(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise CandidateProtocolError("request body must be an object")
        contexts = validate_contexts(
            payload.get("input_ids"),
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            maximum_batch_size=self.maximum_batch_size,
        )
        return {
            "logits": normalized_logits(
                self.module,
                self.model,
                contexts,
                vocab_size=self.vocab_size,
            )
        }


def handler_for(application: CandidateApplication):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AutoTrainCandidate/2"

        def log_message(self, _format, *args):
            del args

        def _json(self, status: int, payload: Dict[str, Any]) -> None:
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path != "/health":
                self._json(404, {"error": "not_found"})
                return
            self._json(200, {"status": "ready"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path != "/v2/next-token-logits":
                self._json(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if length < 0 or length > application.maximum_request_bytes:
                    raise CandidateProtocolError("request body is too large")
                payload = json.loads(self.rfile.read(length))
                response = application.predict(payload)
            except (CandidateProtocolError, json.JSONDecodeError, ValueError) as exc:
                self._json(400, {"error": "invalid_request", "message": str(exc)})
                return
            except Exception:
                self._json(500, {"error": "candidate_failure"})
                return
            self._json(200, response)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument("--context-length", required=True, type=int)
    parser.add_argument("--maximum-batch-size", required=True, type=int)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    application = CandidateApplication(
        candidate_dir=args.candidate,
        device=args.device,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        maximum_batch_size=args.maximum_batch_size,
    )
    server = HTTPServer(("0.0.0.0", args.port), handler_for(application))
    server.serve_forever()


if __name__ == "__main__":
    main()
