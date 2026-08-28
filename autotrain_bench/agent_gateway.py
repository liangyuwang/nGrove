"""Localhost HTTP proxy exposed inside the sandboxed Agent Container."""

from __future__ import annotations

import base64
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .client import Client, ClientError


MAX_BODY_BYTES = 16 * 1024 * 1024


class GatewayHandler(BaseHTTPRequestHandler):
    server_version = "AutoTrainBenchGateway/0"

    def log_message(self, fmt: str, *args) -> None:  # quiet by default
        if os.environ.get("AUTOTRAIN_BENCH_GATEWAY_LOG") == "1":
            super().log_message(fmt, *args)

    def _json_error(self, status: int, message: str) -> None:
        body = json.dumps({"error": {"message": message}}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json_error(400, "invalid Content-Length")
            return
        if length < 0 or length > MAX_BODY_BYTES:
            self._json_error(413, "inference request exceeds the v0 16 MiB limit")
            return
        body = self.rfile.read(length)
        forwarded_headers = {
            "content-type": self.headers.get("Content-Type", "application/json"),
            "accept": self.headers.get("Accept", "application/json"),
        }
        try:
            result = Client().inference(
                method="POST",
                path=self.path,
                headers=forwarded_headers,
                body_base64=base64.b64encode(body).decode("ascii"),
            )
            response_body = base64.b64decode(result["body_base64"])
            status = int(result["status"])
            content_type = str(result.get("content_type") or "application/json")
        except (ClientError, KeyError, TypeError, ValueError) as exc:
            self._json_error(502, str(exc))
            return

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)


def main() -> None:
    host = os.environ.get("AUTOTRAIN_BENCH_GATEWAY_HOST", "127.0.0.1")
    port = int(os.environ.get("AUTOTRAIN_BENCH_GATEWAY_PORT", "8765"))
    server = ThreadingHTTPServer((host, port), GatewayHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
