"""Trusted OpenAI-compatible inference sidecar with streaming audit."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlsplit

from .config import ConfigError, InferenceConfig
from .inference import (
    InferenceFailure,
    MAX_REQUEST_BYTES,
    UsageCollector,
    validate_inference_request,
)


MAX_INFERENCE_AUDIT_BYTES = 64 * 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, code, msg, headers, newurl
        return None


class GatewayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address,
        inference_config: InferenceConfig,
        audit_path: Path,
    ) -> None:
        self.inference_config = inference_config
        self.audit_path = audit_path
        self.audit_lock = threading.Lock()
        self.opener = urllib.request.build_opener(_NoRedirect())
        super().__init__(address, GatewayHandler)

    def audit(self, record: Dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **record,
        }
        encoded = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
        with self.audit_lock:
            current_size = (
                self.audit_path.stat().st_size if self.audit_path.exists() else 0
            )
            if current_size + len(encoded) > MAX_INFERENCE_AUDIT_BYTES:
                raise OSError("inference audit size limit reached")
            with self.audit_path.open("ab") as stream:
                stream.write(encoded)
            self.audit_path.chmod(0o600)


class GatewayHandler(BaseHTTPRequestHandler):
    server: GatewayServer
    server_version = "AutoTrainBenchInferenceGateway/0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        if os.environ.get("AUTOTRAIN_BENCH_GATEWAY_LOG") == "1":
            super().log_message(fmt, *args)

    def _json_error(self, status: int, code: str, message: str) -> None:
        body = json.dumps({"error": {"code": code, "message": message}}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path != "/health":
            self._json_error(404, "not_found", "only /health is available via GET")
            return
        body = b'{"status":"ready"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        request_id = uuid.uuid4().hex
        started = time.monotonic()
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._reject(
                request_id, "length_required", "Content-Length is required", 411
            )
            return
        try:
            length = int(raw_length)
        except ValueError:
            self._reject(
                request_id, "invalid_content_length", "invalid Content-Length", 400
            )
            return
        if length < 0 or length > MAX_REQUEST_BYTES:
            self._reject(
                request_id,
                "inference_request_too_large",
                "inference request exceeds the v0 16 MiB limit",
                413,
            )
            return
        body = self.rfile.read(length)
        if len(body) != length:
            self._reject(
                request_id, "incomplete_request", "request body ended early", 400
            )
            return
        try:
            model = validate_inference_request(
                method="POST",
                path=self.path,
                body=body,
                inference_config=self.server.inference_config,
            )
        except InferenceFailure as exc:
            self._reject(request_id, exc.code, str(exc), exc.http_status)
            return

        config = self.server.inference_config
        base = urlsplit(config.base_url)
        path = urlsplit(self.path).path
        url = f"{base.scheme}://{base.netloc}{path}"
        request_headers = {
            "Content-Type": self.headers.get("Content-Type", "application/json"),
            "Accept": self.headers.get("Accept", "application/json"),
            "User-Agent": "AutoTrain-Bench-Inference-Gateway/0",
        }
        for header in ("OpenAI-Beta", "OpenAI-Organization", "OpenAI-Project"):
            value = self.headers.get(header)
            if value:
                request_headers[header] = value
        if config.api_key_env:
            api_key = os.environ.get(config.api_key_env)
            if not api_key:
                self._reject(
                    request_id,
                    "inference_credential_missing",
                    f"gateway environment variable {config.api_key_env!r} is not set",
                    503,
                )
                return
            request_headers["Authorization"] = f"Bearer {api_key}"

        try:
            self.server.audit(
                {
                    "type": "INFERENCE_STARTED",
                    "request_id": request_id,
                    "model": model,
                    "path": path,
                }
            )
        except OSError:
            self._json_error(
                503,
                "inference_audit_unavailable",
                "the trusted inference trace is not writable",
            )
            return

        request = urllib.request.Request(
            url,
            data=body,
            headers=request_headers,
            method="POST",
        )
        try:
            response = self.server.opener.open(request, timeout=config.timeout_seconds)
        except urllib.error.HTTPError as exc:
            if 300 <= exc.code < 400:
                exc.close()
                self._upstream_failure(
                    request_id,
                    model,
                    path,
                    started,
                    "inference_upstream_redirect_forbidden",
                    "the registered inference upstream attempted a redirect",
                )
                return
            response = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._upstream_failure(
                request_id,
                model,
                path,
                started,
                "upstream_unavailable",
                f"inference upstream failed: {exc}",
            )
            return

        status = int(response.status)
        content_type = response.headers.get("Content-Type", "application/json")
        collector = UsageCollector(content_type)
        disconnected = False
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            for header in ("Retry-After", "x-request-id", "openai-processing-ms"):
                value = response.headers.get(header)
                if value:
                    self.send_header(header, value)
            self.send_header("Connection", "close")
            self.end_headers()
            read_chunk = getattr(response, "read1", response.read)
            while True:
                chunk = read_chunk(64 * 1024)
                if not chunk:
                    break
                collector.feed(chunk)
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            disconnected = True
        except OSError as exc:
            self._audit_terminal(
                "INFERENCE_FAILED",
                request_id,
                model,
                path,
                started,
                status=status,
                code="upstream_stream_failed",
                message=str(exc),
            )
            return
        finally:
            response.close()
            self.close_connection = True

        usage = collector.finish()
        if disconnected:
            self._audit_terminal(
                "INFERENCE_FAILED",
                request_id,
                model,
                path,
                started,
                status=status,
                code="agent_disconnected",
                message="Agent disconnected before the inference response completed",
                usage=usage,
            )
            return
        self._audit_terminal(
            "INFERENCE_COMPLETED",
            request_id,
            model,
            path,
            started,
            status=status,
            usage=usage,
            usage_complete=usage is not None,
        )

    def _reject(self, request_id: str, code: str, message: str, status: int) -> None:
        try:
            self.server.audit(
                {
                    "type": "INFERENCE_REJECTED",
                    "request_id": request_id,
                    "path": self.path,
                    "code": code,
                    "message": message,
                }
            )
        except OSError:
            pass
        self._json_error(status, code, message)

    def _upstream_failure(
        self,
        request_id: str,
        model: str,
        path: str,
        started: float,
        code: str,
        message: str,
    ) -> None:
        self._audit_terminal(
            "INFERENCE_FAILED",
            request_id,
            model,
            path,
            started,
            code=code,
            message=message,
        )
        self._json_error(502, code, message)

    def _audit_terminal(
        self,
        event_type: str,
        request_id: str,
        model: str,
        path: str,
        started: float,
        **data: Any,
    ) -> None:
        try:
            self.server.audit(
                {
                    "type": event_type,
                    "request_id": request_id,
                    "model": model,
                    "path": path,
                    "duration_seconds": round(time.monotonic() - started, 6),
                    **data,
                }
            )
        except OSError:
            # The request already started and may already have streamed bytes. The
            # missing terminal record is deliberately visible as incomplete usage.
            pass


def create_gateway_server(
    host: str,
    port: int,
    inference_config: InferenceConfig,
    audit_path: Path,
) -> GatewayServer:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.touch(exist_ok=True)
    return GatewayServer((host, port), inference_config, audit_path)


def main() -> None:
    raw_config = os.environ.get("AUTOTRAIN_BENCH_INFERENCE_CONFIG_JSON")
    if not raw_config:
        raise SystemExit("AUTOTRAIN_BENCH_INFERENCE_CONFIG_JSON is missing")
    try:
        config = InferenceConfig.load(json.loads(raw_config), "gateway.inference")
    except (ConfigError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid inference gateway configuration: {exc}") from exc
    if config is None:
        raise SystemExit("inference gateway configuration is disabled")
    host = os.environ.get("AUTOTRAIN_BENCH_GATEWAY_HOST", "0.0.0.0")
    port = int(os.environ.get("AUTOTRAIN_BENCH_GATEWAY_PORT", "8765"))
    audit_path = Path(
        os.environ.get("AUTOTRAIN_BENCH_INFERENCE_AUDIT_PATH", "/audit/inference.jsonl")
    )
    server = create_gateway_server(host, port, config, audit_path)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
