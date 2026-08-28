"""Policy validation and accounting helpers for the trusted inference gateway."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlsplit


MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_JSON_USAGE_BYTES = 4 * 1024 * 1024
MAX_SSE_LINE_BYTES = 1024 * 1024


class InferenceFailure(RuntimeError):
    """A request violates the registered inference policy."""

    def __init__(self, code: str, message: str, http_status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status


def validate_inference_request(
    *, method: str, path: str, body: bytes, inference_config: Any
) -> str:
    """Return the registered model after validating one Agent request."""

    if method.upper() != "POST":
        raise InferenceFailure(
            "method_not_allowed",
            "the inference gateway only permits POST requests",
            405,
        )
    parsed_path = urlsplit(path)
    if parsed_path.scheme or parsed_path.netloc or parsed_path.query or parsed_path.fragment:
        raise InferenceFailure(
            "path_not_allowed",
            "inference requests must use an exact configured API path",
            403,
        )
    if parsed_path.path not in inference_config.allowed_paths:
        raise InferenceFailure(
            "path_not_allowed",
            f"inference path is not allowed: {parsed_path.path}",
            403,
        )
    if len(body) > MAX_REQUEST_BYTES:
        raise InferenceFailure(
            "inference_request_too_large",
            "inference body exceeds the v0 16 MiB limit",
            413,
        )
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InferenceFailure(
            "invalid_inference_request", "inference request body must be JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise InferenceFailure(
            "invalid_inference_request", "inference request body must be an object"
        )
    model = payload.get("model")
    if not isinstance(model, str) or model not in inference_config.allowed_models:
        raise InferenceFailure(
            "model_not_registered",
            "inference request must use a model registered before task reveal",
            403,
        )

    # Local function calling is part of the Agent. Provider-hosted tools would
    # create an unregistered external-information channel.
    tools = payload.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            raise InferenceFailure(
                "external_tool_not_allowed", "inference tools must be a list", 403
            )
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                raise InferenceFailure(
                    "external_tool_not_allowed",
                    "only function tools executed by the sandboxed Agent are allowed",
                    403,
                )
    if payload.get("web_search_options") is not None:
        raise InferenceFailure(
            "external_tool_not_allowed",
            "provider-hosted web search is not allowed",
            403,
        )
    return model


def _integer(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def normalize_usage(payload: Any) -> Optional[Dict[str, int]]:
    """Normalize Responses and Chat Completions token accounting fields."""

    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    response = payload.get("response")
    if not isinstance(usage, dict) and isinstance(response, dict):
        usage = response.get("usage")
    if not isinstance(usage, dict):
        return None

    input_tokens = _integer(usage.get("input_tokens"))
    if input_tokens is None:
        input_tokens = _integer(usage.get("prompt_tokens"))
    output_tokens = _integer(usage.get("output_tokens"))
    if output_tokens is None:
        output_tokens = _integer(usage.get("completion_tokens"))
    total_tokens = _integer(usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    if input_tokens is None or output_tokens is None or total_tokens is None:
        return None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


class UsageCollector:
    """Inspect a response for usage while allowing every byte to stream onward."""

    def __init__(self, content_type: str) -> None:
        self.is_sse = "text/event-stream" in content_type.lower()
        self.usage: Optional[Dict[str, int]] = None
        self._buffer = bytearray()
        self._discard_sse_line = False
        self._json_overflow = False

    def feed(self, chunk: bytes) -> None:
        if self.is_sse:
            self._feed_sse(chunk)
            return
        if self._json_overflow:
            return
        if len(self._buffer) + len(chunk) > MAX_JSON_USAGE_BYTES:
            self._buffer.clear()
            self._json_overflow = True
            return
        self._buffer.extend(chunk)

    def _feed_sse(self, chunk: bytes) -> None:
        self._buffer.extend(chunk)
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                if len(self._buffer) > MAX_SSE_LINE_BYTES:
                    self._buffer.clear()
                    self._discard_sse_line = True
                return
            raw_line = bytes(self._buffer[:newline]).rstrip(b"\r")
            del self._buffer[: newline + 1]
            if self._discard_sse_line:
                self._discard_sse_line = False
                continue
            self._inspect_sse_line(raw_line)

    def _inspect_sse_line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            return
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        usage = normalize_usage(payload)
        if usage is not None:
            self.usage = usage

    def finish(self) -> Optional[Dict[str, int]]:
        if self.is_sse:
            if self._buffer and not self._discard_sse_line:
                self._inspect_sse_line(bytes(self._buffer).rstrip(b"\r"))
        elif not self._json_overflow:
            try:
                payload = json.loads(bytes(self._buffer).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = None
            usage = normalize_usage(payload)
            if usage is not None:
                self.usage = usage
        return self.usage


def summarize_inference_trace(path: Path) -> Dict[str, Any]:
    """Aggregate the append-only gateway trace without assuming it is complete."""

    started: Dict[str, Dict[str, Any]] = {}
    terminal: Dict[str, Dict[str, Any]] = {}
    rejected_requests = 0
    malformed_records = 0
    if path.exists():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
            malformed_records = 1
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_records += 1
                continue
            if not isinstance(record, dict) or not isinstance(record.get("request_id"), str):
                malformed_records += 1
                continue
            request_id = record["request_id"]
            event_type = record.get("type")
            if event_type == "INFERENCE_STARTED":
                started[request_id] = record
            elif event_type in {"INFERENCE_COMPLETED", "INFERENCE_FAILED"}:
                terminal[request_id] = record
            elif event_type == "INFERENCE_REJECTED":
                rejected_requests += 1

    request_ids = set(started) | set(terminal)
    completed_calls = 0
    failed_calls = 0
    incomplete_usage_calls = 0
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    for request_id in request_ids:
        record = terminal.get(request_id)
        if record is None:
            failed_calls += 1
            incomplete_usage_calls += 1
            continue
        if record.get("type") == "INFERENCE_COMPLETED":
            completed_calls += 1
        else:
            failed_calls += 1
        normalized = normalize_usage({"usage": record.get("usage")})
        if normalized is None:
            incomplete_usage_calls += 1
            continue
        input_tokens += normalized["input_tokens"]
        output_tokens += normalized["output_tokens"]
        total_tokens += normalized["total_tokens"]

    return {
        "model_calls": len(request_ids),
        "completed_calls": completed_calls,
        "failed_calls": failed_calls,
        "rejected_requests": rejected_requests,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "usage_complete": incomplete_usage_calls == 0 and malformed_records == 0,
        "incomplete_usage_calls": incomplete_usage_calls,
        "malformed_records": malformed_records,
    }
