"""Controlled host-side inference forwarding for sandboxed Agent Containers.

The Agent Container never receives general Internet access.  It talks to a
localhost proxy, which relays inference requests through the trusted Supervisor.
The Supervisor fixes the upstream origin, permitted API paths, and registered
model names before task reveal.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Mapping
from urllib.parse import urlsplit


class InferenceFailure(RuntimeError):
    """A request is disallowed by policy or the configured upstream failed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class InferenceResponse:
    status: int
    content_type: str
    body: bytes


def _validate_payload(body: bytes, *, allowed_models: tuple[str, ...]) -> str:
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
    if not isinstance(model, str) or model not in allowed_models:
        raise InferenceFailure(
            "model_not_registered",
            "inference request must use a model registered before task reveal",
        )

    # Function/tool calling is part of normal agent inference.  Provider-hosted
    # retrieval/browser/computer tools are not: those would create a second
    # external-information channel that bypasses the benchmark boundary.
    tools = payload.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            raise InferenceFailure(
                "external_tool_not_allowed", "inference tools must be a list"
            )
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                raise InferenceFailure(
                    "external_tool_not_allowed",
                    "only function tools executed by the sandboxed agent are allowed",
                )
    if payload.get("web_search_options") is not None:
        raise InferenceFailure(
            "external_tool_not_allowed", "provider-hosted web search is not allowed"
        )
    return model


def forward_inference_request(
    *,
    inference_config: Any,
    method: str,
    path: str,
    headers: Mapping[str, str],
    body_base64: str,
    timeout_seconds: float | None = None,
) -> Dict[str, Any]:
    """Forward one policy-constrained OpenAI-compatible request.

    `inference_config` is intentionally duck-typed so this module stays small and
    does not create a config/import cycle.
    """

    if inference_config is None:
        raise InferenceFailure("inference_disabled", "no inference gateway is configured")
    if method.upper() != "POST":
        raise InferenceFailure(
            "method_not_allowed", "v0 inference gateway only permits POST requests"
        )
    parsed_path = urlsplit(path)
    if parsed_path.scheme or parsed_path.netloc or parsed_path.query or parsed_path.fragment:
        raise InferenceFailure(
            "path_not_allowed", "inference requests must use an exact configured API path"
        )
    if parsed_path.path not in inference_config.allowed_paths:
        raise InferenceFailure(
            "path_not_allowed", f"inference path is not allowed: {parsed_path.path}"
        )
    try:
        body = base64.b64decode(body_base64, validate=True)
    except (ValueError, TypeError) as exc:
        raise InferenceFailure(
            "invalid_inference_request", "request body is not valid base64"
        ) from exc
    if len(body) > 16 * 1024 * 1024:
        raise InferenceFailure(
            "inference_request_too_large", "inference body exceeds the v0 16 MiB limit"
        )

    model = _validate_payload(body, allowed_models=inference_config.allowed_models)

    base = urlsplit(inference_config.base_url)
    if base.scheme not in {"http", "https"} or not base.netloc:
        raise InferenceFailure("invalid_upstream", "configured inference base URL is invalid")
    if base.path not in {"", "/"} or base.query or base.fragment:
        raise InferenceFailure(
            "invalid_upstream", "inference base URL must be an origin without a path"
        )
    url = f"{base.scheme}://{base.netloc}{parsed_path.path}"

    request_headers = {
        "Content-Type": headers.get("content-type", "application/json"),
        "Accept": headers.get("accept", "application/json"),
        "User-Agent": "AutoTrain-Bench-Inference-Gateway/0",
    }
    if inference_config.api_key_env:
        api_key = os.environ.get(inference_config.api_key_env)
        if not api_key:
            raise InferenceFailure(
                "inference_credential_missing",
                f"host environment variable {inference_config.api_key_env!r} is not set",
            )
        request_headers["Authorization"] = f"Bearer {api_key}"

    effective_timeout = float(inference_config.timeout_seconds)
    if timeout_seconds is not None:
        effective_timeout = min(effective_timeout, max(0.1, float(timeout_seconds)))
    request = urllib.request.Request(
        url,
        data=body,
        headers=request_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=effective_timeout) as response:
            response_body = response.read()
            status = int(response.status)
            content_type = response.headers.get("Content-Type", "application/json")
    except urllib.error.HTTPError as exc:
        response_body = exc.read()
        status = int(exc.code)
        content_type = exc.headers.get("Content-Type", "application/json")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise InferenceFailure("upstream_unavailable", f"inference upstream failed: {exc}") from exc

    return {
        "status": status,
        "content_type": content_type,
        "body_base64": base64.b64encode(response_body).decode("ascii"),
        "model": model,
    }
