"""Tiny file-mailbox client exposed to the untrusted Agent Container."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class ClientError(RuntimeError):
    """The supervisor rejected an agent request."""


class Client:
    """Agent-facing API.

    The shared directory is only a command mailbox. Trusted state, timers, lease
    counters, Docker access, and evaluator paths remain outside the container.
    """

    def __init__(
        self,
        control_dir: Optional[Path] = None,
        *,
        poll_seconds: float = 0.05,
    ) -> None:
        configured = control_dir or Path(
            os.environ.get("AUTOTRAIN_BENCH_CONTROL_DIR", "/control")
        )
        self.control_dir = configured
        self.inbox = configured / "inbox"
        self.outbox = configured / "outbox"
        self.poll_seconds = poll_seconds

    def request(
        self,
        operation: str,
        *,
        timeout_seconds: Optional[float] = None,
        **payload: Any,
    ) -> Dict[str, Any]:
        request_id = uuid.uuid4().hex
        message = {"id": request_id, "op": operation, **payload}
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.outbox.mkdir(parents=True, exist_ok=True)
        temporary = self.inbox / f".{request_id}.{os.getpid()}.tmp"
        request_path = self.inbox / f"{request_id}.json"
        temporary.write_text(json.dumps(message, sort_keys=True), encoding="utf-8")
        os.replace(str(temporary), str(request_path))

        response_path = self.outbox / f"{request_id}.json"
        started = time.monotonic()
        while not response_path.exists():
            if timeout_seconds is not None and time.monotonic() - started > timeout_seconds:
                raise ClientError(f"supervisor request timed out: {operation}")
            time.sleep(self.poll_seconds)

        try:
            response = json.loads(response_path.read_text(encoding="utf-8"))
        finally:
            response_path.unlink(missing_ok=True)
        if not response.get("ok"):
            error = response.get("error") or {}
            code = error.get("code", "request_failed")
            message = error.get("message", "supervisor rejected the request")
            raise ClientError(f"{code}: {message}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise ClientError("supervisor returned a malformed response")
        return result

    def status(self) -> Dict[str, Any]:
        return self.request("status")

    def start_lease(self, seconds: Optional[int] = None) -> Dict[str, Any]:
        return self.request("lease_start", seconds=seconds)

    def exec(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not argv:
            raise ClientError("exec requires a command")
        response_timeout = None
        if timeout_seconds is not None:
            response_timeout = timeout_seconds + 30
        return self.request(
            "lease_exec",
            argv=list(argv),
            command_timeout_seconds=timeout_seconds,
            timeout_seconds=response_timeout,
        )

    def logs(self, limit: int = 50) -> Dict[str, Any]:
        return self.request("lease_logs", limit=limit)

    def commit(self, candidate_path: str) -> Dict[str, Any]:
        return self.request("lease_commit", candidate_path=candidate_path)

    def stop_lease(self) -> Dict[str, Any]:
        return self.request("lease_stop")

    def finish(self, candidate_id: Optional[str] = None) -> Dict[str, Any]:
        return self.request("finish", candidate_id=candidate_id)
