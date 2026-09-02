"""Host-native Agent process support and transcript accounting."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Sequence


_USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)
MAX_HOST_AGENT_LOG_BYTES = 10 * 1024 * 1024
_LOG_TRUNCATION_MARKER = b"\n[log truncated by AutoTrain-Bench]\n"


class _BoundedLogSink:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.stream = path.open("wb")
        self.limit = MAX_HOST_AGENT_LOG_BYTES - len(_LOG_TRUNCATION_MARKER)
        self.written = 0
        self.truncated = False

    def drain(self, source: BinaryIO) -> None:
        try:
            while True:
                chunk = source.read(64 * 1024)
                if not chunk:
                    break
                remaining = self.limit - self.written
                if remaining > 0:
                    retained = chunk[:remaining]
                    self.stream.write(retained)
                    self.written += len(retained)
                if len(chunk) > remaining:
                    self.truncated = True
            if self.truncated:
                self.stream.write(_LOG_TRUNCATION_MARKER)
            self.stream.flush()
        finally:
            source.close()
            self.stream.close()


def summarize_host_agent_trace(path: Path) -> Dict[str, Any]:
    """Summarize the standard host-Agent records and known JSONL event streams."""

    summary: Dict[str, Any] = {
        "source": "host_agent_stdout",
        "tool": None,
        "tool_version": None,
        "model": None,
        "reasoning_effort": None,
        "turns_completed": 0,
        "turns_failed": 0,
        "command_executions": 0,
        "file_changes": 0,
        "web_searches": 0,
        "mcp_tool_calls": 0,
        "malformed_records": 0,
        **{field: 0 for field in _USAGE_FIELDS},
        "total_tokens": 0,
        "usage_complete": True,
    }
    if not path.exists():
        return summary

    saw_usage = False
    try:
        encoded = path.read_bytes()
    except OSError:
        summary["malformed_records"] = 1
        summary["usage_complete"] = False
        return summary
    decoded = encoded.decode("utf-8", errors="replace")
    if "\ufffd" in decoded or _LOG_TRUNCATION_MARKER in encoded:
        summary["malformed_records"] += 1
        summary["usage_complete"] = False
    lines = decoded.splitlines()

    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            summary["malformed_records"] += 1
            continue
        if not isinstance(record, dict):
            summary["malformed_records"] += 1
            continue

        record_type = record.get("type")
        if record_type == "autotrain_bench.agent_metadata":
            for key in ("tool", "tool_version", "model", "reasoning_effort"):
                value = record.get(key)
                if isinstance(value, str) and value:
                    summary[key] = value
            continue
        if record_type in {"autotrain_bench.agent_usage", "turn.completed"}:
            if record_type == "turn.completed":
                summary["turns_completed"] += 1
            usage = record.get("usage")
            if not isinstance(usage, dict):
                summary["usage_complete"] = False
                continue
            saw_usage = True
            if (
                record_type == "autotrain_bench.agent_usage"
                and record.get("usage_complete") is False
            ):
                summary["usage_complete"] = False
            for field in _USAGE_FIELDS:
                value = usage.get(field, 0)
                if (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                ):
                    summary[field] += value
                else:
                    summary["usage_complete"] = False
            continue
        if record_type == "turn.failed":
            summary["turns_failed"] += 1
        item = record.get("item")
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if record_type == "item.started" and item_type == "command_execution":
            summary["command_executions"] += 1
        elif record_type == "item.started" and item_type == "file_change":
            summary["file_changes"] += 1
        elif record_type == "item.started" and item_type == "web_search":
            summary["web_searches"] += 1
        elif record_type == "item.started" and item_type == "mcp_tool_call":
            summary["mcp_tool_calls"] += 1

    summary["total_tokens"] = summary["input_tokens"] + summary["output_tokens"]
    if not saw_usage:
        summary["usage_complete"] = False
    if summary["malformed_records"]:
        summary["usage_complete"] = False
    return summary


@dataclass
class HostAgentProcess:
    """A host Agent subprocess whose output is streamed directly to files."""

    process: subprocess.Popen
    stdout_path: Path
    stderr_path: Path
    _stdout_thread: threading.Thread
    _stderr_thread: threading.Thread
    _closed: bool = False

    @property
    def running(self) -> bool:
        return self.process.poll() is None

    @property
    def returncode(self) -> Optional[int]:
        return self.process.poll()

    def stop(self, timeout_seconds: float = 3) -> int:
        if self.process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(self.process.pid, signal.SIGTERM)
                else:  # pragma: no cover - exercised on Windows only
                    self.process.terminate()
                self.process.wait(timeout=timeout_seconds)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if self.process.poll() is None:
                    if os.name == "posix":
                        os.killpg(self.process.pid, signal.SIGKILL)
                    else:  # pragma: no cover - exercised on Windows only
                        self.process.kill()
                    self.process.wait(timeout=timeout_seconds)
        self.close_logs()
        return int(self.process.returncode or 0)

    def close_logs(self) -> None:
        if self._closed:
            return
        self._stdout_thread.join(timeout=5)
        self._stderr_thread.join(timeout=5)
        self._closed = True


def start_host_agent_process(
    *,
    command: Sequence[str],
    workspace: Path,
    control_dir: Path,
    public_task_path: Path,
    package_root: Path,
    forwarded_environment: Sequence[str],
    stdout_path: Path,
    stderr_path: Path,
) -> HostAgentProcess:
    """Launch a host Agent with a deliberately small inherited environment."""

    inherited_names = (
        "CODEX_HOME",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "PATH",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TERM",
        "TMPDIR",
        "USER",
    )
    environment = {
        name: os.environ[name] for name in inherited_names if name in os.environ
    }
    for name in forwarded_environment:
        if name in os.environ:
            environment[name] = os.environ[name]
    environment.update(
        {
            "AUTOTRAIN_BENCH_CONTROL_DIR": str(control_dir.resolve()),
            "AUTOTRAIN_BENCH_FORWARDED_ENVIRONMENT": json.dumps(
                list(forwarded_environment), separators=(",", ":")
            ),
            "AUTOTRAIN_BENCH_HOST_RUNNER": "1",
            "AUTOTRAIN_BENCH_TASK": str(public_task_path.resolve()),
            "AUTOTRAIN_BENCH_TRAIN_DIR": "/benchmark/train",
            "AUTOTRAIN_BENCH_TRAIN_WORKSPACE": "/workspace",
            "AUTOTRAIN_BENCH_WORKSPACE": str(workspace.resolve()),
            "PYTHONPATH": str(package_root.resolve()),
        }
    )

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_sink = _BoundedLogSink(stdout_path)
    stderr_sink = _BoundedLogSink(stderr_path)
    stdout_path.chmod(0o600)
    stderr_path.chmod(0o600)
    try:
        process = subprocess.Popen(
            list(command),
            cwd=str(workspace.resolve()),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError:
        stdout_sink.stream.close()
        stderr_sink.stream.close()
        raise
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_thread = threading.Thread(
        target=stdout_sink.drain, args=(process.stdout,), daemon=True
    )
    stderr_thread = threading.Thread(
        target=stderr_sink.drain, args=(process.stderr,), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()
    return HostAgentProcess(
        process=process,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        _stdout_thread=stdout_thread,
        _stderr_thread=stderr_thread,
    )
