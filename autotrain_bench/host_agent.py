"""Host-native Agent process support and transcript accounting."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, IO, Optional, Sequence


_USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)


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
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        summary["malformed_records"] = 1
        summary["usage_complete"] = False
        return summary

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
            if record_type == "autotrain_bench.agent_usage" and record.get(
                "usage_complete"
            ) is False:
                summary["usage_complete"] = False
            for field in _USAGE_FIELDS:
                value = usage.get(field, 0)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
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
    _stdout: IO[str]
    _stderr: IO[str]
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
        self._stdout.close()
        self._stderr.close()
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
        name: os.environ[name]
        for name in inherited_names
        if name in os.environ
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
    stdout = stdout_path.open("w", encoding="utf-8")
    stderr = stderr_path.open("w", encoding="utf-8")
    stdout_path.chmod(0o600)
    stderr_path.chmod(0o600)
    try:
        process = subprocess.Popen(
            list(command),
            cwd=str(workspace.resolve()),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
    except OSError:
        stdout.close()
        stderr.close()
        raise
    return HostAgentProcess(
        process=process,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        _stdout=stdout,
        _stderr=stderr,
    )
