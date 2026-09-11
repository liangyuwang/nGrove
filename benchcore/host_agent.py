"""Host-native Agent process support and transcript accounting."""

from __future__ import annotations

import errno
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
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        try:
            os.chmod(path, 0o600)
            self.stream = os.fdopen(descriptor, "wb")
            descriptor = -1
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        self.limit = MAX_HOST_AGENT_LOG_BYTES - len(_LOG_TRUNCATION_MARKER)
        self.written = 0
        self.truncated = False
        self.eof = False
        self.finished = False
        self.error: Optional[Dict[str, Any]] = None

    def _record_error(self, operation: str, exc: Exception) -> None:
        # Retain bounded evidence independently of the file that just failed.
        if self.error is None:
            self.error = {
                "operation": operation,
                "error_type": type(exc).__name__,
                "errno": getattr(exc, "errno", None),
            }

    def _write(self, data: bytes) -> None:
        try:
            if self.stream.write(data) != len(data):
                raise OSError(errno.EIO, "short host Agent log write")
        except (OSError, ValueError) as exc:
            self._record_error("write", exc)
            return
        try:
            self.stream.flush()
        except (OSError, ValueError) as exc:
            self._record_error("flush", exc)

    def close(self, source: Optional[BinaryIO] = None) -> None:
        # A failed close/flush on one stream must not skip the other resource.
        for operation, stream in (
            ("close_source", source),
            ("close_sink", self.stream),
        ):
            try:
                if stream is not None and not stream.closed:
                    stream.close()
            except (OSError, ValueError) as exc:
                self._record_error(operation, exc)

    def capture_status(self) -> Dict[str, Any]:
        finished = self.finished
        return {
            "complete": finished
            and self.eof
            and not self.truncated
            and self.error is None,
            "finished": finished,
            "eof": self.eof,
            "truncated": self.truncated,
            "error": dict(self.error) if self.error is not None else None,
        }

    def drain(self, source: BinaryIO) -> None:
        # BufferedReader.read(n) can wait for n bytes or EOF on a live pipe.
        # read1 returns the bytes available from one underlying read instead.
        read = getattr(source, "read1", source.read)
        try:
            while True:
                try:
                    chunk = read(64 * 1024)
                except (OSError, ValueError) as exc:
                    self._record_error("read", exc)
                    break
                if not chunk:
                    self.eof = True
                    break
                # A full/unavailable log filesystem must not close the read
                # pipe or block the Agent. Discard further output until EOF.
                if self.error is not None:
                    continue
                remaining = self.limit - self.written
                if remaining > 0:
                    retained = chunk[:remaining]
                    self._write(retained)
                    if self.error is None:
                        self.written += len(retained)
                if len(chunk) > remaining:
                    self.truncated = True
            if self.truncated and self.error is None:
                self._write(_LOG_TRUNCATION_MARKER)
        finally:
            self.close(source)
            self.finished = True


def summarize_host_agent_trace(path: Path) -> Dict[str, Any]:
    """Parse retained host-Agent records; this cannot detect lost capture bytes.

    Live launchers should use ``HostAgentProcess.summarize_usage()`` after
    stopping the Agent so capture failures also make usage incomplete.
    """

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
        summary["usage_complete"] = False
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
        if record_type == "benchcore.agent_metadata":
            for key in ("tool", "tool_version", "model", "reasoning_effort"):
                value = record.get(key)
                if isinstance(value, str) and value:
                    summary[key] = value
            continue
        if record_type in {"benchcore.agent_usage", "turn.completed"}:
            if record_type == "turn.completed":
                summary["turns_completed"] += 1
            usage = record.get("usage")
            if not isinstance(usage, dict):
                summary["usage_complete"] = False
                continue
            saw_usage = True
            if (
                record_type == "benchcore.agent_usage"
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
    _stdout_sink: _BoundedLogSink
    _stderr_sink: _BoundedLogSink
    _stdout_thread: Optional[threading.Thread] = None
    _stderr_thread: Optional[threading.Thread] = None
    _stdout_thread_started: bool = False
    _stderr_thread_started: bool = False
    _closed: bool = False

    @property
    def running(self) -> bool:
        return self.process.poll() is None

    @property
    def returncode(self) -> Optional[int]:
        return self.process.poll()

    def summarize_usage(self) -> Dict[str, Any]:
        """Combine retained usage records with trusted log-capture evidence."""

        capture = {
            "stdout": self._stdout_sink.capture_status(),
            "stderr": self._stderr_sink.capture_status(),
        }
        summary = summarize_host_agent_trace(self.stdout_path)
        summary["log_capture"] = capture
        # Usage records are sourced from stdout. Loss of stderr diagnostics is
        # recorded separately and does not invalidate intact stdout accounting.
        if self.running or not capture["stdout"]["complete"]:
            summary["usage_complete"] = False
        return summary

    def stop(self, timeout_seconds: float = 3) -> int:
        termination_error: Optional[BaseException] = None
        try:
            if self.process.poll() is None:
                try:
                    try:
                        if os.name == "posix":
                            os.killpg(self.process.pid, signal.SIGTERM)
                        else:  # pragma: no cover - exercised on Windows only
                            self.process.terminate()
                    except ProcessLookupError:
                        pass
                    self.process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    pass
            if self.process.poll() is None:
                try:
                    if os.name == "posix":
                        os.killpg(self.process.pid, signal.SIGKILL)
                    else:  # pragma: no cover - exercised on Windows only
                        self.process.kill()
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=timeout_seconds)
            if self.process.poll() is None:
                raise RuntimeError("host Agent termination was not confirmed")
        except BaseException as exc:
            termination_error = exc
        finally:
            # Closing unowned pipes and joining live drainers is independent of
            # whether process termination was confirmed.  On failure the handle
            # remains retryable and keeps any still-running drainer reachable.
            try:
                self.close_logs()
            except BaseException as exc:
                if termination_error is None:
                    termination_error = exc
        if self.process.poll() is None and termination_error is None:
            termination_error = RuntimeError("host Agent termination was not confirmed")
        if termination_error is not None:
            raise termination_error
        return int(self.process.returncode or 0)

    def close_logs(self) -> None:
        if self._closed:
            return
        complete = True
        for source, sink, thread, started in (
            (
                self.process.stdout,
                self._stdout_sink,
                self._stdout_thread,
                self._stdout_thread_started,
            ),
            (
                self.process.stderr,
                self._stderr_sink,
                self._stderr_thread,
                self._stderr_thread_started,
            ),
        ):
            thread_started = started or (
                thread is not None and thread.ident is not None
            )
            if thread_started and thread is not None:
                thread.join(timeout=5)
                complete = complete and not thread.is_alive()
            else:
                sink.close(source)
        self._closed = complete


class HostAgentStartupFailure(RuntimeError):
    """Post-Popen launch failure that preserves the cleanup-capable handle."""

    def __init__(
        self,
        message: str,
        *,
        handle: HostAgentProcess,
        startup_error: BaseException,
        cleanup_error: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.handle = handle
        self.startup_error = startup_error
        self.cleanup_error = cleanup_error


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
    stdout_sink: Optional[_BoundedLogSink] = None
    stderr_sink: Optional[_BoundedLogSink] = None
    try:
        stdout_sink = _BoundedLogSink(stdout_path)
        stderr_sink = _BoundedLogSink(stderr_path)
        process = subprocess.Popen(
            list(command),
            cwd=str(workspace.resolve()),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except BaseException:
        for sink in (stdout_sink, stderr_sink):
            if sink is not None:
                sink.close()
        raise
    assert stdout_sink is not None
    assert stderr_sink is not None
    handle = HostAgentProcess(
        process=process,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        _stdout_sink=stdout_sink,
        _stderr_sink=stderr_sink,
    )
    try:
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("host Agent pipes were not created")
        handle._stdout_thread = threading.Thread(
            target=stdout_sink.drain, args=(process.stdout,), daemon=True
        )
        handle._stderr_thread = threading.Thread(
            target=stderr_sink.drain, args=(process.stderr,), daemon=True
        )
        handle._stdout_thread.start()
        handle._stdout_thread_started = True
        handle._stderr_thread.start()
        handle._stderr_thread_started = True
        return handle
    except BaseException as startup_error:
        cleanup_error: Optional[BaseException] = None
        try:
            handle.stop()
        except BaseException as exc:
            cleanup_error = exc
        message = "host Agent startup failed after process creation"
        if cleanup_error is not None:
            message += " and termination was not confirmed"
        raise HostAgentStartupFailure(
            message,
            handle=handle,
            startup_error=startup_error,
            cleanup_error=cleanup_error,
        ) from startup_error
