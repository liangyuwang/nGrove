"""Process supervision for the bundled POSIX Docker launcher, outside Core.

The parent owns an unreaped session leader until it has fenced the entire local
process group. Docker ownership is then taken over through the recovery lock.
No PID read from a file, participant message or runtime inventory is signalled.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager, nullcontext
from functools import wraps
from pathlib import Path
from typing import Callable, Optional, Sequence

from .docker_recovery import DEFAULT_RECOVERY_TIMEOUT_SECONDS, recovery_timeout_seconds
from .worker_protocol import WorkerFailure


HEARTBEAT_SECONDS = 0.5
HEARTBEAT_TIMEOUT = 10.0
DEADLINE_GRACE_SECONDS = 5.0
CONTROL_TIMEOUT = 120.0
MAX_MESSAGE_BYTES = 4096
_client: Optional["WatchdogClient"] = None


def current() -> Optional["WatchdogClient"]:
    return _client


def cleanup_notifications():
    """Defer lost supervision notifications across a complete cleanup call."""
    client = current()
    return (
        client.cleanup_scope()
        if client is not None and client.registered
        else nullcontext()
    )


class WatchdogClient:
    """Private launcher channel; never inherited by Docker commands/Agents."""

    def __init__(self, fd: int) -> None:
        self.channel = socket.socket(fileno=fd)
        self.channel.set_inheritable(False)
        self.channel.settimeout(2)
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.registered = False
        self.control_timeout = CONTROL_TIMEOUT
        self._cleanup_context = threading.local()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def _send(self, message: dict) -> None:
        data = (json.dumps(message) + "\n").encode()
        if len(data) > MAX_MESSAGE_BYTES:
            raise WorkerFailure("Worker watchdog message is too large")
        self.channel.sendall(data)

    def _request(self, message: dict) -> None:
        with self.lock:
            if self.closed.is_set():
                raise WorkerFailure("Worker watchdog channel is unavailable")
            try:
                self._send(message)
                # Exactly one caller holds the channel until its acknowledgement.
                if self.channel.recv(1) != b"!":
                    raise OSError("watchdog acknowledgement missing")
            except OSError as exc:
                self.closed.set()
                raise WorkerFailure("Worker watchdog channel failed") from exc

    def _heartbeat(self) -> None:
        while not self.closed.wait(HEARTBEAT_SECONDS):
            try:
                with self.lock:
                    self._send({"type": "heartbeat"})
            except OSError:
                self.closed.set()
                return

    def register(self, journal, docker_binary: str) -> None:
        self._request(
            {
                "type": "register",
                "state_dir": str(journal.root),
                "owner": journal.state["owner"],
                "directory_identity": journal.state["directory_identity"],
                "docker_binary": docker_binary,
            }
        )
        self.registered = True

    def arm(self, key: str, seconds: float, label: str) -> None:
        self._request(
            {"type": "arm", "key": key, "seconds": max(0.0, seconds), "label": label}
        )

    def disarm(self, key: str) -> None:
        try:
            self._request({"type": "disarm", "key": key})
        except WorkerFailure as exc:
            if not self._defer_notification_error(exc):
                raise

    def _defer_notification_error(self, error: WorkerFailure) -> bool:
        context = self._cleanup_context
        if not self.closed.is_set() or not getattr(context, "depth", 0):
            return False
        if context.error is None:
            context.error = error
        return True

    @contextmanager
    def cleanup_scope(self):
        # Independent cleanup threads must neither consume each other's error
        # nor report it from an inner inspect/info call before removal occurs.
        context = self._cleanup_context
        depth = getattr(context, "depth", 0)
        if depth == 0:
            context.error = None
        context.depth = depth + 1
        completed = False
        try:
            yield
            completed = True
        finally:
            context.depth = depth
            if depth == 0:
                error = context.error
                context.error = None
                if completed and error is not None:
                    raise error

    @contextmanager
    def operation(self, label: str, seconds: float, *, cleanup: bool = False):
        with self.cleanup_scope() if cleanup else nullcontext():
            key = uuid.uuid4().hex
            armed = False
            try:
                self.arm(key, seconds, label)
                armed = True
            except WorkerFailure as exc:
                if not self._defer_notification_error(exc):
                    raise
            try:
                yield
            finally:
                if armed:
                    self.disarm(key)

    def close(self) -> None:
        self.closed.set()
        self.thread.join(timeout=3)
        self.channel.close()


def watched(*, timeout: str = "timeout_seconds", fallback=None, cleanup: bool = False):
    """Arm declared operation allowances without changing Worker/Core semantics."""

    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def invoke(*args, **kwargs):
            client = current()
            if client is None or not client.registered:
                return function(*args, **kwargs)
            values = signature.bind(*args, **kwargs).arguments
            seconds = values.get(timeout)
            if seconds is None:
                seconds = (
                    fallback(values)
                    if callable(fallback)
                    else client.control_timeout
                    if fallback is not None
                    else None
                )
            if seconds is None or not math.isfinite(seconds) or seconds <= 0:
                return function(*args, **kwargs)  # Existing validation owns bad input.
            with client.operation(function.__name__, seconds, cleanup=cleanup):
                return function(*args, **kwargs)

        return invoke

    return decorate


def _recover(registration: dict, *, timeout_seconds: float) -> dict:
    from .local_recovery import recover_resources

    return recover_resources(
        Path(registration["state_dir"]),
        docker_binary=registration["docker_binary"],
        registration=registration,
        timeout_seconds=timeout_seconds,
    )


def _write_report(registration: dict, report: dict, *, timeout_seconds: float) -> None:
    from .local_recovery import write_watchdog_report

    write_watchdog_report(registration, report, timeout_seconds=timeout_seconds)


@contextmanager
def _operator_signals(*, propagate: bool = False):
    """Latch cancellation without unwinding process creation or owned cleanup."""
    interruption = []
    previous = {}

    def interrupted(signum, frame):
        if not interruption:
            interruption.append(signum)

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, interrupted)
        yield interruption
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if propagate and interruption:
            signum = interruption[0]
            handler = previous[signum]
            if callable(handler) and handler is not signal.default_int_handler:
                handler(signum, None)


def supervise(
    command: Callable[[int], Sequence[str]],
    *,
    heartbeat_timeout: float = HEARTBEAT_TIMEOUT,
    grace_seconds: float = DEADLINE_GRACE_SECONDS,
    recovery_seconds: float = DEFAULT_RECOVERY_TIMEOUT_SECONDS,
    recover: Callable[..., dict] = _recover,
) -> int:
    """Run one trusted launcher, fence it before reaping, then reclaim ownership."""

    recovery_seconds = recovery_timeout_seconds(recovery_seconds)
    if os.name != "posix" or signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        raise WorkerFailure(
            "Docker watchdog requires POSIX and exclusive child reaping"
        )
    if any(
        not math.isfinite(value) or value <= 0
        for value in (heartbeat_timeout, grace_seconds)
    ):
        raise WorkerFailure("watchdog tolerances must be finite positive seconds")
    # Keep handlers installed through fencing, recovery and report persistence.
    # Raising from a handler can lose a just-created Popen handle or abort a
    # cleanup that has already started. Further signals do not skip that work.
    with _operator_signals() as interruption:
        return _supervise(
            command,
            heartbeat_timeout=heartbeat_timeout,
            grace_seconds=grace_seconds,
            recovery_seconds=recovery_seconds,
            recover=recover,
            interruption=interruption,
        )


def _supervise(
    command: Callable[[int], Sequence[str]],
    *,
    heartbeat_timeout: float,
    grace_seconds: float,
    recovery_seconds: float,
    recover: Callable[..., dict],
    interruption: list[int],
) -> int:
    parent, child = socket.socketpair()
    process = None
    registration = None
    completed_code = None
    reason = "worker_exited"
    detail = None
    failures = []
    deadlines = {}
    buffer = b""
    try:
        process = subprocess.Popen(
            command(child.fileno()),
            pass_fds=(child.fileno(),),
            start_new_session=True,
        )
        child.close()
        parent.settimeout(0.1)
        last_heartbeat = time.monotonic()
        while True:
            if interruption:
                reason = "operator_interrupted"
                break
            now = time.monotonic()
            expired = [
                label for deadline, label in deadlines.values() if now >= deadline
            ]
            if expired:
                reason, detail = "deadline_exceeded", expired[0]
                break
            if now - last_heartbeat >= heartbeat_timeout:
                reason = "worker_unresponsive"
                break
            try:
                data = parent.recv(MAX_MESSAGE_BYTES)
            except socket.timeout:
                continue
            if not data:
                break
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if len(line) > MAX_MESSAGE_BYTES:
                    raise WorkerFailure("oversized watchdog control message")
                message = json.loads(line)
                kind = message["type"]
                now = time.monotonic()
                last_heartbeat = now
                # Late end messages cannot withdraw an already expired deadline.
                if any(now >= deadline for deadline, _ in deadlines.values()):
                    reason = "deadline_exceeded"
                    break
                if kind == "heartbeat":
                    last_heartbeat = now
                    continue
                if kind == "finished":
                    code = message["exit_code"]
                    if not isinstance(code, int) or not 0 <= code <= 255:
                        raise WorkerFailure("invalid completed Worker status")
                    completed_code = code
                    reason = "worker_completed"
                    parent.sendall(b"!")
                    break
                elif kind == "register" and registration is None:
                    root = Path(message["state_dir"])
                    if not root.is_absolute() or len(message["owner"]) != 32:
                        raise WorkerFailure("invalid watchdog registration")
                    registration = message
                elif kind == "arm" and registration is not None:
                    seconds = message["seconds"]
                    if (
                        not isinstance(seconds, (int, float))
                        or not math.isfinite(seconds)
                        or seconds < 0
                        or message["key"] in deadlines
                        or len(deadlines) >= 128
                    ):
                        raise WorkerFailure("invalid watchdog deadline")
                    deadlines[message["key"]] = (
                        now + seconds + grace_seconds,
                        message["label"],
                    )
                    if not math.isfinite(deadlines[message["key"]][0]):
                        raise WorkerFailure("watchdog deadline overflow")
                elif kind == "disarm":
                    deadlines.pop(message["key"], None)
                else:
                    raise WorkerFailure("invalid watchdog control transition")
                parent.sendall(b"!")
            if reason in {"deadline_exceeded", "worker_completed"}:
                break
            if len(buffer) > MAX_MESSAGE_BYTES:
                raise WorkerFailure("oversized watchdog control message")
    except KeyboardInterrupt:
        reason = "operator_interrupted"
    except Exception as exc:
        reason = "watchdog_failure"
        failures.append(str(exc))
    finally:
        child.close()
        if process is not None:
            # NEVER poll/wait before this signal. Even an exited leader remains
            # our unreaped child, pinning its PID/PGID against reuse. All Docker
            # clients inherit this dedicated group, including create clients
            # that retain the journal lock after Worker process loss.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                failures.append("Worker process-group termination failed: " + str(exc))
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                failures.append("Worker process termination remains unconfirmed")
        parent.close()

    cleanup = None
    recovery_deadline = time.monotonic() + recovery_seconds
    if registration is not None:
        try:
            cleanup = recover(registration, timeout_seconds=recovery_seconds)
        except Exception as exc:
            failures.append("Worker recovery failed: " + str(exc))
    if interruption and reason in {"worker_completed", "worker_exited"}:
        reason = "operator_interrupted"
    report = {
        "reason": reason,
        "detail": detail,
        "operator_signal": interruption[0] if interruption else None,
        "worker_exit_code": process.returncode if process is not None else None,
        "completed_exit_code": completed_code,
        "recovery_timeout_seconds": recovery_seconds,
        "cleanup": cleanup,
        "report_persisted": False,
        "failures": failures,
        "resumed": False,
    }
    if registration is not None:
        try:
            remaining = recovery_deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerFailure(
                    "recovery deadline exhausted before report persistence"
                )
            if cleanup is not None and not cleanup.get("process_supervision", {}).get(
                "termination_confirmed", True
            ):
                raise WorkerFailure(
                    "report deferred while recovery process termination is unconfirmed"
                )
            _write_report(
                registration,
                {**report, "report_persisted": True},
                timeout_seconds=remaining,
            )
            report["report_persisted"] = True
        except Exception as exc:
            failures.append("cannot persist watchdog report: " + str(exc))
    cleanup_status = (
        "confirmed"
        if cleanup is not None and cleanup.get("cleanup_complete")
        else "unconfirmed"
    )
    report_location = (
        f"report {registration['state_dir']}/watchdog-report.json"
        if registration is not None and report["report_persisted"]
        else "current report was not persisted; any retained report may be from an earlier attempt"
        if registration is not None
        else "no journal registered; pre-journal resources are outside recovery scope"
    )
    print(
        f"Worker watchdog: {reason}; cleanup {cleanup_status}; {report_location}"
        + (
            f"; recovery attempt {cleanup['process_supervision']['attempt_id']}"
            if cleanup is not None and "process_supervision" in cleanup
            else ""
        ),
        file=sys.stderr,
        flush=True,
    )
    for failure in failures:
        print("Worker watchdog: " + failure, file=sys.stderr, flush=True)
    if (
        interruption
        or reason != "worker_completed"
        or failures
        or (cleanup is not None and not cleanup["cleanup_complete"])
        or process is None
        or process.returncode is None
    ):
        return 2
    return completed_code


def supervise_start(
    argv: Sequence[str],
    *,
    heartbeat_timeout: float = HEARTBEAT_TIMEOUT,
    grace_seconds: float = DEADLINE_GRACE_SECONDS,
    recovery_seconds: float = DEFAULT_RECOVERY_TIMEOUT_SECONDS,
) -> int:
    return supervise(
        lambda fd: [
            sys.executable,
            "-m",
            "autotrain_bench.local_watchdog",
            str(fd),
            *argv,
        ],
        heartbeat_timeout=heartbeat_timeout,
        grace_seconds=grace_seconds,
        recovery_seconds=recovery_seconds,
    )


@watched(fallback=120)
def _flush_output() -> bool:
    flushed = True
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            flushed = False
    return flushed


def worker_main() -> None:
    global _client
    from .cli import main

    _client = WatchdogClient(int(sys.argv[1]))
    try:
        main(sys.argv[2:])
        code = 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 0 if exc.code is None else 2
    except BaseException:
        traceback.print_exc()
        code = 2
    # main has completed its ExitStack/cleanup. Stay an unreaped group leader
    # until the parent fences leftover clients, avoiding PID/PGID reuse even on
    # the normal path. The parent's CLI exit status uses this completed result.
    # SIGKILL cannot run Python's normal buffered-stream shutdown. Flush BEFORE
    # telling the parent it may fence us, and never acknowledge output loss as
    # success. The operation guard also bounds a stuck flush after registration.
    if not _flush_output():
        code = 2
    _client._request({"type": "finished", "exit_code": code})
    while True:
        signal.pause()


if __name__ == "__main__":
    # Use the canonical module instance also imported by cli/runtime.
    from autotrain_bench.local_watchdog import worker_main

    worker_main()
