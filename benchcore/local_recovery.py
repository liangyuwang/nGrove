"""Bounded process supervision for local Docker recovery and report I/O.

The parent never opens recovery storage. A fresh interpreter owns all journal,
runtime and report operations; the parent only exchanges bounded socket frames
and fences its own unreaped process group before waiting for termination.
"""

from __future__ import annotations

import json
import os
import select
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from .docker_recovery import DEFAULT_RECOVERY_TIMEOUT_SECONDS, recovery_timeout_seconds
from .worker_protocol import WorkerFailure


MAX_FRAME_BYTES = 1024 * 1024
TERMINATION_WAIT_SECONDS = 5.0


def _command(fd: int) -> list[str]:
    # A direct integration may import Core from a checkout via sys.path rather
    # than an installed package. Preserve that trusted source in the fresh
    # interpreter without changing the caller's cwd or resolving storage paths.
    return [
        sys.executable,
        "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from autotrain_bench.local_recovery import worker_main; "
        "worker_main(int(sys.argv[2]))",
        str(Path(__file__).parent.parent),
        str(fd),
    ]


def _encode(payload: dict) -> bytes:
    data = json.dumps(payload, allow_nan=False).encode() + b"\n"
    if len(data) > MAX_FRAME_BYTES:
        raise WorkerFailure("recovery process message capacity reached")
    return data


def _exchange(channel: socket.socket, request: bytes, deadline: float) -> dict:
    channel.setblocking(False)
    sent = 0
    received = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("recovery process deadline exceeded")
        readable, writable, _ = select.select(
            [channel] if sent == len(request) else [],
            [channel] if sent < len(request) else [],
            [],
            min(remaining, 0.1),
        )
        if writable:
            sent += channel.send(request[sent:])
        if readable:
            chunk = channel.recv(64 * 1024)
            if not chunk:
                raise WorkerFailure("recovery process exited without a complete reply")
            received.extend(chunk)
            if len(received) > MAX_FRAME_BYTES:
                raise WorkerFailure("recovery process reply capacity reached")
            if b"\n" in received:
                if not received.endswith(b"\n") or received.count(b"\n") != 1:
                    raise WorkerFailure("invalid recovery process reply framing")
                result = json.loads(received)
                if not isinstance(result, dict):
                    raise WorkerFailure("invalid recovery process reply")
                if time.monotonic() >= deadline:
                    raise TimeoutError("recovery process reply arrived after deadline")
                return result


def _run(request: dict, *, timeout_seconds: float) -> dict:
    timeout_seconds = recovery_timeout_seconds(timeout_seconds)
    if (
        os.name != "posix"
        or threading.current_thread() is not threading.main_thread()
        or signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL
    ):
        raise WorkerFailure(
            "Docker recovery supervision requires POSIX, the main thread "
            "and exclusive child reaping"
        )
    # Share the launcher's signal discipline, including nested use under its
    # existing handler. Signals never unwind Popen or interrupt fencing.
    from .local_watchdog import _operator_signals

    request = _encode({**request, "timeout_seconds": timeout_seconds})
    parent, child = socket.socketpair()
    process = None
    reply = None
    failures = []
    expired = False
    terminated = False
    with _operator_signals(propagate=True) as interruption:
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                _command(child.fileno()),
                pass_fds=(child.fileno(),),
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            child.close()
            reply = _exchange(parent, request, started + timeout_seconds)
        except TimeoutError as exc:
            expired = True
            failures.append(str(exc))
        except (OSError, WorkerFailure, ValueError) as exc:
            failures.append(str(exc)[:1024])
        finally:
            child.close()
            parent.close()
            if process is not None:
                # No poll/wait before this fence: our unreaped child pins its
                # PID/PGID even if it exited before acknowledging the operation.
                fenced = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    fenced = False
                    failures.append(
                        "recovery process-group fencing failed: " + str(exc)
                    )
                try:
                    process.wait(timeout=TERMINATION_WAIT_SECONDS)
                    terminated = fenced
                except subprocess.TimeoutExpired:
                    failures.append("recovery process termination remains unconfirmed")
                    # Do not wait indefinitely or release another process's
                    # journal lock. Popen retains unreaped children for polling.
        if reply is not None:
            if set(reply) == {"error"} and isinstance(reply["error"], str):
                failures.append(reply["error"])
                reply = None
            elif set(reply) != {"result"}:
                failures.append("invalid recovery process reply fields")
                reply = None
        return {
            "result": reply["result"] if reply is not None else None,
            "deadline_exceeded": expired,
            "termination_confirmed": terminated,
            "exit_code": process.returncode if process is not None else None,
            "operator_signal": interruption[0] if interruption else None,
            "failures": failures,
        }


def recover_resources(
    state_dir: Path,
    *,
    docker_binary: str,
    timeout_seconds: float = DEFAULT_RECOVERY_TIMEOUT_SECONDS,
    registration: dict | None = None,
) -> dict:
    attempt = uuid.uuid4().hex
    execution = _run(
        {
            "operation": "recover",
            "state_dir": str(state_dir),
            "docker_binary": docker_binary,
            "registration": registration,
        },
        timeout_seconds=timeout_seconds,
    )
    result = execution.pop("result")
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("cleanup_complete"), bool)
        or result.get("resumed") is not False
        or not isinstance(result.get("failures"), list)
        or not isinstance(result.get("deadline_exceeded"), bool)
    ):
        if result is not None:
            execution["failures"].append("invalid recovery result")
        result = {
            "cleanup_complete": False,
            "episode": None,
            "resumed": False,
            "failures": [],
        }
    result["timeout_seconds"] = timeout_seconds
    result["deadline_exceeded"] = (
        result.get("deadline_exceeded", False) or execution["deadline_exceeded"]
    )
    result["process_supervision"] = {"attempt_id": attempt, **execution}
    if execution["failures"] or not execution["termination_confirmed"]:
        result["cleanup_complete"] = False
    result["failures"].extend(
        {"kind": "recovery_process", "name": "helper", "reason": failure}
        for failure in execution["failures"]
    )
    return result


def write_watchdog_report(
    registration: dict, report: dict, *, timeout_seconds: float
) -> None:
    execution = _run(
        {"operation": "report", "registration": registration, "report": report},
        timeout_seconds=timeout_seconds,
    )
    if (
        execution["failures"]
        or not execution["termination_confirmed"]
        or execution["result"] != {"persisted": True}
    ):
        raise WorkerFailure(
            "; ".join(execution["failures"])
            or "watchdog report persistence remains unconfirmed"
        )


def _persist_watchdog_report(registration: dict, report: dict) -> None:
    root = Path(registration["state_dir"])
    metadata = root.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
        or [metadata.st_dev, metadata.st_ino] != registration["directory_identity"]
    ):
        raise WorkerFailure("watchdog report directory ownership changed")
    data = _encode(report)
    temporary = root / "watchdog-report.next"
    fd = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, root / "watchdog-report.json")
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _perform(request: dict):
    if request["operation"] == "report":
        _persist_watchdog_report(request["registration"], request["report"])
        return {"persisted": True}
    if request["operation"] != "recover":
        raise WorkerFailure("unknown recovery process operation")
    from .docker_recovery import DockerRecovery
    from .runtime import DockerRuntime

    runtime = DockerRuntime(docker_binary=request["docker_binary"])
    journal = DockerRecovery(Path(request["state_dir"]))
    runtime._recovery = journal
    runtime._recovery_clients_hold_lock = True
    try:
        registration = request["registration"]
        if registration is not None and (
            journal.state["owner"] != registration["owner"]
            or journal.state["directory_identity"] != registration["directory_identity"]
        ):
            raise WorkerFailure("watchdog Worker ownership differs from registration")
        return journal.recover(
            runtime._run_raw, timeout_seconds=request["timeout_seconds"]
        )
    finally:
        journal.close()


def worker_main(fd: int, *, perform: Callable = _perform) -> None:
    channel = socket.socket(fileno=fd)
    channel.set_inheritable(False)
    try:
        data = bytearray()
        while not data.endswith(b"\n"):
            chunk = channel.recv(64 * 1024)
            if not chunk:
                return
            data.extend(chunk)
            if len(data) > MAX_FRAME_BYTES:
                raise WorkerFailure("recovery process request capacity reached")
        result = perform(json.loads(data))
        reply = _encode({"result": result})
    except Exception as exc:
        reply = _encode({"error": f"{type(exc).__name__}: {exc}"[:1024]})
    channel.sendall(reply)
    # Hold the session-leader PID until the parent fences all inherited clients.
    while True:
        signal.pause()


if __name__ == "__main__":
    worker_main(int(sys.argv[1]))
