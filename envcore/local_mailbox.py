"""Hardened local-file transport for the Agent control API.

This module owns filesystem mechanics only. Benchmark request semantics remain
in :mod:`envcore.supervisor`, and Docker or other workload placement
belongs to a deployment launcher.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import stat
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .agent_control import (
    MAX_AGENT_CONTROL_RESPONSE_BYTES,
    AgentControlCleanupResult,
    AgentControlEvent,
    AgentControlExchange,
    AgentControlStepResult,
    ProtocolError,
)


MAX_MAILBOX_REQUEST_BYTES = 1024 * 1024
MAX_MAILBOX_RESPONSE_BYTES = MAX_AGENT_CONTROL_RESPONSE_BYTES
MAX_MAILBOX_PENDING_ENTRIES = 128
MAX_MAILBOX_PENDING_BYTES = 64 * 1024 * 1024

_AGENT_CONTROLLED_OPEN_ERRNOS = {
    errno.EACCES,
    errno.EISDIR,
    errno.ELOOP,
    errno.ENOENT,
    errno.ENOTDIR,
    errno.EPERM,
}


class MailboxMessageError(RuntimeError):
    """One Agent-controlled mailbox message is invalid."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class MailboxLimitExceeded(RuntimeError):
    """The visible Agent-control surface exceeded its bounded policy."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def compact_json_bytes(payload: Any) -> bytes:
    """Encode transport JSON deterministically and normalize invalid surrogates."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8", errors="replace")


def valid_request_id(request_id: str) -> bool:
    return bool(request_id) and all(
        value in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for value in request_id
    )


def directory_fd_operations_available() -> bool:
    """Return whether Python exposes the required race-resistant dir-fd APIs."""

    return all(
        function in os.supports_dir_fd for function in (os.open, os.rename, os.unlink)
    )


def read_untrusted_request(path: Path, maximum_bytes: int) -> Dict[str, Any]:
    """Read one mailbox request without following Agent-created links."""

    try:
        initial_metadata = path.lstat()
    except FileNotFoundError as exc:
        raise MailboxMessageError("invalid_request", "request is unavailable") from exc
    if not stat.S_ISREG(initial_metadata.st_mode):
        raise MailboxMessageError("invalid_request", "request must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        if exc.errno in _AGENT_CONTROLLED_OPEN_ERRNOS:
            raise MailboxMessageError(
                "invalid_request", "request must be a regular file"
            ) from exc
        raise
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise MailboxMessageError(
                "invalid_request", "request must be a regular file"
            )
        if (metadata.st_dev, metadata.st_ino) != (
            initial_metadata.st_dev,
            initial_metadata.st_ino,
        ):
            raise MailboxMessageError(
                "invalid_request", "request changed while opening"
            )
        if metadata.st_size > maximum_bytes:
            raise MailboxMessageError("request_too_large", "request exceeds 1 MiB")
        try:
            with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                descriptor = -1
                payload = json.load(stream)
        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise MailboxMessageError(
                "invalid_request", "request is not valid UTF-8 JSON"
            ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(payload, dict):
        raise MailboxMessageError("invalid_request", "request must be an object")
    return payload


def bounded_response(response: Dict[str, Any], maximum_bytes: int) -> Dict[str, Any]:
    """The actual response published when a handler result exceeds the limit."""
    if len(compact_json_bytes(response)) <= maximum_bytes:
        return response
    return {
        "ok": False,
        "error": {
            "code": "response_too_large",
            "message": "supervisor response exceeds the mailbox limit",
        },
    }


def write_untrusted_response(
    directory: Path,
    request_id: str,
    response: Dict[str, Any],
    *,
    maximum_bytes: int = MAX_MAILBOX_RESPONSE_BYTES,
) -> None:
    """Publish a mailbox response without following Agent-created links."""

    if not valid_request_id(request_id):
        raise MailboxMessageError(
            "invalid_request", "request id contains invalid characters"
        )
    temporary = f".{request_id}.{uuid.uuid4().hex}.tmp"
    destination = f"{request_id}.json"
    destination_path = directory / destination
    if destination_path.is_dir():
        raise MailboxMessageError(
            "mailbox_conflict", "mailbox response destination is a directory"
        )
    encoded = compact_json_bytes(bounded_response(response, maximum_bytes))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    if not directory_fd_operations_available():
        # An isolated workload sees inbox and outbox as separate mount roots, so
        # it cannot replace this directory itself. Host-native Agents remain
        # explicitly non-compliant.
        if directory.is_symlink() or not directory.is_dir():
            raise MailboxMessageError(
                "invalid_request", "mailbox response directory is invalid"
            )
        temporary_path = directory / temporary
        descriptor = os.open(str(temporary_path), flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(encoded)
            os.replace(str(temporary_path), str(destination_path))
        except Exception as exc:
            temporary_path.unlink(missing_ok=True)
            if destination_path.is_dir():
                raise MailboxMessageError(
                    "mailbox_conflict",
                    "mailbox response destination is a directory",
                ) from exc
            raise
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        return

    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_descriptor = os.open(str(directory), directory_flags)
    descriptor = -1
    try:
        descriptor = os.open(temporary, flags, 0o600, dir_fd=directory_descriptor)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(encoded)
        os.replace(
            temporary,
            destination,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
    except Exception as exc:
        try:
            os.unlink(temporary, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        if destination_path.is_dir():
            raise MailboxMessageError(
                "mailbox_conflict",
                "mailbox response destination is a directory",
            ) from exc
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_descriptor)


class LocalFileMailbox:
    """Bounded local transport storage shared with one Tested Agent."""

    transport_name = "file_mailbox"

    def __init__(
        self,
        control_dir: Path,
        *,
        maximum_request_bytes: int = MAX_MAILBOX_REQUEST_BYTES,
        maximum_response_bytes: int = MAX_MAILBOX_RESPONSE_BYTES,
        maximum_pending_entries: int = MAX_MAILBOX_PENDING_ENTRIES,
        maximum_pending_bytes: int = MAX_MAILBOX_PENDING_BYTES,
    ) -> None:
        self.control_dir = control_dir
        self.inbox = control_dir / "inbox"
        self.outbox = control_dir / "outbox"
        self.rejected = control_dir / "rejected"
        self.maximum_request_bytes = maximum_request_bytes
        self.maximum_response_bytes = maximum_response_bytes
        self.maximum_pending_entries = maximum_pending_entries
        self.maximum_pending_bytes = maximum_pending_bytes

    @staticmethod
    def valid_request_id(request_id: str) -> bool:
        return valid_request_id(request_id)

    def initialize(self) -> None:
        for directory in (self.inbox, self.outbox, self.rejected):
            directory.mkdir(parents=True, exist_ok=True)
        os.chmod(self.control_dir, 0o700)
        os.chmod(self.inbox, 0o777)
        os.chmod(self.outbox, 0o777)
        os.chmod(self.rejected, 0o700)

    def _check_pending_limits(self) -> None:
        """Bound the live queue; deployment storage quotas remain authoritative.

        Nested contents of a hostile non-regular entry are intentionally not
        traversed. Doing so would enlarge the attack surface before the entry
        is quarantined. The Worker must place this directory on quota-bounded,
        disposable storage to contain bytes written before inspection.
        """

        entries = 0
        total_bytes = 0
        for directory in (self.inbox, self.outbox):
            with os.scandir(directory) as children:
                for child in children:
                    try:
                        metadata = child.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    entries += 1
                    if stat.S_ISREG(metadata.st_mode):
                        total_bytes += metadata.st_size
                    if entries > self.maximum_pending_entries:
                        raise MailboxLimitExceeded(
                            "mailbox_entry_limit",
                            "Agent control mailbox contains too many pending entries",
                        )
                    if total_bytes > self.maximum_pending_bytes:
                        raise MailboxLimitExceeded(
                            "mailbox_byte_limit",
                            "Agent control mailbox exceeds its pending byte limit",
                        )

    def next_request(self) -> Optional[Path]:
        self._check_pending_limits()
        paths = sorted(self.inbox.glob("*.json"))
        return paths[0] if paths else None

    def read_request(self, request_path: Path) -> Dict[str, Any]:
        return read_untrusted_request(request_path, self.maximum_request_bytes)

    def write_response(self, request_id: str, response: Dict[str, Any]) -> None:
        write_untrusted_response(
            self.outbox,
            request_id,
            response,
            maximum_bytes=self.maximum_response_bytes,
        )

    def quarantine(self, request_path: Path) -> tuple[bool, bool]:
        """Remove one untrusted entry from the live queue without following it."""

        quarantined = self.rejected / f"{uuid.uuid4().hex}.rejected"
        try:
            os.replace(str(request_path), str(quarantined))
        except FileNotFoundError:
            return True, False
        policy_violation = True
        try:
            metadata = quarantined.lstat()
            policy_violation = not stat.S_ISREG(metadata.st_mode)
            if not stat.S_ISDIR(metadata.st_mode):
                quarantined.unlink()
        except FileNotFoundError:
            policy_violation = True
        return True, policy_violation

    def process_one(
        self,
        handler: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> AgentControlStepResult:
        """Dispatch one local request to a transport-independent handler."""

        events = []
        try:
            request_path = self.next_request()
        except MailboxLimitExceeded as exc:
            events.append(
                AgentControlEvent("MAILBOX_REQUEST_REJECTED", {"code": exc.code})
            )
            return AgentControlStepResult(
                handled=True,
                policy_failure=True,
                events=tuple(events),
            )
        if request_path is None:
            return AgentControlStepResult(handled=False)

        request_id = request_path.stem
        if not self.valid_request_id(request_id):
            isolated, policy_violation = self.quarantine(request_path)
            events.append(
                AgentControlEvent(
                    "MAILBOX_REQUEST_REJECTED",
                    {"code": "invalid_request_id"},
                )
            )
            return AgentControlStepResult(
                handled=True,
                policy_failure=not isolated or policy_violation,
                events=tuple(events),
            )

        try:
            request = self.read_request(request_path)
        except MailboxMessageError as exc:
            code = exc.code
            response = {
                "ok": False,
                "error": {"code": code, "message": str(exc)},
            }
            events.append(
                AgentControlEvent(
                    "MAILBOX_REQUEST_REJECTED",
                    {"code": code},
                )
            )
            infrastructure_failure_code = None
            infrastructure_failure = None
        except Exception as exc:
            response = {
                "ok": False,
                "error": {
                    "code": "infrastructure_failure",
                    "message": "trusted mailbox request read failed",
                },
            }
            infrastructure_failure_code = "agent_control_transport_failure"
            infrastructure_failure = str(exc)
        else:
            try:
                result = handler(request)
            except ProtocolError as exc:
                response = {
                    "ok": False,
                    "error": {"code": exc.code, "message": str(exc)},
                }
                infrastructure_failure_code = None
                infrastructure_failure = None
            except Exception as exc:
                response = {
                    "ok": False,
                    "error": {
                        "code": "infrastructure_failure",
                        "message": "trusted request handler failed",
                    },
                }
                infrastructure_failure_code = "agent_control_handler_failure"
                infrastructure_failure = str(exc)
            else:
                response = {"ok": True, "result": result}
                infrastructure_failure_code = None
                infrastructure_failure = None
        finally:
            isolated, policy_violation = self.quarantine(request_path)

        policy_failure = not isolated or policy_violation
        if policy_failure:
            events.append(
                AgentControlEvent(
                    "MAILBOX_REQUEST_REJECTED",
                    {
                        "code": (
                            "quarantine_failed"
                            if not isolated
                            else "non_regular_request"
                        )
                    },
                )
            )
        delivery = "not_published"
        response = bounded_response(response, self.maximum_response_bytes)
        if isolated:
            try:
                self.write_response(request_id, response)
                delivery = "published"
            except MailboxMessageError as exc:
                # The Agent controls its outbox.  If it prevents publication
                # after a side-effecting request has run, continuing would make
                # retries ambiguous.  Classify that manipulation as policy
                # failure, never as trusted infrastructure failure.
                policy_failure = True
                events.append(
                    AgentControlEvent(
                        "MAILBOX_RESPONSE_REJECTED",
                        {"code": exc.code},
                    )
                )
        return AgentControlStepResult(
            handled=True,
            policy_failure=policy_failure,
            infrastructure_failure_code=infrastructure_failure_code,
            infrastructure_failure=infrastructure_failure,
            events=tuple(events),
            exchange=AgentControlExchange(request_id, response, delivery),
        )

    def cleanup(self) -> AgentControlCleanupResult:
        """Remove all transport artifacts after the Agent has stopped."""

        removed = 0
        agent_policy_violations = 0
        agent_caused_failures = 0
        infrastructure_failures = 0
        for directory in (self.inbox, self.outbox, self.rejected):
            try:
                children = directory.iterdir()
                for entry in children:
                    try:
                        metadata = entry.lstat()
                    except FileNotFoundError:
                        continue
                    except OSError:
                        infrastructure_failures += 1
                        continue
                    regular = stat.S_ISREG(metadata.st_mode)
                    if not regular:
                        agent_policy_violations += 1
                    try:
                        if stat.S_ISDIR(metadata.st_mode):
                            shutil.rmtree(entry)
                        else:
                            entry.unlink()
                        removed += 1
                    except FileNotFoundError:
                        continue
                    except OSError:
                        if regular:
                            infrastructure_failures += 1
                        else:
                            agent_caused_failures += 1
            except OSError:
                infrastructure_failures += 1
        return AgentControlCleanupResult(
            removed=removed,
            agent_policy_violations=agent_policy_violations,
            agent_caused_failures=agent_caused_failures,
            infrastructure_failures=infrastructure_failures,
        )
