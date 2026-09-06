"""Private, cleanup-only recovery for one local Docker Worker session.

The journal is never a Core/Agent API and contains no resumable benchmark state.
Only a local POSIX filesystem with reliable flock/fsync semantics is supported.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import stat
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from .worker_protocol import WorkerFailure


OWNER_LABEL = "autotrain-bench.worker-owner"
RESOURCE_LABEL = "autotrain-bench.resource-token"
MAX_JOURNAL_BYTES = 4 * 1024 * 1024
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}\Z")
_TOKEN = re.compile(r"[a-f0-9]{32}\Z")
_IDENTITY = re.compile(r"[a-f0-9]{64}\Z")
# Bound the serialized volume creation timestamp, including JSON quotes and
# escapes, so its later identity capture has a finite admission reservation.
_MAX_VOLUME_IDENTITY_JSON_BYTES = 128


class DockerRecovery:
    def __init__(self, root: Path, *, episode: Optional[Path] = None) -> None:
        try:
            import fcntl
        except ImportError as exc:
            raise WorkerFailure(
                "Docker recovery requires local POSIX file locking"
            ) from exc
        self._fcntl = fcntl
        self._mutex = threading.RLock()
        self.lock_fd = -1
        self.root = root.expanduser().absolute()
        self.state: Dict[str, Any] = {}
        self._durable_state: Dict[str, Any] = {}
        self._can_create = episode is not None
        try:
            if episode is not None:
                self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            metadata = self.root.lstat()
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_mode & 0o077
            ):
                raise WorkerFailure(
                    "Worker state directory must be private and owned by this UID"
                )
            self.root = self.root.resolve()
            self.lock_fd = os.open(
                self.root / "owner.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
            )
            lock_stat = os.fstat(self.lock_fd)
            if lock_stat.st_uid != os.geteuid() or not stat.S_ISREG(lock_stat.st_mode):
                raise WorkerFailure("invalid Worker ownership lock")
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise WorkerFailure(
                    "Worker or its runtime client still holds ownership"
                ) from exc
            if episode is None:
                self.state = self._read()
                self._durable_state = copy.deepcopy(self.state)
            else:
                if any(path.name != "owner.lock" for path in self.root.iterdir()):
                    raise WorkerFailure(
                        "Worker state already exists; use worker-recover, not resume"
                    )
                self.state = {
                    "schema_version": 1,
                    "owner": uuid.uuid4().hex,
                    "episode": str(episode.resolve()),
                    "state_dir": str(self.root),
                    "directory_identity": [metadata.st_dev, metadata.st_ino],
                    "backend_id": None,
                    "phase": "active",
                    "resources": {},
                }
                self._save()
        except OSError as exc:
            self.close()
            raise WorkerFailure("cannot open private Worker recovery storage") from exc
        except BaseException:
            self.close()
            raise

    def _read(self) -> Dict[str, Any]:
        try:
            path = self.root / "resources.json"
            metadata = path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_mode & 0o077
                or metadata.st_size > MAX_JOURNAL_BYTES
            ):
                raise ValueError("unsafe journal file")
            state = json.loads(path.read_text(encoding="utf-8"))
            if set(state) != {
                "schema_version",
                "owner",
                "episode",
                "state_dir",
                "directory_identity",
                "backend_id",
                "phase",
                "resources",
            }:
                raise ValueError("invalid journal fields")
            if (
                not isinstance(state["schema_version"], int)
                or isinstance(state["schema_version"], bool)
                or state["schema_version"] != 1
                or not _TOKEN.fullmatch(state["owner"])
            ):
                raise ValueError("invalid journal identity")
            if (
                not isinstance(state["episode"], str)
                or not Path(state["episode"]).is_absolute()
            ):
                raise ValueError("invalid episode identity")
            directory = self.root.stat()
            if state["state_dir"] != str(self.root) or state["directory_identity"] != [
                directory.st_dev,
                directory.st_ino,
            ]:
                raise ValueError(
                    "journal was copied or its ownership directory replaced"
                )
            if state["backend_id"] is not None and not isinstance(
                state["backend_id"], str
            ):
                raise ValueError("invalid backend identity")
            if state["phase"] not in {"active", "recovering", "cleaned"}:
                raise ValueError("invalid Worker phase")
            if not isinstance(state["resources"], dict):
                raise ValueError("invalid resource records")
            for key, record in state["resources"].items():
                if set(record) != {"kind", "name", "token", "state", "identity"}:
                    raise ValueError("invalid resource fields")
                if record["kind"] not in {"container", "network", "volume"}:
                    raise ValueError("invalid resource kind")
                if not _NAME.fullmatch(record["name"]) or not _TOKEN.fullmatch(
                    record["token"]
                ):
                    raise ValueError("invalid resource name or token")
                if key != record["kind"] + ":" + record["name"]:
                    raise ValueError("invalid resource key")
                if record["state"] not in {
                    "intent",
                    "completed",
                    "identified",
                    "removed",
                }:
                    raise ValueError("invalid resource state")
                if record["identity"] is not None and not isinstance(
                    record["identity"], str
                ):
                    raise ValueError("invalid resource identity")
                if record["identity"] is not None and record["kind"] != "volume":
                    if not _IDENTITY.fullmatch(record["identity"]):
                        raise ValueError("invalid immutable runtime ID")
                if record["identity"] is not None and record["kind"] == "volume":
                    if (
                        len(json.dumps(record["identity"]))
                        > _MAX_VOLUME_IDENTITY_JSON_BYTES
                    ):
                        raise ValueError("volume identity exceeds its reserved size")
            return state
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise WorkerFailure(
                "Worker recovery journal is missing, damaged or unsafe"
            ) from exc

    @staticmethod
    def _encode(state: Dict[str, Any]) -> bytes:
        data = (json.dumps(state, sort_keys=True) + "\n").encode("utf-8")
        if len(data) > MAX_JOURNAL_BYTES:
            raise WorkerFailure("Worker recovery journal capacity reached")
        return data

    @classmethod
    def _reserve_lifecycle(cls, state: Dict[str, Any]) -> None:
        # All records may grow after admission, including other creates still
        # awaiting their responses. identified and recovering are the longest
        # resource/session states. Existing and removed records retain the same
        # reservation, so interrupted creation/deletion and retry fit as well.
        cls._encode(
            {
                **state,
                "phase": "recovering",
                "resources": {
                    key: {
                        **record,
                        "state": "identified",
                        "identity": "0"
                        * (
                            _MAX_VOLUME_IDENTITY_JSON_BYTES - 2
                            if record["kind"] == "volume"
                            else 64
                        ),
                    }
                    for key, record in state["resources"].items()
                },
            }
        )

    def _save(self) -> None:
        self._require_open()
        data = self._encode(self.state)
        temporary = self.root / "resources.next"
        try:
            fd = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.root / "resources.json")
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as exc:
            # In particular, rename may have succeeded before directory fsync
            # failed. Retain pending records rather than rolling back an intent
            # that could be on disk; they must not become durable proof in RAM.
            raise WorkerFailure("cannot persist Worker recovery journal") from exc
        self._durable_state = copy.deepcopy(self.state)

    def _require_creation(self) -> None:
        if not self._can_create or self.state["phase"] != "active":
            raise WorkerFailure("recovered Worker sessions cannot launch workloads")
        if self.state != self._durable_state:
            raise WorkerFailure(
                "Worker journal has uncommitted changes; only cleanup is allowed"
            )

    def _require_open(self) -> None:
        if self.lock_fd < 0:
            raise WorkerFailure("Worker ownership has been released")

    def close(self) -> None:
        if self.lock_fd >= 0:
            # Do not LOCK_UN: a still-running Docker child inherits this open
            # description and must retain the lock after Worker process loss.
            os.close(self.lock_fd)
            self.lock_fd = -1

    def bind_backend(self, run: Callable) -> None:
        self._require_open()
        actual = run(["info", "--format", "{{.ID}}"], timeout=20).stdout.strip()
        if not actual or len(actual) > 256 or "\n" in actual:
            raise WorkerFailure("cannot establish Docker daemon identity")
        with self._mutex:
            expected = self.state["backend_id"]
            if expected is None:
                if self.state["resources"]:
                    raise WorkerFailure(
                        "resource records lack a Docker daemon identity"
                    )
                self.state["backend_id"] = actual
                self._save()
            elif actual != expected:
                raise WorkerFailure(
                    "Docker daemon identity differs from the recorded Worker"
                )

    def private_directory(self, prefix: str) -> Path:
        with self._mutex:
            self._require_open()
            if not isinstance(prefix, str) or not _NAME.fullmatch(prefix + "x"):
                raise WorkerFailure("invalid private storage prefix")
            self._require_creation()
            parent = self.root / "private"
            if parent.is_symlink():
                raise WorkerFailure("Worker private storage became a symlink")
            parent.mkdir(mode=0o700, exist_ok=True)
            # All possible partial directories are inside this exclusively
            # owned subtree, even if the process dies during mkdir/copy.
            directory = parent / (prefix + uuid.uuid4().hex)
            directory.mkdir(mode=0o700)
            return directory

    @staticmethod
    def _creation(args: Sequence[str]):
        if args[:1] in (["run"], ["create"]):
            if "--name" not in args:
                raise WorkerFailure("recoverable containers require an explicit name")
            return "container", args[args.index("--name") + 1], 1
        if args[:2] in (["network", "create"], ["volume", "create"]):
            return args[0], args[-1], 2
        return None

    @staticmethod
    def _container_create_args(args: Sequence[str]):
        # Parse only the long options emitted by this Worker. Stop at IMAGE,
        # preserving participant command arguments even if named --rm/--detach.
        booleans = {"--detach", "--rm", "--init", "--read-only"}
        index = 1
        while index < len(args) and args[index].startswith("--"):
            index += 1 if args[index] in booleans or "=" in args[index] else 2
        if index >= len(args):
            raise WorkerFailure("recoverable container lacks an image")
        options = args[1:index]
        return (
            [
                "create",
                *(item for item in options if item not in {"--detach", "--rm"}),
                *args[index:],
            ],
            "--detach" in options,
            "--rm" in options,
        )

    def execute(self, args: Sequence[str], run: Callable, **kwargs):
        self._require_open()
        args = list(args)
        creation = self._creation(args)
        if creation is None:
            if args[:1] == ["rm"]:
                self.remove("container", args[-1], run)
                return subprocess.CompletedProcess(args, 0, "", "")
            if args[:2] in (["network", "rm"], ["volume", "rm"]):
                self.remove(args[0], args[-1], run)
                return subprocess.CompletedProcess(args, 0, "", "")
            return run(args, **kwargs)
        deadline = (
            None
            if kwargs.get("timeout") is None
            else time.monotonic() + kwargs["timeout"]
        )
        raw_run = run

        def run(command, **options):
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise WorkerFailure("Worker resource bookkeeping deadline reached")
                options["timeout"] = min(options.get("timeout") or remaining, remaining)
            return raw_run(command, **options)

        kind, name, insertion = creation
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise WorkerFailure("invalid recoverable resource name")
        self.bind_backend(run)
        with self._mutex:
            self._require_creation()
            key = kind + ":" + name
            previous = self.state["resources"].get(key)
            if previous is not None and previous["state"] != "removed":
                raise WorkerFailure("resource has unresolved prior ownership")
            record = {
                "kind": kind,
                "name": name,
                "token": uuid.uuid4().hex,
                "state": "intent",
                "identity": None,
            }
            # Reject before changing RAM/disk or dispatching create. Reserve
            # subsequent identity capture and recovery, not only this intent.
            self._reserve_lifecycle(
                {
                    **self.state,
                    "resources": {**self.state["resources"], key: record},
                }
            )
            self.state["resources"][key] = record
            self._save()  # No runtime create command is sent before this succeeds.
        labels = [
            "--label",
            f"{OWNER_LABEL}={self.state['owner']}",
            "--label",
            f"{RESOURCE_LABEL}={record['token']}",
        ]
        args[insertion:insertion] = labels
        check = kwargs.pop("check", True)
        run_container = kind == "container" and args[0] == "run"
        if run_container:
            create_args, detached, auto_remove = self._container_create_args(args)
        else:
            create_args = args
        result = run(create_args, check=False, **kwargs)
        # A timeout/exception leaves intent, including a request that the
        # daemon may still be processing. Absence alone cannot resolve it.
        with self._mutex:
            completed = result.returncode == 0
            if completed:
                record["state"] = "completed"
            self._save()
            actual = self._inspect(record, run)
            if actual is None and completed:
                record["state"] = "removed"
            elif actual is not None:
                self._bind_identity(record, actual)
            self._save()
        if run_container and result.returncode == 0:
            if actual is None:
                raise WorkerFailure(
                    "created container disappeared before identity capture"
                )
            identity = actual["identity"]
            try:
                # Ownership is durable before start. Attached/waiting clients
                # need not retain the owner lock: they can only address this
                # immutable container ID, which recovery can safely terminate.
                started = run(
                    ["start", *([] if detached else ["--attach"]), identity],
                    check=False,
                    **kwargs,
                )
                if detached or started.returncode != 0:
                    result = subprocess.CompletedProcess(
                        args,
                        started.returncode,
                        identity + "\n" if detached else started.stdout,
                        started.stderr,
                    )
                else:
                    waited = run(["wait", identity], **kwargs)
                    try:
                        exit_code = int(waited.stdout.strip())
                        if not 0 <= exit_code <= 255:
                            raise ValueError("invalid exit code")
                    except ValueError as exc:
                        raise WorkerFailure(
                            "invalid completed container status"
                        ) from exc
                    result = subprocess.CompletedProcess(
                        args, exit_code, started.stdout, started.stderr
                    )
            finally:
                if auto_remove:
                    self.remove(kind, name, raw_run)
        if check and result.returncode != 0:
            raise WorkerFailure(
                "Docker command failed: "
                + (result.stderr.strip() or result.stdout.strip())
            )
        return result

    def _inspect(
        self, record: Dict[str, Any], run: Callable
    ) -> Optional[Dict[str, Any]]:
        kind, name = record["kind"], record["name"]
        target = record["identity"] if kind != "volume" and record["identity"] else name
        result = run([kind, "inspect", target], check=False, timeout=20)
        if result.returncode != 0:
            field = "{{.Names}}" if kind == "container" else "{{.Name}}"
            command = [kind, "ls"] + (["--all"] if kind == "container" else [])
            if kind != "volume":
                command += ["--no-trunc"]
                field = "{{.ID}}\t" + field
            inventory = run([*command, "--format", field], timeout=20)
            for line in inventory.stdout.splitlines():
                if kind == "volume":
                    present = line == name
                else:
                    fields = line.split("\t")
                    if len(fields) != 2 or not _IDENTITY.fullmatch(fields[0]):
                        raise WorkerFailure("invalid runtime identity inventory")
                    present = fields[0] == target or fields[1] == name
                if present:
                    raise WorkerFailure(
                        "refusing unverified resource ownership or inspection"
                    )
            return None
        try:
            payload = json.loads(result.stdout)
            if not isinstance(payload, list) or len(payload) != 1:
                raise ValueError("invalid inspection")
            actual = payload[0]
            labels = (
                actual["Config"].get("Labels")
                if kind == "container"
                else actual.get("Labels")
            )
            if (
                not isinstance(labels, dict)
                or labels.get(OWNER_LABEL) != self.state["owner"]
                or labels.get(RESOURCE_LABEL) != record["token"]
                or actual["Name"].lstrip("/") != name
            ):
                raise ValueError("ownership mismatch")
            identity = actual["CreatedAt"] if kind == "volume" else actual["Id"]
            if not isinstance(identity, str) or not identity:
                raise ValueError("missing resource identity")
            if kind != "volume" and not _IDENTITY.fullmatch(identity):
                raise ValueError("invalid immutable resource ID")
            if (
                kind == "volume"
                and len(json.dumps(identity)) > _MAX_VOLUME_IDENTITY_JSON_BYTES
            ):
                raise ValueError("volume identity exceeds its reserved size")
            if record["identity"] is not None and identity != record["identity"]:
                raise ValueError("resource replaced")
            return {"identity": identity}
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise WorkerFailure(
                f"refusing unverified {kind} ownership: {name}"
            ) from exc

    @staticmethod
    def _bind_identity(record: Dict[str, Any], actual: Dict[str, Any]) -> None:
        record["identity"] = actual["identity"]
        record["state"] = "identified"

    def remove(self, kind: str, name: str, run: Callable) -> None:
        self._require_open()
        self.bind_backend(run)
        with self._mutex:
            record = self.state["resources"].get(kind + ":" + name)
            if record is None:
                # A cleanup path can mention a later step that was never
                # attempted. Verify absence, but never delete an unregistered name.
                field = "{{.Names}}" if kind == "container" else "{{.Name}}"
                command = [kind, "ls"] + (["--all"] if kind == "container" else [])
                inventory = run([*command, "--format", field], timeout=20)
                if name in inventory.stdout.splitlines():
                    raise WorkerFailure(
                        "refusing to remove an unregistered runtime resource"
                    )
                return
            actual = self._inspect(record, run)
            if actual is None:
                if record["state"] == "intent":
                    raise WorkerFailure(
                        "creation outcome is unknown; absence is not a fence"
                    )
            else:
                self._bind_identity(record, actual)
                durable = self._durable_state["resources"].get(kind + ":" + name)
                if durable is None or any(
                    durable[field] != record[field]
                    for field in ("kind", "name", "token", "identity")
                ):
                    # A newly discovered identity must survive interruption.
                    # Already durable identities need no write before stop:
                    # a full/read-only journal must not keep workloads alive.
                    self._save()
                target = name if kind == "volume" else actual["identity"]
                command = (
                    ["rm", "--force", target]
                    if kind == "container"
                    else [kind, "rm", target]
                )
                run(command, check=False, timeout=20)
                if self._inspect(record, run) is not None:
                    raise WorkerFailure(f"{kind} removal remains unconfirmed")
            record["state"] = "removed"
            self._save()

    def recover(self, run: Callable) -> Dict[str, Any]:
        self.bind_backend(run)
        failures = []
        with self._mutex:
            self._can_create = False
            self.state["phase"] = "recovering"
            try:
                self._save()
            except (WorkerFailure, OSError) as exc:
                failures.append(
                    {"kind": "journal", "name": "resources.json", "reason": str(exc)}
                )
                # Still stop durably identified workloads. Loaded journals
                # cannot create resources even if phase persistence failed.
        # Kill every workload/control container before touching networks or
        # storage. The holder is a container too; consumers are stopped first.
        records = list(self.state["resources"].values())
        containers = [
            record
            for record in records
            if record["kind"] == "container"
            and "prediction-holder" not in record["name"]
        ]
        holders = [
            record
            for record in records
            if record["kind"] == "container" and "prediction-holder" in record["name"]
        ]
        for record in containers:
            try:
                self.remove(record["kind"], record["name"], run)
            except (WorkerFailure, OSError) as exc:
                failures.append(
                    {"kind": record["kind"], "name": record["name"], "reason": str(exc)}
                )
        if not failures:
            for record in holders:
                try:
                    self.remove(record["kind"], record["name"], run)
                except (WorkerFailure, OSError) as exc:
                    failures.append(
                        {
                            "kind": record["kind"],
                            "name": record["name"],
                            "reason": str(exc),
                        }
                    )
        if not failures:
            for record in (item for item in records if item["kind"] != "container"):
                try:
                    self.remove(record["kind"], record["name"], run)
                except (WorkerFailure, OSError) as exc:
                    failures.append(
                        {
                            "kind": record["kind"],
                            "name": record["name"],
                            "reason": str(exc),
                        }
                    )
        if not failures:
            private = self.root / "private"
            try:
                if private.is_symlink():
                    raise WorkerFailure("Worker private storage became a symlink")
                if private.exists():
                    shutil.rmtree(private)
                self.state["phase"] = "cleaned"
                self._save()
            except (WorkerFailure, OSError) as exc:
                failures.append(
                    {"kind": "private_storage", "name": "private", "reason": str(exc)}
                )
        return {
            "cleanup_complete": not failures,
            "episode": self.state["episode"],
            "resumed": False,
            "failures": failures,
        }
