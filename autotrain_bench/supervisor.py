"""Trusted host-side episode supervisor for AutoTrain-Bench v0."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .config import TaskConfig
from .inference import InferenceFailure, forward_inference_request
from .runtime import ExecResult, RuntimeFailure


class ProtocolError(RuntimeError):
    """A request violates the benchmark protocol."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class Supervisor:
    """Own timers, lease counters, container lifecycle, and evaluator access."""

    def __init__(
        self,
        *,
        config: TaskConfig,
        track: str,
        episode_dir: Path,
        runtime: Any,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
    ) -> None:
        normalized = track.upper()
        if normalized not in {"A", "B"}:
            raise ValueError("track must be A or B")
        self.config = config
        self.track = normalized
        self.episode_dir = episode_dir.resolve()
        self.runtime = runtime
        self.monotonic = monotonic
        self.wall_time = wall_time
        self.lock = threading.RLock()
        self.started_monotonic: Optional[float] = None

        self.control_dir = self.episode_dir / "control"
        self.inbox = self.control_dir / "inbox"
        self.outbox = self.control_dir / "outbox"
        self.runs_dir = self.episode_dir / "leases"
        self.candidates_dir = self.episode_dir / "candidates"
        self.evaluations_dir = self.episode_dir / "evaluations"
        self.public_dir = self.episode_dir / "public"
        for directory in (
            self.inbox,
            self.outbox,
            self.runs_dir,
            self.candidates_dir,
            self.evaluations_dir,
            self.public_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        os.chmod(self.control_dir, 0o777)
        os.chmod(self.inbox, 0o777)
        os.chmod(self.outbox, 0o777)

        self.state_path = self.episode_dir / "state.json"
        self.events_path = self.episode_dir / "events.jsonl"
        self.public_task_path = self.public_dir / "task.json"
        if self.state_path.exists() or self.events_path.exists():
            raise ValueError(f"episode directory already contains state: {self.episode_dir}")

        self.state: Dict[str, Any] = {
            "protocol_version": 1,
            "episode_id": self.episode_dir.name,
            "task_id": config.task_id,
            "track": self.track,
            "agent_system": {
                "image": config.agent_image,
                "command": list(config.agent_command),
                "mode": config.agent_mode,
                "network": "disabled" if config.agent_mode == "sandboxed" else "unrestricted_dev",
                "policy_compliant": config.policy_compliant,
                "inference": {
                    "enabled": config.inference is not None,
                    "protocol": config.inference.protocol if config.inference else None,
                    "allowed_models": list(config.inference.allowed_models) if config.inference else [],
                    "allowed_paths": list(config.inference.allowed_paths) if config.inference else [],
                },
                "forwarded_environment": list(config.agent_environment),
                "initial_workspace_sha256": self._tree_hash(
                    config.agent_workspace
                ),
            },
            "status": "created",
            "started_at": None,
            "elapsed_seconds": 0.0,
            "lease_serial": 0,
            "leases_used": 0,
            "active_lease": None,
            "last_lease_id": None,
            "candidates": {},
            "incumbent_id": None,
            "final_candidate_id": None,
            "test_nll": None,
            "failure": None,
        }
        self.sequence = 0
        self.public_task_path.write_text(
            json.dumps(config.public_manifest(self.track), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        self._save_state()

    def begin(self) -> None:
        with self.lock:
            if self.state["status"] != "created":
                raise ProtocolError("invalid_state", "episode has already started")
            self.started_monotonic = self.monotonic()
            started_at = datetime.fromtimestamp(
                self.wall_time(), tz=timezone.utc
            ).isoformat()
            self.state.update(status="active", started_at=started_at)
            self._event(
                "TASK_REVEALED",
                track=self.track,
                agent_mode=self.config.agent_mode,
                inference_enabled=self.config.inference is not None,
            )
            self._save_state()

    def _elapsed(self) -> float:
        if self.started_monotonic is None:
            return 0.0
        return max(0.0, self.monotonic() - self.started_monotonic)

    def _track_a_remaining(self) -> float:
        return max(0.0, self.config.track_a_total_seconds - self._elapsed())

    def _event(self, event_type: str, **data: Any) -> None:
        self.sequence += 1
        event = {
            "seq": self.sequence,
            "type": event_type,
            "elapsed_seconds": round(self._elapsed(), 6),
            **data,
        }
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")

    def _save_state(self) -> None:
        self.state["elapsed_seconds"] = round(self._elapsed(), 6)
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(str(temporary), str(self.state_path))
        os.chmod(self.state_path, 0o600)

    def _require_active(self) -> None:
        if self.state["status"] != "active":
            raise ProtocolError(
                "episode_closed", "the episode is not active and cannot be resumed"
            )
        if self.track == "A" and self._track_a_remaining() <= 0:
            self.enforce_deadlines()
            raise ProtocolError("deadline_reached", "Track A deadline has passed")

    def _active_lease(self) -> Dict[str, Any]:
        active = self.state.get("active_lease")
        if not isinstance(active, dict):
            raise ProtocolError("no_active_lease", "there is no active training lease")
        return active

    def public_status(self) -> Dict[str, Any]:
        with self.lock:
            active = self.state.get("active_lease")
            status: Dict[str, Any] = {
                "episode_id": self.state["episode_id"],
                "task_id": self.config.task_id,
                "track": self.track,
                "agent_mode": self.config.agent_mode,
                "agent_network": "disabled" if self.config.agent_mode == "sandboxed" else "unrestricted_dev",
                "policy_compliant": self.config.policy_compliant,
                "inference_enabled": self.config.inference is not None,
                "status": self.state["status"],
                "elapsed_seconds": round(self._elapsed(), 3),
                "leases_used": self.state["leases_used"],
                "active_lease": None,
                "incumbent_id": self.state["incumbent_id"],
                "candidates": self.state["candidates"],
            }
            if active:
                status["active_lease"] = {
                    "lease_id": active["lease_id"],
                    "remaining_seconds": round(
                        max(0.0, active["deadline_monotonic"] - self.monotonic()),
                        3,
                    ),
                }
            if self.track == "A":
                status["remaining_wall_clock_seconds"] = round(
                    self._track_a_remaining(), 3
                )
            else:
                status["leases_remaining"] = max(
                    0,
                    self.config.track_b_max_leases - self.state["leases_used"],
                )
            if self.state["status"] != "active":
                status["final_candidate_id"] = self.state["final_candidate_id"]
                status["test_nll"] = self.state["test_nll"]
                status["failure"] = self.state["failure"]
            return status

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        operation = request.get("op")
        if operation == "status":
            return self.public_status()
        if operation == "lease_start":
            return self.start_lease(request.get("seconds"))
        if operation == "lease_exec":
            return self.exec_lease(
                request.get("argv"), request.get("command_timeout_seconds")
            )
        if operation == "lease_logs":
            return self.lease_logs(request.get("limit", 50))
        if operation == "lease_commit":
            return self.commit_lease(request.get("candidate_path"))
        if operation == "lease_stop":
            return self.stop_lease()
        if operation == "inference":
            return self.inference(
                method=request.get("method"),
                path=request.get("path"),
                headers=request.get("headers"),
                body_base64=request.get("body_base64"),
            )
        if operation == "finish":
            return self.finish(request.get("candidate_id"))
        raise ProtocolError("unknown_operation", f"unknown operation: {operation!r}")

    def start_lease(self, requested_seconds: Any = None) -> Dict[str, Any]:
        with self.lock:
            self._require_active()
            if self.state["active_lease"] is not None:
                raise ProtocolError("lease_active", "a training lease is already active")

            if requested_seconds is not None and (
                not isinstance(requested_seconds, int)
                or isinstance(requested_seconds, bool)
                or requested_seconds <= 0
            ):
                raise ProtocolError("invalid_duration", "lease duration must be positive")

            if self.track == "A":
                maximum = self._track_a_remaining()
            else:
                if self.state["leases_used"] >= self.config.track_b_max_leases:
                    raise ProtocolError("lease_budget_exhausted", "no leases remain")
                maximum = float(self.config.track_b_lease_seconds)

            duration = maximum if requested_seconds is None else float(requested_seconds)
            if duration > maximum + 1e-9:
                raise ProtocolError(
                    "lease_too_long",
                    f"requested {duration:g}s but at most {maximum:g}s is available",
                )

            serial = self.state["lease_serial"] + 1
            lease_id = f"lease-{serial:04d}"
            container_name = (
                f"atb-{self.state['episode_id'][:12]}-{serial:04d}"
            ).lower()
            lease_dir = self.runs_dir / lease_id
            lease_dir.mkdir(parents=True, exist_ok=False)
            try:
                self.runtime.start_training(
                    name=container_name,
                    config=self.config,
                    lease_dir=lease_dir,
                )
            except RuntimeFailure as exc:
                shutil.rmtree(lease_dir, ignore_errors=True)
                self._event("LEASE_INFRA_FAILURE", message=str(exc))
                self._save_state()
                raise ProtocolError("infrastructure_failure", str(exc)) from exc

            started = self.monotonic()
            self.state["lease_serial"] = serial
            self.state["leases_used"] += 1
            self.state["active_lease"] = {
                "lease_id": lease_id,
                "container_name": container_name,
                "started_monotonic": started,
                "deadline_monotonic": started + duration,
                "duration_seconds": duration,
                "exec_count": 0,
            }
            self.state["last_lease_id"] = lease_id
            self._event(
                "LEASE_STARTED", lease_id=lease_id, duration_seconds=duration
            )
            self._save_state()
            return {
                "lease_id": lease_id,
                "duration_seconds": duration,
                "leases_used": self.state["leases_used"],
            }

    def exec_lease(
        self, argv: Any, command_timeout_seconds: Any = None
    ) -> Dict[str, Any]:
        with self.lock:
            self._require_active()
            active = self._active_lease()
            if not isinstance(argv, list) or not argv or not all(
                isinstance(value, str) and value for value in argv
            ):
                raise ProtocolError("invalid_command", "argv must be a string array")
            remaining = active["deadline_monotonic"] - self.monotonic()
            if remaining <= 0:
                self._end_active_lease("timeout")
                raise ProtocolError("lease_expired", "the training lease has expired")
            if command_timeout_seconds is not None:
                if not isinstance(command_timeout_seconds, (int, float)) or isinstance(
                    command_timeout_seconds, bool
                ):
                    raise ProtocolError(
                        "invalid_timeout", "command timeout must be numeric"
                    )
                if command_timeout_seconds <= 0:
                    raise ProtocolError(
                        "invalid_timeout", "command timeout must be positive"
                    )
                remaining = min(remaining, float(command_timeout_seconds))

            result: ExecResult = self.runtime.exec_training(
                name=active["container_name"],
                argv=argv,
                timeout_seconds=remaining,
            )
            active["exec_count"] += 1
            record = {
                "exec": active["exec_count"],
                "argv": argv,
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": result.timed_out,
            }
            log_path = self.runs_dir / active["lease_id"] / "commands.jsonl"
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            self._event(
                "LEASE_COMMAND_FINISHED",
                lease_id=active["lease_id"],
                exec=active["exec_count"],
                exit_code=result.exit_code,
                timed_out=result.timed_out,
            )
            lease_active = True
            if result.timed_out or not self.runtime.container_running(
                active["container_name"]
            ):
                self._end_active_lease(
                    "timeout" if result.timed_out else "container_exited"
                )
                lease_active = False
            self._save_state()
            return {**record, "lease_active": lease_active}

    def lease_logs(self, limit: Any = 50) -> Dict[str, Any]:
        with self.lock:
            if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
                raise ProtocolError("invalid_limit", "log limit must be positive")
            limit = min(limit, 500)
            active = self.state.get("active_lease")
            lease_id = active["lease_id"] if active else self.state["last_lease_id"]
            if not lease_id:
                return {"lease_id": None, "commands": []}
            log_path = self.runs_dir / lease_id / "commands.jsonl"
            if not log_path.exists():
                return {"lease_id": lease_id, "commands": []}
            lines = log_path.read_text(encoding="utf-8").splitlines()[-limit:]
            return {
                "lease_id": lease_id,
                "commands": [json.loads(line) for line in lines],
            }

    def _end_active_lease(self, reason: str) -> Optional[str]:
        active = self.state.get("active_lease")
        if not active:
            return None
        self.runtime.stop_container(active["container_name"])
        lease_id = active["lease_id"]
        self.state["active_lease"] = None
        self.state["last_lease_id"] = lease_id
        self._event("LEASE_ENDED", lease_id=lease_id, reason=reason)
        self._save_state()
        return lease_id

    def stop_lease(self) -> Dict[str, Any]:
        with self.lock:
            self._require_active()
            lease_id = self._active_lease()["lease_id"]
            self._end_active_lease("agent_stopped_without_candidate")
            return {"lease_id": lease_id, "candidate_id": None, "val_nll": None}

    def _candidate_source(self, candidate_path: Any) -> Path:
        if not isinstance(candidate_path, str) or not candidate_path:
            raise ProtocolError("invalid_candidate", "candidate path is required")
        if candidate_path == "/workspace":
            source = self.config.agent_workspace
        elif candidate_path.startswith("/workspace/"):
            source = self.config.agent_workspace / candidate_path[len("/workspace/") :]
        else:
            raw = Path(candidate_path)
            if raw.is_absolute():
                raise ProtocolError(
                    "invalid_candidate", "candidate must be inside /workspace"
                )
            source = self.config.agent_workspace / raw
        if source.is_symlink():
            raise ProtocolError("invalid_candidate", "candidate may not be a symlink")
        resolved = source.resolve()
        try:
            resolved.relative_to(self.config.agent_workspace.resolve())
        except ValueError as exc:
            raise ProtocolError(
                "invalid_candidate", "candidate escapes the agent workspace"
            ) from exc
        if not resolved.is_dir():
            raise ProtocolError(
                "invalid_candidate", f"candidate directory does not exist: {candidate_path}"
            )
        for item in resolved.rglob("*"):
            if item.is_symlink():
                raise ProtocolError(
                    "invalid_candidate", "candidate bundles may not contain symlinks"
                )
        if not (resolved / "candidate.py").is_file():
            raise ProtocolError(
                "invalid_candidate", "v0 candidate must contain candidate.py"
            )
        return resolved

    @staticmethod
    def _tree_hash(root: Path) -> str:
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix().encode("utf-8")
            if path.is_symlink():
                target = os.readlink(path).encode("utf-8")
                digest.update(b"symlink\0")
                digest.update(len(relative).to_bytes(8, "big"))
                digest.update(relative)
                digest.update(len(target).to_bytes(8, "big"))
                digest.update(target)
                continue
            if not path.is_file():
                continue
            digest.update(b"file\0")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    def _snapshot_candidate(self, source: Path) -> Dict[str, Any]:
        stage = self.candidates_dir / f".stage-{uuid.uuid4().hex}"
        shutil.copytree(source, stage)
        digest = self._tree_hash(stage)
        candidate_id = f"sha256:{digest}"
        destination = self.candidates_dir / digest
        if destination.exists():
            shutil.rmtree(stage)
        else:
            os.replace(str(stage), str(destination))
        return {"candidate_id": candidate_id, "path": destination}

    def commit_lease(self, candidate_path: Any) -> Dict[str, Any]:
        with self.lock:
            self._require_active()
            active = self._active_lease()
            source = self._candidate_source(candidate_path)
            snapshot = self._snapshot_candidate(source)
            lease_id = active["lease_id"]
            self._end_active_lease("candidate_committed")
            candidate_id = snapshot["candidate_id"]
            metadata = {
                "candidate_id": candidate_id,
                "source_lease": lease_id,
                "val_nll": None,
                "evaluation_error": None,
            }
            self.state["candidates"][candidate_id] = metadata
            self.state["incumbent_id"] = candidate_id
            self._event(
                "CANDIDATE_COMMITTED",
                lease_id=lease_id,
                candidate_id=candidate_id,
            )
            try:
                val_nll = self.runtime.run_evaluator(
                    config=self.config,
                    candidate_dir=snapshot["path"],
                    split="val",
                    output_dir=self.evaluations_dir / snapshot["path"].name,
                )
                metadata["val_nll"] = val_nll
                self._event(
                    "VAL_EVALUATED",
                    candidate_id=candidate_id,
                    nll=val_nll,
                )
            except RuntimeFailure as exc:
                metadata["evaluation_error"] = str(exc)
                self._event(
                    "VAL_EVALUATION_FAILED",
                    candidate_id=candidate_id,
                    message=str(exc),
                )
            self._save_state()
            return dict(metadata)

    def inference(
        self,
        *,
        method: Any,
        path: Any,
        headers: Any,
        body_base64: Any,
    ) -> Dict[str, Any]:
        with self.lock:
            self._require_active()
            if self.config.inference is None:
                raise ProtocolError(
                    "inference_disabled", "no external inference channel is registered"
                )
            if not isinstance(method, str) or not isinstance(path, str):
                raise ProtocolError(
                    "invalid_inference_request", "method and path must be strings"
                )
            if not isinstance(headers, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in headers.items()
            ):
                raise ProtocolError(
                    "invalid_inference_request", "headers must be a string mapping"
                )
            if not isinstance(body_base64, str):
                raise ProtocolError(
                    "invalid_inference_request", "body_base64 must be a string"
                )
            remaining = None
            if self.track == "A":
                remaining = self._track_a_remaining()
                if remaining <= 0:
                    self.enforce_deadlines()
                    raise ProtocolError("deadline_reached", "Track A deadline has passed")
            try:
                result = forward_inference_request(
                    inference_config=self.config.inference,
                    method=method,
                    path=path,
                    headers=headers,
                    body_base64=body_base64,
                    timeout_seconds=remaining,
                )
            except InferenceFailure as exc:
                self._event("INFERENCE_REJECTED", code=exc.code, message=str(exc))
                self._save_state()
                raise ProtocolError(exc.code, str(exc)) from exc

            # A Track A inference call consumes the same wall clock as all other
            # research.  If the deadline elapsed while the upstream was running,
            # close the episode instead of returning a post-deadline result.
            if self.track == "A" and self._track_a_remaining() <= 0:
                self.enforce_deadlines()
                raise ProtocolError("deadline_reached", "Track A deadline has passed")
            self._event(
                "INFERENCE_COMPLETED",
                model=result.get("model"),
                path=path,
                status=result.get("status"),
            )
            self._save_state()
            return result

    def finish(self, candidate_id: Any = None) -> Dict[str, Any]:
        with self.lock:
            self._require_active()
            if self.state["active_lease"] is not None:
                raise ProtocolError(
                    "lease_active", "commit or stop the active lease before finish"
                )
            selected = candidate_id or self.state["incumbent_id"]
            if selected is not None and selected not in self.state["candidates"]:
                raise ProtocolError("unknown_candidate", "candidate is not registered")
            return self._close_and_test(selected, reason="agent_finished")

    def _close_and_test(
        self, candidate_id: Optional[str], *, reason: str
    ) -> Dict[str, Any]:
        self.state["status"] = "closed"
        self.state["final_candidate_id"] = candidate_id
        self._event(
            "EPISODE_CLOSED", reason=reason, candidate_id=candidate_id
        )
        self._save_state()

        if candidate_id is None:
            self.state["status"] = "finalized"
            self._event("EPISODE_FINALIZED", result="no_candidate")
            self._save_state()
            return self.public_status()

        digest = candidate_id.split(":", 1)[1]
        candidate_dir = self.candidates_dir / digest
        try:
            test_nll = self.runtime.run_evaluator(
                config=self.config,
                candidate_dir=candidate_dir,
                split="test",
                output_dir=self.evaluations_dir / digest,
            )
        except RuntimeFailure as exc:
            self.state["failure"] = f"test evaluation failed: {exc}"
            self._event(
                "TEST_EVALUATION_FAILED",
                candidate_id=candidate_id,
                message=str(exc),
            )
            self._save_state()
            return self.public_status()

        self.state["test_nll"] = test_nll
        self.state["status"] = "finalized"
        self._event("TEST_EVALUATED", candidate_id=candidate_id, nll=test_nll)
        self._event("EPISODE_FINALIZED", result="success")
        self._save_state()
        return self.public_status()

    def enforce_deadlines(self) -> bool:
        with self.lock:
            if self.state["status"] != "active":
                return False
            active = self.state.get("active_lease")
            if active and self.monotonic() >= active["deadline_monotonic"]:
                self._end_active_lease("timeout")
            if self.track == "A" and self._track_a_remaining() <= 0:
                if self.state["active_lease"] is not None:
                    self._end_active_lease("track_a_deadline")
                self._close_and_test(
                    self.state["incumbent_id"], reason="track_a_deadline"
                )
                return True
            return False

    def agent_exited_without_finish(self) -> None:
        with self.lock:
            if self.state["status"] != "active":
                return
            if self.state["active_lease"] is not None:
                self._end_active_lease("agent_exited")
            self.state["status"] = "dnf"
            self.state["failure"] = "agent exited without finishing the episode"
            self._event("EPISODE_DNF", reason="agent_exited_without_finish")
            self._save_state()

    def process_one_request(self) -> bool:
        paths = sorted(self.inbox.glob("*.json"))
        if not paths:
            return False
        request_path = paths[0]
        request_id = request_path.stem
        try:
            if request_path.stat().st_size > 32 * 1024 * 1024:
                raise ProtocolError("request_too_large", "request exceeds 32 MiB")
            request = json.loads(request_path.read_text(encoding="utf-8"))
            if not isinstance(request, dict):
                raise ProtocolError("invalid_request", "request must be an object")
            result = self.handle(request)
            response = {"ok": True, "result": result}
        except ProtocolError as exc:
            response = {
                "ok": False,
                "error": {"code": exc.code, "message": str(exc)},
            }
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            response = {
                "ok": False,
                "error": {"code": "invalid_request", "message": str(exc)},
            }
        except Exception as exc:  # supervisor must return a bounded failure
            response = {
                "ok": False,
                "error": {"code": "internal_error", "message": str(exc)},
            }
            self._event("SUPERVISOR_ERROR", message=str(exc))
            self._save_state()
        finally:
            request_path.unlink(missing_ok=True)

        temporary = self.outbox / f".{request_id}.tmp"
        destination = self.outbox / f"{request_id}.json"
        temporary.write_text(json.dumps(response, sort_keys=True), encoding="utf-8")
        os.replace(str(temporary), str(destination))
        return True
