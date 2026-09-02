"""Trusted host-side episode supervisor for AutoTrain-Bench v0."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .config import (
    AgentConfig,
    TaskConfig,
    TrainingResources,
    immutable_image_reference,
)
from .inference import summarize_inference_trace
from .results import RESULT_SCHEMA_VERSION, perplexity_from_nll, validated_nll
from .worker_protocol import (
    EVALUATION_REQUIREMENTS,
    GPU_DEVICE_ISOLATION,
    RESEARCH_REQUIREMENTS,
    CapabilityRequirement,
    CommandResult,
    EvaluationWorker,
    EnforcementLevel,
    EvaluationTaskSpec,
    HiddenSplitRef,
    LeaseState,
    ResearchTaskSpec,
    ResearchWorker,
    WorkerFailure,
    assess_worker,
)


MAX_COMMAND_STREAM_BYTES = 1024 * 1024
MAX_MAILBOX_RESPONSE_BYTES = 3 * 1024 * 1024
MAX_LOG_READ_BYTES = 4 * 1024 * 1024
MAX_COMMAND_LOG_BYTES = 64 * 1024 * 1024


class ProtocolError(RuntimeError):
    """A request violates the benchmark protocol."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _read_untrusted_request(path: Path, maximum_bytes: int) -> Dict[str, Any]:
    """Read one mailbox request without following Agent-created filesystem links."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise ProtocolError("invalid_request", "request must be a regular file") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ProtocolError("invalid_request", "request must be a regular file")
        if metadata.st_size > maximum_bytes:
            raise ProtocolError("request_too_large", "request exceeds 1 MiB")
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            payload = json.load(stream)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(payload, dict):
        raise ProtocolError("invalid_request", "request must be an object")
    return payload


def _write_untrusted_response(directory: Path, request_id: str, response: Dict[str, Any]) -> None:
    """Publish a mailbox response without following Agent-created symlinks."""

    if not request_id or any(value not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for value in request_id):
        raise ProtocolError("invalid_request", "request id contains invalid characters")
    temporary = f".{request_id}.{uuid.uuid4().hex}.tmp"
    destination = f"{request_id}.json"
    encoded = json.dumps(response, sort_keys=True).encode("utf-8")
    if len(encoded) > MAX_MAILBOX_RESPONSE_BYTES:
        encoded = json.dumps(
            {
                "ok": False,
                "error": {
                    "code": "response_too_large",
                    "message": "supervisor response exceeds the mailbox limit",
                },
            },
            sort_keys=True,
        ).encode("utf-8")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
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
    except Exception:
        try:
            os.unlink(temporary, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_descriptor)


def _bounded_text(value: Any, maximum_bytes: int = MAX_COMMAND_STREAM_BYTES) -> tuple[str, bool]:
    if not isinstance(value, str):
        value = str(value)
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= maximum_bytes:
        return value, False
    marker = b"\n[output truncated by AutoTrain-Bench]\n"
    prefix = encoded[: max(0, maximum_bytes - len(marker))]
    while prefix:
        try:
            result = prefix.decode("utf-8")
            break
        except UnicodeDecodeError:
            prefix = prefix[:-1]
    else:
        result = ""
    return result + marker.decode("ascii"), True


def _tail_json_lines(path: Path, limit: int) -> tuple[list[Dict[str, Any]], bool]:
    size = path.stat().st_size
    offset = max(0, size - MAX_LOG_READ_BYTES)
    with path.open("rb") as stream:
        stream.seek(offset)
        data = stream.read(MAX_LOG_READ_BYTES)
    truncated = offset > 0
    if truncated:
        first_newline = data.find(b"\n")
        data = b"" if first_newline < 0 else data[first_newline + 1 :]
    lines = data.splitlines()[-limit:]
    return [json.loads(line) for line in lines], truncated


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _core_identity() -> Dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        revision = None
        dirty = None
    return {"revision": revision, "dirty": dirty}


class Supervisor:
    """Own benchmark state while delegating enforcement to a semantic Worker."""

    def __init__(
        self,
        *,
        config: TaskConfig,
        agent: AgentConfig,
        resources: TrainingResources,
        track: str,
        episode_dir: Path,
        research_worker: ResearchWorker,
        evaluation_worker: EvaluationWorker,
        research_worker_environment: Optional[Dict[str, Any]] = None,
        evaluation_worker_environment: Optional[Dict[str, Any]] = None,
        comparison_key: Optional[Dict[str, str]] = None,
        challenge_release: Optional[Dict[str, Any]] = None,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
    ) -> None:
        normalized = track.upper()
        if normalized not in {"A", "B"}:
            raise ValueError("track must be A or B")
        self.config = config.validate()
        self.agent = agent.validate()
        self.resources = resources.validate_for(config)
        self.track = normalized
        self.episode_dir = episode_dir.resolve()
        self.research_worker = research_worker
        self.evaluation_worker = evaluation_worker
        self.evaluation_task = EvaluationTaskSpec.from_task(config)
        self.monotonic = monotonic
        self.wall_time = wall_time
        self.lock = threading.RLock()
        self.started_monotonic: Optional[float] = None
        self.closed_monotonic: Optional[float] = None

        self.control_dir = self.episode_dir / "control"
        self.inbox = self.control_dir / "inbox"
        self.outbox = self.control_dir / "outbox"
        self.runs_dir = self.episode_dir / "leases"
        self.candidates_dir = self.episode_dir / "candidates"
        self.evaluations_dir = self.episode_dir / "evaluations"
        self.inference_dir = self.episode_dir / "inference"
        self.inference_trace_path = self.inference_dir / "inference.jsonl"
        self.public_dir = self.episode_dir / "public"
        self.state_path = self.episode_dir / "state.json"
        self.events_path = self.episode_dir / "events.jsonl"
        self.public_task_path = self.public_dir / "task.json"
        self.agent_api_dir = self.public_dir / "agent_api"
        if self.state_path.exists() or self.events_path.exists():
            raise ValueError(f"episode directory already contains state: {self.episode_dir}")
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.episode_dir, 0o700)
        for directory in (
            self.inbox,
            self.outbox,
            self.runs_dir,
            self.candidates_dir,
            self.evaluations_dir,
            self.inference_dir,
            self.public_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        os.chmod(self.control_dir, 0o700)
        os.chmod(self.inbox, 0o777)
        os.chmod(self.outbox, 0o777)
        os.chmod(self.inference_dir, 0o700)
        for directory in (
            self.runs_dir,
            self.candidates_dir,
            self.evaluations_dir,
        ):
            os.chmod(directory, 0o700)

        self._publish_agent_api()
        self.research_task = ResearchTaskSpec.from_task(
            config, self.agent_api_dir
        )

        research_descriptor = research_worker.describe_capabilities()
        evaluation_descriptor = evaluation_worker.describe_capabilities()
        research_requirements = list(RESEARCH_REQUIREMENTS)
        evaluation_requirements = list(EVALUATION_REQUIREMENTS)
        if resources.nvidia_gpu_count:
            research_requirements.append(
                CapabilityRequirement(
                    GPU_DEVICE_ISOLATION, EnforcementLevel.ENFORCED
                )
            )
        if config.evaluator_device == "cuda":
            evaluation_requirements.append(
                CapabilityRequirement(
                    GPU_DEVICE_ISOLATION, EnforcementLevel.ENFORCED
                )
            )
        research_assessment = assess_worker(
            research_descriptor, research_requirements
        )
        evaluation_assessment = assess_worker(
            evaluation_descriptor, evaluation_requirements
        )

        policy_violations = []
        if not agent.policy_compliant:
            policy_violations.append("agent_execution_boundary")
        if not config.evaluator_policy_compliant:
            policy_violations.append("candidate_evaluator_isolation")
        if "research" not in research_descriptor.roles:
            policy_violations.append("research_worker_role_missing")
        if "evaluation" not in evaluation_descriptor.roles:
            policy_violations.append("evaluation_worker_role_missing")
        for role, assessment in (
            ("research", research_assessment),
            ("evaluation", evaluation_assessment),
        ):
            for capability in assessment.missing:
                policy_violations.append(
                    f"{role}_worker_capability_missing:{capability}"
                )
            for capability in assessment.insufficient:
                policy_violations.append(
                    f"{role}_worker_capability_insufficient:{capability}"
                )
        policy_compliant = not policy_violations
        eligibility_violations = []
        if not config.leaderboard_eligible:
            eligibility_violations.append("task_not_leaderboard_eligible")
        if not policy_compliant:
            eligibility_violations.append("policy_not_compliant")
        if not research_assessment.leaderboard_eligible:
            eligibility_violations.append("research_worker_not_certified")
        if not evaluation_assessment.leaderboard_eligible:
            eligibility_violations.append("evaluation_worker_not_certified")
        required_comparison_fields = {
            "challenge_id",
            "track",
            "track_config",
            "runtime_profile",
        }
        if (
            not isinstance(comparison_key, dict)
            or not required_comparison_fields.issubset(comparison_key)
            or any(
                not isinstance(comparison_key.get(field), str)
                or not comparison_key[field].strip()
                for field in required_comparison_fields
            )
        ):
            eligibility_violations.append("comparison_key_missing")
        elif comparison_key.get("track") != self.track:
            eligibility_violations.append("comparison_key_track_mismatch")
        release_ready = isinstance(challenge_release, dict)
        if not release_ready:
            eligibility_violations.append("challenge_release_missing")
        else:
            if challenge_release.get("release_stage") != "official":
                eligibility_violations.append("challenge_release_not_official")
            if challenge_release.get("leaderboard_eligible") is not True:
                eligibility_violations.append("challenge_release_not_eligible")
            if (
                isinstance(comparison_key, dict)
                and challenge_release.get("challenge_id")
                != comparison_key.get("challenge_id")
            ):
                eligibility_violations.append("challenge_release_identity_mismatch")
            for digest_field in (
                "challenge_manifest_sha256",
                "task_manifest_sha256",
            ):
                digest = challenge_release.get(digest_field)
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(value not in "0123456789abcdef" for value in digest)
                ):
                    eligibility_violations.append(f"{digest_field}_missing")
            for field in ("track_config", "runtime_profile"):
                selected = challenge_release.get(field)
                if not isinstance(selected, dict) or selected.get("provisional") is not False:
                    eligibility_violations.append(f"{field}_not_frozen")
            if isinstance(comparison_key, dict):
                for release_field, comparison_field in (
                    ("track_config_id", "track_config"),
                    ("runtime_profile_id", "runtime_profile"),
                ):
                    if challenge_release.get(release_field) != comparison_key.get(
                        comparison_field
                    ):
                        eligibility_violations.append(
                            f"{release_field}_mismatch"
                        )
        if agent.runner != "container" or not immutable_image_reference(agent.image):
            eligibility_violations.append("agent_image_not_digest_pinned")

        leaderboard_eligible = (
            config.leaderboard_eligible
            and policy_compliant
            and research_assessment.leaderboard_eligible
            and evaluation_assessment.leaderboard_eligible
            and not eligibility_violations
        )

        self.state: Dict[str, Any] = {
            "protocol_version": 1,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "episode_id": self.episode_dir.name,
            "task_id": config.task_id,
            "task_kind": config.task_kind,
            "leaderboard_eligible": leaderboard_eligible,
            "leaderboard_eligibility_violations": eligibility_violations,
            "track": self.track,
            "comparison_key": dict(comparison_key) if comparison_key else None,
            "challenge_release": (
                dict(challenge_release) if challenge_release else None
            ),
            "artifact_identities": {
                "core_git": _core_identity(),
                "task_manifest_sha256": _sha256_file(config.source),
                "policy_sha256": _sha256_file(config.source.parent / "POLICY.md"),
                "evaluator_sha256": _sha256_file(config.evaluator_entrypoint),
                "train_manifest_sha256": _sha256_file(
                    config.train_dir / "manifest.json"
                ),
                "hidden_splits": {
                    "val_sha256": _sha256_file(config.val_path),
                    "test_sha256": _sha256_file(config.test_path),
                },
            },
            "policy_compliant": policy_compliant,
            "policy_violations": policy_violations,
            "workers": {
                "research": {
                    **research_descriptor.public_manifest(),
                    "assessment": research_assessment.public_manifest(),
                    "environment": dict(research_worker_environment or {}),
                },
                "evaluation": {
                    **evaluation_descriptor.public_manifest(),
                    "assessment": evaluation_assessment.public_manifest(),
                    "environment": dict(evaluation_worker_environment or {}),
                },
            },
            "agent_system": {
                "runner": agent.runner,
                "image": agent.image if agent.runner == "container" else None,
                "command": list(agent.command),
                "mode": agent.mode,
                "network": agent.network,
                "policy_compliant": agent.policy_compliant,
                "exit_code": None,
                "termination_reason": None,
                "finish_observed": False,
                "drain_seconds": None,
                "inference": {
                    "enabled": agent.inference is not None,
                    "protocol": agent.inference.protocol if agent.inference else None,
                    "gateway_image": (
                        config.inference_gateway_image if agent.inference else None
                    ),
                    "allowed_models": list(agent.inference.allowed_models) if agent.inference else [],
                    "allowed_paths": list(agent.inference.allowed_paths) if agent.inference else [],
                },
                "forwarded_environment": list(agent.environment),
                "initial_workspace_sha256": self._tree_hash(
                    agent.workspace
                ),
            },
            "evaluation_system": {
                "backend": config.evaluator_backend,
                "image": config.evaluator_image,
                "candidate_image": config.candidate_image,
                "boundary": "candidate_evaluator_channel",
                "policy_compliant": (
                    config.evaluator_policy_compliant
                    and evaluation_assessment.satisfied
                ),
            },
            "status": "created",
            "outcome": None,
            "started_at": None,
            "elapsed_seconds": 0.0,
            "timing": {
                "training_lease_seconds": 0.0,
                "validation_evaluation_seconds": 0.0,
                "test_evaluation_seconds": 0.0,
            },
            "lease_serial": 0,
            "leases_used": 0,
            "active_lease": None,
            "last_lease_id": None,
            "candidates": {},
            "incumbent_id": None,
            "final_candidate_id": None,
            "test_nll": None,
            "test_perplexity": None,
            "closure_reason": None,
            "inference_usage": summarize_inference_trace(self.inference_trace_path),
            "agent_usage": None,
            "failure": None,
        }
        self.sequence = 0
        public_manifest = config.public_manifest(self.track)
        public_manifest["agent_environment"] = agent.public_manifest(
            config.inference_gateway_image
        )
        public_manifest["training_environment"]["resources"] = (
            resources.public_manifest()
        )
        public_manifest["policy"] = {
            "compliant": policy_compliant,
            "violations": policy_violations,
        }
        public_manifest["leaderboard_eligible"] = leaderboard_eligible
        public_manifest["comparison_key"] = self.state["comparison_key"]
        if challenge_release is not None:
            public_manifest["challenge_release"] = dict(challenge_release)
        self.public_task_path.write_text(
            json.dumps(public_manifest, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        self._save_state()

    def _publish_agent_api(self) -> None:
        """Publish only the untrusted Agent client, never the Core package."""

        package = self.agent_api_dir / "autotrain_bench"
        package.mkdir(parents=True, exist_ok=False)
        core_package = Path(__file__).resolve().parent
        published = {
            "client.py": "client.py",
            "agent_cli.py": "cli.py",
            "candidate_protocol.py": "candidate_protocol.py",
            "candidate_preflight.py": "candidate_preflight.py",
        }
        for source_name, destination_name in published.items():
            shutil.copy2(
                core_package / source_name,
                package / destination_name,
            )
        (package / "__init__.py").write_text(
            "from .client import Client, ClientError\n"
            "__all__ = ['Client', 'ClientError']\n",
            encoding="utf-8",
        )
        for path in package.iterdir():
            path.chmod(0o444)
        package.chmod(0o555)
        self.agent_api_dir.chmod(0o555)

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
                agent_runner=self.agent.runner,
                agent_mode=self.agent.mode,
                inference_enabled=self.agent.inference is not None,
            )
            self._save_state()

    def _elapsed(self) -> float:
        if self.started_monotonic is None:
            return 0.0
        end = (
            self.closed_monotonic
            if self.closed_monotonic is not None
            else self.monotonic()
        )
        return max(0.0, end - self.started_monotonic)

    def _freeze_elapsed(self, at_monotonic: Optional[float] = None) -> None:
        if self.closed_monotonic is None:
            self.closed_monotonic = (
                self.monotonic() if at_monotonic is None else at_monotonic
            )

    def _track_a_deadline_monotonic(self) -> float:
        if self.started_monotonic is None:
            raise ProtocolError("invalid_state", "episode has not started")
        return self.started_monotonic + self.config.track_a_total_seconds

    def _track_a_remaining(self) -> float:
        return max(0.0, self.config.track_a_total_seconds - self._elapsed())

    def _close_at_track_a_deadline(self) -> Dict[str, Any]:
        """Close Track A at its fixed deadline and TEST the prior incumbent."""

        self._freeze_elapsed(self._track_a_deadline_monotonic())
        if self.state["active_lease"] is not None:
            self._end_active_lease("track_a_deadline")
        return self._close_and_test(
            self.state["incumbent_id"], reason="track_a_deadline"
        )

    def _timing_summary(self) -> Dict[str, Any]:
        elapsed = self._elapsed()
        lease_seconds = float(self.state["timing"]["training_lease_seconds"])
        active = self.state.get("active_lease")
        if active:
            lease_seconds += min(
                active["duration_seconds"],
                max(0.0, self.monotonic() - active["started_monotonic"]),
            )
        validation_seconds = float(
            self.state["timing"]["validation_evaluation_seconds"]
        )
        result: Dict[str, Any] = {
            "official_elapsed_seconds": round(elapsed, 6),
            "training_lease_seconds": round(lease_seconds, 6),
            "validation_evaluation_seconds": round(validation_seconds, 6),
            "agent_and_worker_seconds": round(
                max(0.0, elapsed - lease_seconds - validation_seconds), 6
            ),
            "test_evaluation_seconds": round(
                float(self.state["timing"]["test_evaluation_seconds"]), 6
            ),
        }
        if self.track == "A":
            result["unused_budget_seconds"] = round(
                max(0.0, self.config.track_a_total_seconds - elapsed), 6
            )
        return result

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
        self.events_path.chmod(0o600)

    def _save_state(self) -> None:
        self.state["elapsed_seconds"] = round(self._elapsed(), 6)
        self.state["timing_summary"] = self._timing_summary()
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
                "result_schema_version": self.state["result_schema_version"],
                "episode_id": self.state["episode_id"],
                "task_id": self.config.task_id,
                "task_kind": self.config.task_kind,
                "leaderboard_eligible": self.state["leaderboard_eligible"],
                "leaderboard_eligibility_violations": list(
                    self.state["leaderboard_eligibility_violations"]
                ),
                "track": self.track,
                "comparison_key": self.state["comparison_key"],
                "agent_runner": self.agent.runner,
                "agent_mode": self.agent.mode,
                "agent_network": self.agent.network,
                "policy_compliant": self.state["policy_compliant"],
                "policy_violations": list(self.state["policy_violations"]),
                "evaluator_backend": self.config.evaluator_backend,
                "evaluator_policy_compliant": (
                    self.state["evaluation_system"]["policy_compliant"]
                ),
                "inference_enabled": self.agent.inference is not None,
                "inference_usage": dict(self.state["inference_usage"]),
                "agent_usage": self.state["agent_usage"],
                "agent_exit_code": self.state["agent_system"]["exit_code"],
                "agent_termination_reason": self.state["agent_system"][
                    "termination_reason"
                ],
                "agent_finish_observed": self.state["agent_system"][
                    "finish_observed"
                ],
                "agent_drain_seconds": self.state["agent_system"]["drain_seconds"],
                "status": self.state["status"],
                "outcome": self.state["outcome"],
                "elapsed_seconds": round(self._elapsed(), 3),
                "timing": self._timing_summary(),
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
                status["test_perplexity"] = self.state["test_perplexity"]
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
        if operation == "declare_incumbent":
            return self.declare_incumbent(request.get("candidate_id"))
        if operation == "lease_stop":
            return self.stop_lease()
        if operation == "finish":
            return self.finish()
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
            lease_dir = self.runs_dir / lease_id
            lease_dir.mkdir(parents=True, exist_ok=False)
            try:
                self.research_worker.start_lease(
                    episode_id=self.state["episode_id"],
                    lease_id=lease_id,
                    duration_seconds=duration,
                    task=self.research_task,
                    workspace=self.agent.workspace,
                    resources=self.resources,
                    lease_dir=lease_dir,
                )
            except WorkerFailure as exc:
                shutil.rmtree(lease_dir, ignore_errors=True)
                self._event("LEASE_INFRA_FAILURE", message=str(exc))
                self._save_state()
                raise ProtocolError("infrastructure_failure", str(exc)) from exc

            started = self.monotonic()
            self.state["lease_serial"] = serial
            self.state["leases_used"] += 1
            self.state["active_lease"] = {
                "lease_id": lease_id,
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

            log_path = self.runs_dir / active["lease_id"] / "commands.jsonl"
            if log_path.exists() and log_path.stat().st_size >= MAX_COMMAND_LOG_BYTES:
                raise ProtocolError(
                    "command_log_limit",
                    "the lease command log size limit has been reached",
                )

            result: CommandResult = self.research_worker.exec_lease(
                episode_id=self.state["episode_id"],
                lease_id=active["lease_id"],
                argv=argv,
                timeout_seconds=remaining,
            )
            stdout, local_stdout_truncated = _bounded_text(result.stdout)
            stderr, local_stderr_truncated = _bounded_text(result.stderr)
            stdout_truncated = (
                result.stdout_truncated or local_stdout_truncated
            )
            stderr_truncated = (
                result.stderr_truncated or local_stderr_truncated
            )
            active["exec_count"] += 1
            record = {
                "exec": active["exec_count"],
                "argv": argv,
                "exit_code": result.exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "timed_out": result.timed_out,
            }
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
            lease_state = self.research_worker.lease_state(
                episode_id=self.state["episode_id"],
                lease_id=active["lease_id"],
            )
            if result.timed_out or lease_state != LeaseState.RUNNING:
                self._end_active_lease(
                    "timeout" if result.timed_out else "worker_lease_stopped"
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
            commands, truncated = _tail_json_lines(log_path, limit)
            return {
                "lease_id": lease_id,
                "commands": commands,
                "history_truncated": truncated,
            }

    def _end_active_lease(self, reason: str) -> Optional[str]:
        active = self.state.get("active_lease")
        if not active:
            return None
        ended = self.monotonic()
        used_seconds = min(
            active["duration_seconds"],
            max(0.0, ended - active["started_monotonic"]),
        )
        lease_id = active["lease_id"]
        self.research_worker.terminate_lease(
            episode_id=self.state["episode_id"],
            lease_id=lease_id,
        )
        self.state["timing"]["training_lease_seconds"] += used_seconds
        self.state["active_lease"] = None
        self.state["last_lease_id"] = lease_id
        self._event(
            "LEASE_ENDED",
            lease_id=lease_id,
            reason=reason,
            used_seconds=round(used_seconds, 6),
        )
        self._save_state()
        return lease_id

    def stop_lease(self) -> Dict[str, Any]:
        with self.lock:
            self._require_active()
            lease_id = self._active_lease()["lease_id"]
            self._end_active_lease("agent_stopped_without_candidate")
            return {"lease_id": lease_id, "candidate_id": None, "val_nll": None}

    def _candidate_source(self, candidate_path: Any, lease_id: str) -> Path:
        if not isinstance(candidate_path, str) or not candidate_path:
            raise ProtocolError("invalid_candidate", "candidate path is required")
        output_root = (self.runs_dir / lease_id / "output").resolve()
        if candidate_path == "/output":
            source = output_root
        elif candidate_path.startswith("/output/"):
            source = output_root / candidate_path[len("/output/") :]
        else:
            raise ProtocolError(
                "invalid_candidate",
                "candidate must be an absolute path inside /output",
            )
        if source.is_symlink():
            raise ProtocolError("invalid_candidate", "candidate may not be a symlink")
        resolved = source.resolve()
        try:
            resolved.relative_to(output_root)
        except ValueError as exc:
            raise ProtocolError(
                "invalid_candidate", "candidate escapes the lease output directory"
            ) from exc
        if not resolved.is_dir():
            raise ProtocolError(
                "invalid_candidate", f"candidate directory does not exist: {candidate_path}"
            )
        entry_count = 0
        total_bytes = 0
        for item in resolved.rglob("*"):
            if item.is_symlink():
                raise ProtocolError(
                    "invalid_candidate", "candidate bundles may not contain symlinks"
                )
            metadata = item.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                entry_count += 1
            elif stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink != 1:
                    raise ProtocolError(
                        "invalid_candidate",
                        "candidate bundle files may not be hard-linked",
                    )
                entry_count += 1
                total_bytes += metadata.st_size
            else:
                raise ProtocolError(
                    "invalid_candidate",
                    "candidate bundles may contain only directories and regular files",
                )
            if (
                entry_count > self.config.candidate_max_files
                or total_bytes > self.config.candidate_max_bytes
            ):
                raise ProtocolError(
                    "invalid_candidate",
                    "candidate bundle exceeds the task artifact limit",
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
            metadata = path.lstat()
            mode = stat.S_IMODE(metadata.st_mode).to_bytes(4, "big")
            if path.is_symlink():
                target = os.readlink(path).encode("utf-8")
                digest.update(b"symlink\0")
                digest.update(len(relative).to_bytes(8, "big"))
                digest.update(relative)
                digest.update(len(target).to_bytes(8, "big"))
                digest.update(target)
                continue
            if path.is_dir():
                digest.update(b"dir\0")
                digest.update(len(relative).to_bytes(8, "big"))
                digest.update(relative)
                digest.update(mode)
                continue
            if not path.is_file():
                continue
            digest.update(b"file\0")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(mode)
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
            lease_id = active["lease_id"]
            self._end_active_lease("candidate_commit_requested")
            try:
                source = self._candidate_source(candidate_path, lease_id)
                snapshot = self._snapshot_candidate(source)
            except ProtocolError as exc:
                self._event(
                    "CANDIDATE_REJECTED",
                    lease_id=lease_id,
                    code=exc.code,
                    message=str(exc),
                )
                self._save_state()
                raise
            candidate_id = snapshot["candidate_id"]
            metadata = {
                "candidate_id": candidate_id,
                "source_lease": lease_id,
                "val_nll": None,
                "evaluation_error": None,
            }
            self.state["candidates"][candidate_id] = metadata
            self._event(
                "CANDIDATE_COMMITTED",
                lease_id=lease_id,
                candidate_id=candidate_id,
            )
            evaluation_started = self.monotonic()
            deadline_reached = False
            try:
                timeout_seconds = float(self.config.evaluator_timeout_seconds)
                if self.track == "A":
                    remaining = self._track_a_remaining()
                    if remaining <= 0:
                        deadline_reached = True
                    else:
                        timeout_seconds = min(timeout_seconds, remaining)
                if deadline_reached:
                    raise WorkerFailure("evaluation deadline reached")
                val_nll = validated_nll(
                    self.evaluation_worker.evaluate_candidate(
                        episode_id=self.state["episode_id"],
                        task=self.evaluation_task,
                        resources=self.resources,
                        candidate_dir=snapshot["path"],
                        hidden_split=HiddenSplitRef("val", self.config.val_path),
                        output_dir=self.evaluations_dir / snapshot["path"].name,
                        timeout_seconds=timeout_seconds,
                    )
                )
                deadline_reached = (
                    self.track == "A" and self._track_a_remaining() <= 0
                )
                if not deadline_reached:
                    metadata["val_nll"] = val_nll
                    self._event(
                        "VAL_EVALUATED",
                        candidate_id=candidate_id,
                        nll=val_nll,
                    )
            except (ValueError, WorkerFailure):
                deadline_reached = (
                    self.track == "A" and self._track_a_remaining() <= 0
                )
                if not deadline_reached:
                    metadata["evaluation_error"] = "evaluation_failed"
                    self._event(
                        "VAL_EVALUATION_FAILED",
                        candidate_id=candidate_id,
                        code="evaluation_failed",
                    )
            finally:
                evaluation_seconds = max(
                    0.0, self.monotonic() - evaluation_started
                )
                if self.track == "A":
                    evaluation_seconds = min(
                        evaluation_seconds,
                        max(
                            0.0,
                            self._track_a_deadline_monotonic()
                            - evaluation_started,
                        ),
                    )
                self.state["timing"][
                    "validation_evaluation_seconds"
                ] += evaluation_seconds
            if deadline_reached:
                metadata["evaluation_error"] = "track_a_deadline"
                self._event(
                    "VAL_EVALUATION_DISCARDED",
                    candidate_id=candidate_id,
                    code="track_a_deadline",
                )
            self._save_state()
            if deadline_reached:
                self._close_at_track_a_deadline()
            return dict(metadata)

    def declare_incumbent(self, candidate_id: Any) -> Dict[str, Any]:
        """Record the candidate the Agent would submit if the episode ended now."""

        with self.lock:
            self._require_active()
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ProtocolError(
                    "invalid_candidate", "candidate_id must be a non-empty string"
                )
            candidate = self.state["candidates"].get(candidate_id)
            if candidate is None:
                raise ProtocolError(
                    "unknown_candidate", "candidate is not registered"
                )
            if candidate.get("val_nll") is None:
                raise ProtocolError(
                    "candidate_not_validated",
                    "candidate has no successful VAL result",
                )
            previous_id = self.state["incumbent_id"]
            self.state["incumbent_id"] = candidate_id
            self._event(
                "INCUMBENT_DECLARED",
                candidate_id=candidate_id,
                previous_candidate_id=previous_id,
                source_lease=candidate["source_lease"],
                val_nll=candidate["val_nll"],
                leases_used=self.state["leases_used"],
            )
            self._save_state()
            return {
                "incumbent_id": candidate_id,
                "previous_incumbent_id": previous_id,
                "val_nll": candidate["val_nll"],
            }

    def refresh_inference_usage(self) -> Dict[str, Any]:
        with self.lock:
            summary = summarize_inference_trace(self.inference_trace_path)
            self.state["inference_usage"] = summary
            self._save_state()
            return dict(summary)

    def record_agent_execution(
        self,
        *,
        exit_code: Optional[int],
        usage: Optional[Dict[str, Any]] = None,
        termination_reason: str = "exited",
        drain_seconds: float = 0.0,
    ) -> None:
        """Attach trusted runner metadata after the Agent process exits."""

        with self.lock:
            normalized_exit_code = (
                int(exit_code) if exit_code is not None else None
            )
            self.state["agent_system"]["exit_code"] = normalized_exit_code
            self.state["agent_system"]["termination_reason"] = termination_reason
            self.state["agent_system"]["finish_observed"] = (
                self.state["closure_reason"] == "agent_finished"
            )
            self.state["agent_system"]["drain_seconds"] = round(
                max(0.0, float(drain_seconds)), 6
            )
            self.state["agent_usage"] = dict(usage) if usage is not None else None
            self._event(
                "AGENT_PROCESS_EXITED",
                runner=self.agent.runner,
                exit_code=normalized_exit_code,
                termination_reason=termination_reason,
                finish_observed=self.state["agent_system"]["finish_observed"],
                drain_seconds=self.state["agent_system"]["drain_seconds"],
            )
            self._save_state()

    def finish(self) -> Dict[str, Any]:
        with self.lock:
            self._require_active()
            if self.state["active_lease"] is not None:
                raise ProtocolError(
                    "lease_active", "commit or stop the active lease before finish"
                )
            return self._close_and_test(
                self.state["incumbent_id"], reason="agent_finished"
            )

    def _close_and_test(
        self, candidate_id: Optional[str], *, reason: str
    ) -> Dict[str, Any]:
        self._freeze_elapsed()
        self.state["status"] = "closed"
        self.state["closure_reason"] = reason
        self.state["agent_system"]["finish_observed"] = reason == "agent_finished"
        self.state["final_candidate_id"] = candidate_id
        self._event(
            "EPISODE_CLOSED", reason=reason, candidate_id=candidate_id
        )
        self._save_state()

        if candidate_id is None:
            return self._finalize_failed_test("no_candidate", candidate_id=None)

        digest = candidate_id.split(":", 1)[1]
        candidate_dir = self.candidates_dir / digest
        evaluation_started = self.monotonic()
        test_nll: Optional[float] = None
        test_failure: Optional[str] = None
        try:
            test_nll = validated_nll(
                self.evaluation_worker.evaluate_candidate(
                    episode_id=self.state["episode_id"],
                    task=self.evaluation_task,
                    resources=self.resources,
                    candidate_dir=candidate_dir,
                    hidden_split=HiddenSplitRef("test", self.config.test_path),
                    output_dir=self.evaluations_dir / digest,
                    timeout_seconds=float(self.config.evaluator_timeout_seconds),
                )
            )
        except WorkerFailure:
            test_failure = "test_evaluation_failed"
            self._event(
                "TEST_EVALUATION_FAILED",
                candidate_id=candidate_id,
                code="evaluation_failed",
            )
        except ValueError:
            test_failure = "invalid_test_metric"
            self._event(
                "TEST_EVALUATION_FAILED",
                candidate_id=candidate_id,
                code="invalid_metric",
            )
        finally:
            self.state["timing"]["test_evaluation_seconds"] += max(
                0.0, self.monotonic() - evaluation_started
            )

        if test_failure is not None or test_nll is None:
            return self._finalize_failed_test(
                test_failure or "test_evaluation_failed", candidate_id=candidate_id
            )

        try:
            test_perplexity = perplexity_from_nll(test_nll)
        except ValueError:
            self._event(
                "TEST_EVALUATION_FAILED",
                candidate_id=candidate_id,
                code="perplexity_overflow",
            )
            return self._finalize_failed_test(
                "perplexity_overflow", candidate_id=candidate_id
            )

        self.state["test_nll"] = test_nll
        self.state["test_perplexity"] = test_perplexity
        self.state["status"] = "finalized"
        self.state["outcome"] = "success"
        self.state["failure"] = None
        self._event(
            "TEST_EVALUATED",
            candidate_id=candidate_id,
            nll=test_nll,
            perplexity=test_perplexity,
        )
        self._event("EPISODE_FINALIZED", result="success")
        self._save_state()
        return self.public_status()

    def _finalize_failed_test(
        self, failure: str, *, candidate_id: Optional[str]
    ) -> Dict[str, Any]:
        """Finalize an episode without manufacturing a missing TEST metric."""

        self.state["status"] = "finalized"
        self.state["outcome"] = "failed"
        self.state["failure"] = failure
        self.state["test_nll"] = None
        self.state["test_perplexity"] = None
        self._event(
            "EPISODE_FAILED",
            reason=failure,
            candidate_id=candidate_id,
        )
        self._event("EPISODE_FINALIZED", result="failure", reason=failure)
        self._save_state()
        return self.public_status()

    def enforce_deadlines(self) -> bool:
        with self.lock:
            if self.state["status"] != "active":
                return False
            active = self.state.get("active_lease")
            if self.track == "A" and self._track_a_remaining() <= 0:
                self._close_at_track_a_deadline()
                return True
            if active and self.monotonic() >= active["deadline_monotonic"]:
                self._end_active_lease("timeout")
            return False

    def agent_exited_without_finish(self) -> None:
        with self.lock:
            if self.state["status"] != "active":
                return
            self._freeze_elapsed()
            if self.state["active_lease"] is not None:
                self._end_active_lease("agent_exited")
            self.state["status"] = "dnf"
            self.state["outcome"] = "failed"
            self.state["closure_reason"] = "agent_exited_without_finish"
            self.state["failure"] = "agent exited without finishing the episode"
            self._event("EPISODE_DNF", reason="agent_exited_without_finish")
            self._save_state()

    def infrastructure_failed(self, message: str) -> None:
        with self.lock:
            if self.state["status"] not in {"created", "active"}:
                return
            if self.state["status"] == "active":
                self._freeze_elapsed()
            if self.state["active_lease"] is not None:
                self._end_active_lease("infrastructure_failure")
            self.state["status"] = "infrastructure_failure"
            self.state["outcome"] = "failed"
            self.state["closure_reason"] = "infrastructure_failure"
            self.state["failure"] = message
            self._event("EPISODE_INFRASTRUCTURE_FAILURE", message=message)
            self._save_state()

    def process_one_request(self) -> bool:
        paths = sorted(self.inbox.glob("*.json"))
        if not paths:
            return False
        request_path = paths[0]
        request_id = request_path.stem
        try:
            request = _read_untrusted_request(request_path, 1024 * 1024)
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

        _write_untrusted_response(self.outbox, request_id, response)
        return True
