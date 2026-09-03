"""Trusted host-side episode supervisor for AutoTrain-Bench v0."""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

from .agent_control import (
    MAX_AGENT_CONTROL_RESPONSE_BYTES,
    AgentControlTransport,
    ProtocolError,
)
from .agent_execution import AgentExecutionEvidence
from .challenge import (
    FORMAL_ARTIFACT_DIGEST_FIELDS,
    ReleaseVerificationReceipt,
)
from .config import (
    AGENT_PROTOCOL_VERSION,
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
    WorkerTeardownFailure,
    assess_worker,
)


MAX_COMMAND_STREAM_BYTES = 1024 * 1024
MAX_LOG_READ_BYTES = 4 * 1024 * 1024
MAX_COMMAND_LOG_BYTES = 64 * 1024 * 1024
_OUTPUT_TRUNCATION_MARKER = "\n[output truncated by AutoTrain-Bench]\n"
_TREE_HASH_DOMAIN = b"autotrain-bench-tree-v2\0"
_PARTICIPANT_ARTIFACT_ERRNOS = {
    errno.EACCES,
    errno.ELOOP,
    errno.ENOENT,
    errno.ENOTDIR,
    errno.EPERM,
}


def _safe_protocol_text(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\0" in value:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8", errors="replace")


def _bounded_text(
    value: Any, maximum_bytes: int = MAX_COMMAND_STREAM_BYTES
) -> tuple[str, bool]:
    if not isinstance(value, str):
        value = str(value)
    encoded = value.encode("utf-8", errors="replace")
    normalized = encoded.decode("utf-8")
    sanitized = normalized != value
    if len(encoded) <= maximum_bytes:
        return normalized, sanitized
    marker = _OUTPUT_TRUNCATION_MARKER.encode("ascii")
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


def _fit_command_record(
    record: Dict[str, Any], *, lease_active: bool
) -> Dict[str, Any]:
    """Fit a completed command into one successful mailbox response.

    JSON escaping can expand a stream well beyond its UTF-8 byte length.  A
    command has already executed by this point, so replacing the response with
    ``response_too_large`` would make a non-idempotent operation ambiguous.
    """

    original_stdout = record["stdout"]
    original_stderr = record["stderr"]

    def candidate(fraction: float) -> Dict[str, Any]:
        result = dict(record)
        for field in ("stdout", "stderr"):
            original = original_stdout if field == "stdout" else original_stderr
            kept = min(len(original), int(len(original) * fraction))
            locally_truncated = kept < len(original)
            text = original[:kept]
            if locally_truncated:
                text += _OUTPUT_TRUNCATION_MARKER
            result[field] = text
            flag = f"{field}_truncated"
            result[flag] = bool(record[flag] or locally_truncated)
        return result

    def encoded_size(value: Dict[str, Any]) -> int:
        return len(
            _json_bytes(
                {
                    "ok": True,
                    "result": {**value, "lease_active": lease_active},
                }
            )
        )

    if encoded_size(record) <= MAX_AGENT_CONTROL_RESPONSE_BYTES:
        return record

    smallest = candidate(0.0)
    if encoded_size(smallest) > MAX_AGENT_CONTROL_RESPONSE_BYTES:
        raise ProtocolError(
            "internal_error", "command metadata exceeds the mailbox limit"
        )

    low = 0.0
    high = 1.0
    best = smallest
    for _ in range(32):
        midpoint = (low + high) / 2
        current = candidate(midpoint)
        if encoded_size(current) <= MAX_AGENT_CONTROL_RESPONSE_BYTES:
            best = current
            low = midpoint
        else:
            high = midpoint
    return best


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
        agent_api_dir: Path,
        agent_execution: AgentExecutionEvidence,
        research_worker: ResearchWorker,
        evaluation_worker: EvaluationWorker,
        agent_control_transport: Optional[AgentControlTransport] = None,
        research_worker_environment: Optional[Dict[str, Any]] = None,
        evaluation_worker_environment: Optional[Dict[str, Any]] = None,
        comparison_key: Optional[Dict[str, str]] = None,
        challenge_release: Optional[Dict[str, Any]] = None,
        release_verification: Optional[ReleaseVerificationReceipt] = None,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
    ) -> None:
        normalized = track.upper()
        if normalized not in {"A", "B"}:
            raise ValueError("track must be A or B")
        self.config = config.validate()
        self.agent = agent.validate()
        if agent_execution.mechanism != self.agent.runner:
            raise ValueError("Agent execution mechanism does not match Agent config")
        if self.agent.runner == "container":
            if agent_execution.workload_identity != self.agent.image:
                raise ValueError(
                    "Agent execution workload identity does not match the selected "
                    "Agent image"
                )
            if (
                agent_execution.immutable_workload_identity
                and not immutable_image_reference(self.agent.image)
            ):
                raise ValueError(
                    "Agent execution evidence claims a mutable image is immutable"
                )
        elif (
            agent_execution.workload_identity is not None
            or agent_execution.immutable_workload_identity
        ):
            raise ValueError(
                "host_cli Agent execution evidence may not claim a workload image"
            )
        self.agent_execution = agent_execution
        self.resources = resources.validate_for(config)
        self.track = normalized
        self.episode_dir = episode_dir.resolve()
        self.research_worker = research_worker
        self.evaluation_worker = evaluation_worker
        self.agent_control_transport = agent_control_transport
        agent_control_name = (
            agent_control_transport.transport_name
            if agent_control_transport is not None
            else None
        )
        if agent_control_name is not None and (
            not isinstance(agent_control_name, str) or not agent_control_name.strip()
        ):
            raise ValueError("Agent control transport name must be a non-empty string")
        self.evaluation_task = EvaluationTaskSpec.from_task(config)
        if release_verification is None:
            self.release_verification = None
        else:
            verified_release = release_verification.validate()
            self.release_verification = ReleaseVerificationReceipt(
                authority=verified_release.authority,
                release_id=verified_release.release_id,
                core_revision=verified_release.core_revision,
                artifact_sha256=dict(verified_release.artifact_sha256),
            )
        self.monotonic = monotonic
        self.wall_time = wall_time
        self.lock = threading.RLock()
        self.started_monotonic: Optional[float] = None
        self.closed_monotonic: Optional[float] = None

        self.runs_dir = self.episode_dir / "leases"
        self.candidates_dir = self.episode_dir / "candidates"
        self.evaluations_dir = self.episode_dir / "evaluations"
        self.inference_dir = self.episode_dir / "inference"
        self.inference_trace_path = self.inference_dir / "inference.jsonl"
        self.public_dir = self.episode_dir / "public"
        self.state_path = self.episode_dir / "state.json"
        self.events_path = self.episode_dir / "events.jsonl"
        self.public_task_path = self.public_dir / "task.json"
        if self.state_path.exists() or self.events_path.exists():
            raise ValueError(
                f"episode directory already contains state: {self.episode_dir}"
            )
        self.agent_api_dir = agent_api_dir.resolve()
        if not (self.agent_api_dir / "autotrain_bench" / "client.py").is_file():
            raise ValueError(f"Agent API bundle is missing: {self.agent_api_dir}")
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.episode_dir, 0o700)
        for directory in (
            self.runs_dir,
            self.candidates_dir,
            self.evaluations_dir,
            self.inference_dir,
            self.public_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        os.chmod(self.inference_dir, 0o700)
        for directory in (
            self.runs_dir,
            self.candidates_dir,
            self.evaluations_dir,
        ):
            os.chmod(directory, 0o700)

        self.research_task = ResearchTaskSpec.from_task(config, self.agent_api_dir)

        research_descriptor = research_worker.describe_capabilities()
        evaluation_descriptor = evaluation_worker.describe_capabilities()
        research_requirements = list(RESEARCH_REQUIREMENTS)
        evaluation_requirements = list(EVALUATION_REQUIREMENTS)
        if resources.nvidia_gpu_count:
            research_requirements.append(
                CapabilityRequirement(GPU_DEVICE_ISOLATION, EnforcementLevel.ENFORCED)
            )
        if config.evaluator_device == "cuda":
            evaluation_requirements.append(
                CapabilityRequirement(GPU_DEVICE_ISOLATION, EnforcementLevel.ENFORCED)
            )
        research_assessment = assess_worker(research_descriptor, research_requirements)
        evaluation_assessment = assess_worker(
            evaluation_descriptor, evaluation_requirements
        )
        core_identity = _core_identity()
        artifact_identities = {
            "challenge_manifest_sha256": (
                challenge_release.get("challenge_manifest_sha256")
                if isinstance(challenge_release, dict)
                else None
            ),
            "task_manifest_sha256": _sha256_file(config.source),
            "policy_sha256": _sha256_file(config.source.parent / "POLICY.md"),
            "evaluator_sha256": _sha256_file(config.evaluator_entrypoint),
            "train_manifest_sha256": _sha256_file(config.train_dir / "manifest.json"),
            "val_sha256": _sha256_file(config.val_path),
            "test_sha256": _sha256_file(config.test_path),
        }

        policy_violations = []
        policy_violations.extend(agent_execution.limitations)
        if not agent_execution.policy_compliant and not agent_execution.limitations:
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
            if isinstance(comparison_key, dict) and challenge_release.get(
                "challenge_id"
            ) != comparison_key.get("challenge_id"):
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
            if (
                challenge_release.get("task_manifest_sha256")
                != artifact_identities["task_manifest_sha256"]
            ):
                eligibility_violations.append("task_manifest_identity_mismatch")
            for field in ("track_config", "runtime_profile"):
                selected = challenge_release.get(field)
                if (
                    not isinstance(selected, dict)
                    or selected.get("provisional") is not False
                ):
                    eligibility_violations.append(f"{field}_not_frozen")
            if isinstance(comparison_key, dict):
                for release_field, comparison_field in (
                    ("track_config_id", "track_config"),
                    ("runtime_profile_id", "runtime_profile"),
                ):
                    if challenge_release.get(release_field) != comparison_key.get(
                        comparison_field
                    ):
                        eligibility_violations.append(f"{release_field}_mismatch")
        if not agent_execution.immutable_workload_identity:
            eligibility_violations.append("agent_workload_identity_not_immutable")
        if not agent_execution.leaderboard_eligible:
            eligibility_violations.append("agent_execution_not_certified")
        core_revision = core_identity.get("revision")
        if not isinstance(core_revision, str) or not core_revision:
            eligibility_violations.append("core_identity_unknown")
        if core_identity.get("dirty") is not False:
            eligibility_violations.append("core_worktree_not_clean")
        if self.release_verification is None:
            eligibility_violations.append("formal_release_verification_missing")
        else:
            receipt = self.release_verification
            if not isinstance(
                comparison_key, dict
            ) or receipt.release_id != comparison_key.get("challenge_id"):
                eligibility_violations.append(
                    "formal_release_verification_identity_mismatch"
                )
            if receipt.core_revision != core_revision:
                eligibility_violations.append(
                    "formal_release_verification_core_mismatch"
                )
            for field in FORMAL_ARTIFACT_DIGEST_FIELDS:
                if receipt.artifact_sha256[field] != artifact_identities.get(field):
                    eligibility_violations.append(
                        f"formal_release_verification_artifact_mismatch:{field}"
                    )

        leaderboard_eligible = (
            config.leaderboard_eligible
            and policy_compliant
            and research_assessment.leaderboard_eligible
            and evaluation_assessment.leaderboard_eligible
            and not eligibility_violations
        )

        self.state: Dict[str, Any] = {
            "protocol_version": AGENT_PROTOCOL_VERSION,
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
            "formal_release_verification": (
                self.release_verification.private_manifest()
                if self.release_verification
                else None
            ),
            "artifact_identities": {
                "core_git": core_identity,
                "challenge_manifest_sha256": artifact_identities[
                    "challenge_manifest_sha256"
                ],
                "task_manifest_sha256": artifact_identities["task_manifest_sha256"],
                "policy_sha256": artifact_identities["policy_sha256"],
                "evaluator_sha256": artifact_identities["evaluator_sha256"],
                "train_manifest_sha256": artifact_identities["train_manifest_sha256"],
                "hidden_splits": {
                    "val_sha256": artifact_identities["val_sha256"],
                    "test_sha256": artifact_identities["test_sha256"],
                },
            },
            "artifact_identity_algorithms": {
                "candidate_tree": "autotrain-bench-tree-v2-sha256",
                "file": "sha256",
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
                "egress": agent.egress,
                "network": agent.network,
                "policy_compliant": agent_execution.policy_compliant,
                "execution_evidence": agent_execution.public_manifest(),
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
                    "allowed_models": list(agent.inference.allowed_models)
                    if agent.inference
                    else [],
                    "allowed_paths": list(agent.inference.allowed_paths)
                    if agent.inference
                    else [],
                },
                "forwarded_environment": list(agent.environment),
                "initial_workspace_sha256": self._tree_hash(agent.workspace),
            },
            "agent_control": {
                "transport": agent_control_name,
                "cleanup_complete": None,
                "cleanup_removed": 0,
                "cleanup_failures": 0,
                "cleanup_policy_violations": 0,
                "cleanup_agent_caused_failures": 0,
                "cleanup_infrastructure_failures": 0,
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
            # Semantic Lease IDs whose Worker-side startup result was
            # uncertain and whose compensating termination was not confirmed.
            # Platform resource handles remain private to the Worker.
            "pending_lease_teardowns": [],
            "last_lease_id": None,
            "deployment_teardown_failures": [],
            "candidates": {},
            "incumbent_id": None,
            "final_candidate_id": None,
            "test_nll": None,
            "test_perplexity": None,
            "pending_result": None,
            "result_publication": None,
            "closure_reason": None,
            "inference_usage": summarize_inference_trace(self.inference_trace_path),
            "agent_usage": None,
            "failure": None,
        }
        self.sequence = 0
        public_manifest = config.public_manifest(self.track)
        public_manifest["agent_environment"] = agent.public_manifest(
            config.inference_gateway_image,
            policy_compliant=agent_execution.policy_compliant,
            control_transport=agent_control_name,
        )
        public_manifest["training_environment"]["resources"] = (
            resources.public_manifest()
        )
        public_manifest["policy"] = {
            "compliant": policy_compliant,
            "violations": policy_violations,
        }
        public_manifest["leaderboard_eligible"] = leaderboard_eligible
        public_manifest["candidate_artifact_identity"] = (
            "autotrain-bench-tree-v2-sha256"
        )
        public_manifest["formal_release_verified"] = (
            self.release_verification is not None
            and not any(
                violation.startswith("formal_release_verification")
                for violation in eligibility_violations
            )
        )
        public_manifest["comparison_key"] = self.state["comparison_key"]
        if challenge_release is not None:
            public_manifest["challenge_release"] = dict(challenge_release)
        self.public_task_path.write_text(
            json.dumps(public_manifest, indent=2, sort_keys=True) + "\n",
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
                agent_runner=self.agent.runner,
                agent_egress=self.agent.egress,
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

    def _event(self, event_type: str, **data: Any) -> int:
        self.sequence += 1
        event = {
            "seq": self.sequence,
            "type": event_type,
            "elapsed_seconds": round(self._elapsed(), 6),
            **data,
        }
        descriptor = os.open(
            self.events_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            # Tighten a pre-existing file before appending private event data.
            os.chmod(self.events_path, 0o600)
            with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
                descriptor = -1
                stream.write(json.dumps(event, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        return self.sequence

    def _save_state(self) -> None:
        self._validate_state_invariants()
        self.state["elapsed_seconds"] = round(self._elapsed(), 6)
        self.state["timing_summary"] = self._timing_summary()
        temporary = self.state_path.with_suffix(".json.tmp")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        try:
            # Tighten a leftover temporary before writing private state.  Do
            # every fallible file operation before replace; after replace
            # returns, callers can treat the authoritative commit as complete.
            os.chmod(temporary, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                descriptor = -1
                stream.write(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        os.replace(str(temporary), str(self.state_path))

    def _validate_state_invariants(self) -> None:
        """Reject internally contradictory lifecycle state before publication."""

        status = self.state["status"]
        pending_teardowns = self.state.get("pending_lease_teardowns", [])
        if not isinstance(pending_teardowns, list) or any(
            not isinstance(lease_id, str) or not lease_id
            for lease_id in pending_teardowns
        ):
            raise RuntimeError("pending Lease teardowns must be semantic Lease IDs")
        if len(set(pending_teardowns)) != len(pending_teardowns):
            raise RuntimeError("pending Lease teardown IDs must be unique")
        termination = self.state["agent_system"]["termination_reason"]
        cleanup_complete = self.state["agent_control"]["cleanup_complete"]
        outcome = self.state["outcome"]
        result_publication = self.state.get("result_publication")
        if status in {"created", "active", "closed"} and outcome is not None:
            raise RuntimeError("a non-terminal episode cannot publish an outcome")
        if status in {"dnf", "infrastructure_failure"} and outcome != "failed":
            raise RuntimeError("a failure terminal must publish a failed outcome")
        if status == "finalized" and outcome not in {"success", "failed"}:
            raise RuntimeError("a finalized episode requires an outcome")
        if status != "finalized" and result_publication is not None:
            raise RuntimeError("only a finalized episode may publish a result")
        if self.state.get("pending_result") is not None and status != "closed":
            raise RuntimeError("a staged TEST result requires the closed state")
        if outcome == "failed" and (
            self.state.get("test_nll") is not None
            or self.state.get("test_perplexity") is not None
        ):
            raise RuntimeError("a failed outcome cannot retain TEST metrics")
        if cleanup_complete is True and termination is None:
            raise RuntimeError("Agent control cannot be clean before Agent termination")
        if status == "active" and termination is not None:
            raise RuntimeError("an active episode cannot have a terminated Agent")
        if status == "finalized":
            if not isinstance(result_publication, dict):
                raise RuntimeError(
                    "a finalized result requires an audit publication reference"
                )
            publication_id = result_publication.get("publication_id")
            prepared_event_seq = result_publication.get("prepared_event_seq")
            if not _safe_protocol_text(publication_id):
                raise RuntimeError("result publication ID must be a safe string")
            if (
                isinstance(prepared_event_seq, bool)
                or not isinstance(prepared_event_seq, int)
                or prepared_event_seq < 1
                or prepared_event_seq > self.sequence
            ):
                raise RuntimeError(
                    "result publication must reference a prepared audit event"
                )
            if termination is None or cleanup_complete is not True:
                raise RuntimeError("a finalized result requires completed teardown")
            if self.state.get("active_lease") is not None or pending_teardowns:
                raise RuntimeError("a finalized result cannot retain Worker Leases")
            if self.state.get("pending_result") is not None:
                raise RuntimeError("a finalized result cannot remain staged")

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
                "agent_egress": self.agent.egress,
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
                "agent_finish_observed": self.state["agent_system"]["finish_observed"],
                "agent_drain_seconds": self.state["agent_system"]["drain_seconds"],
                "status": self.state["status"],
                "outcome": self.state["outcome"],
                "elapsed_seconds": round(self._elapsed(), 3),
                "timing": self._timing_summary(),
                "leases_used": self.state["leases_used"],
                "active_lease": None,
                "deployment_teardown_failures": list(
                    self.state.get("deployment_teardown_failures", [])
                ),
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
                raise ProtocolError(
                    "lease_active", "a training lease is already active"
                )

            if requested_seconds is not None and (
                not isinstance(requested_seconds, int)
                or isinstance(requested_seconds, bool)
                or requested_seconds <= 0
            ):
                raise ProtocolError(
                    "invalid_duration", "lease duration must be positive"
                )

            if self.track == "A":
                maximum = self._track_a_remaining()
            else:
                if self.state["leases_used"] >= self.config.track_b_max_leases:
                    raise ProtocolError("lease_budget_exhausted", "no leases remain")
                maximum = float(self.config.track_b_lease_seconds)

            duration = (
                maximum if requested_seconds is None else float(requested_seconds)
            )
            if duration > maximum + 1e-9:
                raise ProtocolError(
                    "lease_too_long",
                    f"requested {duration:g}s but at most {maximum:g}s is available",
                )

            serial = self.state["lease_serial"] + 1
            lease_id = f"lease-{serial:04d}"
            lease_dir = self.runs_dir / lease_id
            lease_dir.mkdir(parents=True, exist_ok=False)
            # Persist a semantic cleanup intent before asking the Worker to
            # create anything.  If the process dies after an acknowledgement is
            # lost, state.json still identifies what terminate_lease must revoke.
            self.state["lease_serial"] = serial
            pending = self.state["pending_lease_teardowns"]
            pending.append(lease_id)
            self._save_state()
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
            except WorkerTeardownFailure as exc:
                self._event(
                    "LEASE_INFRA_FAILURE",
                    lease_id=lease_id,
                    code="worker_teardown_failure",
                )
                self.deployment_teardown_failed(
                    "research_worker_teardown_failure",
                    private_detail=str(exc),
                )
                raise ProtocolError(
                    "infrastructure_failure",
                    "research worker teardown failed",
                ) from exc
            except WorkerFailure as exc:
                pending.remove(lease_id)
                try:
                    shutil.rmtree(lease_dir)
                except FileNotFoundError:
                    pass
                except OSError as cleanup_exc:
                    self._event(
                        "LEASE_INFRA_FAILURE",
                        lease_id=lease_id,
                        code="lease_directory_cleanup_failure",
                    )
                    self.deployment_teardown_failed(
                        "research_lease_directory_cleanup_failure",
                        private_detail=str(cleanup_exc),
                    )
                    raise ProtocolError(
                        "infrastructure_failure",
                        "training lease startup cleanup failed",
                    ) from cleanup_exc
                self._event("LEASE_INFRA_FAILURE", message=str(exc))
                self._save_state()
                raise ProtocolError(
                    "infrastructure_failure", "training lease startup failed"
                ) from exc

            started = self.monotonic()
            self.state["leases_used"] += 1
            self.state["active_lease"] = {
                "lease_id": lease_id,
                "started_monotonic": started,
                "deadline_monotonic": started + duration,
                "duration_seconds": duration,
                "exec_count": 0,
            }
            pending.remove(lease_id)
            self.state["last_lease_id"] = lease_id
            self._event("LEASE_STARTED", lease_id=lease_id, duration_seconds=duration)
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
            if (
                not isinstance(argv, list)
                or not argv
                or not all(_safe_protocol_text(value) for value in argv)
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
                if not math.isfinite(float(command_timeout_seconds)):
                    raise ProtocolError(
                        "invalid_timeout", "command timeout must be finite"
                    )
                remaining = min(remaining, float(command_timeout_seconds))

            log_path = self.runs_dir / active["lease_id"] / "commands.jsonl"
            if log_path.exists() and log_path.stat().st_size >= MAX_COMMAND_LOG_BYTES:
                raise ProtocolError(
                    "command_log_limit",
                    "the lease command log size limit has been reached",
                )

            try:
                result: CommandResult = self.research_worker.exec_lease(
                    episode_id=self.state["episode_id"],
                    lease_id=active["lease_id"],
                    argv=argv,
                    timeout_seconds=remaining,
                )
            except WorkerTeardownFailure as exc:
                self._event(
                    "LEASE_INFRA_FAILURE",
                    lease_id=active["lease_id"],
                    code="worker_teardown_failure",
                )
                self.deployment_teardown_failed(
                    "research_worker_teardown_failure",
                    private_detail=str(exc),
                )
                raise ProtocolError(
                    "infrastructure_failure",
                    "research worker teardown failed",
                ) from exc
            except Exception as exc:
                self.infrastructure_failed(
                    "research_worker_execution_failure",
                    private_detail=str(exc),
                )
                raise ProtocolError(
                    "infrastructure_failure",
                    "research worker execution failed",
                ) from None
            if not isinstance(result, CommandResult):
                self.infrastructure_failed(
                    "research_worker_protocol_failure",
                    private_detail=("exec_lease returned " f"{type(result).__name__}"),
                )
                raise ProtocolError(
                    "infrastructure_failure",
                    "research worker returned an invalid command result",
                )
            stdout, local_stdout_truncated = _bounded_text(result.stdout)
            stderr, local_stderr_truncated = _bounded_text(result.stderr)
            stdout_truncated = result.stdout_truncated or local_stdout_truncated
            stderr_truncated = result.stderr_truncated or local_stderr_truncated
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
            self._event(
                "LEASE_COMMAND_FINISHED",
                lease_id=active["lease_id"],
                exec=active["exec_count"],
                exit_code=result.exit_code,
                timed_out=result.timed_out,
            )
            lease_active = True
            try:
                lease_state = self.research_worker.lease_state(
                    episode_id=self.state["episode_id"],
                    lease_id=active["lease_id"],
                )
            except WorkerTeardownFailure as exc:
                self.deployment_teardown_failed(
                    "research_worker_teardown_failure",
                    private_detail=str(exc),
                )
                raise ProtocolError(
                    "infrastructure_failure",
                    "research worker teardown failed",
                ) from None
            except Exception as exc:
                self.infrastructure_failed(
                    "research_worker_state_failure",
                    private_detail=str(exc),
                )
                raise ProtocolError(
                    "infrastructure_failure",
                    "research worker state query failed",
                ) from None
            if not isinstance(lease_state, LeaseState):
                self.infrastructure_failed(
                    "research_worker_protocol_failure",
                    private_detail=(
                        "lease_state returned " f"{type(lease_state).__name__}"
                    ),
                )
                raise ProtocolError(
                    "infrastructure_failure",
                    "research worker returned an invalid Lease state",
                )
            if result.timed_out or lease_state != LeaseState.RUNNING:
                self._end_active_lease(
                    "timeout" if result.timed_out else "worker_lease_stopped"
                )
                lease_active = False
            record = _fit_command_record(record, lease_active=lease_active)
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(_json_bytes(record).decode("utf-8") + "\n")
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
            result = {
                "lease_id": lease_id,
                "commands": commands,
                "history_truncated": truncated,
            }
            while (
                commands
                and len(_json_bytes({"ok": True, "result": result}))
                > MAX_AGENT_CONTROL_RESPONSE_BYTES
            ):
                commands.pop(0)
                result["history_truncated"] = True
            return result

    def _end_active_lease(
        self,
        reason: str,
        *,
        ended_monotonic: Optional[float] = None,
        record_teardown_failure: bool = True,
    ) -> Optional[str]:
        active = self.state.get("active_lease")
        if not active:
            return None
        ended = self.monotonic() if ended_monotonic is None else ended_monotonic
        used_seconds = min(
            active["duration_seconds"],
            max(0.0, ended - active["started_monotonic"]),
        )
        lease_id = active["lease_id"]
        try:
            self.research_worker.terminate_lease(
                episode_id=self.state["episode_id"],
                lease_id=lease_id,
            )
        except Exception as exc:
            teardown_exc = (
                exc
                if isinstance(exc, WorkerTeardownFailure)
                else WorkerTeardownFailure(str(exc))
            )
            if record_teardown_failure:
                self._event(
                    "LEASE_INFRA_FAILURE",
                    lease_id=lease_id,
                    code="worker_teardown_failure",
                )
                self.deployment_teardown_failed(
                    "research_worker_teardown_failure",
                    private_detail=str(teardown_exc),
                )
            if teardown_exc is exc:
                raise
            raise teardown_exc from exc
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

    def terminate_worker_leases_for_teardown(self) -> bool:
        """Retry uncertain starts and the active semantic Worker Lease."""

        with self.lock:
            pending = self.state.setdefault("pending_lease_teardowns", [])
            active = self.state.get("active_lease")
            if not pending and active is None:
                return False
            teardown_failures = []
            reconciliation_failures = []
            cleaned_startups = []
            state_changed = False
            for lease_id in list(pending):
                try:
                    self.research_worker.terminate_lease(
                        episode_id=self.state["episode_id"],
                        lease_id=lease_id,
                    )
                except Exception as exc:
                    teardown_failures.append(f"{lease_id}: {type(exc).__name__}: {exc}")
                else:
                    pending.remove(lease_id)
                    cleaned_startups.append(lease_id)
                    state_changed = True

            if active is not None:
                ended = (
                    self.closed_monotonic
                    if self.closed_monotonic is not None
                    else self.monotonic()
                )
                try:
                    self._end_active_lease(
                        "deployment_teardown",
                        ended_monotonic=ended,
                        record_teardown_failure=False,
                    )
                except WorkerTeardownFailure as exc:
                    teardown_failures.append(
                        f"{active['lease_id']}: {type(exc).__name__}: {exc}"
                    )
                except Exception as exc:
                    reconciliation_failures.append(
                        f"{active['lease_id']}: {type(exc).__name__}: {exc}"
                    )

            for lease_id in cleaned_startups:
                self._event("LEASE_STARTUP_CLEANED", lease_id=lease_id)
            if state_changed:
                self._save_state()
            if teardown_failures:
                raise WorkerTeardownFailure(
                    "Worker Lease teardown failed: " + "; ".join(teardown_failures)
                )
            if reconciliation_failures:
                raise RuntimeError(
                    "Worker Lease state reconciliation failed: "
                    + "; ".join(reconciliation_failures)
                )
            return True

    def stop_lease(self) -> Dict[str, Any]:
        with self.lock:
            self._require_active()
            lease_id = self._active_lease()["lease_id"]
            self._end_active_lease("agent_stopped_without_candidate")
            return {"lease_id": lease_id, "candidate_id": None, "val_nll": None}

    def _candidate_source(self, candidate_path: Any, lease_id: str) -> Path:
        if not isinstance(candidate_path, str) or not candidate_path:
            raise ProtocolError("invalid_candidate", "candidate path is required")
        if not _safe_protocol_text(candidate_path):
            raise ProtocolError(
                "invalid_candidate", "candidate path contains invalid characters"
            )
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
                "invalid_candidate",
                f"candidate directory does not exist: {candidate_path}",
            )
        entry_count = 0
        total_bytes = 0
        for item, metadata in self._tree_entries(resolved):
            relative = item.relative_to(resolved).as_posix()
            if not _safe_protocol_text(relative):
                raise ProtocolError(
                    "invalid_candidate",
                    "candidate bundle contains an invalid path encoding",
                )
            if stat.S_ISLNK(metadata.st_mode):
                raise ProtocolError(
                    "invalid_candidate", "candidate bundles may not contain symlinks"
                )
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
                try:
                    with item.open("rb") as stream:
                        stream.read(1)
                except OSError as exc:
                    if exc.errno in _PARTICIPANT_ARTIFACT_ERRNOS:
                        raise ProtocolError(
                            "invalid_candidate",
                            "candidate bundle contains an unreadable file",
                        ) from exc
                    raise
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
    def _tree_entries(root: Path) -> Iterator[Tuple[Path, os.stat_result]]:
        """Walk a tree without links or suppressed directory-read failures."""

        pending = [root]
        while pending:
            directory = pending.pop()
            child_directories = []
            with os.scandir(directory) as children:
                for child in children:
                    metadata = child.stat(follow_symlinks=False)
                    path = Path(child.path)
                    yield path, metadata
                    if stat.S_ISDIR(metadata.st_mode):
                        child_directories.append(path)
            pending.extend(reversed(child_directories))

    @staticmethod
    def _tree_hash(root: Path) -> str:
        digest = hashlib.sha256()
        digest.update(_TREE_HASH_DOMAIN)
        entries = sorted(
            Supervisor._tree_entries(root),
            key=lambda entry: entry[0].relative_to(root).as_posix(),
        )
        for path, metadata in entries:
            relative = path.relative_to(root).as_posix().encode("utf-8")
            mode = stat.S_IMODE(metadata.st_mode).to_bytes(4, "big")
            if stat.S_ISLNK(metadata.st_mode):
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
            digest.update(metadata.st_size.to_bytes(8, "big"))
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _remove_staging_tree(stage: Path) -> None:
        """Remove a trusted copy even when source directory modes were read-only."""

        os.chmod(stage, 0o700)
        for directory, child_directories, _ in os.walk(stage, followlinks=False):
            os.chmod(directory, 0o700)
            for name in child_directories:
                child = Path(directory) / name
                if not child.is_symlink():
                    os.chmod(child, 0o700)
        shutil.rmtree(stage)

    def _snapshot_candidate(self, source: Path) -> Dict[str, Any]:
        stage = self.candidates_dir / f".stage-{uuid.uuid4().hex}"
        try:
            shutil.copytree(source, stage)
            digest = self._tree_hash(stage)
            candidate_id = f"sha256:{digest}"
            destination = self.candidates_dir / digest
            if not destination.exists():
                os.replace(str(stage), str(destination))
            return {"candidate_id": candidate_id, "path": destination}
        finally:
            if stage.exists():
                self._remove_staging_tree(stage)

    def commit_lease(self, candidate_path: Any) -> Dict[str, Any]:
        with self.lock:
            self._require_active()
            active = self._active_lease()
            lease_id = active["lease_id"]
            self._end_active_lease("candidate_commit_requested")
            try:
                try:
                    source = self._candidate_source(candidate_path, lease_id)
                except OSError as exc:
                    if exc.errno in _PARTICIPANT_ARTIFACT_ERRNOS:
                        raise ProtocolError(
                            "invalid_candidate",
                            "candidate bundle cannot be inspected safely",
                        ) from exc
                    self._event(
                        "CANDIDATE_INSPECTION_FAILURE",
                        lease_id=lease_id,
                        code="coordinator_storage_failure",
                        exception_type=type(exc).__name__,
                        message=str(exc),
                    )
                    self.infrastructure_failed(
                        "candidate_inspection_infrastructure_failure",
                        private_detail=str(exc),
                    )
                    raise ProtocolError(
                        "infrastructure_failure",
                        "candidate inspection infrastructure failed",
                    ) from None
            except ProtocolError as exc:
                self._event(
                    "CANDIDATE_REJECTED",
                    lease_id=lease_id,
                    code=exc.code,
                    message=str(exc),
                )
                self._save_state()
                raise
            try:
                snapshot = self._snapshot_candidate(source)
            except OSError as exc:
                self._event(
                    "CANDIDATE_SNAPSHOT_FAILURE",
                    lease_id=lease_id,
                    code="coordinator_storage_failure",
                    exception_type=type(exc).__name__,
                    message=str(exc),
                )
                self.infrastructure_failed(
                    "candidate_snapshot_infrastructure_failure",
                    private_detail=str(exc),
                )
                raise ProtocolError(
                    "infrastructure_failure",
                    "candidate snapshot infrastructure failed",
                ) from None
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
            teardown_failure: Optional[str] = None
            infrastructure_failure: Optional[str] = None
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
                deadline_reached = self.track == "A" and self._track_a_remaining() <= 0
                if not deadline_reached:
                    metadata["val_nll"] = val_nll
                    self._event(
                        "VAL_EVALUATED",
                        candidate_id=candidate_id,
                        nll=val_nll,
                    )
            except WorkerTeardownFailure as exc:
                metadata["evaluation_error"] = "evaluation_worker_teardown_failure"
                teardown_failure = str(exc)
                self._event(
                    "VAL_INFRASTRUCTURE_FAILURE",
                    candidate_id=candidate_id,
                    code="worker_teardown_failure",
                )
            except (ValueError, WorkerFailure):
                deadline_reached = self.track == "A" and self._track_a_remaining() <= 0
                if not deadline_reached:
                    metadata["evaluation_error"] = "evaluation_failed"
                    self._event(
                        "VAL_EVALUATION_FAILED",
                        candidate_id=candidate_id,
                        code="evaluation_failed",
                    )
            except Exception as exc:
                metadata["evaluation_error"] = "evaluation_infrastructure_failure"
                infrastructure_failure = str(exc)
                self._event(
                    "VAL_INFRASTRUCTURE_FAILURE",
                    candidate_id=candidate_id,
                    code="unexpected_worker_failure",
                    exception_type=type(exc).__name__,
                    message=str(exc),
                )
            finally:
                evaluation_seconds = max(0.0, self.monotonic() - evaluation_started)
                if self.track == "A":
                    evaluation_seconds = min(
                        evaluation_seconds,
                        max(
                            0.0,
                            self._track_a_deadline_monotonic() - evaluation_started,
                        ),
                    )
                self.state["timing"]["validation_evaluation_seconds"] += (
                    evaluation_seconds
                )
            if teardown_failure is not None:
                # Persist the timing update and the failure transition in one
                # authoritative state write.  Raising inside the except block
                # would skip the save that normally follows this finally.
                self.deployment_teardown_failed(
                    "evaluation_worker_teardown_failure",
                    private_detail=teardown_failure,
                )
                raise ProtocolError(
                    "infrastructure_failure",
                    "evaluation worker teardown failed",
                ) from None
            if infrastructure_failure is not None:
                self.infrastructure_failed(
                    "val_evaluation_infrastructure_failure",
                    private_detail=infrastructure_failure,
                )
                raise ProtocolError(
                    "infrastructure_failure",
                    "validation evaluation infrastructure failed",
                ) from None
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
                raise ProtocolError("unknown_candidate", "candidate is not registered")
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
            if self.state["status"] in {"created", "active"}:
                raise RuntimeError(
                    "episode closure must be recorded before Agent termination"
                )
            if (
                not isinstance(termination_reason, str)
                or not termination_reason.strip()
            ):
                raise ValueError("Agent termination reason must be a non-empty string")
            normalized_exit_code = int(exit_code) if exit_code is not None else None
            normalized_drain_seconds = round(max(0.0, float(drain_seconds)), 6)
            normalized_usage = dict(usage) if usage is not None else None
            previous_reason = self.state["agent_system"]["termination_reason"]
            if previous_reason is not None:
                if (
                    self.state["agent_system"]["exit_code"] != normalized_exit_code
                    or previous_reason != termination_reason
                    or self.state["agent_system"]["drain_seconds"]
                    != normalized_drain_seconds
                    or self.state["agent_usage"] != normalized_usage
                ):
                    raise RuntimeError("Agent termination metadata is immutable")
                if not self._publish_staged_result():
                    self._save_state()
                return
            self.state["agent_system"]["exit_code"] = normalized_exit_code
            self.state["agent_system"]["termination_reason"] = termination_reason
            self.state["agent_system"]["finish_observed"] = (
                self.state["closure_reason"] == "agent_finished"
            )
            self.state["agent_system"]["drain_seconds"] = normalized_drain_seconds
            self.state["agent_usage"] = normalized_usage
            self._event(
                "AGENT_PROCESS_EXITED",
                runner=self.agent.runner,
                exit_code=normalized_exit_code,
                termination_reason=termination_reason,
                finish_observed=self.state["agent_system"]["finish_observed"],
                drain_seconds=self.state["agent_system"]["drain_seconds"],
            )
            if not self._publish_staged_result():
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
        self._event("EPISODE_CLOSED", reason=reason, candidate_id=candidate_id)
        self._save_state()

        if candidate_id is None:
            return self._stage_failed_test("no_candidate")

        digest = candidate_id.split(":", 1)[1]
        candidate_dir = self.candidates_dir / digest
        evaluation_started = self.monotonic()
        test_nll: Optional[float] = None
        test_failure: Optional[str] = None
        test_infrastructure_failure = False
        test_teardown_failure: Optional[str] = None
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
        except WorkerTeardownFailure as exc:
            test_teardown_failure = str(exc)
            self._event(
                "TEST_INFRASTRUCTURE_FAILURE",
                candidate_id=candidate_id,
                code="worker_teardown_failure",
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
        except Exception as exc:
            self._event(
                "TEST_INFRASTRUCTURE_FAILURE",
                candidate_id=candidate_id,
                exception_type=type(exc).__name__,
                message=str(exc),
            )
            test_infrastructure_failure = True
        finally:
            self.state["timing"]["test_evaluation_seconds"] += max(
                0.0, self.monotonic() - evaluation_started
            )

        if test_teardown_failure is not None:
            self.deployment_teardown_failed(
                "evaluation_worker_teardown_failure",
                private_detail=test_teardown_failure,
            )
            return self.public_status()

        if test_infrastructure_failure:
            self.infrastructure_failed("test_evaluation_infrastructure_failure")
            return self.public_status()

        if test_failure is not None or test_nll is None:
            return self._stage_failed_test(test_failure or "test_evaluation_failed")

        try:
            test_perplexity = perplexity_from_nll(test_nll)
        except ValueError:
            self._event(
                "TEST_EVALUATION_FAILED",
                candidate_id=candidate_id,
                code="perplexity_overflow",
            )
            return self._stage_failed_test("perplexity_overflow")

        self.state["pending_result"] = {
            "outcome": "success",
            "failure": None,
            "test_nll": test_nll,
            "test_perplexity": test_perplexity,
            "publication_id": uuid.uuid4().hex,
        }
        self._event(
            "TEST_EVALUATED",
            candidate_id=candidate_id,
            nll=test_nll,
            perplexity=test_perplexity,
        )
        self._save_state()
        return self.public_status()

    def _stage_failed_test(self, failure: str) -> Dict[str, Any]:
        """Stage a failed result without manufacturing a missing TEST metric."""

        self.state["pending_result"] = {
            "outcome": "failed",
            "failure": failure,
            "test_nll": None,
            "test_perplexity": None,
            "publication_id": uuid.uuid4().hex,
        }
        self._save_state()
        return self.public_status()

    def _publish_staged_result(self) -> bool:
        """Atomically persist one terminal result after required cleanup."""

        pending = self.state.get("pending_result")
        if self.state["status"] != "closed" or not isinstance(pending, dict):
            return False
        if self.state["agent_system"]["termination_reason"] is None:
            return False
        if self.state["agent_control"]["cleanup_complete"] is not True:
            return False
        if self.state.get("deployment_teardown_failures"):
            return False

        outcome = pending["outcome"]
        failure = pending["failure"]
        publication_id = pending.get("publication_id")
        if not _safe_protocol_text(publication_id):
            # Compatibility for an in-memory staged result created before the
            # publication protocol was introduced.  New staged results persist
            # their stable ID before teardown begins.
            publication_id = uuid.uuid4().hex
            pending["publication_id"] = publication_id

        prepared_event_seq = pending.get("prepared_event_seq")
        if (
            isinstance(prepared_event_seq, bool)
            or not isinstance(prepared_event_seq, int)
            or prepared_event_seq < 1
            or prepared_event_seq > self.sequence
        ):
            prepared_event_seq = self._event(
                "RESULT_PUBLICATION_PREPARED",
                publication_id=publication_id,
                result=outcome,
                reason=failure,
                candidate_id=self.state["final_candidate_id"],
            )
            pending["prepared_event_seq"] = prepared_event_seq

        # Preserve the preparation reference in memory if the authoritative
        # replace fails.  A retry in the same Coordinator process reuses the
        # existing append-only record instead of duplicating it.
        previous_state = copy.deepcopy(self.state)
        self.state["status"] = "finalized"
        self.state["outcome"] = outcome
        self.state["failure"] = failure
        self.state["test_nll"] = pending["test_nll"]
        self.state["test_perplexity"] = pending["test_perplexity"]
        self.state["pending_result"] = None
        self.state["result_publication"] = {
            "publication_id": publication_id,
            "prepared_event_seq": prepared_event_seq,
        }
        try:
            # The append-only record above is explicitly a non-terminal
            # preparation.  state.json becomes the sole commit point by
            # referencing that record.  A failed replace leaves an orphaned
            # preparation, never a ghost terminal event, and sequence numbers
            # remain monotonic.
            self._save_state()
        except BaseException:
            self.state = previous_state
            raise
        return True

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

    def infrastructure_failed(
        self, code: str, *, private_detail: Optional[str] = None
    ) -> None:
        with self.lock:
            status = self.state["status"]
            if status not in {"created", "active", "closed"}:
                return
            lease_teardown_detail: Optional[str] = None
            lease_reconciliation_detail: Optional[str] = None
            if status == "active":
                self._freeze_elapsed()
            if self.state["active_lease"] is not None:
                try:
                    self._end_active_lease(
                        "infrastructure_failure",
                        record_teardown_failure=False,
                    )
                except WorkerTeardownFailure as exc:
                    lease_teardown_detail = str(exc)
                    # A failure transition must still become authoritative when
                    # the Worker cannot confirm lease termination.  Keep the
                    # authoritative state reached so far and retain details only
                    # in the trusted event stream.  Deployment teardown can retry
                    # when the active lease record remains present.
                    self._event(
                        "LEASE_TERMINATION_FAILED",
                        reason="infrastructure_failure",
                        exception_type=type(exc).__name__,
                        message=str(exc),
                    )
                except Exception as exc:
                    # terminate_lease was confirmed and the failure came from
                    # Coordinator reconciliation (event/state persistence).
                    # Do not mislabel it as a Worker resource still running.
                    lease_reconciliation_detail = f"{type(exc).__name__}: {exc}"
            self.state["status"] = "infrastructure_failure"
            self.state["outcome"] = "failed"
            self.state["closure_reason"] = "infrastructure_failure"
            self.state["failure"] = code
            self.state["test_nll"] = None
            self.state["test_perplexity"] = None
            self.state["pending_result"] = None
            event = {"code": code}
            if private_detail is not None:
                event["message"] = private_detail
            if lease_reconciliation_detail is not None:
                event["lease_reconciliation_error"] = lease_reconciliation_detail
            self._event("EPISODE_INFRASTRUCTURE_FAILURE", **event)
            if lease_teardown_detail is not None:
                self._record_deployment_teardown_failure(
                    "research_worker_teardown_failure",
                    private_detail=lease_teardown_detail,
                )
            self._save_state()

    def _record_deployment_teardown_failure(
        self, code: str, *, private_detail: Optional[str] = None
    ) -> None:
        """Mutate teardown evidence without recursively changing lifecycle state."""

        failures = self.state.setdefault("deployment_teardown_failures", [])
        if code not in failures:
            failures.append(code)
        violation = "deployment_teardown_incomplete"
        if violation not in self.state["policy_violations"]:
            self.state["policy_violations"].append(violation)
        if violation not in self.state["leaderboard_eligibility_violations"]:
            self.state["leaderboard_eligibility_violations"].append(violation)
        self.state["policy_compliant"] = False
        self.state["leaderboard_eligible"] = False
        event = {"code": code}
        if private_detail is not None:
            event["message"] = private_detail
        self._event("DEPLOYMENT_TEARDOWN_FAILED", **event)

    def deployment_teardown_failed(
        self, code: str, *, private_detail: Optional[str] = None
    ) -> None:
        """Record trusted teardown failure even after the outcome is terminal."""

        with self.lock:
            self._record_deployment_teardown_failure(
                code, private_detail=private_detail
            )
            if self.state["status"] in {"created", "active", "closed"}:
                self.infrastructure_failed(code)
            else:
                self._save_state()

    def _agent_control_policy_failed(self) -> None:
        """Fail an episode after a dynamic Agent-control policy violation."""

        if self.state["status"] not in {"created", "active", "closed"}:
            return
        if self.state["status"] == "active":
            self._freeze_elapsed()
        if self.state["active_lease"] is not None:
            self._end_active_lease("agent_policy_violation")
        self.state["status"] = "dnf"
        self.state["outcome"] = "failed"
        self.state["closure_reason"] = "agent_policy_violation"
        self.state["failure"] = "agent_control_policy_violation"
        self.state["test_nll"] = None
        self.state["test_perplexity"] = None
        self.state["pending_result"] = None
        self._event("EPISODE_DNF", reason="agent_control_policy_violation")
        self._save_state()

    def cleanup_agent_control(self) -> bool:
        """Clean deployment-owned Agent-control artifacts after Agent stop."""

        if self.state["agent_system"]["termination_reason"] is None:
            raise RuntimeError(
                "Agent termination must be recorded before control cleanup"
            )
        if self.state["agent_control"]["cleanup_complete"] is True:
            self._publish_staged_result()
            return True
        if self.agent_control_transport is None:
            self.state["agent_control"].update(
                cleanup_complete=True,
                cleanup_removed=0,
                cleanup_failures=0,
                cleanup_policy_violations=0,
                cleanup_agent_caused_failures=0,
                cleanup_infrastructure_failures=0,
            )
            self._event(
                "AGENT_CONTROL_CLEANED",
                removed=0,
                failures=0,
                policy_violations=0,
                agent_caused_failures=0,
                infrastructure_failures=0,
            )
            if not self._publish_staged_result():
                self._save_state()
            return True
        try:
            cleanup = self.agent_control_transport.cleanup()
        except Exception as exc:
            self.state["agent_control"].update(
                cleanup_complete=False,
                cleanup_failures=1,
                cleanup_infrastructure_failures=1,
            )
            violation = "agent_control_cleanup_incomplete"
            if violation not in self.state["policy_violations"]:
                self.state["policy_violations"].append(violation)
            if violation not in self.state["leaderboard_eligibility_violations"]:
                self.state["leaderboard_eligibility_violations"].append(violation)
            self.state["policy_compliant"] = False
            self.state["leaderboard_eligible"] = False
            self.deployment_teardown_failed(
                "agent_control_teardown_failure", private_detail=str(exc)
            )
            return False
        cleanup_complete = cleanup.complete
        self.state["agent_control"].update(
            cleanup_complete=cleanup_complete,
            cleanup_removed=cleanup.removed,
            cleanup_failures=cleanup.failures,
            cleanup_policy_violations=cleanup.agent_policy_violations,
            cleanup_agent_caused_failures=cleanup.agent_caused_failures,
            cleanup_infrastructure_failures=cleanup.infrastructure_failures,
        )
        cleanup_policy_failure = bool(
            cleanup.agent_policy_violations or cleanup.agent_caused_failures
        )
        if not cleanup_complete:
            violation = "agent_control_cleanup_incomplete"
            if violation not in self.state["policy_violations"]:
                self.state["policy_violations"].append(violation)
            if violation not in self.state["leaderboard_eligibility_violations"]:
                self.state["leaderboard_eligibility_violations"].append(violation)
            self.state["policy_compliant"] = False
            self.state["leaderboard_eligible"] = False
        if cleanup_policy_failure:
            violation = "agent_control_cleanup_policy_violation"
            if violation not in self.state["policy_violations"]:
                self.state["policy_violations"].append(violation)
            if violation not in self.state["leaderboard_eligibility_violations"]:
                self.state["leaderboard_eligibility_violations"].append(violation)
            self.state["policy_compliant"] = False
            self.state["leaderboard_eligible"] = False
        self._event(
            "AGENT_CONTROL_CLEANED",
            removed=cleanup.removed,
            failures=cleanup.failures,
            policy_violations=cleanup.agent_policy_violations,
            agent_caused_failures=cleanup.agent_caused_failures,
            infrastructure_failures=cleanup.infrastructure_failures,
        )
        if cleanup_policy_failure:
            self._agent_control_policy_failed()
            if not cleanup_complete:
                self.deployment_teardown_failed("agent_control_teardown_failure")
        elif cleanup.infrastructure_failures:
            self.deployment_teardown_failed("agent_control_teardown_failure")
        elif self._publish_staged_result():
            return cleanup_complete
        self._save_state()
        return cleanup_complete

    def process_one_request(self) -> bool:
        if self.agent_control_transport is None:
            return False
        try:
            result = self.agent_control_transport.process_one(self.handle)
        except Exception as exc:
            self.infrastructure_failed(
                "agent_control_transport_failure", private_detail=str(exc)
            )
            return True
        for event in result.events:
            self._event(event.name, **event.fields)
        if result.policy_failure:
            self._agent_control_policy_failed()
        elif result.infrastructure_failure_code is not None:
            self.infrastructure_failed(
                result.infrastructure_failure_code,
                private_detail=result.infrastructure_failure,
            )
        if result.handled or result.events:
            self._save_state()
        return result.handled
