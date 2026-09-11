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
from .artifact_identity import (
    CANDIDATE_TREE_IDENTITY_ALGORITHM,
    FILE_IDENTITY_ALGORITHM,
    HIDDEN_BUNDLE_IDENTITY_ALGORITHM,
)
from .challenge import (
    FORMAL_ARTIFACT_DIGEST_FIELDS,
    ReleaseVerificationReceipt,
    agent_release_manifest,
)
from .config import (
    AGENT_PROTOCOL_VERSION,
    AgentConfig,
    TaskConfig,
    TrainingResources,
    immutable_image_reference,
    operator_research_limit,
)
from .inference import summarize_inference_trace
from .gateway_execution import (
    GatewayExecutionReceipt,
    GatewayExecutionRequest,
    gateway_eligibility_violations,
    inference_policy_sha256,
    manifest_sha256,
)
from .results import (
    RESULT_SCHEMA_VERSION,
    normalize_result_report,
)
from .worker_protocol import (
    EVALUATION_REQUIREMENTS,
    GPU_DEVICE_ISOLATION,
    RESEARCH_REQUIREMENTS,
    CapabilityRequirement,
    CandidateEvaluationResult,
    CommandResult,
    ConformanceReceipt,
    EvaluationWorker,
    EnforcementLevel,
    EvaluationTaskSpec,
    EvaluationResourceFailure,
    assess_evaluation_resources,
    HiddenDatasetIdentityMismatch,
    HiddenDatasetRequest,
    HiddenDatasetResolutionFailure,
    LeaseState,
    ResourceEnvelope,
    ResearchTaskSpec,
    ResearchWorker,
    ResolvedHiddenDataset,
    ResolvedWorkload,
    WorkloadRequest,
    WorkloadResolutionFailure,
    WorkerFailure,
    WorkerTeardownFailure,
    assess_resource_allocation,
    assess_worker,
    v0_resource_envelope,
)


MAX_COMMAND_STREAM_BYTES = 1024 * 1024
MAX_LOG_READ_BYTES = 4 * 1024 * 1024
MAX_COMMAND_LOG_BYTES = 64 * 1024 * 1024
MAX_INCUMBENT_DECLARATIONS = 4096
DEFAULT_WORKER_RESOLUTION_TIMEOUT_SECONDS = 300.0
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
        training_data_verification: Dict[str, Any],
        resource_envelope: Optional[ResourceEnvelope] = None,
        evaluation_resource_envelope: Optional[ResourceEnvelope] = None,
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
        research_worker_conformance: Optional[ConformanceReceipt] = None,
        evaluation_worker_conformance: Optional[ConformanceReceipt] = None,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
        worker_resolution_timeout_seconds: float = (
            DEFAULT_WORKER_RESOLUTION_TIMEOUT_SECONDS
        ),
        operator_max_research_seconds: Optional[float] = None,
    ) -> None:
        normalized = track.upper()
        if normalized not in {"A", "B"}:
            raise ValueError("track must be A or B")
        self.operator_max_research_seconds = operator_research_limit(
            operator_max_research_seconds, normalized
        )
        self.config = config.validate()
        # The trusted launcher verifies the data view and the Worker keeps it
        # immutable. Core binds that evidence to its small manifest identity;
        # it does not inventory or read the dataset files.
        train_manifest_sha256 = _sha256_file(self.config.train_dir / "manifest.json")
        if (
            not isinstance(training_data_verification, dict)
            or training_data_verification.get("status") != "verified"
            or train_manifest_sha256 is None
            or training_data_verification.get("manifest_sha256")
            != train_manifest_sha256
        ):
            raise ValueError(
                "training data verification must bind the current TRAIN manifest"
            )
        training_data_verification = copy.deepcopy(training_data_verification)
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
        self.resource_envelope = resource_envelope or v0_resource_envelope(
            self.resources,
            minimum_enforcement=(
                EnforcementLevel.ENFORCED
                if config.leaderboard_eligible
                else EnforcementLevel.BEST_EFFORT
            ),
        )
        if not isinstance(self.resource_envelope, ResourceEnvelope):
            raise TypeError("resource_envelope must be a ResourceEnvelope")
        if evaluation_resource_envelope is not None and not isinstance(
            evaluation_resource_envelope, ResourceEnvelope
        ):
            raise TypeError("evaluation_resource_envelope must be a ResourceEnvelope")
        release_evaluation_envelope = (
            challenge_release.get("evaluation_resource_envelope")
            if isinstance(challenge_release, dict)
            else None
        )
        frozen_evaluation_envelope = None
        if release_evaluation_envelope is not None:
            frozen_evaluation_envelope = ResourceEnvelope.from_manifest(
                release_evaluation_envelope
            )
            if evaluation_resource_envelope is None:
                evaluation_resource_envelope = frozen_evaluation_envelope
            elif (
                evaluation_resource_envelope.sha256 != frozen_evaluation_envelope.sha256
            ):
                raise ValueError(
                    "evaluation resource envelope differs from the release"
                )
        self.evaluation_resource_envelope = evaluation_resource_envelope
        self.track = normalized
        if (
            not math.isfinite(worker_resolution_timeout_seconds)
            or worker_resolution_timeout_seconds <= 0
        ):
            raise ValueError("worker resolution timeout must be finite and positive")
        self.worker_resolution_timeout_seconds = float(
            worker_resolution_timeout_seconds
        )
        self.episode_dir = episode_dir.resolve()
        self.research_worker = research_worker
        self.evaluation_worker = evaluation_worker
        for receipt, field_name in (
            (research_worker_conformance, "research_worker_conformance"),
            (evaluation_worker_conformance, "evaluation_worker_conformance"),
        ):
            if receipt is not None and not isinstance(receipt, ConformanceReceipt):
                raise TypeError(f"{field_name} must be a ConformanceReceipt")
        self.research_worker_conformance = research_worker_conformance
        self.evaluation_worker_conformance = evaluation_worker_conformance
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
                artifact_identity_algorithms=dict(
                    verified_release.artifact_identity_algorithms
                ),
                workload_resolution_acceptance=dict(
                    verified_release.workload_resolution_acceptance
                ),
                inference_gateway_artifact=verified_release.inference_gateway_artifact,
            )
        self.monotonic = monotonic
        self.wall_time = wall_time
        self.lock = threading.RLock()
        self.started_monotonic: Optional[float] = None
        self.closed_monotonic: Optional[float] = None
        self._resolved_workloads: Dict[str, ResolvedWorkload] = {}

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
        if not (self.agent_api_dir / "benchcore" / "client.py").is_file():
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
        self.research_worker_id = research_descriptor.worker_id
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
        research_assessment = assess_worker(
            research_descriptor,
            research_requirements,
            conformance_receipt=self.research_worker_conformance,
        )
        evaluation_assessment = assess_worker(
            evaluation_descriptor,
            evaluation_requirements,
            conformance_receipt=self.evaluation_worker_conformance,
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
            "train_manifest_sha256": train_manifest_sha256,
            # Hidden identities are deliberately resolved only after the
            # research phase is irreversibly sealed. A same-host deployment
            # may keep the bytes physically absent until then.
            "analysis_sha256": None,
            "test_sha256": None,
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
            release_resource_envelope = challenge_release.get("resource_envelope")
            if release_resource_envelope is None:
                eligibility_violations.append("resource_envelope_not_frozen")
            elif release_resource_envelope != self.resource_envelope.public_manifest():
                eligibility_violations.append("resource_envelope_mismatch")
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
        # The launcher supplies actual Gateway evidence after startup and before
        # begin(). A configured image is never proof of an executed artifact.
        if config.leaderboard_eligible and agent.inference is not None:
            eligibility_violations.extend(
                gateway_eligibility_violations(
                    None,
                    self.release_verification.inference_gateway_artifact
                    if self.release_verification is not None
                    else None,
                )
            )
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
                if field in {"analysis_sha256", "test_sha256"}:
                    continue
                if receipt.artifact_sha256[field] != artifact_identities.get(field):
                    eligibility_violations.append(
                        f"formal_release_verification_artifact_mismatch:{field}"
                    )
            # Hidden bytes may be absent until research teardown.  TEST is the
            # primary leaderboard measurement, so eligibility remains
            # provisional only until the TEST identity is resolved and
            # matched.  ANALYSIS identity/completeness is reported on its own
            # diagnostic axis and never gates an otherwise valid TEST result.
            eligibility_violations.append("formal_test_evaluation_pending")
        if config.leaderboard_eligible:
            if (
                self.evaluation_resource_envelope is None
                or not self.evaluation_resource_envelope.requirements
                or any(
                    item.minimum_enforcement is not EnforcementLevel.ENFORCED
                    for item in self.evaluation_resource_envelope.requirements
                )
            ):
                eligibility_violations.append("evaluation_resource_envelope_not_formal")
            if self.evaluation_resource_envelope is not None and (
                frozen_evaluation_envelope is None
                or frozen_evaluation_envelope.sha256
                != self.evaluation_resource_envelope.sha256
            ):
                eligibility_violations.append("evaluation_resource_envelope_not_frozen")
            if not self.resource_envelope.requirements or any(
                requirement.minimum_enforcement is not EnforcementLevel.ENFORCED
                for requirement in self.resource_envelope.requirements
            ):
                eligibility_violations.append("resource_envelope_not_formal")
            eligibility_violations.extend(
                f"workload_resolution_pending:{role}"
                for role in ("training", "candidate", "evaluator")
            )
            eligibility_violations.append("resource_allocation_pending")
            eligibility_violations.append("evaluation_resource_allocation_pending")

        leaderboard_eligible = (
            config.leaderboard_eligible
            and policy_compliant
            and research_assessment.leaderboard_eligible
            and evaluation_assessment.leaderboard_eligible
            and not eligibility_violations
        )

        episode_id = "episode:" + uuid.uuid4().hex
        self._gateway_execution_request = None
        if self.agent.inference is not None:
            inference = self.agent.inference
            # Hash private registration values, never credential contents. Do
            # not publish even this fingerprint to the Tested Agent.
            self._gateway_execution_request = GatewayExecutionRequest(
                episode_id=episode_id,
                requested_identity=config.inference_gateway_image,
                core_revision=core_identity.get("revision"),
                inference_policy_sha256=inference_policy_sha256(
                    inference.protocol,
                    inference.allowed_models,
                    inference.allowed_paths,
                ),
                inference_registration_sha256=manifest_sha256(
                    {
                        "protocol": inference.protocol,
                        "base_url": inference.base_url,
                        "api_key_env": inference.api_key_env,
                        "allowed_models": sorted(inference.allowed_models),
                        "allowed_paths": sorted(inference.allowed_paths),
                        "timeout_seconds": inference.timeout_seconds,
                    }
                ),
            )
        self.state: Dict[str, Any] = {
            "protocol_version": AGENT_PROTOCOL_VERSION,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            # A deployment path is not a semantic protocol identifier.  In
            # particular, an operator-selected directory may contain spaces or
            # Unicode that are valid on the host but invalid in opaque Worker
            # receipt IDs.  Keep the protocol identity stable and path-free.
            "episode_id": episode_id,
            "gateway_execution": (
                {
                    "request": self._gateway_execution_request.private_manifest(),
                    "receipt": None,
                }
                if self._gateway_execution_request is not None
                else None
            ),
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
            "training_data_verification": training_data_verification,
            "artifact_identities": {
                "core_git": core_identity,
                "challenge_manifest_sha256": artifact_identities[
                    "challenge_manifest_sha256"
                ],
                "task_manifest_sha256": artifact_identities["task_manifest_sha256"],
                "policy_sha256": artifact_identities["policy_sha256"],
                "evaluator_sha256": artifact_identities["evaluator_sha256"],
                "train_manifest_sha256": artifact_identities["train_manifest_sha256"],
                "hidden_evaluation": {
                    "analysis_sha256": artifact_identities["analysis_sha256"],
                    "test_sha256": artifact_identities["test_sha256"],
                },
            },
            "artifact_identity_algorithms": {
                "candidate_tree": CANDIDATE_TREE_IDENTITY_ALGORITHM,
                "file": FILE_IDENTITY_ALGORITHM,
                "hidden_evaluation": HIDDEN_BUNDLE_IDENTITY_ALGORITHM,
            },
            "workload_resolutions": {
                "training": None,
                "candidate": None,
                "evaluator": None,
            },
            "resource_envelope": self.resource_envelope.public_manifest(),
            "resource_allocations": [],
            "evaluation_resource_envelope": (
                self.evaluation_resource_envelope.public_manifest()
                if self.evaluation_resource_envelope is not None
                else None
            ),
            "evaluation_resource_allocations": [],
            "policy_compliant": policy_compliant,
            "policy_violations": policy_violations,
            "workers": {
                "research": {
                    **research_descriptor.public_manifest(),
                    "conformance": (
                        self.research_worker_conformance.public_manifest()
                        if self.research_worker_conformance
                        else None
                    ),
                    "assessment": research_assessment.public_manifest(),
                    "environment": dict(research_worker_environment or {}),
                },
                "evaluation": {
                    **evaluation_descriptor.public_manifest(),
                    "conformance": (
                        self.evaluation_worker_conformance.public_manifest()
                        if self.evaluation_worker_conformance
                        else None
                    ),
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
                "candidate_channel": self.evaluation_task.candidate_channel.public_manifest(),
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
                "benchmark_overhead_seconds": 0.0,
                "training_lease_seconds": 0.0,
                "analysis_evaluation_seconds": 0.0,
                "test_evaluation_seconds": 0.0,
            },
            # Trusted audit only; never copied into the public task or status.
            "operator_limits": {
                "max_research_seconds": self.operator_max_research_seconds,
                "clock": "monotonic_since_task_reveal_without_refunds",
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
            "candidate_commits": [],
            "incumbent_declarations": [],
            "incumbent_declaration_limit": MAX_INCUMBENT_DECLARATIONS,
            "incumbent_id": None,
            "final_candidate_id": None,
            "research_sealed_at": None,
            "analysis_complete": None,
            "analysis_status": None,
            "analysis_failure": None,
            "analysis_identity_verified": None,
            "analysis_policy": {
                "candidate_selection": config.analysis_candidate_selection,
                "max_candidates": config.max_analysis_candidates,
                "result_visibility": config.analysis_result_visibility,
            },
            "analysis_trajectory": [],
            "analysis_selected_points": 0,
            "analysis_evaluated_points": 0,
            "test_status": None,
            "performance_score_contract": config.performance_score.public_manifest(),
            "cost_contract": config.cost.public_manifest(),
            "test_performance_score": None,
            "test_raw_metrics": None,
            "test_cost": None,
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
        public_manifest["training_environment"]["resource_envelope"] = (
            self.resource_envelope.public_manifest()
        )
        public_manifest["policy"] = {
            "compliant": policy_compliant,
            "violations": policy_violations,
        }
        public_manifest["leaderboard_eligible"] = leaderboard_eligible
        public_manifest["candidate_artifact_identity"] = (
            "autotrain-bench-tree-v2-sha256"
        )
        public_manifest["incumbent_declaration_limit"] = MAX_INCUMBENT_DECLARATIONS
        public_manifest["formal_release_verified"] = (
            self.release_verification is not None
            and not any(
                violation == "formal_test_evaluation_pending"
                or violation.startswith("formal_release_verification")
                for violation in eligibility_violations
            )
        )
        public_manifest["comparison_key"] = self.state["comparison_key"]
        if challenge_release is not None:
            public_manifest["challenge_release"] = agent_release_manifest(
                challenge_release
            )
        self.public_task_path.write_text(
            json.dumps(public_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._save_state()

    def _refresh_leaderboard_eligibility(self) -> None:
        research = self.state["workers"]["research"]["assessment"]
        evaluation = self.state["workers"]["evaluation"]["assessment"]
        self.state["leaderboard_eligible"] = bool(
            self.config.leaderboard_eligible
            and self.state["policy_compliant"]
            and research["leaderboard_eligible"]
            and evaluation["leaderboard_eligible"]
            and not self.state["leaderboard_eligibility_violations"]
        )

    @property
    def gateway_execution_request(self) -> Optional[GatewayExecutionRequest]:
        """Private, immutable request for the trusted Gateway launcher."""
        return self._gateway_execution_request

    def record_gateway_execution(self, receipt: GatewayExecutionReceipt) -> None:
        """Freeze actual launcher evidence once, before task reveal.

        The caller is the trusted composition, never the Tested Agent. Starting,
        inspecting, enforcing and stopping the Gateway remain deployment-owned.
        """
        with self.lock:
            if self.state["status"] != "created":
                raise ProtocolError(
                    "invalid_state", "Gateway evidence must precede task reveal"
                )
            request = self._gateway_execution_request
            if request is None:
                raise ValueError("Gateway inference is not enabled")
            if not isinstance(receipt, GatewayExecutionReceipt):
                raise TypeError("receipt must be a GatewayExecutionReceipt")
            if self.state["gateway_execution"]["receipt"] is not None:
                raise ValueError("Gateway execution evidence is already fixed")
            receipt.validate_request(request)
            self.state["gateway_execution"]["receipt"] = receipt.private_manifest()
            if self.config.leaderboard_eligible:
                violations = self.state["leaderboard_eligibility_violations"]
                violations[:] = [
                    item
                    for item in violations
                    if not item.startswith("inference_gateway_")
                ]
                violations.extend(
                    gateway_eligibility_violations(
                        receipt,
                        self.release_verification.inference_gateway_artifact
                        if self.release_verification is not None
                        else None,
                    )
                )
                self._refresh_leaderboard_eligibility()
            self._event(
                "INFERENCE_GATEWAY_EXECUTION_RECORDED", receipt_id=receipt.receipt_id
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

    def _benchmark_overhead(self) -> float:
        """Wall clock spent on benchmark-owned preparation, not on the Agent.

        `docs/benchmark-reference.md` places "benchmark infrastructure startup"
        outside participant timing. Worker Lease materialization -- scheduler
        queueing, image conversion, workload startup -- is that preparation,
        and its cost varies by deployment, so billing it to the participant
        would make the same budget mean different things at different sites.
        """

        return float(self.state["timing"]["benchmark_overhead_seconds"])

    def _record_benchmark_overhead(self, phase: str, since_monotonic: float) -> None:
        """Refund a phase only when the release declares it benchmark-owned."""

        if phase not in self.config.benchmark_owned_preparation:
            return
        self.state["timing"]["benchmark_overhead_seconds"] += max(
            0.0, self.monotonic() - since_monotonic
        )

    def _elapsed(self) -> float:
        if self.started_monotonic is None:
            return 0.0
        end = (
            self.closed_monotonic
            if self.closed_monotonic is not None
            else self.monotonic()
        )
        return max(0.0, end - self.started_monotonic - self._benchmark_overhead())

    def _freeze_elapsed(self, at_monotonic: Optional[float] = None) -> None:
        if self.closed_monotonic is None:
            self.closed_monotonic = (
                self.monotonic() if at_monotonic is None else at_monotonic
            )

    def _track_a_deadline_monotonic(self) -> float:
        if self.started_monotonic is None:
            raise ProtocolError("invalid_state", "episode has not started")
        return (
            self.started_monotonic
            + self.config.track_a_total_seconds
            + self._benchmark_overhead()
        )

    def _track_a_remaining(self) -> float:
        return max(0.0, self.config.track_a_total_seconds - self._elapsed())

    def _close_at_track_a_deadline(self) -> Dict[str, Any]:
        """Irreversibly seal Track A at its fixed participant deadline."""

        self._freeze_elapsed(self._track_a_deadline_monotonic())
        if self.state["active_lease"] is not None:
            self._end_active_lease("track_a_deadline")
        return self._seal_research(
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
        result: Dict[str, Any] = {
            "official_elapsed_seconds": round(elapsed, 6),
            "benchmark_overhead_seconds": round(self._benchmark_overhead(), 6),
            "training_lease_seconds": round(lease_seconds, 6),
            "agent_and_worker_seconds": round(max(0.0, elapsed - lease_seconds), 6),
            "analysis_evaluation_seconds": round(
                float(self.state["timing"]["analysis_evaluation_seconds"]), 6
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
            "timestamp": datetime.fromtimestamp(
                self.wall_time(), tz=timezone.utc
            ).isoformat(),
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
        declarations = self.state.get("incumbent_declarations")
        declaration_limit = self.state.get("incumbent_declaration_limit")
        if (
            isinstance(declaration_limit, bool)
            or not isinstance(declaration_limit, int)
            or declaration_limit < 1
        ):
            raise RuntimeError("invalid incumbent declaration limit")
        if not isinstance(declarations, list) or len(declarations) > declaration_limit:
            raise RuntimeError("incumbent declaration history exceeds its limit")
        termination = self.state["agent_system"]["termination_reason"]
        cleanup_complete = self.state["agent_control"]["cleanup_complete"]
        outcome = self.state["outcome"]
        result_publication = self.state.get("result_publication")
        if status in {"created", "active", "closed"} and outcome is not None:
            raise RuntimeError("a non-terminal episode cannot publish an outcome")
        if status in {"dnf", "infrastructure_failure"} and outcome != "failed":
            raise RuntimeError("a failure terminal must publish a failed outcome")
        if (status == "operator_stopped") != (outcome == "aborted"):
            raise RuntimeError("only an operator stop may have an aborted outcome")
        if status == "operator_stopped" and (
            self.state.get("research_sealed_at") is None
            or self.state.get("test_status") is not None
            or self.state.get("analysis_status") is not None
            or self.state.get("final_candidate_id") is not None
            or self.state.get("leaderboard_eligible") is not False
        ):
            raise RuntimeError("an operator stop must seal research without evaluation")
        if status == "finalized" and outcome not in {"success", "failed"}:
            raise RuntimeError("a finalized episode requires an outcome")
        if status != "finalized" and result_publication is not None:
            raise RuntimeError("only a finalized episode may publish a result")
        if self.state.get("pending_result") is not None and status != "closed":
            raise RuntimeError("a staged TEST result requires the closed state")
        sealed_at = self.state.get("research_sealed_at")
        if status in {"created", "active"} and sealed_at is not None:
            raise RuntimeError("active research cannot already be sealed")
        if status in {"closed", "finalized"} and not _safe_protocol_text(sealed_at):
            raise RuntimeError("closed research requires a sealing timestamp")
        analysis_complete = self.state.get("analysis_complete")
        if analysis_complete is not None and not isinstance(analysis_complete, bool):
            raise RuntimeError("analysis_complete must be null or boolean")
        if analysis_complete is not None and status in {"created", "active"}:
            raise RuntimeError("ANALYSIS state requires sealed research")
        analysis_status = self.state.get("analysis_status")
        if analysis_status not in {None, "pending", "complete", "incomplete"}:
            raise RuntimeError("invalid ANALYSIS lifecycle status")
        if analysis_complete is True and analysis_status != "complete":
            raise RuntimeError("completed ANALYSIS requires complete status")
        if analysis_status == "complete" and analysis_complete is not True:
            raise RuntimeError("complete ANALYSIS status requires completion")
        analysis_failure = self.state.get("analysis_failure")
        if analysis_status == "incomplete":
            if analysis_complete is not False:
                raise RuntimeError("incomplete ANALYSIS cannot be complete")
            if not _safe_protocol_text(analysis_failure):
                raise RuntimeError("incomplete ANALYSIS requires a failure code")
        elif analysis_failure is not None:
            raise RuntimeError("only incomplete ANALYSIS may have a failure code")
        analysis_identity_verified = self.state.get("analysis_identity_verified")
        if analysis_identity_verified is not None and not isinstance(
            analysis_identity_verified, bool
        ):
            raise RuntimeError("ANALYSIS identity evidence must be null or boolean")
        test_status = self.state.get("test_status")
        if test_status not in {None, "pending", "evaluated", "failed"}:
            raise RuntimeError("invalid TEST lifecycle status")
        if self.state.get("pending_result") is not None and test_status not in {
            "evaluated",
            "failed",
        }:
            raise RuntimeError("a staged result requires a completed TEST decision")
        if outcome in {"failed", "aborted"} and (
            self.state.get("test_performance_score") is not None
            or self.state.get("test_raw_metrics") is not None
            or self.state.get("test_cost") is not None
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
            if analysis_status not in {"complete", "incomplete"}:
                raise RuntimeError(
                    "a finalized result requires an ANALYSIS disposition"
                )
            if outcome == "success" and test_status != "evaluated":
                raise RuntimeError("a successful result requires evaluated TEST")
            if outcome == "failed" and test_status != "failed":
                raise RuntimeError("a failed result requires failed TEST")
        # Keep persisted state and the public result contract on one set of
        # structural invariants.  The normalizer works on a copy and removes
        # private staged TEST data before validating it.
        normalize_result_report(self.state)

    def _require_active(self) -> None:
        self.enforce_operator_limit()
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
                "candidate_commits": list(self.state["candidate_commits"]),
                "incumbent_declarations": list(self.state["incumbent_declarations"]),
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
                status["research_sealed_at"] = self.state["research_sealed_at"]
                status["analysis_complete"] = self.state["analysis_complete"]
                status["analysis_status"] = self.state["analysis_status"]
                status["analysis_failure"] = self.state["analysis_failure"]
                status["analysis_identity_verified"] = self.state[
                    "analysis_identity_verified"
                ]
                status["analysis_trajectory"] = list(self.state["analysis_trajectory"])
                status["analysis_selected_points"] = self.state[
                    "analysis_selected_points"
                ]
                status["analysis_evaluated_points"] = self.state[
                    "analysis_evaluated_points"
                ]
                status["final_candidate_id"] = self.state["final_candidate_id"]
                status["test_status"] = self.state["test_status"]
                status["test_performance_score"] = self.state["test_performance_score"]
                status["test_raw_metrics"] = self.state["test_raw_metrics"]
                status["test_cost"] = self.state["test_cost"]
                status["failure"] = self.state["failure"]
            return status

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        self.enforce_operator_limit()
        if self.state["status"] == "operator_stopped":
            raise ProtocolError("episode_closed", "the episode is closed")
        try:
            result = self._handle_request(request)
        finally:
            # A Worker call can consume the remaining operator allowance.
            # Never return its late acknowledgement as continued research.
            self.enforce_operator_limit()
        if self.state["status"] == "operator_stopped":
            raise ProtocolError("episode_closed", "the episode is closed")
        return result

    def _handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
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

    def _workload_request(self, role: str) -> WorkloadRequest:
        requested = {
            "training": self.config.training_image,
            "candidate": self.config.candidate_image,
            "evaluator": self.config.evaluator_image,
        }[role]
        return WorkloadRequest(role=role, requested_identity=requested)

    @staticmethod
    def _identity_workload_resolution(
        request: WorkloadRequest,
        resolved: ResolvedWorkload,
    ) -> bool:
        """Return whether the Worker executes the canonical OCI identity itself."""

        if not immutable_image_reference(request.requested_identity):
            return False
        requested_sha256 = request.requested_identity.rsplit("@sha256:", 1)[1]
        return bool(
            resolved.effective_format == "oci"
            and resolved.effective_sha256 == requested_sha256
            and resolved.transformation_sha256 is None
        )

    def _resolve_workload(
        self,
        role: str,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> ResolvedWorkload:
        """Resolve and accept one workload before passing it to an operation.

        The returned opaque handle is the only workload authorization accepted
        by Worker Protocol v14 operations. Core records semantic evidence but
        never learns the Worker's path, scheduler job, image cache, or runtime
        handle.
        """

        effective_timeout = (
            self.worker_resolution_timeout_seconds
            if timeout_seconds is None
            else float(timeout_seconds)
        )
        if not math.isfinite(effective_timeout) or effective_timeout <= 0:
            raise WorkloadResolutionFailure("workload resolution deadline reached")
        cached = self._resolved_workloads.get(role)
        if cached is not None:
            return cached
        request = self._workload_request(role)
        worker = self.research_worker if role == "training" else self.evaluation_worker
        worker_scope = "research" if role == "training" else "evaluation"
        resolution_deadline = self.monotonic() + effective_timeout
        try:
            resolved = worker.resolve_workload(
                episode_id=self.state["episode_id"],
                workload=request,
                timeout_seconds=effective_timeout,
            )
            if self.monotonic() >= resolution_deadline:
                raise WorkloadResolutionFailure(
                    f"{role} workload resolution deadline reached"
                )
        except WorkloadResolutionFailure as exc:
            self._event(
                "WORKLOAD_RESOLUTION_FAILED",
                role=role,
                code="workload_resolution_failure",
                message=str(exc),
            )
            self._save_state()
            raise
        except Exception as exc:
            self._event(
                "WORKLOAD_RESOLUTION_FAILED",
                role=role,
                code="workload_resolver_state_unknown",
                exception_type=type(exc).__name__,
                message=str(exc),
            )
            self._save_state()
            raise WorkerTeardownFailure(
                f"{role} workload resolver state is unknown: {exc}"
            ) from exc

        if not isinstance(resolved, ResolvedWorkload):
            detail = (
                f"{worker_scope} Worker returned {type(resolved).__name__}, "
                "expected ResolvedWorkload"
            )
            self._event(
                "WORKLOAD_RESOLUTION_FAILED",
                role=role,
                code="workload_resolution_receipt_invalid",
                message=detail,
            )
            self._save_state()
            raise WorkloadResolutionFailure(detail)
        if (
            resolved.role != request.role
            or resolved.requested_identity != request.requested_identity
        ):
            detail = f"{worker_scope} Worker resolved a different workload request"
            self._event(
                "WORKLOAD_RESOLUTION_FAILED",
                role=role,
                code="workload_resolution_receipt_invalid",
                message=detail,
            )
            self._save_state()
            raise WorkloadResolutionFailure(detail)

        if self._identity_workload_resolution(request, resolved):
            acceptance = "canonical_identity"
        else:
            accepted = (
                self.release_verification.workload_resolution_acceptance.get(role)
                if self.release_verification is not None
                else None
            )
            acceptance = (
                "organizer_receipt"
                if accepted is not None and accepted.matches(resolved)
                else "unverified"
            )

        evidence = {
            **resolved.evidence_manifest(),
            "worker_id": self.state["workers"][worker_scope]["worker_id"],
            "acceptance": acceptance,
        }
        self.state["workload_resolutions"][role] = evidence
        pending = f"workload_resolution_pending:{role}"
        if pending in self.state["leaderboard_eligibility_violations"]:
            self.state["leaderboard_eligibility_violations"].remove(pending)
        if self.config.leaderboard_eligible and acceptance == "unverified":
            violation = f"workload_resolution_not_accepted:{role}"
            if violation not in self.state["leaderboard_eligibility_violations"]:
                self.state["leaderboard_eligibility_violations"].append(violation)
            self.state["leaderboard_eligible"] = False
            self._event(
                "WORKLOAD_RESOLUTION_REJECTED",
                role=role,
                requested_identity=request.requested_identity,
                effective_format=resolved.effective_format,
                effective_sha256=resolved.effective_sha256,
            )
            self._save_state()
            raise WorkloadResolutionFailure(
                f"{role} workload resolution is not accepted by the release"
            )

        self._resolved_workloads[role] = resolved
        self._refresh_leaderboard_eligibility()
        self._event(
            "WORKLOAD_RESOLVED",
            role=role,
            requested_identity=request.requested_identity,
            effective_format=resolved.effective_format,
            effective_sha256=resolved.effective_sha256,
            transformation_sha256=resolved.transformation_sha256,
            acceptance=acceptance,
        )
        self._save_state()
        return resolved

    def _reject_started_lease_allocation(
        self,
        *,
        lease_id: str,
        lease_dir: Path,
        failure_code: str,
        private_detail: str,
    ) -> None:
        """Revoke a workload whose allocation receipt cannot be accepted."""

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
            self._event(
                "LEASE_INFRA_FAILURE",
                lease_id=lease_id,
                code="resource_allocation_revoke_failure",
            )
            self.deployment_teardown_failed(
                "research_worker_teardown_failure",
                private_detail=str(teardown_exc),
            )
            raise ProtocolError(
                "infrastructure_failure",
                "resource allocation rejection could not revoke the workload",
            ) from exc

        pending = self.state["pending_lease_teardowns"]
        if lease_id in pending:
            pending.remove(lease_id)
        try:
            shutil.rmtree(lease_dir)
        except FileNotFoundError:
            pass
        except OSError as exc:
            self.deployment_teardown_failed(
                "research_lease_directory_cleanup_failure",
                private_detail=str(exc),
            )
            raise ProtocolError(
                "infrastructure_failure",
                "resource allocation rejection cleanup failed",
            ) from exc
        self.infrastructure_failed(failure_code, private_detail=private_detail)
        raise ProtocolError(
            "infrastructure_failure",
            "training resource allocation evidence was rejected",
        )

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

            worker_call_started = self.monotonic()
            try:
                training_workload = self._resolve_workload("training")
            except WorkerTeardownFailure as exc:
                self._record_benchmark_overhead(
                    "worker_lease_startup", worker_call_started
                )
                self.deployment_teardown_failed(
                    "research_worker_teardown_failure",
                    private_detail=str(exc),
                )
                raise ProtocolError(
                    "infrastructure_failure",
                    "training workload resolution left unknown Worker state",
                ) from exc
            except WorkloadResolutionFailure as exc:
                self._record_benchmark_overhead(
                    "worker_lease_startup", worker_call_started
                )
                self.infrastructure_failed(
                    "training_workload_resolution_failure",
                    private_detail=str(exc),
                )
                raise ProtocolError(
                    "infrastructure_failure",
                    "training workload resolution failed",
                ) from exc

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
            resource_pending = "resource_allocation_pending"
            if (
                self.config.leaderboard_eligible
                and resource_pending
                not in self.state["leaderboard_eligibility_violations"]
            ):
                self.state["leaderboard_eligibility_violations"].append(
                    resource_pending
                )
                self._refresh_leaderboard_eligibility()
            self._save_state()
            try:
                allocation_receipt = self.research_worker.start_lease(
                    episode_id=self.state["episode_id"],
                    lease_id=lease_id,
                    duration_seconds=duration,
                    task=self.research_task,
                    workspace=self.agent.workspace,
                    resources=self.resources,
                    resource_envelope=self.resource_envelope,
                    lease_dir=lease_dir,
                    workload=training_workload,
                )
            except WorkerTeardownFailure as exc:
                self._record_benchmark_overhead(
                    "worker_lease_startup", worker_call_started
                )
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
                self._record_benchmark_overhead(
                    "worker_lease_startup", worker_call_started
                )
                pending.remove(lease_id)
                if resource_pending in self.state["leaderboard_eligibility_violations"]:
                    self.state["leaderboard_eligibility_violations"].remove(
                        resource_pending
                    )
                    self._refresh_leaderboard_eligibility()
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

            try:
                assessment = assess_resource_allocation(
                    self.resource_envelope,
                    allocation_receipt,
                    episode_id=self.state["episode_id"],
                    lease_id=lease_id,
                    worker_id=self.research_worker_id,
                )
            except (TypeError, ValueError) as exc:
                self._record_benchmark_overhead(
                    "worker_lease_startup", worker_call_started
                )
                self._reject_started_lease_allocation(
                    lease_id=lease_id,
                    lease_dir=lease_dir,
                    failure_code="resource_allocation_receipt_invalid",
                    private_detail=f"{type(exc).__name__}: {exc}",
                )
                raise AssertionError("unreachable")

            allocation_record = {
                "receipt": allocation_receipt.public_manifest(),
                "assessment": assessment.public_manifest(),
                "accepted": assessment.satisfied,
            }
            self.state["resource_allocations"].append(allocation_record)
            self._event(
                "RESOURCE_ALLOCATION_RECORDED",
                lease_id=lease_id,
                receipt_id=allocation_receipt.receipt_id,
                resource_request_sha256=self.resource_envelope.sha256,
                accepted=assessment.satisfied,
            )
            if not assessment.satisfied:
                self._record_benchmark_overhead(
                    "worker_lease_startup", worker_call_started
                )
                self._reject_started_lease_allocation(
                    lease_id=lease_id,
                    lease_dir=lease_dir,
                    failure_code="resource_allocation_mismatch",
                    private_detail=json.dumps(
                        assessment.public_manifest(),
                        sort_keys=True,
                    ),
                )
                raise AssertionError("unreachable")

            self._record_benchmark_overhead("worker_lease_startup", worker_call_started)
            started = self.monotonic()
            if self.track == "A":
                # Defence in depth: whatever else consumed budget between the
                # admission check and here, never promise more than remains.
                duration = min(duration, self._track_a_remaining())
            self.state["leases_used"] += 1
            self.state["active_lease"] = {
                "lease_id": lease_id,
                "started_monotonic": started,
                "deadline_monotonic": started + duration,
                "duration_seconds": duration,
                "exec_count": 0,
                "resource_allocation_receipt_id": allocation_receipt.receipt_id,
            }
            pending.remove(lease_id)
            if resource_pending in self.state["leaderboard_eligibility_violations"]:
                self.state["leaderboard_eligibility_violations"].remove(
                    resource_pending
                )
                self._refresh_leaderboard_eligibility()
            self.state["last_lease_id"] = lease_id
            self._event("LEASE_STARTED", lease_id=lease_id, duration_seconds=duration)
            self._save_state()
            return {
                "lease_id": lease_id,
                "duration_seconds": duration,
                "leases_used": self.state["leases_used"],
                "resource_allocation_verified": True,
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
            return {"lease_id": lease_id, "candidate_id": None}

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
            created = False
            if not destination.exists():
                os.replace(str(stage), str(destination))
                created = True
            return {
                "candidate_id": candidate_id,
                "path": destination,
                "created": created,
            }
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
            if (
                len(self.state["candidate_commits"])
                >= self.config.max_analysis_candidates
            ):
                self._event(
                    "CANDIDATE_REJECTED",
                    lease_id=lease_id,
                    code="candidate_limit_reached",
                )
                self._save_state()
                raise ProtocolError(
                    "candidate_limit_reached",
                    "the release Candidate commit limit has been reached",
                )
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
            existing = self.state["candidates"].get(candidate_id)
            if existing is None:
                metadata = {
                    "candidate_id": candidate_id,
                    "source_leases": [lease_id],
                    "declared_incumbent": False,
                    "analysis_status": "pending",
                    "analysis_performance_score": None,
                    "analysis_raw_metrics": None,
                    "analysis_cost": None,
                    "analysis_error": None,
                }
                self.state["candidates"][candidate_id] = metadata
            else:
                metadata = existing
                if lease_id not in metadata["source_leases"]:
                    metadata["source_leases"].append(lease_id)

            commit_index = len(self.state["candidate_commits"]) + 1
            committed_at = datetime.fromtimestamp(
                self.wall_time(), tz=timezone.utc
            ).isoformat()
            commit_event_seq = self._event(
                "CANDIDATE_COMMITTED",
                lease_id=lease_id,
                candidate_id=candidate_id,
                commit_index=commit_index,
                timestamp=committed_at,
            )
            commit = {
                "commit_index": commit_index,
                "event_seq": commit_event_seq,
                "candidate_id": candidate_id,
                "source_lease": lease_id,
                "timestamp": committed_at,
                "elapsed_seconds": round(self._elapsed(), 6),
            }
            self.state["candidate_commits"].append(commit)
            self._save_state()
            return dict(commit)

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
            if (
                len(self.state["incumbent_declarations"])
                >= self.state["incumbent_declaration_limit"]
            ):
                raise ProtocolError(
                    "incumbent_declaration_limit_reached",
                    "the episode incumbent declaration limit has been reached",
                )
            previous_id = self.state["incumbent_id"]
            self.state["incumbent_id"] = candidate_id
            candidate["declared_incumbent"] = True
            declaration_index = len(self.state["incumbent_declarations"]) + 1
            declared_at = datetime.fromtimestamp(
                self.wall_time(), tz=timezone.utc
            ).isoformat()
            declaration_event_seq = self._event(
                "INCUMBENT_DECLARED",
                declaration_index=declaration_index,
                candidate_id=candidate_id,
                previous_candidate_id=previous_id,
                source_leases=list(candidate["source_leases"]),
                leases_used=self.state["leases_used"],
                timestamp=declared_at,
            )
            declaration = {
                "declaration_index": declaration_index,
                "event_seq": declaration_event_seq,
                "candidate_id": candidate_id,
                "previous_candidate_id": previous_id,
                "source_leases": list(candidate["source_leases"]),
                "leases_used": self.state["leases_used"],
                "timestamp": declared_at,
                "elapsed_seconds": round(self._elapsed(), 6),
            }
            self.state["incumbent_declarations"].append(declaration)
            self._save_state()
            return {
                "incumbent_id": candidate_id,
                "previous_incumbent_id": previous_id,
                "declaration_index": declaration_index,
                "event_seq": declaration_event_seq,
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
            self._save_state()

    def finish(self) -> Dict[str, Any]:
        with self.lock:
            self._require_active()
            if self.state["active_lease"] is not None:
                raise ProtocolError(
                    "lease_active", "commit or stop the active lease before finish"
                )
            return self._seal_research(
                self.state["incumbent_id"], reason="agent_finished"
            )

    def _seal_research(
        self, candidate_id: Optional[str], *, reason: str
    ) -> Dict[str, Any]:
        if self.enforce_operator_limit():
            return self.public_status()
        self._freeze_elapsed()
        self.state["status"] = "closed"
        self.state["closure_reason"] = reason
        self.state["agent_system"]["finish_observed"] = reason == "agent_finished"
        self.state["final_candidate_id"] = candidate_id
        self.state["research_sealed_at"] = datetime.fromtimestamp(
            self.wall_time(), tz=timezone.utc
        ).isoformat()
        self.state["analysis_complete"] = False
        self.state["analysis_status"] = "pending"
        self.state["analysis_failure"] = None
        self.state["analysis_identity_verified"] = None
        self.state["test_status"] = "pending"
        self._event(
            "RESEARCH_SEALED",
            reason=reason,
            candidate_id=candidate_id,
            candidate_inventory=list(self.state["candidate_commits"]),
            incumbent_declarations=list(self.state["incumbent_declarations"]),
        )
        self._save_state()
        return self.public_status()

    def _hidden_dataset_request(self, name: str) -> HiddenDatasetRequest:
        field = f"{name}_sha256"
        expected = (
            self.release_verification.artifact_sha256[field]
            if self.release_verification is not None
            else None
        )
        return HiddenDatasetRequest(name, expected)

    def _hidden_dataset_resolution_failed(
        self,
        name: str,
        *,
        primary: bool,
        code: str,
        private_detail: Optional[str] = None,
        teardown_code: Optional[str] = None,
    ) -> None:
        event: Dict[str, Any] = {"dataset": name, "code": code}
        if private_detail is not None:
            event["message"] = private_detail
        self._event("HIDDEN_DATASET_RESOLUTION_FAILED", **event)
        if primary:
            self.state["test_status"] = "failed"
            if teardown_code is not None:
                self.deployment_teardown_failed(
                    teardown_code,
                    private_detail=private_detail,
                )
            else:
                self.infrastructure_failed(code, private_detail=private_detail)
        else:
            self.state["analysis_identity_verified"] = False
            self._mark_analysis_incomplete(
                code,
                private_detail=private_detail,
                teardown_code=teardown_code,
            )

    def _record_hidden_dataset_resolution(
        self,
        request: HiddenDatasetRequest,
        resolved: Any,
        *,
        primary: bool,
    ) -> bool:
        """Validate Worker evidence without learning its private data location."""

        if not isinstance(resolved, ResolvedHiddenDataset):
            self._hidden_dataset_resolution_failed(
                request.name,
                primary=primary,
                code="hidden_dataset_receipt_invalid",
                private_detail=(
                    "Evaluation Worker returned "
                    f"{type(resolved).__name__}, expected ResolvedHiddenDataset"
                ),
            )
            return False
        if resolved.name != request.name:
            self._hidden_dataset_resolution_failed(
                request.name,
                primary=primary,
                code="hidden_dataset_receipt_invalid",
                private_detail=(
                    "Evaluation Worker resolved "
                    f"{resolved.name!r} for {request.name!r}"
                ),
            )
            return False
        if resolved.identity_algorithm != request.identity_algorithm:
            self._hidden_dataset_resolution_failed(
                request.name,
                primary=primary,
                code="hidden_dataset_receipt_invalid",
                private_detail=(
                    "Evaluation Worker used hidden dataset identity algorithm "
                    f"{resolved.identity_algorithm!r}, expected "
                    f"{request.identity_algorithm!r}"
                ),
            )
            return False

        field = f"{request.name}_sha256"
        identities = self.state["artifact_identities"]["hidden_evaluation"]
        previously_resolved = identities[field]
        expected = request.expected_sha256 or previously_resolved
        if expected is not None and expected != resolved.sha256:
            self._event(
                "HIDDEN_DATASET_IDENTITY_MISMATCH",
                dataset=request.name,
                expected_sha256=expected,
                actual_sha256=resolved.sha256,
            )
            if primary:
                violation = f"formal_release_verification_artifact_mismatch:{field}"
                if violation not in self.state["leaderboard_eligibility_violations"]:
                    self.state["leaderboard_eligibility_violations"].append(violation)
                self.state["leaderboard_eligible"] = False
                self.infrastructure_failed("hidden_dataset_identity_mismatch")
            else:
                self.state["analysis_identity_verified"] = False
                self._mark_analysis_incomplete("hidden_dataset_identity_mismatch")
            return False
        identities[field] = resolved.sha256

        if request.name == "test":
            violation = "formal_test_evaluation_pending"
            if violation in self.state["leaderboard_eligibility_violations"]:
                self.state["leaderboard_eligibility_violations"].remove(violation)
            self._refresh_leaderboard_eligibility()
        elif request.expected_sha256 is not None:
            self.state["analysis_identity_verified"] = True
        if previously_resolved is None:
            self._event(
                "HIDDEN_DATASET_RESOLVED",
                dataset=request.name,
                sha256=resolved.sha256,
                identity_algorithm=resolved.identity_algorithm,
            )
        self._save_state()
        return True

    def _record_hidden_dataset_mismatch(
        self,
        request: HiddenDatasetRequest,
        receipt: Any,
        *,
        primary: bool,
    ) -> None:
        """Accept a mismatch signal only when its receipt proves a mismatch."""

        field = f"{request.name}_sha256"
        previously_resolved = self.state["artifact_identities"]["hidden_evaluation"][
            field
        ]
        expected = request.expected_sha256 or previously_resolved
        if (
            not isinstance(receipt, ResolvedHiddenDataset)
            or receipt.name != request.name
            or receipt.identity_algorithm != request.identity_algorithm
            or expected is None
            or receipt.sha256 == expected
        ):
            self._hidden_dataset_resolution_failed(
                request.name,
                primary=primary,
                code="hidden_dataset_receipt_invalid",
                private_detail=(
                    "Evaluation Worker raised HiddenDatasetIdentityMismatch "
                    "without evidence of a different hidden dataset identity"
                ),
            )
            return
        self._record_hidden_dataset_resolution(
            request,
            receipt,
            primary=primary,
        )

    def _resolve_hidden_dataset(
        self,
        request: HiddenDatasetRequest,
        *,
        primary: bool,
        timeout_seconds: float,
    ) -> Optional[ResolvedHiddenDataset]:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            self._hidden_dataset_resolution_failed(
                request.name,
                primary=primary,
                code="hidden_dataset_resolution_deadline",
                private_detail="hidden dataset resolution deadline reached",
            )
            return None
        resolution_deadline = self.monotonic() + timeout_seconds
        try:
            resolved = self.evaluation_worker.resolve_hidden_dataset(
                episode_id=self.state["episode_id"],
                task=self.evaluation_task,
                hidden_dataset=request,
                timeout_seconds=timeout_seconds,
            )
            if self.monotonic() >= resolution_deadline:
                raise HiddenDatasetResolutionFailure(
                    "hidden dataset resolution deadline reached"
                )
        except HiddenDatasetIdentityMismatch as exc:
            self._record_hidden_dataset_mismatch(
                request,
                exc.receipt,
                primary=primary,
            )
            return None
        except HiddenDatasetResolutionFailure as exc:
            self._hidden_dataset_resolution_failed(
                request.name,
                primary=primary,
                code="hidden_dataset_resolution_failure",
                private_detail=str(exc),
            )
            return None
        except Exception as exc:
            self._hidden_dataset_resolution_failed(
                request.name,
                primary=primary,
                code="hidden_dataset_resolution_infrastructure_failure",
                private_detail=f"{type(exc).__name__}: {exc}",
                teardown_code="evaluation_worker_state_unknown",
            )
            return None
        if not self._record_hidden_dataset_resolution(
            request,
            resolved,
            primary=primary,
        ):
            return None
        return resolved

    def _store_analysis_trajectory(self) -> list[Dict[str, Any]]:
        """Write the trajectory and the summary counts that describe it.

        The counts exist so a reader of the summary fields alone can see that a
        `complete` ANALYSIS pass still has gaps.  They are derived here, in one
        place, so they cannot drift from the trajectory.
        """

        trajectory = self._analysis_trajectory()
        self.state["analysis_trajectory"] = trajectory
        self.state["analysis_selected_points"] = len(trajectory)
        self.state["analysis_evaluated_points"] = sum(
            point["analysis_status"] == "evaluated" for point in trajectory
        )
        return trajectory

    def _analysis_trajectory(self) -> list[Dict[str, Any]]:
        declaration_events: Dict[str, list[int]] = {}
        for declaration in self.state["incumbent_declarations"]:
            declaration_events.setdefault(declaration["candidate_id"], []).append(
                declaration["event_seq"]
            )
        trajectory = []
        for commit in self.state["candidate_commits"]:
            metadata = self.state["candidates"][commit["candidate_id"]]
            status = metadata["analysis_status"]
            error = metadata["analysis_error"]
            if status == "pending":
                status = "not_run"
                error = self.state.get("analysis_failure") or "analysis_not_run"
            trajectory.append(
                {
                    **commit,
                    "analysis_status": status,
                    "analysis_performance_score": metadata[
                        "analysis_performance_score"
                    ],
                    "analysis_raw_metrics": metadata["analysis_raw_metrics"],
                    "analysis_cost": metadata["analysis_cost"],
                    "analysis_error": error,
                    "declared_incumbent": metadata["declared_incumbent"],
                    "declaration_event_seqs": list(
                        declaration_events.get(commit["candidate_id"], [])
                    ),
                }
            )
        return trajectory

    def _mark_analysis_incomplete(
        self,
        code: str,
        *,
        private_detail: Optional[str] = None,
        teardown_code: Optional[str] = None,
    ) -> None:
        self.state["analysis_complete"] = False
        self.state["analysis_status"] = "incomplete"
        self.state["analysis_failure"] = code
        self._store_analysis_trajectory()
        event: Dict[str, Any] = {"code": code}
        if private_detail is not None:
            event["message"] = private_detail
        self._event("ANALYSIS_INCOMPLETE", **event)
        if teardown_code is not None:
            self._record_deployment_teardown_failure(
                teardown_code,
                private_detail=private_detail,
            )
        self._save_state()

    def _evaluate_candidate_with_resources(
        self, *, candidate_id: str, **kwargs: Any
    ) -> CandidateEvaluationResult:
        evaluation_id = "evaluation:" + uuid.uuid4().hex
        result = self.evaluation_worker.evaluate_candidate(
            **kwargs,
            evaluation_id=evaluation_id,
            resource_envelope=self.evaluation_resource_envelope,
        )
        if not isinstance(result, CandidateEvaluationResult):
            raise WorkerFailure("Evaluation Worker returned an invalid result type")
        record = {
            "evaluation_id": evaluation_id,
            "candidate_id": candidate_id,
            "split": kwargs["hidden_dataset"].name,
            "receipt": None,
            "assessment": None,
            "accepted": False,
        }
        self.state["evaluation_resource_allocations"].append(record)
        envelope = self.evaluation_resource_envelope
        try:
            if envelope is None:
                if result.resource_receipt is not None:
                    raise ValueError(
                        "unexpected evaluation resource receipt without a contract"
                    )
            else:
                assessment = assess_evaluation_resources(
                    envelope,
                    result.resource_receipt,
                    episode_id=self.state["episode_id"],
                    evaluation_id=evaluation_id,
                    worker_id=self.state["workers"]["evaluation"]["worker_id"],
                )
                if any(
                    previous["receipt"] is not None
                    and previous["receipt"]["receipt_id"]
                    == result.resource_receipt.receipt_id
                    for previous in self.state["evaluation_resource_allocations"][:-1]
                ):
                    raise ValueError("evaluation resource receipt_id was reused")
                record["receipt"] = result.resource_receipt.public_manifest()
                record["assessment"] = assessment.public_manifest()
                record["accepted"] = assessment.satisfied
                if not assessment.satisfied:
                    raise ValueError("evaluation resource contract is not satisfied")
                if record["split"] == "test":
                    violation = "evaluation_resource_allocation_pending"
                    if violation in self.state["leaderboard_eligibility_violations"]:
                        self.state["leaderboard_eligibility_violations"].remove(
                            violation
                        )
                        self._refresh_leaderboard_eligibility()
        except (TypeError, ValueError) as exc:
            self._event(
                "EVALUATION_RESOURCE_REJECTED",
                evaluation_id=evaluation_id,
                candidate_id=candidate_id,
                split=record["split"],
                message=str(exc),
            )
            raise EvaluationResourceFailure(str(exc)) from exc
        finally:
            self._save_state()
        return result

    def _evaluate_test_for_postmortem(self) -> None:
        """Durably stage the primary TEST decision before any diagnostics."""

        candidate_id = self.state["final_candidate_id"]
        if candidate_id is None:
            self._stage_failed_test("no_candidate")
            return

        digest = candidate_id.split(":", 1)[1]
        evaluation_started = self.monotonic()
        evaluation_deadline = evaluation_started + float(
            self.config.evaluator_timeout_seconds
        )

        def remaining_seconds() -> float:
            remaining = evaluation_deadline - self.monotonic()
            if remaining <= 0:
                raise WorkloadResolutionFailure(
                    "TEST evaluation operation deadline reached"
                )
            return remaining

        request = self._hidden_dataset_request("test")
        resolved: Optional[ResolvedHiddenDataset] = None
        candidate_workload: Optional[ResolvedWorkload] = None
        evaluator_workload: Optional[ResolvedWorkload] = None
        test_performance_score: Optional[float] = None
        test_raw_metrics: Optional[Dict[str, float]] = None
        test_cost: Optional[Dict[str, Any]] = None
        test_failure: Optional[str] = None
        test_infrastructure_failure: Optional[str] = None
        test_infrastructure_failure_code = "test_evaluation_infrastructure_failure"
        test_teardown_failure: Optional[str] = None
        try:
            candidate_workload = self._resolve_workload(
                "candidate", timeout_seconds=remaining_seconds()
            )
            evaluator_workload = self._resolve_workload(
                "evaluator", timeout_seconds=remaining_seconds()
            )
            resolved = self._resolve_hidden_dataset(
                request,
                primary=True,
                timeout_seconds=remaining_seconds(),
            )
            if resolved is not None:
                assert candidate_workload is not None
                assert evaluator_workload is not None
                evaluation_result = self._evaluate_candidate_with_resources(
                    candidate_id=candidate_id,
                    episode_id=self.state["episode_id"],
                    task=self.evaluation_task,
                    resources=self.resources,
                    candidate_dir=self.candidates_dir / digest,
                    hidden_dataset=resolved,
                    output_dir=self.evaluations_dir / "test" / digest,
                    candidate_workload=candidate_workload,
                    evaluator_workload=evaluator_workload,
                    timeout_seconds=remaining_seconds(),
                )
                if self.monotonic() >= evaluation_deadline:
                    raise WorkerFailure("TEST evaluation operation deadline reached")
                if not isinstance(evaluation_result, CandidateEvaluationResult):
                    raise WorkerFailure(
                        "Evaluation Worker returned an invalid result type"
                    )
                performance = evaluation_result.performance_manifest(
                    self.evaluation_task.performance_score
                )
                test_performance_score = performance["performance_score"]
                test_raw_metrics = performance["raw_metrics"]
                test_cost = evaluation_result.cost_manifest(self.evaluation_task.cost)
        except HiddenDatasetIdentityMismatch as exc:
            self._record_hidden_dataset_mismatch(
                request,
                exc.receipt,
                primary=True,
            )
        except HiddenDatasetResolutionFailure as exc:
            self._hidden_dataset_resolution_failed(
                "test",
                primary=True,
                code="hidden_dataset_resolution_failure",
                private_detail=str(exc),
            )
        except WorkloadResolutionFailure as exc:
            test_infrastructure_failure = str(exc)
            test_infrastructure_failure_code = "evaluation_workload_resolution_failure"
            self._event(
                "TEST_INFRASTRUCTURE_FAILURE",
                candidate_id=candidate_id,
                code="workload_resolution_failure",
            )
        except EvaluationResourceFailure as exc:
            test_infrastructure_failure = str(exc)
            test_infrastructure_failure_code = "evaluation_resource_contract_failure"
            self._event(
                "TEST_INFRASTRUCTURE_FAILURE",
                candidate_id=candidate_id,
                code="evaluation_resource_contract_failure",
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
            test_infrastructure_failure = str(exc)
            self._event(
                "TEST_INFRASTRUCTURE_FAILURE",
                candidate_id=candidate_id,
                exception_type=type(exc).__name__,
                message=str(exc),
            )
        finally:
            self.state["timing"]["test_evaluation_seconds"] += max(
                0.0, self.monotonic() - evaluation_started
            )

        if test_teardown_failure is not None:
            self.state["test_status"] = "failed"
            self.deployment_teardown_failed(
                "evaluation_worker_teardown_failure",
                private_detail=test_teardown_failure,
            )
            return
        if test_infrastructure_failure is not None:
            self.state["test_status"] = "failed"
            self.infrastructure_failed(
                test_infrastructure_failure_code,
                private_detail=test_infrastructure_failure,
            )
            return
        if resolved is None or self.state["status"] != "closed":
            self._save_state()
            return
        if (
            test_failure is not None
            or test_performance_score is None
            or test_raw_metrics is None
            or test_cost is None
        ):
            self._stage_failed_test(test_failure or "test_evaluation_failed")
            return

        self.state["test_status"] = "evaluated"
        self.state["pending_result"] = {
            "outcome": "success",
            "failure": None,
            "test_performance_score": test_performance_score,
            "test_raw_metrics": test_raw_metrics,
            "test_cost": test_cost,
            "publication_id": uuid.uuid4().hex,
        }
        self._event(
            "TEST_EVALUATED",
            candidate_id=candidate_id,
            performance_score=test_performance_score,
            raw_metrics=test_raw_metrics,
            cost=test_cost,
        )
        self._save_state()

    def _evaluate_analysis_for_postmortem(self) -> None:
        """Attempt diagnostics without invalidating a staged TEST decision."""

        candidate_ids = list(self.state["candidates"])
        analysis_started = self.monotonic()
        request = self._hidden_dataset_request("analysis")
        resolved: Optional[ResolvedHiddenDataset] = None
        incomplete: Optional[tuple[str, Optional[str], Optional[str]]] = None
        try:
            if candidate_ids:
                resolved = self._resolve_hidden_dataset(
                    request,
                    primary=False,
                    timeout_seconds=float(self.config.evaluator_timeout_seconds),
                )
            for candidate_id in candidate_ids if resolved is not None else []:
                metadata = self.state["candidates"][candidate_id]
                if metadata["analysis_status"] in {"evaluated", "failed"}:
                    continue
                digest = candidate_id.split(":", 1)[1]
                candidate_deadline = self.monotonic() + float(
                    self.config.evaluator_timeout_seconds
                )

                def remaining_seconds() -> float:
                    remaining = candidate_deadline - self.monotonic()
                    if remaining <= 0:
                        raise WorkloadResolutionFailure(
                            "ANALYSIS evaluation operation deadline reached"
                        )
                    return remaining

                try:
                    # TEST may have been skipped when no incumbent was declared.
                    # Resolve independently; cached handles avoid repeated work.
                    candidate_workload = self._resolve_workload(
                        "candidate", timeout_seconds=remaining_seconds()
                    )
                    evaluator_workload = self._resolve_workload(
                        "evaluator", timeout_seconds=remaining_seconds()
                    )
                    evaluation_result = self._evaluate_candidate_with_resources(
                        candidate_id=candidate_id,
                        episode_id=self.state["episode_id"],
                        task=self.evaluation_task,
                        resources=self.resources,
                        candidate_dir=self.candidates_dir / digest,
                        hidden_dataset=resolved,
                        output_dir=self.evaluations_dir / "analysis" / digest,
                        candidate_workload=candidate_workload,
                        evaluator_workload=evaluator_workload,
                        timeout_seconds=remaining_seconds(),
                    )
                    if self.monotonic() >= candidate_deadline:
                        raise WorkerFailure(
                            "ANALYSIS evaluation operation deadline reached"
                        )
                    if not isinstance(evaluation_result, CandidateEvaluationResult):
                        raise WorkerFailure(
                            "Evaluation Worker returned an invalid result type"
                        )
                    performance = evaluation_result.performance_manifest(
                        self.evaluation_task.performance_score
                    )
                    cost = evaluation_result.cost_manifest(self.evaluation_task.cost)
                except HiddenDatasetIdentityMismatch as exc:
                    self._record_hidden_dataset_mismatch(
                        request,
                        exc.receipt,
                        primary=False,
                    )
                    break
                except HiddenDatasetResolutionFailure as exc:
                    self._hidden_dataset_resolution_failed(
                        "analysis",
                        primary=False,
                        code="hidden_dataset_resolution_failure",
                        private_detail=str(exc),
                    )
                    break
                except WorkloadResolutionFailure as exc:
                    metadata["analysis_status"] = "infrastructure_failure"
                    metadata["analysis_error"] = "workload_resolution_failure"
                    self._event(
                        "ANALYSIS_INFRASTRUCTURE_FAILURE",
                        candidate_id=candidate_id,
                        code="workload_resolution_failure",
                    )
                    incomplete = (
                        "analysis_workload_resolution_failure",
                        str(exc),
                        None,
                    )
                    break
                except EvaluationResourceFailure as exc:
                    metadata["analysis_status"] = "infrastructure_failure"
                    metadata["analysis_error"] = "evaluation_resource_contract_failure"
                    self._event(
                        "ANALYSIS_INFRASTRUCTURE_FAILURE",
                        candidate_id=candidate_id,
                        code="evaluation_resource_contract_failure",
                    )
                    incomplete = (
                        "analysis_evaluation_resource_contract_failure",
                        str(exc),
                        None,
                    )
                    break
                except WorkerTeardownFailure as exc:
                    metadata["analysis_status"] = "infrastructure_failure"
                    metadata["analysis_error"] = "worker_teardown_failure"
                    self._event(
                        "ANALYSIS_INFRASTRUCTURE_FAILURE",
                        candidate_id=candidate_id,
                        code="worker_teardown_failure",
                    )
                    incomplete = (
                        "analysis_worker_teardown_failure",
                        str(exc),
                        "evaluation_worker_teardown_failure",
                    )
                    break
                except WorkerFailure:
                    metadata["analysis_status"] = "failed"
                    metadata["analysis_error"] = "evaluation_failed"
                    self._event(
                        "ANALYSIS_EVALUATION_FAILED",
                        candidate_id=candidate_id,
                        code="evaluation_failed",
                    )
                except ValueError:
                    metadata["analysis_status"] = "failed"
                    metadata["analysis_error"] = "invalid_metric"
                    self._event(
                        "ANALYSIS_EVALUATION_FAILED",
                        candidate_id=candidate_id,
                        code="invalid_metric",
                    )
                except Exception as exc:
                    metadata["analysis_status"] = "infrastructure_failure"
                    metadata["analysis_error"] = "unexpected_worker_failure"
                    self._event(
                        "ANALYSIS_INFRASTRUCTURE_FAILURE",
                        candidate_id=candidate_id,
                        exception_type=type(exc).__name__,
                        message=str(exc),
                    )
                    incomplete = (
                        "analysis_evaluation_infrastructure_failure",
                        str(exc),
                        "evaluation_worker_state_unknown",
                    )
                    break
                else:
                    metadata["analysis_status"] = "evaluated"
                    metadata["analysis_performance_score"] = performance[
                        "performance_score"
                    ]
                    metadata["analysis_raw_metrics"] = performance["raw_metrics"]
                    metadata["analysis_cost"] = cost
                    self._event(
                        "ANALYSIS_EVALUATED",
                        candidate_id=candidate_id,
                        performance_score=performance["performance_score"],
                        raw_metrics=performance["raw_metrics"],
                        cost=cost,
                    )
                self._store_analysis_trajectory()
                self._save_state()
        finally:
            self.state["timing"]["analysis_evaluation_seconds"] += max(
                0.0, self.monotonic() - analysis_started
            )

        if candidate_ids and resolved is None:
            self._save_state()
            return
        if self.state["analysis_status"] == "incomplete":
            self._save_state()
            return
        if incomplete is not None:
            code, private_detail, teardown_code = incomplete
            self._mark_analysis_incomplete(
                code,
                private_detail=private_detail,
                teardown_code=teardown_code,
            )
            return

        self._store_analysis_trajectory()
        self.state["analysis_complete"] = True
        self.state["analysis_status"] = "complete"
        self.state["analysis_failure"] = None
        self._event(
            "ANALYSIS_COMPLETED",
            candidates=len(candidate_ids),
            failed=(
                self.state["analysis_selected_points"]
                - self.state["analysis_evaluated_points"]
            ),
        )
        self._save_state()

    def evaluate_postmortem(self) -> Dict[str, Any]:
        """Stage TEST first, then attempt independent postmortem diagnostics.

        This method deliberately does not publish a terminal result.  The
        trusted composition layer must first finish Evaluation Worker and
        private-materialization cleanup, record any teardown failures, and
        then call :meth:`publish_postmortem_result`.
        """

        with self.lock:
            if self.state["status"] != "closed":
                raise RuntimeError("postmortem evaluation requires sealed research")
            if self.state["agent_system"]["termination_reason"] is None:
                raise RuntimeError("Agent termination must precede hidden evaluation")
            if self.state["agent_control"]["cleanup_complete"] is not True:
                raise RuntimeError(
                    "Agent-control cleanup must precede hidden evaluation"
                )
            if self.state.get("active_lease") is not None or self.state.get(
                "pending_lease_teardowns"
            ):
                raise RuntimeError("Research Worker teardown must precede evaluation")
            if (
                self.state.get("deployment_teardown_failures")
                and self.state.get("pending_result") is None
            ):
                raise RuntimeError("deployment teardown failure blocks TEST")

            if self.state.get("pending_result") is None:
                self._evaluate_test_for_postmortem()
                if self.state["status"] != "closed":
                    return self.public_status()

            if self.state["analysis_status"] not in {"complete", "incomplete"}:
                self._evaluate_analysis_for_postmortem()

            return self.public_status()

    def publish_postmortem_result(self) -> Dict[str, Any]:
        """Publish one stable terminal result after deployment cleanup."""

        with self.lock:
            if self.state["status"] != "closed":
                return self.public_status()
            if self.state.get("pending_result") is None:
                raise RuntimeError("postmortem TEST decision has not been staged")
            if self.state.get("analysis_status") not in {"complete", "incomplete"}:
                raise RuntimeError("postmortem ANALYSIS disposition is incomplete")
            if self._publish_staged_result() is not True:
                raise RuntimeError("postmortem result is not ready for publication")
            return self.public_status()

    def _stage_failed_test(self, failure: str) -> Dict[str, Any]:
        """Stage a failed result without manufacturing a missing TEST metric."""

        self.state["test_status"] = "failed"
        self.state["pending_result"] = {
            "outcome": "failed",
            "failure": failure,
            "test_performance_score": None,
            "test_raw_metrics": None,
            "test_cost": None,
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
        # Pre-TEST teardown failures prevent TEST from starting.  A teardown
        # failure discovered while running optional ANALYSIS diagnostics does
        # not erase an already persisted TEST decision; it is represented
        # independently by policy/eligibility and teardown evidence.  The
        # composition layer nevertheless calls this method only after all
        # postmortem cleanup attempts have finished, so the published terminal
        # state will not be revised by a later required cleanup step.

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
        self.state["test_performance_score"] = pending["test_performance_score"]
        self.state["test_raw_metrics"] = pending["test_raw_metrics"]
        self.state["test_cost"] = pending["test_cost"]
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
            if self.enforce_operator_limit():
                return True
            if self.state["status"] != "active":
                return False
            active = self.state.get("active_lease")
            if self.track == "A" and self._track_a_remaining() <= 0:
                self._close_at_track_a_deadline()
                return True
            if active and self.monotonic() >= active["deadline_monotonic"]:
                self._end_active_lease("timeout")
            return False

    def operator_limit_remaining(self) -> Optional[float]:
        """Trusted launcher clock; never acquire the synchronous Worker lock.

        None disarms the launcher guard once research closes normally. The deployment
        may terminate workloads independently while a Worker call holds lock.
        """

        if (
            self.operator_max_research_seconds is None
            or self.started_monotonic is None
            or self.state["status"] not in {"active", "operator_stopped"}
        ):
            return None
        return (
            self.started_monotonic
            + self.operator_max_research_seconds
            - self.monotonic()
        )

    def enforce_operator_limit(self) -> bool:
        """Seal an operator-aborted run without producing a benchmark result.

        Persist the stop before potentially blocking teardown. The trusted
        launcher/Worker owns termination and retries using retained Lease IDs.
        """

        with self.lock:
            if self.state["status"] != "active":
                return False
            remaining = self.operator_limit_remaining()
            if remaining is None or remaining > 0:
                return False
            self._freeze_elapsed()
            self.state.update(
                status="operator_stopped",
                outcome="aborted",
                closure_reason="operator_time_limit_reached",
                failure="operator_time_limit_reached",
                research_sealed_at=datetime.fromtimestamp(
                    self.wall_time(), tz=timezone.utc
                ).isoformat(),
                leaderboard_eligible=False,
            )
            self.state["leaderboard_eligibility_violations"].append(
                "operator_time_limit_reached"
            )
            self._event(
                "EPISODE_OPERATOR_STOPPED",
                code="operator_time_limit_reached",
                max_research_seconds=self.operator_max_research_seconds,
                actual_research_seconds=self.monotonic() - self.started_monotonic,
            )
            self._save_state()
            return True

    def agent_exited_without_finish(self) -> None:
        with self.lock:
            self.enforce_operator_limit()
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
            # Workload termination by the operator guard can surface as a
            # Worker error. Preserve the operator cause rather than blaming
            # the participant or reclassifying it as an infrastructure fault.
            self.enforce_operator_limit()
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
            self.state["test_performance_score"] = None
            self.state["test_raw_metrics"] = None
            self.state["test_cost"] = None
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
        self.state["test_performance_score"] = None
        self.state["test_raw_metrics"] = None
        self.state["test_cost"] = None
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
