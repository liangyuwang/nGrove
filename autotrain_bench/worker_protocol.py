"""Platform-neutral execution contract used by the benchmark coordinator.

The core protocol describes effects that a trusted Worker must provide.  It
does not describe how Docker, a scheduler, a container runtime, or a remote job
service implements those effects.  Platform implementations belong in
separately distributed Worker packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:
    from .config import TaskConfig, TrainingResources


WORKER_PROTOCOL_VERSION = 3
WORKER_CONFORMANCE_SUITE_VERSION = f"worker-v{WORKER_PROTOCOL_VERSION}"


class WorkerFailure(RuntimeError):
    """A Worker operation failed with no unconfirmed workload boundary left."""


class WorkerTeardownFailure(WorkerFailure):
    """A Worker could not confirm teardown of an isolated workload boundary."""


class EnforcementLevel(str, Enum):
    """How strongly a Worker deployment enforces one semantic capability."""

    UNSUPPORTED = "unsupported"
    UNVERIFIED = "unverified"
    BEST_EFFORT = "best_effort"
    ENFORCED = "enforced"

    @property
    def rank(self) -> int:
        return {
            EnforcementLevel.UNSUPPORTED: 0,
            EnforcementLevel.UNVERIFIED: 1,
            EnforcementLevel.BEST_EFFORT: 2,
            EnforcementLevel.ENFORCED: 3,
        }[self]


class TrustLevel(str, Enum):
    """Who controls the concrete Worker deployment."""

    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    CERTIFIED = "certified"

    @property
    def rank(self) -> int:
        return {
            TrustLevel.DEVELOPMENT: 0,
            TrustLevel.CALIBRATION: 1,
            TrustLevel.CERTIFIED: 2,
        }[self]


class LeaseState(str, Enum):
    PROVISIONING = "provisioning"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int):
            raise TypeError("command exit_code must be an integer")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise TypeError("command stdout and stderr must be strings")
        for field_name in ("timed_out", "stdout_truncated", "stderr_truncated"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"command {field_name} must be a boolean")


@dataclass(frozen=True)
class ResearchTaskSpec:
    """Only task fields that a Research Worker is allowed to receive."""

    task_id: str
    training_image: str
    train_dir: Path
    public_api_dir: Path

    @classmethod
    def from_task(cls, task: "TaskConfig", public_api_dir: Path) -> "ResearchTaskSpec":
        return cls(
            task_id=task.task_id,
            training_image=task.training_image,
            train_dir=task.train_dir,
            public_api_dir=public_api_dir,
        )


@dataclass(frozen=True)
class CandidateChannelRequirements:
    """Release-owned application semantics, not a transport selection."""

    protocol: str
    hidden_data_to_candidate: bool = False
    candidate_egress_allowed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, str) or not self.protocol.strip():
            raise ValueError("Candidate channel protocol must not be empty")
        if self.hidden_data_to_candidate is not False:
            raise ValueError("Candidate channel may not expose hidden data")
        if self.candidate_egress_allowed is not False:
            raise ValueError("Candidate channel may not grant Candidate egress")

    def public_manifest(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol,
            "mechanism": "worker_defined",
            "hidden_data_to_candidate": "forbidden",
            "candidate_egress": "denied",
        }


@dataclass(frozen=True)
class EvaluationTaskSpec:
    """Evaluator configuration without either hidden-dataset location."""

    task_id: str
    backend: str
    candidate_image: str
    evaluator_image: str
    evaluator_entrypoint: Path
    metric: str
    vocab_size: int
    context_length: int
    batch_size: int
    device: str
    timeout_seconds: int
    candidate_channel: CandidateChannelRequirements

    @classmethod
    def from_task(cls, task: "TaskConfig") -> "EvaluationTaskSpec":
        return cls(
            task_id=task.task_id,
            backend=task.evaluator_backend,
            candidate_image=task.candidate_image,
            evaluator_image=task.evaluator_image,
            evaluator_entrypoint=task.evaluator_entrypoint,
            metric=task.metric,
            vocab_size=task.vocab_size,
            context_length=task.context_length,
            batch_size=task.evaluator_batch_size,
            device=task.evaluator_device,
            timeout_seconds=task.evaluator_timeout_seconds,
            candidate_channel=CandidateChannelRequirements(
                protocol=task.candidate_protocol
            ),
        )


@dataclass(frozen=True)
class HiddenDatasetRef:
    """One organizer-selected hidden dataset visible only after research seal.

    ``path`` remains a version 3 local/shared-filesystem deployment handle. It
    is not required to exist while the Agent or Research Worker can still act;
    the Evaluation Worker may materialize it immediately before evaluation.
    """

    name: str
    path: Path

    def __post_init__(self) -> None:
        if self.name not in {"analysis", "test"}:
            raise ValueError("hidden dataset name must be 'analysis' or 'test'")


@dataclass(frozen=True)
class Capability:
    """One semantic guarantee and optional deployment-supplied evidence."""

    level: EnforcementLevel
    mechanism: Optional[str] = None

    def public_manifest(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"level": self.level.value}
        if self.mechanism:
            payload["mechanism"] = self.mechanism
        return payload


@dataclass(frozen=True)
class ConformanceReceipt:
    suite_version: str
    passed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.suite_version, str) or not self.suite_version.strip():
            raise ValueError("conformance suite_version must not be empty")
        if not isinstance(self.passed, bool):
            raise TypeError("conformance passed must be a boolean")

    def public_manifest(self) -> Dict[str, Any]:
        return {"suite_version": self.suite_version, "passed": self.passed}


@dataclass(frozen=True)
class WorkerDescriptor:
    """Self-description of a concrete Worker deployment.

    Capabilities and deployment trust are intentionally independent.  A
    user-owned Worker can enforce every technical capability without becoming
    a certified leaderboard deployment.
    """

    worker_id: str
    trust_level: TrustLevel
    roles: Tuple[str, ...]
    capabilities: Mapping[str, Capability]
    protocol_version: int = WORKER_PROTOCOL_VERSION
    conformance: Optional[ConformanceReceipt] = None

    def __post_init__(self) -> None:
        if self.protocol_version != WORKER_PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported Worker protocol version: {self.protocol_version}"
            )
        if not self.worker_id.strip():
            raise ValueError("worker_id must not be empty")
        if not self.roles or any(not role.strip() for role in self.roles):
            raise ValueError("Worker roles must not be empty")
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("Worker roles must be unique")
        if self.conformance is not None and not isinstance(
            self.conformance, ConformanceReceipt
        ):
            raise TypeError("conformance must be a ConformanceReceipt")
        for name, capability in self.capabilities.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("capability names must not be empty")
            if not isinstance(capability, Capability):
                raise TypeError("capabilities must contain Capability values")

    def public_manifest(self) -> Dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "worker_id": self.worker_id,
            "trust_level": self.trust_level.value,
            "roles": list(self.roles),
            "capabilities": {
                name: capability.public_manifest()
                for name, capability in sorted(self.capabilities.items())
            },
            "conformance": (
                self.conformance.public_manifest() if self.conformance else None
            ),
        }


@dataclass(frozen=True)
class CapabilityRequirement:
    name: str
    minimum: EnforcementLevel = EnforcementLevel.ENFORCED

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("capability requirement name must not be empty")


@dataclass(frozen=True)
class CapabilityAssessment:
    satisfied: bool
    missing: Tuple[str, ...]
    insufficient: Tuple[str, ...]
    trust_satisfied: bool
    conformance_satisfied: bool

    @property
    def leaderboard_eligible(self) -> bool:
        return self.satisfied and self.trust_satisfied and self.conformance_satisfied

    def public_manifest(self) -> Dict[str, Any]:
        return {
            "capabilities_satisfied": self.satisfied,
            "missing_capabilities": list(self.missing),
            "insufficient_capabilities": list(self.insufficient),
            "trust_satisfied": self.trust_satisfied,
            "conformance_satisfied": self.conformance_satisfied,
            "leaderboard_eligible": self.leaderboard_eligible,
        }


IMMUTABLE_WORKLOAD_IDENTITY = "immutable_workload_identity"
GPU_DEVICE_ISOLATION = "gpu_device_isolation"
TRAINING_DATA_READ_ONLY = "training_data_read_only"
TRAINING_NETWORK_ISOLATION = "training_network_isolation"
HIDDEN_DATASET_ISOLATION = "hidden_dataset_isolation"
CANDIDATE_EVALUATOR_CHANNEL_ISOLATION = "candidate_evaluator_channel_isolation"
DEADLINE_TERMINATION = "deadline_termination"
IMMUTABLE_CANDIDATE_ARTIFACT = "immutable_candidate_artifact"


RESEARCH_REQUIREMENTS: Tuple[CapabilityRequirement, ...] = tuple(
    CapabilityRequirement(name)
    for name in (
        IMMUTABLE_WORKLOAD_IDENTITY,
        TRAINING_DATA_READ_ONLY,
        TRAINING_NETWORK_ISOLATION,
        DEADLINE_TERMINATION,
        IMMUTABLE_CANDIDATE_ARTIFACT,
    )
)

EVALUATION_REQUIREMENTS: Tuple[CapabilityRequirement, ...] = tuple(
    CapabilityRequirement(name)
    for name in (
        IMMUTABLE_WORKLOAD_IDENTITY,
        HIDDEN_DATASET_ISOLATION,
        CANDIDATE_EVALUATOR_CHANNEL_ISOLATION,
    )
)

BASELINE_REQUIREMENTS: Tuple[CapabilityRequirement, ...] = (
    RESEARCH_REQUIREMENTS + EVALUATION_REQUIREMENTS
)


def assess_worker(
    descriptor: WorkerDescriptor,
    requirements: Sequence[CapabilityRequirement],
    *,
    minimum_trust: TrustLevel = TrustLevel.CERTIFIED,
    require_conformance: bool = True,
) -> CapabilityAssessment:
    """Evaluate capabilities separately from deployment trust."""

    missing = []
    insufficient = []
    for requirement in requirements:
        capability = descriptor.capabilities.get(requirement.name)
        if capability is None:
            missing.append(requirement.name)
        elif capability.level.rank < requirement.minimum.rank:
            insufficient.append(requirement.name)
    return CapabilityAssessment(
        satisfied=not missing and not insufficient,
        # A capability may be required by more than one role, so a shortfall
        # can be recorded more than once.  Report each name once: callers that
        # need per-role attribution assess each role separately.
        missing=tuple(sorted(set(missing))),
        insufficient=tuple(sorted(set(insufficient))),
        trust_satisfied=descriptor.trust_level.rank >= minimum_trust.rank,
        conformance_satisfied=(
            not require_conformance
            or bool(
                descriptor.conformance
                and descriptor.conformance.suite_version
                == WORKER_CONFORMANCE_SUITE_VERSION
                and descriptor.conformance.passed is True
            )
        ),
    )


@runtime_checkable
class ResearchWorker(Protocol):
    """Semantic execution surface required by the research Supervisor."""

    def describe_capabilities(self) -> WorkerDescriptor: ...

    def start_lease(
        self,
        *,
        episode_id: str,
        lease_id: str,
        duration_seconds: float,
        task: ResearchTaskSpec,
        workspace: Path,
        resources: "TrainingResources",
        lease_dir: Path,
    ) -> None: ...

    def exec_lease(
        self,
        *,
        episode_id: str,
        lease_id: str,
        argv: Sequence[str],
        timeout_seconds: float,
    ) -> CommandResult: ...

    def lease_state(self, *, episode_id: str, lease_id: str) -> LeaseState: ...

    def terminate_lease(self, *, episode_id: str, lease_id: str) -> None: ...


@runtime_checkable
class EvaluationWorker(Protocol):
    """Semantic hidden-evaluation surface, separable from ResearchWorker."""

    def describe_capabilities(self) -> WorkerDescriptor: ...

    def evaluate_candidate(
        self,
        *,
        episode_id: str,
        task: EvaluationTaskSpec,
        resources: "TrainingResources",
        candidate_dir: Path,
        hidden_dataset: HiddenDatasetRef,
        output_dir: Path,
        timeout_seconds: Optional[float] = None,
    ) -> float: ...


@runtime_checkable
class Worker(ResearchWorker, EvaluationWorker, Protocol):
    """Phase-one in-process Worker contract.

    A future transport can implement the same semantics asynchronously without
    adding scheduler or container concepts to the benchmark Supervisor.
    """
