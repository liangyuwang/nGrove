"""Platform-neutral execution contract used by the benchmark coordinator.

The core protocol describes effects that a trusted Worker must provide.  It
does not describe how Docker, a scheduler, a container runtime, or a remote job
service implements those effects.  Platform implementations belong in
separately distributed Worker packages.
"""

from __future__ import annotations

import hashlib
import json
import math
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


WORKER_PROTOCOL_VERSION = 7
WORKER_CONFORMANCE_SUITE_VERSION = f"worker-v{WORKER_PROTOCOL_VERSION}"
RESOURCE_ENVELOPE_SCHEMA_VERSION = 1
RESOURCE_ALLOCATION_RECEIPT_VERSION = 1


class WorkerFailure(RuntimeError):
    """A Worker operation failed with no unconfirmed workload boundary left."""


class WorkerTeardownFailure(WorkerFailure):
    """A Worker could not confirm teardown of an isolated workload boundary."""


class HiddenDatasetResolutionFailure(WorkerFailure):
    """An Evaluation Worker could not resolve an authorized hidden dataset."""


class WorkloadResolutionFailure(WorkerFailure):
    """A Worker could not resolve a release-owned workload identity."""


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
    request_content: str = "token_prefixes"
    hidden_artifact_visible: bool = False
    current_target_visible: bool = False
    future_tokens_visible: bool = False
    candidate_egress_allowed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, str) or not self.protocol.strip():
            raise ValueError("Candidate channel protocol must not be empty")
        if self.request_content != "token_prefixes":
            raise ValueError("Candidate channel must carry token prefixes")
        if self.hidden_artifact_visible is not False:
            raise ValueError("Candidate channel may not expose the hidden artifact")
        if self.current_target_visible is not False:
            raise ValueError("Candidate channel may not expose the current target")
        if self.future_tokens_visible is not False:
            raise ValueError("Candidate channel may not expose future tokens")
        if self.candidate_egress_allowed is not False:
            raise ValueError("Candidate channel may not grant Candidate egress")

    def public_manifest(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol,
            "mechanism": "worker_defined",
            "request_content": self.request_content,
            "hidden_artifact_visibility": "denied",
            "current_target_visibility": "denied",
            "future_token_visibility": "denied",
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


def _validate_sha256(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


_WORKLOAD_ROLES = {"training", "candidate", "evaluator"}


def _validate_workload_role(value: str) -> None:
    if value not in _WORKLOAD_ROLES:
        raise ValueError(
            "workload role must be 'training', 'candidate', or 'evaluator'"
        )


def _validate_workload_identity(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 2048
        or any(character.isspace() or character == "\0" for character in value)
    ):
        raise ValueError(f"{field} must be a safe non-empty identity")


def _validate_workload_format(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 64
        or not value[0].isalnum()
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789._+-"
            for character in value
        )
    ):
        raise ValueError("effective workload format must be a lowercase identifier")


def _validate_opaque_id(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-"
            for character in value
        )
    ):
        raise ValueError(f"{field} must be an opaque identifier")


def _validate_resource_property_name(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not value[0].isalnum()
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789._-"
            for character in value
        )
    ):
        raise ValueError(f"{field} must be a lowercase semantic property name")


def _resource_scalar(value: Any, field: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        if isinstance(value, str) and (not value or len(value) > 2048 or "\0" in value):
            raise ValueError(f"{field} contains an invalid string")
        return value
    if isinstance(value, int):
        if value.bit_length() > 127:
            raise ValueError(f"{field} integer is outside the supported range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
        return value
    raise TypeError(f"{field} must be a JSON scalar")


def _resource_value(value: Any, field: str) -> Any:
    """Normalize one path-free resource value used by the semantic protocol."""

    if isinstance(value, (tuple, list)):
        if len(value) > 1024:
            raise ValueError(f"{field} contains too many values")
        return tuple(_resource_scalar(item, field) for item in value)
    return _resource_scalar(value, field)


def _resource_value_manifest(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_resource_value_manifest(item) for item in value]
    return value


def _resource_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, tuple) or isinstance(right, tuple):
        return (
            isinstance(left, tuple)
            and isinstance(right, tuple)
            and len(left) == len(right)
            and all(
                _resource_values_equal(left_item, right_item)
                for left_item, right_item in zip(left, right)
            )
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    return type(left) is type(right) and left == right


def _resource_relation_satisfied(relation: str, required: Any, actual: Any) -> bool:
    values = actual if isinstance(actual, tuple) else (actual,)
    if relation == "exact":
        return _resource_values_equal(actual, required)
    if relation == "at_least":
        if isinstance(required, bool) or not isinstance(required, (int, float)):
            return False
        return bool(values) and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and value >= required
            for value in values
        )
    if relation == "contains":
        return bool(values) and all(
            isinstance(value, str) and required in value for value in values
        )
    raise ValueError(f"unsupported resource relation: {relation}")


@dataclass(frozen=True)
class ResourceRequirement:
    """One release-selected semantic property, independent of its mechanism."""

    name: str
    relation: str
    value: Any
    minimum_enforcement: EnforcementLevel = EnforcementLevel.ENFORCED

    def __post_init__(self) -> None:
        _validate_resource_property_name(self.name, "resource requirement name")
        if self.relation not in {"exact", "at_least", "contains"}:
            raise ValueError("resource relation must be exact, at_least, or contains")
        normalized = _resource_value(self.value, "resource requirement value")
        if self.relation == "at_least" and (
            isinstance(normalized, bool) or not isinstance(normalized, (int, float))
        ):
            raise TypeError("at_least resource requirements must be numeric")
        if self.relation == "contains" and not isinstance(normalized, str):
            raise TypeError("contains resource requirements must be strings")
        if not isinstance(self.minimum_enforcement, EnforcementLevel):
            raise TypeError("minimum resource enforcement must be an EnforcementLevel")
        if self.minimum_enforcement.rank < EnforcementLevel.BEST_EFFORT.rank:
            raise ValueError(
                "normative resources require at least best_effort enforcement"
            )
        object.__setattr__(self, "value", normalized)

    def public_manifest(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "relation": self.relation,
            "value": _resource_value_manifest(self.value),
            "minimum_enforcement": self.minimum_enforcement.value,
        }

    @classmethod
    def from_manifest(cls, value: Any) -> "ResourceRequirement":
        if not isinstance(value, dict):
            raise TypeError("resource requirement must be an object")
        if set(value) != {"name", "relation", "value", "minimum_enforcement"}:
            raise ValueError("resource requirement has unexpected fields")
        try:
            level = EnforcementLevel(value["minimum_enforcement"])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid minimum resource enforcement") from exc
        return cls(
            name=value["name"],
            relation=value["relation"],
            value=value["value"],
            minimum_enforcement=level,
        )


@dataclass(frozen=True)
class ResourceEnvelope:
    """Release-owned resource semantics requested for one operation.

    Only properties named here become normative or observational. An absent
    platform property is deliberately unspecified rather than silently fixed
    by Core.
    """

    requirements: Tuple[ResourceRequirement, ...]
    observational_properties: Tuple[str, ...] = ()
    schema_version: int = RESOURCE_ENVELOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != RESOURCE_ENVELOPE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported resource envelope schema version")
        if not isinstance(self.requirements, tuple) or any(
            not isinstance(requirement, ResourceRequirement)
            for requirement in self.requirements
        ):
            raise TypeError("resource requirements must be a tuple")
        names = [requirement.name for requirement in self.requirements]
        if len(set(names)) != len(names):
            raise ValueError("normative resource properties must be unique")
        if not isinstance(self.observational_properties, tuple):
            raise TypeError("observational resource properties must be a tuple")
        for name in self.observational_properties:
            _validate_resource_property_name(name, "observational resource property")
        if len(set(self.observational_properties)) != len(
            self.observational_properties
        ):
            raise ValueError("observational resource properties must be unique")
        if set(names).intersection(self.observational_properties):
            raise ValueError("a resource property cannot have two classifications")

    def public_manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "normative": [
                requirement.public_manifest()
                for requirement in sorted(self.requirements, key=lambda item: item.name)
            ],
            "observational": sorted(self.observational_properties),
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.public_manifest(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_manifest(cls, value: Any) -> "ResourceEnvelope":
        if not isinstance(value, dict):
            raise TypeError("resource envelope must be an object")
        if set(value) != {"schema_version", "normative", "observational"}:
            raise ValueError("resource envelope has unexpected fields")
        normative = value["normative"]
        observational = value["observational"]
        if not isinstance(normative, list) or not isinstance(observational, list):
            raise TypeError("resource envelope property sets must be arrays")
        return cls(
            schema_version=value["schema_version"],
            requirements=tuple(
                ResourceRequirement.from_manifest(requirement)
                for requirement in normative
            ),
            observational_properties=tuple(observational),
        )


def v0_resource_envelope(
    resources: "TrainingResources",
    *,
    minimum_enforcement: EnforcementLevel,
) -> ResourceEnvelope:
    """Translate the bundled v0 launch flags into semantic resource properties.

    This compatibility helper is deliberately not the universal resource
    vocabulary. New releases can publish any properties through
    :class:`ResourceEnvelope` without adding fields here.
    """

    resources.validate()
    requirements = [
        ResourceRequirement(
            "host.count",
            "exact",
            1,
            minimum_enforcement,
        ),
        ResourceRequirement(
            "accelerator.count",
            "exact",
            resources.nvidia_gpu_count,
            minimum_enforcement,
        ),
    ]
    if resources.nvidia_gpu_count:
        requirements.append(
            ResourceRequirement(
                "accelerator.vendor",
                "exact",
                "nvidia",
                minimum_enforcement,
            )
        )
        if resources.required_gpu_model is not None:
            requirements.append(
                ResourceRequirement(
                    "accelerator.model",
                    "contains",
                    resources.required_gpu_model,
                    minimum_enforcement,
                )
            )
        if resources.minimum_gpu_memory_mib is not None:
            requirements.append(
                ResourceRequirement(
                    "accelerator.memory_mib_per_device",
                    "at_least",
                    resources.minimum_gpu_memory_mib,
                    minimum_enforcement,
                )
            )
    return ResourceEnvelope(requirements=tuple(requirements))


@dataclass(frozen=True)
class ResourceObservation:
    """One effective resource view supplied by an evidence authority."""

    status: str
    value: Any = None
    authority: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in {"observed", "unavailable", "not_applicable"}:
            raise ValueError("resource observation has an invalid status")
        if self.status == "observed":
            normalized = _resource_value(self.value, "resource observation value")
            if not isinstance(self.authority, str) or not self.authority.strip():
                raise ValueError("observed resources require an evidence authority")
            if len(self.authority) > 256 or "\0" in self.authority:
                raise ValueError("resource evidence authority is invalid")
            object.__setattr__(self, "value", normalized)
        elif self.value is not None or self.authority is not None:
            raise ValueError("an unavailable resource view cannot carry evidence")

    def public_manifest(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "value": _resource_value_manifest(self.value),
            "authority": self.authority,
        }

    @classmethod
    def from_manifest(cls, value: Any) -> "ResourceObservation":
        if not isinstance(value, dict):
            raise TypeError("resource observation must be an object")
        if set(value) != {"status", "value", "authority"}:
            raise ValueError("resource observation has unexpected fields")
        return cls(
            status=value["status"],
            value=value["value"],
            authority=value["authority"],
        )


@dataclass(frozen=True)
class ResourcePropertyEvidence:
    """Allocated, workload-visible, and enforced views of one property."""

    name: str
    allocator_allocated: ResourceObservation
    workload_visible: ResourceObservation
    enforcement_level: EnforcementLevel
    enforcement_authority: Optional[str]

    def __post_init__(self) -> None:
        _validate_resource_property_name(self.name, "resource evidence name")
        if not isinstance(
            self.allocator_allocated, ResourceObservation
        ) or not isinstance(self.workload_visible, ResourceObservation):
            raise TypeError(
                "resource evidence views must be ResourceObservation values"
            )
        if not isinstance(self.enforcement_level, EnforcementLevel):
            raise TypeError("resource enforcement level must be an EnforcementLevel")
        if self.enforcement_level is EnforcementLevel.UNSUPPORTED:
            if self.enforcement_authority is not None:
                raise ValueError("unsupported enforcement cannot name an authority")
        elif (
            not isinstance(self.enforcement_authority, str)
            or not self.enforcement_authority.strip()
            or len(self.enforcement_authority) > 256
            or "\0" in self.enforcement_authority
        ):
            raise ValueError("resource enforcement requires an authority")

    def public_manifest(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "allocator_allocated": self.allocator_allocated.public_manifest(),
            "workload_visible": self.workload_visible.public_manifest(),
            "enforcement_level": self.enforcement_level.value,
            "enforcement_authority": self.enforcement_authority,
        }

    @classmethod
    def from_manifest(cls, value: Any) -> "ResourcePropertyEvidence":
        if not isinstance(value, dict):
            raise TypeError("resource property evidence must be an object")
        expected = {
            "name",
            "allocator_allocated",
            "workload_visible",
            "enforcement_level",
            "enforcement_authority",
        }
        if set(value) != expected:
            raise ValueError("resource property evidence has unexpected fields")
        try:
            level = EnforcementLevel(value["enforcement_level"])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid resource enforcement level") from exc
        return cls(
            name=value["name"],
            allocator_allocated=ResourceObservation.from_manifest(
                value["allocator_allocated"]
            ),
            workload_visible=ResourceObservation.from_manifest(
                value["workload_visible"]
            ),
            enforcement_level=level,
            enforcement_authority=value["enforcement_authority"],
        )


@dataclass(frozen=True)
class ResourceAllocationReceipt:
    """Operation-bound Worker evidence for one effective allocation."""

    episode_id: str
    lease_id: str
    worker_id: str
    resource_request_sha256: str
    properties: Tuple[ResourcePropertyEvidence, ...]
    receipt_id: str
    receipt_version: int = RESOURCE_ALLOCATION_RECEIPT_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.receipt_version, bool)
            or not isinstance(self.receipt_version, int)
            or self.receipt_version != RESOURCE_ALLOCATION_RECEIPT_VERSION
        ):
            raise ValueError("unsupported resource allocation receipt version")
        _validate_opaque_id(self.episode_id, "resource receipt episode_id")
        _validate_opaque_id(self.lease_id, "resource receipt lease_id")
        if (
            not isinstance(self.worker_id, str)
            or not self.worker_id.strip()
            or len(self.worker_id) > 256
            or "\0" in self.worker_id
        ):
            raise ValueError("resource receipt worker_id is invalid")
        _validate_sha256(
            self.resource_request_sha256,
            "resource receipt request sha256",
        )
        _validate_opaque_id(self.receipt_id, "resource receipt_id")
        if not isinstance(self.properties, tuple) or any(
            not isinstance(item, ResourcePropertyEvidence) for item in self.properties
        ):
            raise TypeError("resource receipt properties must be a tuple")
        names = [item.name for item in self.properties]
        if len(set(names)) != len(names):
            raise ValueError("resource receipt properties must be unique")

    def public_manifest(self) -> Dict[str, Any]:
        return {
            "receipt_version": self.receipt_version,
            "receipt_id": self.receipt_id,
            "episode_id": self.episode_id,
            "lease_id": self.lease_id,
            "worker_id": self.worker_id,
            "resource_request_sha256": self.resource_request_sha256,
            "properties": [
                item.public_manifest()
                for item in sorted(self.properties, key=lambda value: value.name)
            ],
        }

    @classmethod
    def from_manifest(cls, value: Any) -> "ResourceAllocationReceipt":
        if not isinstance(value, dict):
            raise TypeError("resource allocation receipt must be an object")
        expected = {
            "receipt_version",
            "receipt_id",
            "episode_id",
            "lease_id",
            "worker_id",
            "resource_request_sha256",
            "properties",
        }
        if set(value) != expected:
            raise ValueError("resource allocation receipt has unexpected fields")
        properties = value["properties"]
        if not isinstance(properties, list):
            raise TypeError("resource allocation properties must be an array")
        return cls(
            receipt_version=value["receipt_version"],
            receipt_id=value["receipt_id"],
            episode_id=value["episode_id"],
            lease_id=value["lease_id"],
            worker_id=value["worker_id"],
            resource_request_sha256=value["resource_request_sha256"],
            properties=tuple(
                ResourcePropertyEvidence.from_manifest(item) for item in properties
            ),
        )


@dataclass(frozen=True)
class ResourceAllocationAssessment:
    satisfied: bool
    missing: Tuple[str, ...]
    mismatched: Tuple[str, ...]
    insufficient_enforcement: Tuple[str, ...]
    unexpected: Tuple[str, ...]

    def public_manifest(self) -> Dict[str, Any]:
        return {
            "satisfied": self.satisfied,
            "missing_properties": list(self.missing),
            "mismatched_properties": list(self.mismatched),
            "insufficient_enforcement": list(self.insufficient_enforcement),
            "unexpected_properties": list(self.unexpected),
        }


def assess_resource_allocation(
    envelope: ResourceEnvelope,
    receipt: ResourceAllocationReceipt,
    *,
    episode_id: str,
    lease_id: str,
    worker_id: str,
) -> ResourceAllocationAssessment:
    """Validate receipt binding, then compare every release-normative property."""

    if not isinstance(envelope, ResourceEnvelope):
        raise TypeError("resource envelope must be a ResourceEnvelope")
    if not isinstance(receipt, ResourceAllocationReceipt):
        raise TypeError("resource receipt must be a ResourceAllocationReceipt")
    if receipt.episode_id != episode_id:
        raise ValueError("resource receipt is bound to a different episode")
    if receipt.lease_id != lease_id:
        raise ValueError("resource receipt is bound to a different lease")
    if receipt.worker_id != worker_id:
        raise ValueError("resource receipt is bound to a different Worker")
    if receipt.resource_request_sha256 != envelope.sha256:
        raise ValueError("resource receipt is bound to a different request")

    requirements = {item.name: item for item in envelope.requirements}
    evidence = {item.name: item for item in receipt.properties}
    allowed = set(requirements).union(envelope.observational_properties)
    missing = sorted(set(requirements).difference(evidence))
    unexpected = sorted(set(evidence).difference(allowed))
    mismatched = []
    insufficient = []
    for name, requirement in requirements.items():
        item = evidence.get(name)
        if item is None:
            continue
        observed = [
            view.value
            for view in (item.allocator_allocated, item.workload_visible)
            if view.status == "observed"
        ]
        if not observed or any(
            not _resource_relation_satisfied(
                requirement.relation,
                requirement.value,
                value,
            )
            for value in observed
        ):
            mismatched.append(name)
        if item.enforcement_level.rank < requirement.minimum_enforcement.rank:
            insufficient.append(name)
    return ResourceAllocationAssessment(
        satisfied=not missing
        and not mismatched
        and not insufficient
        and not unexpected,
        missing=tuple(missing),
        mismatched=tuple(sorted(mismatched)),
        insufficient_enforcement=tuple(sorted(insufficient)),
        unexpected=tuple(unexpected),
    )


@dataclass(frozen=True)
class WorkloadRequest:
    """One canonical workload selected by the Challenge Release."""

    role: str
    requested_identity: str

    def __post_init__(self) -> None:
        _validate_workload_role(self.role)
        _validate_workload_identity(
            self.requested_identity, "requested workload identity"
        )


@dataclass(frozen=True)
class ResolvedWorkload:
    """A Worker resolution bound to one immutable effective artifact.

    ``resolution_id`` is an opaque operation handle, not a scheduler, runtime,
    path, or cache identifier.  Platform handles remain private to the Worker.
    """

    role: str
    requested_identity: str
    effective_format: str
    effective_sha256: str
    transformation_sha256: Optional[str]
    resolution_id: str

    def __post_init__(self) -> None:
        _validate_workload_role(self.role)
        _validate_workload_identity(
            self.requested_identity, "resolved requested workload identity"
        )
        _validate_workload_format(self.effective_format)
        _validate_sha256(self.effective_sha256, "effective workload sha256")
        if self.transformation_sha256 is not None:
            _validate_sha256(
                self.transformation_sha256,
                "workload transformation provenance sha256",
            )
        _validate_opaque_id(self.resolution_id, "workload resolution_id")

    def evidence_manifest(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "requested_identity": self.requested_identity,
            "effective_format": self.effective_format,
            "effective_sha256": self.effective_sha256,
            "transformation_sha256": self.transformation_sha256,
        }


@dataclass(frozen=True)
class WorkloadResolutionAcceptance:
    """Organizer acceptance for one non-identity workload transformation."""

    role: str
    requested_identity: str
    effective_format: str
    effective_sha256: str
    transformation_sha256: Optional[str]

    def __post_init__(self) -> None:
        # Reuse the resolution validator with a non-secret synthetic handle so
        # the accepted and observed evidence have exactly the same schema.
        ResolvedWorkload(
            role=self.role,
            requested_identity=self.requested_identity,
            effective_format=self.effective_format,
            effective_sha256=self.effective_sha256,
            transformation_sha256=self.transformation_sha256,
            resolution_id="acceptance",
        )

    def matches(self, resolved: ResolvedWorkload) -> bool:
        return isinstance(resolved, ResolvedWorkload) and self.public_manifest() == (
            resolved.evidence_manifest()
        )

    def public_manifest(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "requested_identity": self.requested_identity,
            "effective_format": self.effective_format,
            "effective_sha256": self.effective_sha256,
            "transformation_sha256": self.transformation_sha256,
        }


@dataclass(frozen=True)
class HiddenDatasetRequest:
    """Release-owned identity requested after research sealing.

    No path, URL, credential, mount, cache, or scheduler handle crosses the
    Core/Worker boundary.  The Evaluation Worker privately maps ``task_id``
    from :class:`EvaluationTaskSpec` plus ``name`` to its deployment source.
    """

    name: str
    expected_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if self.name not in {"analysis", "test"}:
            raise ValueError("hidden dataset name must be 'analysis' or 'test'")
        if self.expected_sha256 is not None:
            _validate_sha256(
                self.expected_sha256,
                "hidden dataset expected_sha256",
            )


@dataclass(frozen=True)
class ResolvedHiddenDataset:
    """Worker-owned resolved input with an opaque, non-platform handle."""

    name: str
    sha256: str
    resolution_id: str

    def __post_init__(self) -> None:
        if self.name not in {"analysis", "test"}:
            raise ValueError("resolved hidden dataset has an invalid name")
        _validate_sha256(self.sha256, "resolved hidden dataset sha256")
        _validate_opaque_id(
            self.resolution_id,
            "hidden dataset resolution_id",
        )


class HiddenDatasetIdentityMismatch(HiddenDatasetResolutionFailure):
    """A Worker resolved bytes that do not match the frozen release identity."""

    def __init__(self, receipt: ResolvedHiddenDataset) -> None:
        if not isinstance(receipt, ResolvedHiddenDataset):
            raise TypeError("identity mismatch requires a resolved hidden dataset")
        super().__init__("hidden dataset identity mismatch")
        self.receipt = receipt


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
HIDDEN_DATASET_IDENTITY_VERIFICATION = "hidden_dataset_identity_verification"
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
        HIDDEN_DATASET_IDENTITY_VERIFICATION,
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

    def resolve_workload(
        self,
        *,
        episode_id: str,
        workload: WorkloadRequest,
        timeout_seconds: float,
    ) -> ResolvedWorkload: ...

    def start_lease(
        self,
        *,
        episode_id: str,
        lease_id: str,
        duration_seconds: float,
        task: ResearchTaskSpec,
        workspace: Path,
        resources: "TrainingResources",
        resource_envelope: ResourceEnvelope,
        lease_dir: Path,
        workload: ResolvedWorkload,
    ) -> ResourceAllocationReceipt: ...

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

    def resolve_workload(
        self,
        *,
        episode_id: str,
        workload: WorkloadRequest,
        timeout_seconds: float,
    ) -> ResolvedWorkload: ...

    def resolve_hidden_dataset(
        self,
        *,
        episode_id: str,
        task: EvaluationTaskSpec,
        hidden_dataset: HiddenDatasetRequest,
        timeout_seconds: float,
    ) -> ResolvedHiddenDataset: ...

    def evaluate_candidate(
        self,
        *,
        episode_id: str,
        task: EvaluationTaskSpec,
        resources: "TrainingResources",
        candidate_dir: Path,
        hidden_dataset: ResolvedHiddenDataset,
        output_dir: Path,
        candidate_workload: ResolvedWorkload,
        evaluator_workload: ResolvedWorkload,
        timeout_seconds: Optional[float] = None,
    ) -> float: ...


@runtime_checkable
class Worker(ResearchWorker, EvaluationWorker, Protocol):
    """Phase-one in-process Worker contract.

    A future transport can implement the same semantics asynchronously without
    adding scheduler or container concepts to the benchmark Supervisor.
    """
