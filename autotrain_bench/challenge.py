"""Challenge release profiles and leaderboard comparison identities.

A challenge fixes data and evaluation semantics without fixing one hardware
platform.  A release instead publishes a deliberately small set of track and
runtime profiles.  Only episodes with the same comparison key are directly
comparable.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .artifact_identity import (
    FILE_IDENTITY_ALGORITHM,
    HIDDEN_BUNDLE_IDENTITY_ALGORITHM,
)
from .config import ConfigError, TaskConfig, TrainingResources
from .worker_protocol import (
    EnforcementLevel,
    ResourceEnvelope,
    ResourceRequirement,
    WorkloadResolutionAcceptance,
)


RELEASE_STAGES = {"preview", "official"}
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
FORMAL_ARTIFACT_DIGEST_FIELDS = (
    "challenge_manifest_sha256",
    "task_manifest_sha256",
    "policy_sha256",
    "evaluator_sha256",
    "train_manifest_sha256",
    "analysis_sha256",
    "test_sha256",
)
FORMAL_ARTIFACT_IDENTITY_ALGORITHMS = {
    "file": FILE_IDENTITY_ALGORITHM,
    "hidden_evaluation": HIDDEN_BUNDLE_IDENTITY_ALGORITHM,
}


def agent_release_manifest(release: Mapping[str, Any]) -> Dict[str, Any]:
    """Expose release identifiers without fingerprints of private configuration."""

    # Private release records retain full file identities and verification
    # evidence. New audit fields must not silently enter the Agent boundary.
    return {
        name: release[name]
        for name in (
            "challenge_id",
            "release_stage",
            "leaderboard_eligible",
            "track_config_id",
            "runtime_profile_id",
        )
        if name in release
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be an object")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty string")
    return value


def _identifier(value: Any, field: str) -> str:
    result = _string(value, field)
    if not _IDENTIFIER.fullmatch(result):
        raise ConfigError(
            f"{field} must contain only lowercase letters, digits, '.', '_', or '-'"
        )
    return result


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{field} must be a positive integer")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be a boolean")
    return value


@dataclass(frozen=True)
class TrackConfig:
    """One named, immutable budget configuration within a challenge release."""

    config_id: str
    label: str
    track: str
    total_seconds: Optional[int] = None
    max_leases: Optional[int] = None
    lease_seconds: Optional[int] = None
    provisional: bool = False

    @classmethod
    def load(cls, config_id: str, value: Any) -> "TrackConfig":
        field = f"track_configs.{config_id}"
        _identifier(config_id, f"{field} key")
        raw = _mapping(value, field)
        track = _string(raw.get("track"), f"{field}.track").upper()
        if track not in {"A", "B"}:
            raise ConfigError(f"{field}.track must be A or B")
        label = _string(raw.get("label", config_id), f"{field}.label")
        provisional = _boolean(raw.get("provisional", False), f"{field}.provisional")
        if track == "A":
            if "max_leases" in raw or "lease_seconds" in raw:
                raise ConfigError(f"{field} may not contain Track B lease fields")
            return cls(
                config_id=config_id,
                label=label,
                track=track,
                total_seconds=_positive_int(
                    raw.get("total_seconds"), f"{field}.total_seconds"
                ),
                provisional=provisional,
            )
        if "total_seconds" in raw:
            raise ConfigError(f"{field} may not contain Track A total_seconds")
        return cls(
            config_id=config_id,
            label=label,
            track=track,
            max_leases=_positive_int(raw.get("max_leases"), f"{field}.max_leases"),
            lease_seconds=_positive_int(
                raw.get("lease_seconds"), f"{field}.lease_seconds"
            ),
            provisional=provisional,
        )

    def apply(self, task: TaskConfig) -> TaskConfig:
        if self.track == "A":
            return replace(task, track_a_total_seconds=self.total_seconds).validate()
        return replace(
            task,
            track_b_max_leases=self.max_leases,
            track_b_lease_seconds=self.lease_seconds,
        ).validate()

    def public_manifest(self) -> Dict[str, Any]:
        budget: Dict[str, Any]
        if self.track == "A":
            budget = {"total_seconds": self.total_seconds}
        else:
            budget = {
                "max_leases": self.max_leases,
                "lease_seconds": self.lease_seconds,
            }
        return {
            "label": self.label,
            "track": self.track,
            "provisional": self.provisional,
            **budget,
        }


@dataclass(frozen=True)
class RuntimeProfile:
    """A named episode-time hardware lane, not a property of the task."""

    profile_id: str
    label: str
    host_count: int
    accelerator_vendor: Optional[str]
    accelerator_model: Optional[str]
    accelerator_count: int
    minimum_memory_mib: Optional[int]
    gpu_model_match: Optional[str]
    provisional: bool = False
    resource_contract: Optional[ResourceEnvelope] = None
    evaluation_resource_contract: Optional[ResourceEnvelope] = None

    @classmethod
    def load(cls, profile_id: str, value: Any) -> "RuntimeProfile":
        field = f"runtime_profiles.{profile_id}"
        _identifier(profile_id, f"{field} key")
        raw = _mapping(value, field)
        if "accelerator" not in raw:
            raise ConfigError(f"{field}.accelerator must be an object or null")
        accelerator_value = raw["accelerator"]
        accelerator = (
            None
            if accelerator_value is None
            else _mapping(accelerator_value, f"{field}.accelerator")
        )
        host_count = _positive_int(raw.get("host_count"), f"{field}.host_count")
        if host_count != 1:
            raise ConfigError(f"{field}.host_count must be 1 in the v0 runtime")
        provisional = _boolean(raw.get("provisional", False), f"{field}.provisional")
        label = _string(raw.get("label", profile_id), f"{field}.label")
        try:
            resource_contract = (
                ResourceEnvelope.from_manifest(raw["resource_contract"])
                if "resource_contract" in raw
                else None
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{field}.resource_contract is invalid: {exc}") from exc
        try:
            evaluation_resource_contract = (
                ResourceEnvelope.from_manifest(raw["evaluation_resource_contract"])
                if "evaluation_resource_contract" in raw
                else None
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"{field}.evaluation_resource_contract is invalid: {exc}"
            ) from exc
        if accelerator is None:
            return cls(
                profile_id=profile_id,
                label=label,
                host_count=host_count,
                accelerator_vendor=None,
                accelerator_model=None,
                accelerator_count=0,
                minimum_memory_mib=None,
                gpu_model_match=None,
                provisional=provisional,
                resource_contract=resource_contract,
                evaluation_resource_contract=evaluation_resource_contract,
            )
        vendor = _string(
            accelerator.get("vendor"), f"{field}.accelerator.vendor"
        ).lower()
        if vendor != "nvidia":
            raise ConfigError(
                f"{field}.accelerator.vendor must be 'nvidia' in the v0 runtime"
            )
        return cls(
            profile_id=profile_id,
            label=label,
            host_count=host_count,
            accelerator_vendor=vendor,
            accelerator_model=_string(
                accelerator.get("model"), f"{field}.accelerator.model"
            ),
            accelerator_count=_positive_int(
                accelerator.get("count"), f"{field}.accelerator.count"
            ),
            minimum_memory_mib=_positive_int(
                accelerator.get("minimum_memory_mib"),
                f"{field}.accelerator.minimum_memory_mib",
            ),
            gpu_model_match=_string(
                accelerator.get("model_match"),
                f"{field}.accelerator.model_match",
            ),
            provisional=provisional,
            resource_contract=resource_contract,
            evaluation_resource_contract=evaluation_resource_contract,
        )

    def training_resources(self) -> TrainingResources:
        return TrainingResources(
            nvidia_gpu_count=self.accelerator_count,
            required_gpu_model=self.gpu_model_match,
            minimum_gpu_memory_mib=self.minimum_memory_mib,
        ).validate()

    def resource_envelope(
        self,
        *,
        minimum_enforcement: EnforcementLevel,
    ) -> ResourceEnvelope:
        """Return one contract consistent with the v0 profile projection."""

        requirements = [
            ResourceRequirement(
                "host.count",
                "exact",
                self.host_count,
                minimum_enforcement,
            ),
            ResourceRequirement(
                "accelerator.count",
                "exact",
                self.accelerator_count,
                minimum_enforcement,
            ),
        ]
        if self.accelerator_vendor is not None:
            requirements.extend(
                (
                    ResourceRequirement(
                        "accelerator.vendor",
                        "exact",
                        self.accelerator_vendor,
                        minimum_enforcement,
                    ),
                    ResourceRequirement(
                        "accelerator.model",
                        "contains",
                        self.gpu_model_match,
                        minimum_enforcement,
                    ),
                    ResourceRequirement(
                        "accelerator.memory_mib_per_device",
                        "at_least",
                        self.minimum_memory_mib,
                        minimum_enforcement,
                    ),
                )
            )
        baseline = ResourceEnvelope(requirements=tuple(requirements))
        if self.resource_contract is None:
            return baseline

        declared = {
            requirement.name: requirement
            for requirement in self.resource_contract.requirements
        }
        for required in baseline.requirements:
            actual = declared.get(required.name)
            if actual is None:
                raise ConfigError(
                    f"runtime profile {self.profile_id!r} resource_contract must "
                    f"include {required.name!r} from its compatibility fields"
                )
            if actual.relation != required.relation or actual.value != required.value:
                raise ConfigError(
                    f"runtime profile {self.profile_id!r} resource_contract "
                    f"conflicts with compatibility field {required.name!r}"
                )
        return self.resource_contract

    def public_manifest(self) -> Dict[str, Any]:
        manifest = {
            "label": self.label,
            "host_count": self.host_count,
            "provisional": self.provisional,
        }
        manifest["accelerator"] = (
            None
            if self.accelerator_vendor is None
            else {
                "vendor": self.accelerator_vendor,
                "model": self.accelerator_model,
                "count": self.accelerator_count,
                "minimum_memory_mib": self.minimum_memory_mib,
                "model_match": self.gpu_model_match,
            }
        )
        if self.resource_contract is not None:
            manifest["resource_contract"] = self.resource_contract.public_manifest()
        if self.evaluation_resource_contract is not None:
            manifest["evaluation_resource_contract"] = (
                self.evaluation_resource_contract.public_manifest()
            )
        return manifest


@dataclass(frozen=True)
class ComparisonKey:
    challenge_id: str
    track: str
    track_config: str
    runtime_profile: str

    def public_manifest(self) -> Dict[str, str]:
        return {
            "challenge_id": self.challenge_id,
            "track": self.track,
            "track_config": self.track_config,
            "runtime_profile": self.runtime_profile,
        }


@dataclass(frozen=True)
class ReleaseVerificationReceipt:
    """Trusted organizer-side verification supplied by a formal deployment.

    Challenge manifests can describe and hash themselves, but that does not
    prove that the files came from the organizer.  A local preview therefore
    cannot manufacture this receipt; a formal Coordinator receives it from its
    trusted release-verification boundary.
    """

    authority: str
    release_id: str
    core_revision: str
    artifact_sha256: Mapping[str, str]
    artifact_identity_algorithms: Mapping[str, str]
    workload_resolution_acceptance: Mapping[str, WorkloadResolutionAcceptance] = field(
        default_factory=dict
    )

    def validate(self) -> "ReleaseVerificationReceipt":
        _string(self.authority, "release verification authority")
        _identifier(self.release_id, "release verification release_id")
        if len(self.core_revision) not in {40, 64} or any(
            value not in "0123456789abcdef" for value in self.core_revision
        ):
            raise ConfigError(
                "release verification core_revision must be a Git object ID"
            )
        missing = [
            field
            for field in FORMAL_ARTIFACT_DIGEST_FIELDS
            if field not in self.artifact_sha256
        ]
        invalid = [
            field
            for field in FORMAL_ARTIFACT_DIGEST_FIELDS
            if field in self.artifact_sha256
            and (
                not isinstance(self.artifact_sha256[field], str)
                or len(self.artifact_sha256[field]) != 64
                or any(
                    value not in "0123456789abcdef"
                    for value in self.artifact_sha256[field]
                )
            )
        ]
        if missing or invalid:
            details = sorted(set(missing + invalid))
            raise ConfigError(
                "release verification requires SHA-256 identities for: "
                + ", ".join(details)
            )
        if dict(self.artifact_identity_algorithms) != (
            FORMAL_ARTIFACT_IDENTITY_ALGORITHMS
        ):
            raise ConfigError(
                "release verification artifact identity algorithms must match "
                "the active protocol"
            )
        for role, acceptance in self.workload_resolution_acceptance.items():
            if role not in {"training", "candidate", "evaluator"}:
                raise ConfigError(
                    "release verification workload resolution has an invalid role: "
                    f"{role}"
                )
            if not isinstance(acceptance, WorkloadResolutionAcceptance):
                raise ConfigError(
                    "release verification workload resolutions must contain "
                    "WorkloadResolutionAcceptance values"
                )
            if acceptance.role != role:
                raise ConfigError(
                    "release verification workload resolution role does not match "
                    f"its key: {role}"
                )
        return self

    def private_manifest(self) -> Dict[str, Any]:
        return {
            "authority": self.authority,
            "release_id": self.release_id,
            "core_revision": self.core_revision,
            "artifact_sha256": dict(self.artifact_sha256),
            "artifact_identity_algorithms": dict(self.artifact_identity_algorithms),
            "workload_resolution_acceptance": {
                role: acceptance.public_manifest()
                for role, acceptance in sorted(
                    self.workload_resolution_acceptance.items()
                )
            },
        }


@dataclass(frozen=True)
class ResolvedChallengeRun:
    task: TaskConfig
    track: str
    resources: TrainingResources
    resource_envelope: ResourceEnvelope
    comparison_key: ComparisonKey
    release_manifest: Dict[str, Any]


@dataclass(frozen=True)
class ChallengeRelease:
    """A finite set of supported comparison lanes for one challenge release."""

    source: Path
    manifest_sha256: str
    task_manifest_sha256: str
    schema_version: int
    challenge_id: str
    task_id: str
    task_file: str
    release_stage: str
    leaderboard_eligible: bool
    track_configs: Dict[str, TrackConfig]
    runtime_profiles: Dict[str, RuntimeProfile]

    @classmethod
    def load(cls, path: Path) -> "ChallengeRelease":
        source = path.expanduser().resolve()
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(f"challenge release not found: {source}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON in {source}: {exc}") from exc
        root = _mapping(raw, "root")
        schema_version = _positive_int(root.get("schema_version"), "schema_version")
        if schema_version not in {2, 3}:
            raise ConfigError("challenge schema_version must be 2 or 3")
        release_stage = _string(root.get("release_stage"), "release_stage")
        if release_stage not in RELEASE_STAGES:
            allowed = ", ".join(sorted(RELEASE_STAGES))
            raise ConfigError(f"release_stage must be one of: {allowed}")
        raw_tracks = _mapping(root.get("track_configs"), "track_configs")
        raw_runtimes = _mapping(root.get("runtime_profiles"), "runtime_profiles")
        if not raw_tracks:
            raise ConfigError("track_configs must not be empty")
        if not raw_runtimes:
            raise ConfigError("runtime_profiles must not be empty")
        track_configs = {
            key: TrackConfig.load(key, value) for key, value in raw_tracks.items()
        }
        runtime_profiles = {
            key: RuntimeProfile.load(key, value) for key, value in raw_runtimes.items()
        }
        if schema_version < 3 and any(
            profile.evaluation_resource_contract is not None
            for profile in runtime_profiles.values()
        ):
            raise ConfigError(
                "evaluation_resource_contract requires challenge schema_version 3"
            )
        task_file = _string(root.get("task"), "task")
        configured_task = Path(task_file)
        if configured_task.is_absolute() or ".." in configured_task.parts:
            raise ConfigError("task must be a relative path inside the challenge")
        task_path = (source.parent / configured_task).resolve()
        try:
            task_path.relative_to(source.parent)
        except ValueError as exc:
            raise ConfigError("task escapes the challenge directory") from exc
        if not task_path.is_file():
            raise ConfigError(f"challenge task not found: {task_path}")
        release = cls(
            source=source,
            manifest_sha256=_sha256_file(source),
            task_manifest_sha256=_sha256_file(task_path),
            schema_version=schema_version,
            challenge_id=_identifier(root.get("challenge_id"), "challenge_id"),
            task_id=_identifier(root.get("task_id"), "task_id"),
            task_file=task_file,
            release_stage=release_stage,
            leaderboard_eligible=_boolean(
                root.get("leaderboard_eligible"), "leaderboard_eligible"
            ),
            track_configs=track_configs,
            runtime_profiles=runtime_profiles,
        )
        release.validate()
        return release

    def validate(self) -> "ChallengeRelease":
        minimum_resource_enforcement = (
            EnforcementLevel.ENFORCED
            if self.leaderboard_eligible
            else EnforcementLevel.BEST_EFFORT
        )
        for profile in self.runtime_profiles.values():
            profile.resource_envelope(minimum_enforcement=minimum_resource_enforcement)
        if self.leaderboard_eligible:
            if self.release_stage != "official":
                raise ConfigError(
                    "a leaderboard-eligible release must have release_stage='official'"
                )
            provisional_tracks = sorted(
                key for key, value in self.track_configs.items() if value.provisional
            )
            provisional_runtimes = sorted(
                key for key, value in self.runtime_profiles.items() if value.provisional
            )
            if provisional_tracks or provisional_runtimes:
                details = provisional_tracks + provisional_runtimes
                raise ConfigError(
                    "a leaderboard-eligible release may not contain provisional "
                    "configurations: " + ", ".join(details)
                )
            weak_resource_requirements = sorted(
                f"{profile_id}:{requirement.name}"
                for profile_id, profile in self.runtime_profiles.items()
                for requirement in profile.resource_envelope(
                    minimum_enforcement=EnforcementLevel.ENFORCED
                ).requirements
                if requirement.minimum_enforcement is not EnforcementLevel.ENFORCED
            )
            if weak_resource_requirements:
                raise ConfigError(
                    "a leaderboard-eligible release requires enforced normative "
                    "resources: " + ", ".join(weak_resource_requirements)
                )
            for profile in self.runtime_profiles.values():
                envelope = profile.evaluation_resource_contract
                if (
                    envelope is None
                    or not envelope.requirements
                    or any(
                        item.minimum_enforcement is not EnforcementLevel.ENFORCED
                        for item in envelope.requirements
                    )
                ):
                    raise ConfigError(
                        "a leaderboard-eligible release requires an enforced "
                        f"evaluation_resource_contract: {profile.profile_id}"
                    )
        return self

    @property
    def task_path(self) -> Path:
        return (self.source.parent / self.task_file).resolve()

    def resolve(
        self,
        task: TaskConfig,
        *,
        track_config_id: str,
        runtime_profile_id: str,
    ) -> ResolvedChallengeRun:
        if task.task_id != self.task_id:
            raise ConfigError(
                f"challenge expects task_id {self.task_id!r}, received {task.task_id!r}"
            )
        if task.task_kind == "development_smoke" and self.leaderboard_eligible:
            raise ConfigError(
                "development smoke releases may not be leaderboard eligible"
            )
        if self.leaderboard_eligible and not task.leaderboard_eligible:
            raise ConfigError(
                "a leaderboard-eligible release requires a leaderboard-eligible task"
            )
        if self.leaderboard_eligible:
            if _sha256_file(self.source) != self.manifest_sha256:
                raise ConfigError("challenge manifest changed after it was loaded")
            if (
                task.source != self.task_path
                or _sha256_file(task.source) != self.task_manifest_sha256
            ):
                raise ConfigError(
                    "leaderboard release must use its unchanged task manifest"
                )
        try:
            track_config = self.track_configs[track_config_id]
        except KeyError as exc:
            raise ConfigError(f"unknown track config: {track_config_id}") from exc
        try:
            runtime_profile = self.runtime_profiles[runtime_profile_id]
        except KeyError as exc:
            raise ConfigError(f"unknown runtime profile: {runtime_profile_id}") from exc
        resolved_task = track_config.apply(task)
        resources = runtime_profile.training_resources().validate_for(resolved_task)
        resource_envelope = runtime_profile.resource_envelope(
            minimum_enforcement=(
                EnforcementLevel.ENFORCED
                if self.leaderboard_eligible
                else EnforcementLevel.BEST_EFFORT
            )
        )
        comparison_key = ComparisonKey(
            challenge_id=self.challenge_id,
            track=track_config.track,
            track_config=track_config.config_id,
            runtime_profile=runtime_profile.profile_id,
        )
        return ResolvedChallengeRun(
            task=resolved_task,
            track=track_config.track,
            resources=resources,
            resource_envelope=resource_envelope,
            comparison_key=comparison_key,
            release_manifest={
                "challenge_id": self.challenge_id,
                "challenge_manifest_sha256": self.manifest_sha256,
                "task_manifest_sha256": self.task_manifest_sha256,
                "release_stage": self.release_stage,
                "leaderboard_eligible": self.leaderboard_eligible,
                "track_config_id": track_config.config_id,
                "runtime_profile_id": runtime_profile.profile_id,
                "track_config": track_config.public_manifest(),
                "runtime_profile": runtime_profile.public_manifest(),
                "resource_envelope": resource_envelope.public_manifest(),
                "evaluation_resource_envelope": (
                    runtime_profile.evaluation_resource_contract.public_manifest()
                    if runtime_profile.evaluation_resource_contract is not None
                    else None
                ),
            },
        )

    def public_manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "challenge_id": self.challenge_id,
            "task_id": self.task_id,
            "task": self.task_file,
            "release_stage": self.release_stage,
            "leaderboard_eligible": self.leaderboard_eligible,
            "comparison_key_fields": [
                "challenge_id",
                "track",
                "track_config",
                "runtime_profile",
            ],
            "track_configs": {
                key: value.public_manifest()
                for key, value in self.track_configs.items()
            },
            "runtime_profiles": {
                key: value.public_manifest()
                for key, value in self.runtime_profiles.items()
            },
        }
