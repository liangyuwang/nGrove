"""Benchmark release profiles and leaderboard comparison identities.

A benchmark fixes data and evaluation semantics without fixing one hardware
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
from .config import ConfigError, TaskConfig
from .gateway_execution import GatewayArtifactIdentity, gateway_eligibility_violations
from .environment import (
    EnvironmentConfig,
    ResearchBudget,
    RuntimeProfile as RuntimeProfile,
)
from .run_config import EnvironmentRun, resolve_environment_run
from .worker_protocol import (
    EnforcementLevel,
    ResourceEnvelope,
    WorkloadResolutionAcceptance,
)


RELEASE_STAGES = {"preview", "official"}
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
FORMAL_ARTIFACT_DIGEST_FIELDS = (
    "benchmark_manifest_sha256",
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
            "benchmark_id",
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
    """One named, immutable budget configuration within a benchmark release."""

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
class ComparisonKey:
    benchmark_id: str
    track: str
    track_config: str
    runtime_profile: str

    def public_manifest(self) -> Dict[str, str]:
        return {
            "benchmark_id": self.benchmark_id,
            "track": self.track,
            "track_config": self.track_config,
            "runtime_profile": self.runtime_profile,
        }


@dataclass(frozen=True)
class ReleaseVerificationReceipt:
    """Trusted organizer-side verification supplied by a formal deployment.

    Benchmark manifests can describe and hash themselves, but that does not
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
    inference_gateway_artifact: Optional[GatewayArtifactIdentity] = None

    def validate(self) -> "ReleaseVerificationReceipt":
        if self.inference_gateway_artifact is not None and not isinstance(
            self.inference_gateway_artifact, GatewayArtifactIdentity
        ):
            raise ConfigError(
                "inference_gateway_artifact must be a GatewayArtifactIdentity"
            )
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
            "inference_gateway_artifact": (
                self.inference_gateway_artifact.private_manifest()
                if self.inference_gateway_artifact is not None
                else None
            ),
        }


@dataclass(frozen=True)
class ResolvedBenchmarkRun(EnvironmentRun):
    comparison_key: Optional[ComparisonKey]
    release_manifest: Dict[str, Any]


@dataclass(frozen=True)
class BenchmarkRelease:
    """A finite set of supported comparison lanes for one benchmark release."""

    source: Path
    manifest_sha256: str
    task_manifest_sha256: str
    schema_version: int
    benchmark_id: str
    task_id: str
    task_file: str
    release_stage: str
    leaderboard_eligible: bool
    track_configs: Dict[str, TrackConfig]
    runtime_profiles: Dict[str, RuntimeProfile]

    @classmethod
    def load(cls, path: Path) -> "BenchmarkRelease":
        source = path.expanduser().resolve()
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(f"benchmark release not found: {source}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON in {source}: {exc}") from exc
        root = _mapping(raw, "root")
        schema_version = _positive_int(root.get("schema_version"), "schema_version")
        if schema_version not in {2, 3, 4}:
            raise ConfigError("benchmark schema_version must be 2, 3, or 4")
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
        if schema_version < 4 and any(
            profile.gpu_model_match_mode != "contains"
            for profile in runtime_profiles.values()
        ):
            raise ConfigError(
                "exact GPU model matching requires benchmark schema_version 4"
            )
        if schema_version < 3 and any(
            profile.evaluation_resource_contract is not None
            for profile in runtime_profiles.values()
        ):
            raise ConfigError(
                "evaluation_resource_contract requires benchmark schema_version 3"
            )
        task_file = _string(root.get("task"), "task")
        configured_task = Path(task_file)
        if configured_task.is_absolute() or ".." in configured_task.parts:
            raise ConfigError("task must be a relative path inside the benchmark")
        task_path = (source.parent / configured_task).resolve()
        try:
            task_path.relative_to(source.parent)
        except ValueError as exc:
            raise ConfigError("task escapes the benchmark directory") from exc
        if not task_path.is_file():
            raise ConfigError(f"benchmark task not found: {task_path}")
        release = cls(
            source=source,
            manifest_sha256=_sha256_file(source),
            task_manifest_sha256=_sha256_file(task_path),
            schema_version=schema_version,
            benchmark_id=_identifier(root.get("benchmark_id"), "benchmark_id"),
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

    def validate(self) -> "BenchmarkRelease":
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
        runtime_profile_id: Optional[str] = None,
        environment: Optional[EnvironmentConfig] = None,
    ) -> ResolvedBenchmarkRun:
        if task.task_id != self.task_id:
            raise ConfigError(
                f"benchmark expects task_id {self.task_id!r}, received {task.task_id!r}"
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
                raise ConfigError("benchmark manifest changed after it was loaded")
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
        runtime_profile = None
        if runtime_profile_id is not None:
            try:
                runtime_profile = self.runtime_profiles[runtime_profile_id]
            except KeyError as exc:
                raise ConfigError(
                    f"unknown runtime profile: {runtime_profile_id}"
                ) from exc
        if runtime_profile is None and environment is None:
            raise ConfigError("select --runtime-profile or --environment")
        task = track_config.apply(task)
        published = None
        if runtime_profile is not None:
            # Published requirements are a preset for the same environment resolver.
            profile = replace(
                runtime_profile,
                resource_contract=runtime_profile.resource_envelope(
                    minimum_enforcement=(
                        EnforcementLevel.ENFORCED
                        if self.leaderboard_eligible
                        else EnforcementLevel.BEST_EFFORT
                    )
                ),
            )
            published = resolve_environment_run(
                task, track=track_config.track, environment=EnvironmentConfig(profile)
            )
        run = (
            resolve_environment_run(
                task, track=track_config.track, environment=environment
            )
            if environment is not None
            else published
        )
        matches = published is not None and run.environment == published.environment
        comparison_key = (
            ComparisonKey(
                benchmark_id=self.benchmark_id,
                track=track_config.track,
                track_config=track_config.config_id,
                runtime_profile=runtime_profile.profile_id,
            )
            if matches
            else None
        )
        return ResolvedBenchmarkRun(
            task=replace(
                run.task, leaderboard_eligible=self.leaderboard_eligible and matches
            ),
            track=run.track,
            environment=run.environment,
            comparison_key=comparison_key,
            release_manifest={
                "benchmark_id": self.benchmark_id,
                "benchmark_manifest_sha256": self.manifest_sha256,
                "task_manifest_sha256": self.task_manifest_sha256,
                "release_stage": self.release_stage,
                "leaderboard_eligible": self.leaderboard_eligible,
                "track_config_id": track_config.config_id,
                "track_config": track_config.public_manifest(),
                "runtime_profile_id": runtime_profile_id,
                "runtime_profile": runtime_profile.public_manifest()
                if runtime_profile is not None
                else None,
                "comparison_eligible": matches,
                "environment_sha256": run.environment.configuration_sha256,
                "resource_envelope": published.resource_envelope.public_manifest()
                if matches
                else None,
                "evaluation_resource_envelope": published.evaluation_resource_envelope.public_manifest()
                if matches and published.evaluation_resource_envelope is not None
                else None,
            },
        )

    def public_manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "benchmark_id": self.benchmark_id,
            "task_id": self.task_id,
            "task": self.task_file,
            "release_stage": self.release_stage,
            "leaderboard_eligible": self.leaderboard_eligible,
            "comparison_key_fields": [
                "benchmark_id",
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


def benchmark_eligibility_violations(
    *,
    config,
    policy_compliant,
    research_assessment,
    evaluation_assessment,
    comparison_key,
    benchmark_release,
    agent_execution,
    agent,
    release_verification,
    artifact_identities,
    core_identity,
    track,
    resource_envelope,
    evaluation_resource_envelope,
    frozen_evaluation_envelope,
):
    """Assess optional benchmark evidence at one application boundary."""
    if (
        not config.leaderboard_eligible
        and benchmark_release is None
        and comparison_key is None
        and release_verification is None
    ):
        return []
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
        "benchmark_id",
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
    elif comparison_key.get("track") != track:
        eligibility_violations.append("comparison_key_track_mismatch")
    release_ready = isinstance(benchmark_release, dict)
    if not release_ready:
        eligibility_violations.append("benchmark_release_missing")
    else:
        if benchmark_release.get("release_stage") != "official":
            eligibility_violations.append("benchmark_release_not_official")
        if benchmark_release.get("leaderboard_eligible") is not True:
            eligibility_violations.append("benchmark_release_not_eligible")
        if isinstance(comparison_key, dict) and benchmark_release.get(
            "benchmark_id"
        ) != comparison_key.get("benchmark_id"):
            eligibility_violations.append("benchmark_release_identity_mismatch")
        for digest_field in (
            "benchmark_manifest_sha256",
            "task_manifest_sha256",
        ):
            digest = benchmark_release.get(digest_field)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(value not in "0123456789abcdef" for value in digest)
            ):
                eligibility_violations.append(f"{digest_field}_missing")
        if (
            benchmark_release.get("task_manifest_sha256")
            != artifact_identities["task_manifest_sha256"]
        ):
            eligibility_violations.append("task_manifest_identity_mismatch")
        for field in ("track_config", "runtime_profile"):
            selected = benchmark_release.get(field)
            if (
                not isinstance(selected, dict)
                or selected.get("provisional") is not False
            ):
                eligibility_violations.append(f"{field}_not_frozen")
        release_resource_envelope = benchmark_release.get("resource_envelope")
        if release_resource_envelope is None:
            eligibility_violations.append("resource_envelope_not_frozen")
        elif release_resource_envelope != resource_envelope.public_manifest():
            eligibility_violations.append("resource_envelope_mismatch")
        if isinstance(comparison_key, dict):
            for release_field, comparison_field in (
                ("track_config_id", "track_config"),
                ("runtime_profile_id", "runtime_profile"),
            ):
                if benchmark_release.get(release_field) != comparison_key.get(
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
                release_verification.inference_gateway_artifact
                if release_verification is not None
                else None,
            )
        )
    core_revision = core_identity.get("revision")
    if not isinstance(core_revision, str) or not core_revision:
        eligibility_violations.append("core_identity_unknown")
    if core_identity.get("dirty") is not False:
        eligibility_violations.append("core_worktree_not_clean")
    if release_verification is None:
        eligibility_violations.append("formal_release_verification_missing")
    else:
        receipt = release_verification
        if not isinstance(
            comparison_key, dict
        ) or receipt.release_id != comparison_key.get("benchmark_id"):
            eligibility_violations.append(
                "formal_release_verification_identity_mismatch"
            )
        if receipt.core_revision != core_revision:
            eligibility_violations.append("formal_release_verification_core_mismatch")
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
            evaluation_resource_envelope is None
            or not evaluation_resource_envelope.requirements
            or any(
                item.minimum_enforcement is not EnforcementLevel.ENFORCED
                for item in evaluation_resource_envelope.requirements
            )
        ):
            eligibility_violations.append("evaluation_resource_envelope_not_formal")
        if evaluation_resource_envelope is not None and (
            frozen_evaluation_envelope is None
            or frozen_evaluation_envelope.sha256 != evaluation_resource_envelope.sha256
        ):
            eligibility_violations.append("evaluation_resource_envelope_not_frozen")
        if not resource_envelope.requirements or any(
            requirement.minimum_enforcement is not EnforcementLevel.ENFORCED
            for requirement in resource_envelope.requirements
        ):
            eligibility_violations.append("resource_envelope_not_formal")
        eligibility_violations.extend(
            f"workload_resolution_pending:{role}"
            for role in ("training", "candidate", "evaluator")
        )
        eligibility_violations.append("resource_allocation_pending")
        eligibility_violations.append("evaluation_resource_allocation_pending")
    return eligibility_violations


def resolve_release_evaluation_envelope(release, explicit):
    """Bind an optional published evaluation preset before execution freezes."""
    if explicit is not None and not isinstance(explicit, ResourceEnvelope):
        raise TypeError("evaluation_resource_envelope must be a ResourceEnvelope")
    raw = (
        release.get("evaluation_resource_envelope")
        if isinstance(release, dict)
        else None
    )
    frozen = ResourceEnvelope.from_manifest(raw) if raw is not None else None
    if frozen is not None:
        if explicit is None:
            explicit = frozen
        elif explicit.sha256 != frozen.sha256:
            raise ValueError("evaluation resource envelope differs from the release")
    return explicit, frozen


def validate_benchmark_binding(result):
    """Bind benchmark evidence and comparison labels to the frozen environment."""
    key = result.get("comparison_key")
    release = result.get("benchmark_release")
    # Custom runs retain their source release and its execution identity even
    # though they do not claim a published comparison lane.
    if isinstance(release, dict):
        expected_identity = release.get("environment_sha256")
        environment = result.get("environment")
        if expected_identity is not None and (
            not isinstance(environment, dict)
            or expected_identity != environment.get("configuration_sha256")
        ):
            raise ValueError("benchmark environment identity mismatch")
    if key is None:
        if result.get("leaderboard_eligible") is True:
            raise ValueError(
                "leaderboard eligibility requires a benchmark comparison key"
            )
        return
    fields = {"benchmark_id", "track", "track_config", "runtime_profile"}
    if (
        not isinstance(key, dict)
        or set(key) != fields
        or any(not isinstance(value, str) or not value for value in key.values())
        or not isinstance(release, dict)
    ):
        raise ValueError("invalid benchmark comparison key or release")
    if key["track"] != result.get("track") or any(
        release.get(release_field) != key[key_field]
        for release_field, key_field in (
            ("benchmark_id", "benchmark_id"),
            ("track_config_id", "track_config"),
            ("runtime_profile_id", "runtime_profile"),
        )
    ):
        raise ValueError("benchmark comparison key does not match the selected release")
    # A comparison key requires matching budget and resource contracts even
    # for preview runs or deployments without a separate environment hash.
    budget = TrackConfig.load(key["track_config"], release.get("track_config"))

    expected_budget = ResearchBudget(
        budget.track, budget.total_seconds, budget.max_leases, budget.lease_seconds
    )
    if budget.track != result.get(
        "track"
    ) or expected_budget.public_budget() != result.get("budget"):
        raise ValueError(
            "benchmark comparison budget does not match the frozen environment"
        )
    for contract_field in ("resource_envelope", "evaluation_resource_envelope"):
        declared, actual = release.get(contract_field), result.get(contract_field)
        if (declared is None) != (actual is None) or (
            declared is not None
            and ResourceEnvelope.from_manifest(declared).sha256
            != ResourceEnvelope.from_manifest(actual).sha256
        ):
            raise ValueError(
                "benchmark comparison resources do not match the frozen environment"
            )
