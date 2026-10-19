"""Reusable compute profiles and run-owned resource/time configurations.

Time budgets use the current training protocol's Track A/B clocks. Workers own
admission and enforcement; configuration does not allocate any resources.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

from .config import ConfigError, TaskConfig, TrainingResources
from .worker_protocol import EnforcementLevel, ResourceEnvelope, ResourceRequirement

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


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
    gpu_model_match_mode: str = "contains"

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
        model_match_mode = accelerator.get("model_match_mode", "contains")
        if model_match_mode not in ("contains", "exact"):
            raise ConfigError(
                f"{field}.accelerator.model_match_mode must be 'contains' or 'exact'"
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
            gpu_model_match_mode=model_match_mode,
        )

    def training_resources(self) -> TrainingResources:
        return TrainingResources(
            nvidia_gpu_count=self.accelerator_count,
            required_gpu_model=self.gpu_model_match,
            minimum_gpu_memory_mib=self.minimum_memory_mib,
            gpu_model_match_mode=self.gpu_model_match_mode,
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
                        self.gpu_model_match_mode,
                        (self.gpu_model_match,) * self.accelerator_count
                        if self.gpu_model_match_mode == "exact"
                        else self.gpu_model_match,
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
        if (
            self.accelerator_vendor is not None
            and self.gpu_model_match_mode != "contains"
        ):
            manifest["accelerator"]["model_match_mode"] = self.gpu_model_match_mode
        if self.evaluation_resource_contract is not None:
            manifest["evaluation_resource_contract"] = (
                self.evaluation_resource_contract.public_manifest()
            )
        return manifest


@dataclass(frozen=True)
class ResearchBudget:
    """A fixed participant budget, separate from private operator protection."""

    track: str
    total_seconds: Optional[int] = None
    max_leases: Optional[int] = None
    lease_seconds: Optional[int] = None

    def __post_init__(self) -> None:
        if self.track == "A":
            fields = {"total_seconds": self.total_seconds}
            if self.max_leases is not None or self.lease_seconds is not None:
                raise ConfigError("Track A budget may not contain Track B lease fields")
        elif self.track == "B":
            fields = {
                "max_leases": self.max_leases,
                "lease_seconds": self.lease_seconds,
            }
            if self.total_seconds is not None:
                raise ConfigError("Track B budget may not contain total_seconds")
        else:
            raise ConfigError("environment.budget.track must be A or B")
        for name, value in fields.items():
            _positive_int(value, f"environment.budget.{name}")
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ConfigError(f"environment.budget.{name} is too large")

    @classmethod
    def from_manifest(cls, value: Any) -> "ResearchBudget":
        raw = _mapping(value, "environment.budget")
        track = raw.get("track")
        expected = (
            {"track", "total_seconds"}
            if track == "A"
            else {"track", "max_leases", "lease_seconds"}
        )
        if track not in ("A", "B"):
            raise ConfigError("environment.budget.track must be A or B")
        if set(raw) != expected:
            raise ConfigError(
                f"environment.budget for Track {track} requires exactly {sorted(expected)}"
            )
        return cls(**raw)

    @classmethod
    def from_task(cls, task: TaskConfig, track: str) -> "ResearchBudget":
        if track == "A":
            return cls(track, total_seconds=task.track_a_total_seconds)
        return cls(
            track,
            max_leases=task.track_b_max_leases,
            lease_seconds=task.track_b_lease_seconds,
        )

    def apply(self, task: TaskConfig) -> TaskConfig:
        if self.track == "A":
            return replace(task, track_a_total_seconds=self.total_seconds).validate()
        return replace(
            task,
            track_b_max_leases=self.max_leases,
            track_b_lease_seconds=self.lease_seconds,
        ).validate()

    def manifest(self) -> Dict[str, Any]:
        if self.track == "A":
            return {"track": self.track, "total_seconds": self.total_seconds}
        return {
            "track": self.track,
            "max_leases": self.max_leases,
            "lease_seconds": self.lease_seconds,
        }

    def public_budget(self) -> Dict[str, int]:
        if self.track == "A":
            return {"total_wall_clock_seconds": self.total_seconds}
        return {
            "max_leases": self.max_leases,
            "max_seconds_per_lease": self.lease_seconds,
        }


@dataclass(frozen=True)
class EnvironmentConfig:
    """Run-owned compute resources and an optional participant time budget."""

    profile: RuntimeProfile
    budget: Optional[ResearchBudget] = None

    @classmethod
    def load(cls, path: Path) -> "EnvironmentConfig":
        try:
            value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError(f"cannot read environment configuration: {exc}") from exc
        return cls.from_manifest(value)

    @classmethod
    def from_manifest(cls, value: Any) -> "EnvironmentConfig":
        raw = _mapping(value, "environment")
        required = {"schema_version", "name", "runtime_profile"}
        if not required.issubset(raw) or set(raw) - required - {"budget"}:
            raise ConfigError(
                "environment requires schema_version, name and runtime_profile; "
                "budget is optional"
            )
        version = raw["schema_version"]
        if isinstance(version, bool) or not isinstance(version, int) or version != 1:
            raise ConfigError("environment schema_version must be 1")
        name = _identifier(raw["name"], "environment.name")
        profile = _mapping(raw["runtime_profile"], "environment.runtime_profile")
        unknown = set(profile) - {
            "label",
            "host_count",
            "accelerator",
            "provisional",
            "resource_contract",
            "evaluation_resource_contract",
        }
        if unknown:
            raise ConfigError(f"unknown environment profile fields: {sorted(unknown)}")
        accelerator = profile.get("accelerator")
        if isinstance(accelerator, dict):
            unknown = set(accelerator) - {
                "vendor",
                "model",
                "count",
                "minimum_memory_mib",
                "model_match",
                "model_match_mode",
            }
            if unknown:
                raise ConfigError(f"unknown accelerator fields: {sorted(unknown)}")
        result = cls(
            RuntimeProfile.load(name, profile),
            ResearchBudget.from_manifest(raw["budget"]) if "budget" in raw else None,
        )
        # Validate compatibility projections even when no Worker is contacted.
        result.resource_envelope()
        return result

    def resource_envelope(self) -> ResourceEnvelope:
        return self.profile.resource_envelope(
            minimum_enforcement=EnforcementLevel.BEST_EFFORT
        )

    def manifest(self) -> Dict[str, Any]:
        result = {
            "schema_version": 1,
            "name": self.profile.profile_id,
            "runtime_profile": self.profile.public_manifest(),
        }
        if self.budget is not None:
            result["budget"] = self.budget.manifest()
        return result

    def resolve_budget(self, task: TaskConfig, track: str) -> "EnvironmentConfig":
        """Freeze an explicit budget or inherit the selected task/Track budget."""
        budget = (
            ResearchBudget.from_task(task, track)
            if self.budget is None
            else self.budget
        )
        if not isinstance(budget, ResearchBudget):
            raise ConfigError("environment.budget must be a ResearchBudget")
        if budget.track != track:
            raise ConfigError(
                "environment budget track differs from the selected track"
            )
        return replace(self, budget=budget)

    @property
    def configuration_sha256(self) -> str:
        # Display names and provisional status do not change execution conditions.
        configuration = self.manifest()
        configuration.pop("name")
        configuration["runtime_profile"].pop("label")
        configuration["runtime_profile"].pop("provisional")
        encoded = json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def record(self) -> Dict[str, Any]:
        return {
            "configuration": self.manifest(),
            "configuration_sha256": self.configuration_sha256,
        }


def validate_environment_record(record: Any) -> EnvironmentConfig:
    if not isinstance(record, dict) or set(record) != {
        "configuration",
        "configuration_sha256",
    }:
        raise ConfigError("invalid environment record")
    config = EnvironmentConfig.from_manifest(record["configuration"])
    if config.configuration_sha256 != record["configuration_sha256"]:
        raise ConfigError("environment configuration identity mismatch")
    return config


@dataclass(frozen=True)
class FrozenEnvironment:
    """Canonical execution settings, detached from mutable launcher inputs.

    Profiles are configuration conveniences. Every execution freezes the same
    budget, launch resources and Worker contracts, including deployments that
    supply an extensible resource envelope without a named runtime profile.
    """

    _configuration_json: str

    @classmethod
    def create(
        cls,
        *,
        budget: ResearchBudget,
        resources: TrainingResources,
        resource_envelope: ResourceEnvelope,
        evaluation_resource_envelope: Optional[ResourceEnvelope] = None,
    ) -> "FrozenEnvironment":
        if not isinstance(budget, ResearchBudget):
            raise ConfigError("frozen environment requires a resolved research budget")
        resources.validate()
        configuration = {
            "schema_version": 1,
            "budget": budget.manifest(),
            "resources": asdict(resources),
            "resource_envelope": resource_envelope.public_manifest(),
            "evaluation_resource_envelope": (
                evaluation_resource_envelope.public_manifest()
                if evaluation_resource_envelope is not None
                else None
            ),
        }
        return cls(
            json.dumps(
                configuration, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        )

    @classmethod
    def from_config(
        cls, config: EnvironmentConfig, task: TaskConfig, track: str
    ) -> "FrozenEnvironment":
        resolved = config.resolve_budget(task, track)
        return cls.create(
            budget=resolved.budget,
            resources=resolved.profile.training_resources(),
            resource_envelope=resolved.resource_envelope(),
            evaluation_resource_envelope=resolved.profile.evaluation_resource_contract,
        )

    @classmethod
    def from_record(cls, record: Any) -> "FrozenEnvironment":
        if not isinstance(record, dict) or set(record) != {
            "configuration",
            "configuration_sha256",
        }:
            raise ConfigError("invalid frozen environment record")
        raw = _mapping(record["configuration"], "frozen environment")
        if (
            set(raw)
            != {
                "schema_version",
                "budget",
                "resources",
                "resource_envelope",
                "evaluation_resource_envelope",
            }
            or isinstance(raw["schema_version"], bool)
            or not isinstance(raw["schema_version"], int)
            or raw["schema_version"] != 1
        ):
            raise ConfigError("invalid frozen environment configuration")
        resources = _mapping(raw["resources"], "frozen resources")
        if set(resources) != set(asdict(TrainingResources())) or not isinstance(
            resources["devices"], list
        ):
            raise ConfigError("invalid frozen training resources")
        evaluation = raw["evaluation_resource_envelope"]
        result = cls.create(
            budget=ResearchBudget.from_manifest(raw["budget"]),
            resources=TrainingResources(**resources),
            resource_envelope=ResourceEnvelope.from_manifest(raw["resource_envelope"]),
            evaluation_resource_envelope=(
                ResourceEnvelope.from_manifest(evaluation)
                if evaluation is not None
                else None
            ),
        )
        if result.configuration_sha256 != record["configuration_sha256"]:
            raise ConfigError("environment configuration identity mismatch")
        return result

    def manifest(self) -> Dict[str, Any]:
        return json.loads(self._configuration_json)

    @property
    def budget(self) -> ResearchBudget:
        return ResearchBudget.from_manifest(self.manifest()["budget"])

    @property
    def resources(self) -> TrainingResources:
        return TrainingResources(**self.manifest()["resources"])

    @property
    def resource_envelope(self) -> ResourceEnvelope:
        return ResourceEnvelope.from_manifest(self.manifest()["resource_envelope"])

    @property
    def evaluation_resource_envelope(self) -> Optional[ResourceEnvelope]:
        raw = self.manifest()["evaluation_resource_envelope"]
        return ResourceEnvelope.from_manifest(raw) if raw is not None else None

    @property
    def configuration_sha256(self) -> str:
        return hashlib.sha256(self._configuration_json.encode("utf-8")).hexdigest()

    def record(self) -> Dict[str, Any]:
        return {
            "configuration": self.manifest(),
            "configuration_sha256": self.configuration_sha256,
        }
