"""Reusable hardware profiles and run-owned environment configurations.

This layer contains no challenge, Track or leaderboard policy. Workers own
admission and enforcement; configuration does not allocate any resources.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .config import ConfigError, TrainingResources
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
class EnvironmentConfig:
    """A user-owned hardware configuration, independent of a benchmark release."""

    profile: RuntimeProfile

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
        if set(raw) != {"schema_version", "name", "runtime_profile"}:
            raise ConfigError(
                "environment requires exactly schema_version, name and runtime_profile"
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
        result = cls(RuntimeProfile.load(name, profile))
        # Validate compatibility projections even when no Worker is contacted.
        result.resource_envelope()
        return result

    def resource_envelope(self) -> ResourceEnvelope:
        return self.profile.resource_envelope(
            minimum_enforcement=EnforcementLevel.BEST_EFFORT
        )

    def manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.profile.profile_id,
            "runtime_profile": self.profile.public_manifest(),
        }

    @property
    def configuration_sha256(self) -> str:
        # Display names and benchmark provisional status do not change hardware.
        hardware = self.profile.public_manifest()
        hardware.pop("label")
        hardware.pop("provisional")
        encoded = json.dumps(
            {"schema_version": 1, "runtime_profile": hardware},
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
