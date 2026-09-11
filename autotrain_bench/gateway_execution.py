"""Trusted launcher evidence for the registered inference Gateway.

These records describe an execution boundary, not a Worker or a launch API.
The deployment must observe the running workload before issuing a receipt.
An artifact identifies *all* executable inputs, including injected code; an
image digest alone is insufficient when code is mounted from elsewhere.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .worker_protocol import EnforcementLevel, TrustLevel


GATEWAY_EXECUTION_VERSION = 1


def manifest_sha256(value: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _sha256(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a SHA-256 digest")


def _fields(value: Any, expected: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("invalid Gateway execution manifest fields")


def inference_policy_sha256(
    protocol: str, allowed_models: Any, allowed_paths: Any
) -> str:
    """Bind reportable inference policy without publishing the private origin."""
    _text(protocol, "inference protocol")
    for name, values in (("models", allowed_models), ("paths", allowed_paths)):
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"inference {name} must be a non-empty array")
        for value in values:
            _text(value, f"inference {name}")
    return manifest_sha256(
        {
            "protocol": protocol,
            "allowed_models": sorted(allowed_models),
            "allowed_paths": sorted(allowed_paths),
        }
    )


@dataclass(frozen=True)
class GatewayArtifactIdentity:
    """Identity of the complete executable artifact, defined by its format.

    The organizer and launcher agree on the format's hashing procedure. Core
    compares their independent identities; it does not resolve platform files.
    """

    format: str
    sha256: str

    def __post_init__(self) -> None:
        _text(self.format, "Gateway artifact format")
        _sha256(self.sha256, "Gateway artifact sha256")

    def private_manifest(self) -> Dict[str, Any]:
        return {"format": self.format, "sha256": self.sha256}

    @classmethod
    def from_manifest(cls, value: Any) -> GatewayArtifactIdentity:
        _fields(value, {"format", "sha256"})
        return cls(**value)


@dataclass(frozen=True)
class GatewayExecutionRequest:
    episode_id: str
    requested_identity: str
    core_revision: Optional[str]
    inference_registration_sha256: str
    inference_policy_sha256: str

    def __post_init__(self) -> None:
        _text(self.episode_id, "Gateway episode_id")
        _text(self.requested_identity, "Gateway requested identity")
        if self.core_revision is not None and (
            not isinstance(self.core_revision, str)
            or len(self.core_revision) not in {40, 64}
            or any(char not in "0123456789abcdef" for char in self.core_revision)
        ):
            raise ValueError("Gateway core_revision must be a Git object ID or null")
        _sha256(self.inference_registration_sha256, "inference registration sha256")
        _sha256(self.inference_policy_sha256, "inference policy sha256")

    @property
    def sha256(self) -> str:
        return manifest_sha256(self.private_manifest())

    def private_manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": GATEWAY_EXECUTION_VERSION,
            "episode_id": self.episode_id,
            "requested_identity": self.requested_identity,
            "core_revision": self.core_revision,
            "inference_registration_sha256": self.inference_registration_sha256,
            "inference_policy_sha256": self.inference_policy_sha256,
        }

    @classmethod
    def from_manifest(cls, value: Any) -> GatewayExecutionRequest:
        _fields(
            value,
            {
                "schema_version",
                "episode_id",
                "requested_identity",
                "core_revision",
                "inference_registration_sha256",
                "inference_policy_sha256",
            },
        )
        fields = dict(value)
        version = fields.pop("schema_version")
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != GATEWAY_EXECUTION_VERSION
        ):
            raise ValueError("unsupported Gateway request version")
        return cls(**fields)


@dataclass(frozen=True)
class GatewayExecutionReceipt:
    """A trusted launcher's observation of an active, fixed Gateway boundary.

    ``ENFORCED`` means the registered inference configuration and artifact are
    enforced for the execution lifetime. ``CERTIFIED`` is supplied only across
    the organizer's trusted launcher boundary; this is not self-certification
    or cryptographic verification of an arbitrary receipt file.
    """

    receipt_id: str
    request_sha256: str
    source: str
    artifact: GatewayArtifactIdentity
    enforcement: EnforcementLevel
    trust_level: TrustLevel
    limitations: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.receipt_id, "Gateway receipt_id")
        _text(self.source, "Gateway receipt source")
        _sha256(self.request_sha256, "Gateway request sha256")
        if not isinstance(self.artifact, GatewayArtifactIdentity):
            raise TypeError("Gateway artifact must be a GatewayArtifactIdentity")
        if not isinstance(self.enforcement, EnforcementLevel):
            raise TypeError("Gateway enforcement must be an EnforcementLevel")
        if not isinstance(self.trust_level, TrustLevel):
            raise TypeError("Gateway trust_level must be a TrustLevel")
        if not isinstance(self.limitations, tuple):
            raise ValueError("Gateway limitations must be a tuple")
        for limitation in self.limitations:
            _text(limitation, "Gateway limitation")
        if len(set(self.limitations)) != len(self.limitations):
            raise ValueError("Gateway limitations must be unique")

    @property
    def certified(self) -> bool:
        return (
            self.enforcement is EnforcementLevel.ENFORCED
            and self.trust_level is TrustLevel.CERTIFIED
            and not self.limitations
        )

    def validate_request(self, request: GatewayExecutionRequest) -> None:
        if self.request_sha256 != request.sha256:
            raise ValueError("Gateway execution receipt does not match this request")

    def private_manifest(self) -> Dict[str, Any]:
        return {
            "receipt_version": GATEWAY_EXECUTION_VERSION,
            "receipt_id": self.receipt_id,
            "request_sha256": self.request_sha256,
            "source": self.source,
            "artifact": self.artifact.private_manifest(),
            "enforcement": self.enforcement.value,
            "trust_level": self.trust_level.value,
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_manifest(cls, value: Any) -> GatewayExecutionReceipt:
        _fields(
            value,
            {
                "receipt_version",
                "receipt_id",
                "request_sha256",
                "source",
                "artifact",
                "enforcement",
                "trust_level",
                "limitations",
            },
        )
        fields = dict(value)
        version = fields.pop("receipt_version")
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != GATEWAY_EXECUTION_VERSION
        ):
            raise ValueError("unsupported Gateway receipt version")
        if not isinstance(fields["limitations"], list):
            raise ValueError("Gateway limitations must be an array")
        fields["limitations"] = tuple(fields["limitations"])
        fields["artifact"] = GatewayArtifactIdentity.from_manifest(fields["artifact"])
        fields["enforcement"] = EnforcementLevel(fields["enforcement"])
        fields["trust_level"] = TrustLevel(fields["trust_level"])
        return cls(**fields)


def gateway_eligibility_violations(
    receipt: Optional[GatewayExecutionReceipt],
    expected_artifact: Optional[GatewayArtifactIdentity],
) -> list[str]:
    violations = []
    if receipt is None:
        violations.append("inference_gateway_workload_evidence_missing")
    elif not receipt.certified:
        violations.append("inference_gateway_execution_not_certified")
    if expected_artifact is None:
        violations.append("inference_gateway_release_artifact_missing")
    elif receipt is not None and receipt.artifact != expected_artifact:
        violations.append("inference_gateway_artifact_mismatch")
    return violations
