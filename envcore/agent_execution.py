"""Semantic evidence about the deployment boundary around a Tested Agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .worker_protocol import EnforcementLevel, TrustLevel


@dataclass(frozen=True)
class AgentExecutionEvidence:
    """A trusted launcher's v0 assessment, independent of mechanism names.

    This is intentionally smaller than a future cross-platform attestation
    format.  It establishes the dependency direction without guessing every
    deployment's evidence model.
    """

    source: str
    mechanism: str
    enforcement: EnforcementLevel
    trust_level: TrustLevel
    workload_identity: Optional[str]
    immutable_workload_identity: bool
    limitations: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Agent execution evidence source must not be empty")
        if not isinstance(self.mechanism, str) or not self.mechanism.strip():
            raise ValueError("Agent execution mechanism must not be empty")
        if not isinstance(self.enforcement, EnforcementLevel):
            raise TypeError("Agent execution enforcement must be an EnforcementLevel")
        if not isinstance(self.trust_level, TrustLevel):
            raise TypeError("Agent execution trust_level must be a TrustLevel")
        if not isinstance(self.immutable_workload_identity, bool):
            raise TypeError("immutable_workload_identity must be a boolean")
        if self.workload_identity is not None and (
            not isinstance(self.workload_identity, str)
            or not self.workload_identity.strip()
        ):
            raise ValueError(
                "Agent workload identity must be null or a non-empty string"
            )
        if self.immutable_workload_identity and not self.workload_identity:
            raise ValueError("an immutable Agent workload must have an identity")
        if not isinstance(self.limitations, tuple) or any(
            not isinstance(item, str) or not item for item in self.limitations
        ):
            raise ValueError("Agent execution limitations must be non-empty strings")
        if len(set(self.limitations)) != len(self.limitations):
            raise ValueError("Agent execution limitations must be unique")

    def public_manifest(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "mechanism": self.mechanism,
            "enforcement": self.enforcement.value,
            "trust_level": self.trust_level.value,
            "workload_identity": self.workload_identity,
            "immutable_workload_identity": self.immutable_workload_identity,
            "limitations": list(self.limitations),
        }
