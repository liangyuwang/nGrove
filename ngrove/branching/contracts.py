"""Portable contracts for research-controlled branching, independent of Core."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Tuple


class BranchError(ValueError):
    """A stable, public error code, without deployment exception details."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def encoded(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise BranchError("invalid_json") from exc


def identity(value: Any) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def detached(value: Any) -> Any:
    return json.loads(encoded(value))


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,160}", value):
        raise BranchError("invalid_identifier")
    return value


def number(value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BranchError("invalid_quantity")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise BranchError("invalid_quantity") from exc
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise BranchError("invalid_quantity")
    return result


def vector(value: Any) -> dict:
    if not isinstance(value, dict):
        raise BranchError("invalid_resource_vector")
    return {identifier(k): number(v, positive=True) for k, v in value.items()}


def components(value: Any, required: Tuple[str, ...]) -> dict:
    """References describe immutable research-visible exports, never live handles.

    Opaque locators are private adapter inputs. A digest identifies exported bytes;
    the adapter must verify them and pin them independently of episode retention.
    """
    if not isinstance(value, dict) or not set(required).issubset(value):
        raise BranchError("missing_state_components")
    for name, ref in value.items():
        identifier(name)
        if not isinstance(ref, dict) or set(ref) != {"format", "sha256", "locator"}:
            raise BranchError("invalid_component")
        identifier(ref["format"])
        if not isinstance(ref["sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", ref["sha256"]
        ):
            raise BranchError("invalid_component_digest")
        if (
            not isinstance(ref["locator"], str)
            or not ref["locator"]
            or len(ref["locator"]) > 4096
            or "\0" in ref["locator"]
        ):
            raise BranchError("invalid_component_locator")
    return detached(value)


@dataclass(frozen=True)
class RunnerCapabilities:
    """Trusted admission evidence; this is not certification of a provider.

    Formats and fidelity must be tested on the selected runner before registration.
    No proprietary runner is assumed to support context export or restoration.
    """

    runner_id: str
    evidence_id: str
    component_formats: Mapping[str, Tuple[str, ...]]
    context_modes: Tuple[str, ...] = ()
    prefix_modes: Tuple[str, ...] = ("shared",)

    def manifest(self) -> dict:
        identifier(self.runner_id)
        identifier(self.evidence_id)
        if (
            not self.context_modes
            or not set(self.context_modes) <= {"exact", "replay"}
            or not self.prefix_modes
            or not set(self.prefix_modes) <= {"shared", "copy"}
        ):
            raise BranchError("unsupported_runner_capabilities")
        for name, formats in self.component_formats.items():
            identifier(name)
            if not formats or isinstance(formats, str):
                raise BranchError("invalid_component_formats")
            for value in formats:
                identifier(value)
        return detached(asdict(self))


@dataclass(frozen=True)
class SessionPolicy:
    """Frozen logical limits. Physical admission/enforcement remain deployment-owned.

    Resource quantities are platform-neutral capacity units chosen by the operator.
    resource_seconds bounds quantity × allocated wall seconds, including preparation.
    """

    capacity: Mapping[str, float]
    wall_seconds: float
    research_seconds: float
    branch_seconds: float
    resource_seconds: Mapping[str, float] = field(default_factory=dict)
    max_branches: int = 128
    max_requests: int = 10000
    max_payload_bytes: int = 4 * 1024 * 1024
    required_components: Tuple[str, ...] = ("agent_context", "workspace")

    def manifest(self) -> dict:
        vector(dict(self.capacity))
        for value in (self.wall_seconds, self.research_seconds, self.branch_seconds):
            number(value, positive=True)
        vector(dict(self.resource_seconds))
        if not set(self.resource_seconds) <= set(self.capacity):
            raise BranchError("unknown_resource")
        for value in (self.max_branches, self.max_requests, self.max_payload_bytes):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise BranchError("invalid_limit")
        if not {"agent_context", "workspace"} <= set(self.required_components):
            raise BranchError("missing_required_components")
        for name in self.required_components:
            identifier(name)
        return detached(asdict(self))
