"""Versioned episode-result schema and metric normalization."""

from __future__ import annotations

import math
from typing import Any, Dict

from .config import AGENT_EGRESS_POLICIES, AGENT_RUNNERS


RESULT_SCHEMA_VERSION = 3
LEGACY_RESULT_SCHEMA_VERSION = 1
# Version 2 recorded the Agent egress policy as ``agent_system.mode`` with the
# value ``"sandboxed"``.  Version 3 renames that field to ``agent_system.egress``
# with the value ``"denied"`` so a schema version identifies one structure.
AGENT_MODE_RESULT_SCHEMA_VERSION = 2
_SUPPORTED_INPUT_SCHEMA_VERSIONS = frozenset(
    {
        LEGACY_RESULT_SCHEMA_VERSION,
        AGENT_MODE_RESULT_SCHEMA_VERSION,
        RESULT_SCHEMA_VERSION,
    }
)
_VALID_STATUSES = {
    "created",
    "active",
    "closed",
    "dnf",
    "finalized",
    "infrastructure_failure",
}
_PENDING_OUTCOME_STATUSES = {"created", "active", "closed"}
_FAILED_OUTCOME_STATUSES = {
    "dnf",
    "finalized",
    "infrastructure_failure",
}


# The complete set of ``agent_system.mode`` values a version 2 state could hold.
# ``None`` belongs to the host_cli runner, which never had an egress policy.
_V2_EGRESS_BY_AGENT_MODE = {
    None: None,
    "sandboxed": "denied",
    "unrestricted-dev": "unrestricted-dev",
}


class ResultSchemaError(ValueError):
    """An episode state cannot be represented by the requested result schema."""


def validated_nll(value: Any) -> float:
    """Return a finite, non-negative NLL without coercing non-numeric JSON."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("NLL must be numeric")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("NLL is outside the supported numeric range") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError("NLL must be finite and non-negative")
    return result


def perplexity_from_nll(value: Any) -> float:
    """Return finite token perplexity for a validated mean token NLL."""

    nll = validated_nll(value)
    try:
        result = math.exp(nll)
    except OverflowError as exc:
        raise ValueError("perplexity overflowed") from exc
    if not math.isfinite(result):
        raise ValueError("perplexity must be finite")
    return result


def normalize_result_report(payload: Any) -> Dict[str, Any]:
    """Return a current report without mutating the persisted episode state."""

    if not isinstance(payload, dict):
        raise ResultSchemaError("episode state must be an object")
    version = payload.get("result_schema_version", LEGACY_RESULT_SCHEMA_VERSION)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ResultSchemaError("result_schema_version must be an integer")
    if version not in _SUPPORTED_INPUT_SCHEMA_VERSIONS:
        raise ResultSchemaError(f"unsupported result schema version: {version}")
    if version == LEGACY_RESULT_SCHEMA_VERSION:
        result = _migrate_legacy_result(payload)
    else:
        result = dict(payload)
    if version < RESULT_SCHEMA_VERSION:
        result = _migrate_agent_mode_to_egress(result)
        result["result_schema_version"] = RESULT_SCHEMA_VERSION
        result["migrated_from_result_schema_version"] = version
    result.setdefault("deployment_teardown_failures", [])
    result.pop("pending_result", None)
    _validate_current_result(result)
    return result


def _migrate_agent_mode_to_egress(result: Dict[str, Any]) -> Dict[str, Any]:
    """Rename the pre-v3 Agent egress field without inventing a value."""

    agent_system = result.get("agent_system")
    if not isinstance(agent_system, dict) or "mode" not in agent_system:
        return result
    migrated = dict(agent_system)
    mode = migrated.pop("mode")
    if "egress" not in migrated:
        try:
            migrated["egress"] = _V2_EGRESS_BY_AGENT_MODE[mode]
        except (KeyError, TypeError) as exc:
            raise ResultSchemaError(
                f"cannot migrate an unknown Agent mode: {mode!r}"
            ) from exc
    result["agent_system"] = migrated
    return result


def _validate_agent_system(result: Dict[str, Any]) -> None:
    """Pin the current ``agent_system`` structure to the current schema version."""

    agent_system = result.get("agent_system")
    if agent_system is None:
        return
    if not isinstance(agent_system, dict):
        raise ResultSchemaError("agent_system must be an object")
    if "mode" in agent_system:
        raise ResultSchemaError(
            "agent_system.mode is not part of result schema version "
            f"{RESULT_SCHEMA_VERSION}; it was renamed to agent_system.egress"
        )
    runner = agent_system.get("runner")
    if runner is not None and runner not in AGENT_RUNNERS:
        raise ResultSchemaError(f"invalid agent_system.runner: {runner!r}")
    egress = agent_system.get("egress")
    if egress is not None and egress not in AGENT_EGRESS_POLICIES:
        raise ResultSchemaError(f"invalid agent_system.egress: {egress!r}")
    if runner == "host_cli":
        if egress is not None:
            raise ResultSchemaError(
                "a host_cli Agent result may not declare an Agent egress policy"
            )
    elif runner is not None and egress is None:
        raise ResultSchemaError(
            f"a {runner} Agent result must declare an Agent egress policy"
        )


def _migrate_legacy_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(payload)
    result.setdefault("deployment_teardown_failures", [])
    result.setdefault("test_nll", None)
    result["test_perplexity"] = None

    status = result.get("status")
    candidate_id = result.get("final_candidate_id")
    if status == "finalized" and isinstance(candidate_id, str) and candidate_id.strip():
        try:
            nll = validated_nll(result.get("test_nll"))
        except ValueError:
            return _mark_migrated_failure(result, "invalid_test_metric")
        try:
            perplexity = perplexity_from_nll(nll)
        except ValueError:
            return _mark_migrated_failure(result, "perplexity_overflow")
        if result.get("failure") is None:
            result["outcome"] = "success"
            result["test_nll"] = nll
            result["test_perplexity"] = perplexity
            return result

    if status in _FAILED_OUTCOME_STATUSES:
        failure = result.get("failure")
        if not isinstance(failure, str) or not failure.strip():
            failure = "no_candidate" if not candidate_id else "legacy_unscored_result"
        return _mark_migrated_failure(result, failure)

    if status == "closed":
        # ``closed`` is persisted before post-mortem TEST begins.  Reporting a
        # legacy episode during that window must not manufacture a failure.
        result["outcome"] = None
        result["failure"] = None
        result["test_nll"] = None
        return result

    result["outcome"] = None
    result["test_nll"] = None
    return result


def _mark_migrated_failure(result: Dict[str, Any], failure: str) -> Dict[str, Any]:
    result["outcome"] = "failed"
    result["failure"] = failure
    result["test_nll"] = None
    result["test_perplexity"] = None
    return result


def _validate_current_result(result: Dict[str, Any]) -> None:
    _validate_agent_system(result)
    status = result.get("status")
    if status not in _VALID_STATUSES:
        raise ResultSchemaError(f"invalid result status: {status!r}")
    candidate_id = result.get("final_candidate_id")
    if candidate_id is not None and (
        not isinstance(candidate_id, str) or not candidate_id.strip()
    ):
        raise ResultSchemaError("final_candidate_id must be null or a string")
    teardown_failures = result.get("deployment_teardown_failures", [])
    if not isinstance(teardown_failures, list) or any(
        not isinstance(code, str) or not code.strip() for code in teardown_failures
    ):
        raise ResultSchemaError("deployment_teardown_failures must be a string array")
    pending_lease_teardowns = result.get("pending_lease_teardowns", [])
    if not isinstance(pending_lease_teardowns, list) or any(
        not isinstance(lease_id, str) or not lease_id
        for lease_id in pending_lease_teardowns
    ):
        raise ResultSchemaError("pending_lease_teardowns must be a string array")
    if len(set(pending_lease_teardowns)) != len(pending_lease_teardowns):
        raise ResultSchemaError("pending_lease_teardowns must be unique")
    publication_present = "result_publication" in result
    result_publication = result.get("result_publication")
    if result_publication is not None:
        if status != "finalized" or not isinstance(result_publication, dict):
            raise ResultSchemaError(
                "result_publication must be an object on a finalized result"
            )
        publication_id = result_publication.get("publication_id")
        prepared_event_seq = result_publication.get("prepared_event_seq")
        if not isinstance(publication_id, str) or not publication_id.strip():
            raise ResultSchemaError("result publication ID must be a string")
        if (
            isinstance(prepared_event_seq, bool)
            or not isinstance(prepared_event_seq, int)
            or prepared_event_seq < 1
        ):
            raise ResultSchemaError(
                "result publication must reference a prepared event sequence"
            )
    elif publication_present and status == "finalized":
        raise ResultSchemaError(
            "a finalized result cannot contain an empty result_publication"
        )

    outcome = result.get("outcome")
    if outcome not in {None, "success", "failed"}:
        raise ResultSchemaError(f"invalid result outcome: {outcome!r}")
    if outcome is None:
        if status not in _PENDING_OUTCOME_STATUSES:
            raise ResultSchemaError(
                f"status {status!r} requires a success or failed outcome"
            )
        if (
            result.get("test_nll") is not None
            or result.get("test_perplexity") is not None
        ):
            raise ResultSchemaError("a pending result may not contain TEST metrics")
        if result.get("failure") is not None:
            raise ResultSchemaError("a pending result may not contain a failure")
        if status in {"created", "active"} and candidate_id is not None:
            raise ResultSchemaError(
                f"status {status!r} may not contain a final candidate"
            )
    elif outcome == "success":
        if status != "finalized":
            raise ResultSchemaError("a successful result must be finalized")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ResultSchemaError(
                "a successful result must contain a final candidate"
            )
        try:
            nll = validated_nll(result.get("test_nll"))
            expected_perplexity = perplexity_from_nll(nll)
            perplexity = validated_nll(result.get("test_perplexity"))
        except ValueError as exc:
            raise ResultSchemaError(f"invalid successful result metric: {exc}") from exc
        if not math.isclose(perplexity, expected_perplexity, rel_tol=1e-12):
            raise ResultSchemaError("test_perplexity does not equal exp(test_nll)")
        if result.get("failure") is not None:
            raise ResultSchemaError("a successful result may not contain a failure")
        if result.get("active_lease") is not None or pending_lease_teardowns:
            raise ResultSchemaError("a successful result may not retain Worker Leases")
        agent_control = result.get("agent_control")
        if (
            isinstance(agent_control, dict)
            and agent_control.get("cleanup_complete") is not True
        ):
            raise ResultSchemaError(
                "a successful result requires completed Agent-control cleanup"
            )
    elif outcome == "failed":
        if status not in _FAILED_OUTCOME_STATUSES:
            raise ResultSchemaError(
                f"status {status!r} may not contain a failed outcome"
            )
        if (
            result.get("test_nll") is not None
            or result.get("test_perplexity") is not None
        ):
            raise ResultSchemaError("a failed result may not contain TEST metrics")
        failure = result.get("failure")
        if not isinstance(failure, str) or not failure.strip():
            raise ResultSchemaError("a failed result must contain a failure reason")
