"""Versioned episode-result schema and metric normalization."""

from __future__ import annotations

import math
from typing import Any, Dict


RESULT_SCHEMA_VERSION = 2
LEGACY_RESULT_SCHEMA_VERSION = 1
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
    if version == RESULT_SCHEMA_VERSION:
        result = dict(payload)
        _validate_current_result(result)
        return result
    if version != LEGACY_RESULT_SCHEMA_VERSION:
        raise ResultSchemaError(f"unsupported result schema version: {version}")
    result = _migrate_legacy_result(payload)
    _validate_current_result(result)
    return result


def _migrate_legacy_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(payload)
    result["result_schema_version"] = RESULT_SCHEMA_VERSION
    result["migrated_from_result_schema_version"] = LEGACY_RESULT_SCHEMA_VERSION
    result.setdefault("test_nll", None)
    result["test_perplexity"] = None

    status = result.get("status")
    candidate_id = result.get("final_candidate_id")
    if (
        status == "finalized"
        and isinstance(candidate_id, str)
        and candidate_id.strip()
    ):
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
    status = result.get("status")
    if status not in _VALID_STATUSES:
        raise ResultSchemaError(f"invalid result status: {status!r}")
    candidate_id = result.get("final_candidate_id")
    if candidate_id is not None and (
        not isinstance(candidate_id, str) or not candidate_id.strip()
    ):
        raise ResultSchemaError("final_candidate_id must be null or a string")

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
