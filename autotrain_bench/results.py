"""Versioned episode-result schema and metric normalization."""

from __future__ import annotations

import math
from typing import Any, Dict


RESULT_SCHEMA_VERSION = 2
LEGACY_RESULT_SCHEMA_VERSION = 1
_TERMINAL_STATUSES = {
    "closed",
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
    return _migrate_legacy_result(payload)


def _migrate_legacy_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(payload)
    result["result_schema_version"] = RESULT_SCHEMA_VERSION
    result["migrated_from_result_schema_version"] = LEGACY_RESULT_SCHEMA_VERSION
    result.setdefault("test_nll", None)
    result["test_perplexity"] = None

    status = result.get("status")
    candidate_id = result.get("final_candidate_id")
    if status == "finalized" and isinstance(candidate_id, str) and candidate_id:
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

    if status in _TERMINAL_STATUSES:
        failure = result.get("failure")
        if not isinstance(failure, str) or not failure:
            failure = "no_candidate" if not candidate_id else "legacy_unscored_result"
        return _mark_migrated_failure(result, failure)

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
    outcome = result.get("outcome")
    if outcome not in {None, "success", "failed"}:
        raise ResultSchemaError(f"invalid result outcome: {outcome!r}")
    if outcome == "success":
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
        if (
            result.get("test_nll") is not None
            or result.get("test_perplexity") is not None
        ):
            raise ResultSchemaError("a failed result may not contain TEST metrics")
        failure = result.get("failure")
        if not isinstance(failure, str) or not failure:
            raise ResultSchemaError("a failed result must contain a failure reason")
