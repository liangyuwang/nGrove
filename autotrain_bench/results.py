"""Versioned episode-result schema and metric normalization."""

from __future__ import annotations

import math
from typing import Any, Dict

from .config import AGENT_EGRESS_POLICIES, AGENT_RUNNERS


RESULT_SCHEMA_VERSION = 7
LEGACY_RESULT_SCHEMA_VERSION = 1
# Version 2 recorded the Agent egress policy as ``agent_system.mode`` with the
# value ``"sandboxed"``.  Version 3 renames that field to ``agent_system.egress``
# with the value ``"denied"`` so a schema version identifies one structure.
AGENT_MODE_RESULT_SCHEMA_VERSION = 2
PRE_ANALYSIS_RESULT_SCHEMA_VERSION = 3
POSTMORTEM_RESULT_SCHEMA_VERSION = 4
# Version 5 carried the ANALYSIS trajectory but no summary counts, so a reader
# of the summary fields alone could not see that a `complete` pass had gaps.
UNCOUNTED_ANALYSIS_RESULT_SCHEMA_VERSION = 5
PRE_WORKLOAD_RESOLUTION_RESULT_SCHEMA_VERSION = 6
_SUPPORTED_INPUT_SCHEMA_VERSIONS = frozenset(
    {
        LEGACY_RESULT_SCHEMA_VERSION,
        AGENT_MODE_RESULT_SCHEMA_VERSION,
        PRE_ANALYSIS_RESULT_SCHEMA_VERSION,
        POSTMORTEM_RESULT_SCHEMA_VERSION,
        UNCOUNTED_ANALYSIS_RESULT_SCHEMA_VERSION,
        PRE_WORKLOAD_RESOLUTION_RESULT_SCHEMA_VERSION,
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
    if version < PRE_ANALYSIS_RESULT_SCHEMA_VERSION:
        result = _migrate_agent_mode_to_egress(result)
    if version < POSTMORTEM_RESULT_SCHEMA_VERSION:
        result = _migrate_pre_analysis_result(result)
    if version < UNCOUNTED_ANALYSIS_RESULT_SCHEMA_VERSION:
        result = _migrate_v4_analysis_result(result)
    if version < PRE_WORKLOAD_RESOLUTION_RESULT_SCHEMA_VERSION:
        result = _migrate_uncounted_analysis_result(result)
    if version < RESULT_SCHEMA_VERSION:
        result = _migrate_pre_workload_resolution_result(result)
        result["result_schema_version"] = RESULT_SCHEMA_VERSION
        result["migrated_from_result_schema_version"] = version
    result.setdefault("deployment_teardown_failures", [])
    result.pop("pending_result", None)
    # Validation compatibility follows the schema version actually presented
    # to this call.  An input field that merely claims an earlier migration may
    # be useful provenance, but must not switch off current-schema invariants.
    _validate_current_result(
        result,
        migrated_legacy=version < RESULT_SCHEMA_VERSION,
    )
    return result


def _migrate_uncounted_analysis_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Derive the ANALYSIS summary counts a pre-v6 result did not carry.

    The counts are a lossless function of the trajectory that version 5 already
    required, so this migration invents nothing.
    """

    migrated = dict(result)
    trajectory = migrated.get("analysis_trajectory")
    if not isinstance(trajectory, list):
        trajectory = []
    # These fields did not belong to the older schemas.  Always replace any
    # colliding extension fields instead of trusting them as migration input.
    migrated["analysis_selected_points"] = len(trajectory)
    migrated["analysis_evaluated_points"] = sum(
        isinstance(point, dict) and point.get("analysis_status") == "evaluated"
        for point in trajectory
    )
    return migrated


def _migrate_pre_workload_resolution_result(
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Represent missing pre-v7 evidence explicitly instead of inventing it."""

    migrated = dict(result)
    migrated["workload_resolutions"] = {
        "training": None,
        "candidate": None,
        "evaluator": None,
    }
    return migrated


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


def _migrate_pre_analysis_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Preserve legacy VAL evidence without relabelling it as ANALYSIS."""

    migrated = dict(result)
    migrated["legacy_semantics"] = {
        "organizer_validation_feedback": True,
        "postmortem_analysis": False,
    }
    migrated.setdefault("research_sealed_at", None)
    migrated.setdefault("analysis_complete", False)
    migrated.setdefault("analysis_status", "legacy_not_run")
    migrated.setdefault("analysis_failure", None)
    migrated.setdefault("analysis_identity_verified", None)
    migrated.setdefault("analysis_trajectory", [])
    migrated.setdefault("candidate_commits", [])
    migrated.setdefault("incumbent_declarations", [])
    if migrated.get("outcome") == "success":
        migrated.setdefault("test_status", "evaluated")
    elif migrated.get("status") in _FAILED_OUTCOME_STATUSES:
        migrated.setdefault("test_status", "failed")
    else:
        migrated.setdefault("test_status", None)

    candidates = migrated.get("candidates")
    if isinstance(candidates, dict):
        migrated_candidates = {}
        for candidate_id, metadata in candidates.items():
            if not isinstance(metadata, dict):
                migrated_candidates[candidate_id] = metadata
                continue
            item = dict(metadata)
            legacy_validation = {
                "nll": item.pop("val_nll", None),
                "perplexity": item.pop("val_perplexity", None),
                "error": item.pop("evaluation_error", None),
            }
            if any(value is not None for value in legacy_validation.values()):
                item["legacy_validation"] = legacy_validation
            item.setdefault("analysis_status", "legacy_not_run")
            item.setdefault("analysis_nll", None)
            item.setdefault("analysis_perplexity", None)
            item.setdefault("analysis_error", None)
            migrated_candidates[candidate_id] = item
        migrated["candidates"] = migrated_candidates

    timing = migrated.get("timing")
    if isinstance(timing, dict):
        migrated_timing = dict(timing)
        if "validation_evaluation_seconds" in migrated_timing:
            migrated_timing["legacy_validation_evaluation_seconds"] = (
                migrated_timing.pop("validation_evaluation_seconds")
            )
        migrated_timing.setdefault("analysis_evaluation_seconds", 0.0)
        migrated["timing"] = migrated_timing

    artifacts = migrated.get("artifact_identities")
    if isinstance(artifacts, dict) and "hidden_splits" in artifacts:
        migrated_artifacts = dict(artifacts)
        legacy_hidden = migrated_artifacts.pop("hidden_splits")
        migrated_artifacts["legacy_hidden_splits"] = legacy_hidden
        test_digest = (
            legacy_hidden.get("test_sha256")
            if isinstance(legacy_hidden, dict)
            else None
        )
        migrated_artifacts.setdefault(
            "hidden_evaluation",
            {"analysis_sha256": None, "test_sha256": test_digest},
        )
        migrated["artifact_identities"] = migrated_artifacts
    return migrated


def _migrate_v4_analysis_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Add v5 lifecycle axes without inventing missing declaration history."""

    migrated = dict(result)
    analysis_complete = migrated.get("analysis_complete")
    status = migrated.get("status")
    if analysis_complete is True:
        analysis_status = "complete"
    elif status in {"closed", "finalized"}:
        analysis_status = "incomplete" if status == "finalized" else "pending"
    else:
        analysis_status = None
    migrated.setdefault("analysis_status", analysis_status)
    migrated.setdefault(
        "analysis_failure",
        "legacy_analysis_incomplete" if analysis_status == "incomplete" else None,
    )
    migrated.setdefault("analysis_identity_verified", None)
    migrated.setdefault("incumbent_declarations", [])
    if migrated.get("outcome") == "success":
        migrated.setdefault("test_status", "evaluated")
    elif status == "finalized":
        migrated.setdefault("test_status", "failed")
    elif status == "closed":
        migrated.setdefault("test_status", "pending")
    else:
        migrated.setdefault("test_status", None)

    trajectory = migrated.get("analysis_trajectory")
    if isinstance(trajectory, list):
        migrated_trajectory = []
        for point in trajectory:
            if not isinstance(point, dict):
                migrated_trajectory.append(point)
                continue
            item = dict(point)
            item.setdefault("declaration_event_seqs", [])
            migrated_trajectory.append(item)
        migrated["analysis_trajectory"] = migrated_trajectory
    migrated.setdefault(
        "migration_limitations", ["incumbent_declaration_history_unavailable"]
    )
    return migrated


def _validate_workload_resolutions(
    value: Any,
    *,
    require_evidence: bool,
    leaderboard_eligible: bool,
    release_verification: Any,
) -> None:
    roles = {"training", "candidate", "evaluator"}
    if value is None and not require_evidence:
        return
    if not isinstance(value, dict) or set(value) != roles:
        raise ResultSchemaError(
            "workload_resolutions must contain training, candidate, and evaluator"
        )
    for role in sorted(roles):
        evidence = value[role]
        if evidence is None:
            if leaderboard_eligible and require_evidence:
                raise ResultSchemaError(
                    "a leaderboard-eligible result requires every workload resolution"
                )
            continue
        if not isinstance(evidence, dict) or evidence.get("role") != role:
            raise ResultSchemaError(
                f"workload resolution {role} must be an object with a matching role"
            )
        requested = evidence.get("requested_identity")
        if (
            not isinstance(requested, str)
            or not requested
            or len(requested) > 2048
            or any(character.isspace() or character == "\0" for character in requested)
        ):
            raise ResultSchemaError(
                f"workload resolution {role} has an invalid requested identity"
            )
        effective_format = evidence.get("effective_format")
        if (
            not isinstance(effective_format, str)
            or not effective_format
            or len(effective_format) > 64
            or not effective_format[0].isalnum()
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789._+-"
                for character in effective_format
            )
        ):
            raise ResultSchemaError(
                f"workload resolution {role} has an invalid effective format"
            )
        for field in ("effective_sha256", "transformation_sha256"):
            digest = evidence.get(field)
            if field == "transformation_sha256" and digest is None:
                continue
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ResultSchemaError(
                    f"workload resolution {role} has an invalid {field}"
                )
        worker_id = evidence.get("worker_id")
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ResultSchemaError(
                f"workload resolution {role} requires a Worker identity"
            )
        acceptance = evidence.get("acceptance")
        if acceptance not in {
            "canonical_identity",
            "organizer_receipt",
            "unverified",
        }:
            raise ResultSchemaError(
                f"workload resolution {role} has an invalid acceptance status"
            )
        if acceptance == "canonical_identity":
            marker = "@sha256:"
            requested_digest = (
                requested.rsplit(marker, 1)[1] if marker in requested else None
            )
            if (
                requested_digest != evidence["effective_sha256"]
                or evidence["effective_format"] != "oci"
                or evidence.get("transformation_sha256") is not None
            ):
                raise ResultSchemaError(
                    f"workload resolution {role} is not a canonical identity"
                )
        if acceptance == "organizer_receipt":
            accepted = (
                release_verification.get("workload_resolution_acceptance", {}).get(role)
                if isinstance(release_verification, dict)
                else None
            )
            observed = {
                field: evidence.get(field)
                for field in (
                    "role",
                    "requested_identity",
                    "effective_format",
                    "effective_sha256",
                    "transformation_sha256",
                )
            }
            if accepted != observed:
                raise ResultSchemaError(
                    f"workload resolution {role} lacks matching organizer acceptance"
                )
        if leaderboard_eligible and require_evidence and acceptance == "unverified":
            raise ResultSchemaError(
                "a leaderboard-eligible result cannot use an unverified workload"
            )


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


def _validate_current_result(
    result: Dict[str, Any], *, migrated_legacy: bool = False
) -> None:
    _validate_agent_system(result)
    _validate_workload_resolutions(
        result.get("workload_resolutions"),
        require_evidence=not migrated_legacy,
        leaderboard_eligible=result.get("leaderboard_eligible") is True,
        release_verification=result.get("formal_release_verification"),
    )
    status = result.get("status")
    if status not in _VALID_STATUSES:
        raise ResultSchemaError(f"invalid result status: {status!r}")
    analysis_complete = result.get("analysis_complete")
    analysis_status = result.get("analysis_status")
    analysis_failure = result.get("analysis_failure")
    analysis_identity_verified = result.get("analysis_identity_verified")
    analysis_trajectory = result.get("analysis_trajectory")
    research_sealed_at = result.get("research_sealed_at")
    if analysis_complete is not None and not isinstance(analysis_complete, bool):
        raise ResultSchemaError("analysis_complete must be null or boolean")
    if analysis_trajectory is not None and not isinstance(analysis_trajectory, list):
        raise ResultSchemaError("analysis_trajectory must be an array")
    if isinstance(analysis_trajectory, list):
        _validate_analysis_trajectory(analysis_trajectory)
        _validate_analysis_summary_counts(result, analysis_trajectory)
    if not migrated_legacy:
        if analysis_status not in {None, "pending", "complete", "incomplete"}:
            raise ResultSchemaError("invalid ANALYSIS lifecycle status")
        if analysis_status == "complete" and analysis_complete is not True:
            raise ResultSchemaError("complete ANALYSIS requires analysis_complete")
        if analysis_status == "incomplete":
            if analysis_complete is not False:
                raise ResultSchemaError("incomplete ANALYSIS cannot be complete")
            if not isinstance(analysis_failure, str) or not analysis_failure.strip():
                raise ResultSchemaError("incomplete ANALYSIS requires a failure code")
        elif analysis_failure is not None:
            raise ResultSchemaError(
                "only incomplete ANALYSIS may contain a failure code"
            )
        if analysis_identity_verified is not None and not isinstance(
            analysis_identity_verified, bool
        ):
            raise ResultSchemaError(
                "analysis_identity_verified must be null or boolean"
            )
        if not isinstance(analysis_trajectory, list):
            raise ResultSchemaError(
                "current results require an ANALYSIS trajectory array"
            )
        if status in {"closed", "finalized"} and (
            not isinstance(research_sealed_at, str) or not research_sealed_at.strip()
        ):
            raise ResultSchemaError("sealed research requires research_sealed_at")
        if status == "finalized" and analysis_status not in {
            "complete",
            "incomplete",
        }:
            raise ResultSchemaError(
                "a finalized result requires a postmortem ANALYSIS disposition"
            )
        test_status = result.get("test_status")
        if test_status not in {None, "pending", "evaluated", "failed"}:
            raise ResultSchemaError("invalid TEST lifecycle status")
        candidate_commits = result.get("candidate_commits")
        incumbent_declarations = result.get("incumbent_declarations")
        if not isinstance(candidate_commits, list):
            raise ResultSchemaError("current results require candidate_commits")
        if not isinstance(incumbent_declarations, list):
            raise ResultSchemaError("current results require incumbent_declarations")
        _validate_candidate_commits(candidate_commits)
        _validate_incumbent_declarations(
            incumbent_declarations,
            candidate_commits,
        )
        _validate_analysis_binding(
            analysis_trajectory,
            candidate_commits,
            incumbent_declarations,
            require_complete=status == "finalized",
        )
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
        if not migrated_legacy and result.get("test_status") != "evaluated":
            raise ResultSchemaError("a successful result requires evaluated TEST")
        if not migrated_legacy and not any(
            point["candidate_id"] == candidate_id for point in analysis_trajectory
        ):
            raise ResultSchemaError(
                "the final Candidate must appear in the ANALYSIS trajectory"
            )
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
        if (
            not migrated_legacy
            and status == "finalized"
            and result.get("test_status") != "failed"
        ):
            raise ResultSchemaError("a failed finalized result requires failed TEST")


def _validate_analysis_summary_counts(
    result: Dict[str, Any], trajectory: list[Dict[str, Any]]
) -> None:
    """Bind v6 summary fields even when the report originated in an old schema."""

    selected = result.get("analysis_selected_points")
    evaluated = result.get("analysis_evaluated_points")
    for name, value in (
        ("analysis_selected_points", selected),
        ("analysis_evaluated_points", evaluated),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ResultSchemaError(f"{name} must be a non-negative integer")
    if selected != len(trajectory):
        raise ResultSchemaError(
            "analysis_selected_points must count the ANALYSIS trajectory"
        )
    measured = sum(point["analysis_status"] == "evaluated" for point in trajectory)
    if evaluated != measured:
        raise ResultSchemaError(
            "analysis_evaluated_points must count evaluated trajectory points"
        )


def _validate_analysis_trajectory(trajectory: list[Any]) -> None:
    """Validate the auditable postmortem measurement attached to result v6."""

    commit_indices = []
    for point in trajectory:
        if not isinstance(point, dict):
            raise ResultSchemaError("ANALYSIS trajectory points must be objects")
        candidate_id = point.get("candidate_id")
        source_lease = point.get("source_lease")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ResultSchemaError("ANALYSIS trajectory candidate_id must be a string")
        if not isinstance(source_lease, str) or not source_lease.strip():
            raise ResultSchemaError("ANALYSIS trajectory source_lease must be a string")
        for field in ("commit_index", "event_seq"):
            value = point.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ResultSchemaError(
                    f"ANALYSIS trajectory {field} must be a positive integer"
                )
        elapsed = point.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or elapsed < 0
        ):
            raise ResultSchemaError(
                "ANALYSIS trajectory elapsed_seconds must be finite and non-negative"
            )
        if not isinstance(point.get("declared_incumbent"), bool):
            raise ResultSchemaError(
                "ANALYSIS trajectory declared_incumbent must be a boolean"
            )
        declaration_event_seqs = point.get("declaration_event_seqs")
        if not isinstance(declaration_event_seqs, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in declaration_event_seqs
        ):
            raise ResultSchemaError(
                "ANALYSIS trajectory declaration_event_seqs must be positive integers"
            )
        status = point.get("analysis_status")
        if status == "evaluated":
            try:
                nll = validated_nll(point.get("analysis_nll"))
                expected_perplexity = perplexity_from_nll(nll)
                perplexity = validated_nll(point.get("analysis_perplexity"))
            except ValueError as exc:
                raise ResultSchemaError(
                    f"invalid ANALYSIS trajectory metric: {exc}"
                ) from exc
            if not math.isclose(perplexity, expected_perplexity, rel_tol=1e-12):
                raise ResultSchemaError(
                    "ANALYSIS perplexity does not equal exp(ANALYSIS NLL)"
                )
            if point.get("analysis_error") is not None:
                raise ResultSchemaError(
                    "an evaluated ANALYSIS point may not contain an error"
                )
        elif status in {"failed", "infrastructure_failure", "not_run"}:
            if (
                point.get("analysis_nll") is not None
                or point.get("analysis_perplexity") is not None
            ):
                raise ResultSchemaError(
                    "an unmeasured ANALYSIS point may not contain metrics"
                )
            error = point.get("analysis_error")
            if not isinstance(error, str) or not error.strip():
                raise ResultSchemaError(
                    "an unmeasured ANALYSIS point requires a stable error code"
                )
        else:
            raise ResultSchemaError("invalid ANALYSIS trajectory status")
        commit_indices.append(point["commit_index"])

    if commit_indices != list(range(1, len(commit_indices) + 1)):
        raise ResultSchemaError(
            "ANALYSIS trajectory commit_index values must be contiguous and ordered"
        )


def _validate_candidate_commits(commits: list[Any]) -> None:
    indices = []
    event_sequences = []
    for commit in commits:
        if not isinstance(commit, dict):
            raise ResultSchemaError("candidate commits must be objects")
        for field in ("candidate_id", "source_lease"):
            if not isinstance(commit.get(field), str) or not commit[field].strip():
                raise ResultSchemaError(f"candidate commit {field} must be a string")
        if (
            not isinstance(commit.get("timestamp"), str)
            or not commit["timestamp"].strip()
        ):
            raise ResultSchemaError("candidate commit timestamp must be a string")
        for field in ("commit_index", "event_seq"):
            value = commit.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ResultSchemaError(
                    f"candidate commit {field} must be a positive integer"
                )
        elapsed = commit.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or elapsed < 0
        ):
            raise ResultSchemaError(
                "candidate commit elapsed_seconds must be finite and non-negative"
            )
        indices.append(commit["commit_index"])
        event_sequences.append(commit["event_seq"])
    if indices != list(range(1, len(commits) + 1)):
        raise ResultSchemaError("candidate commit indices must be contiguous")
    if event_sequences != sorted(set(event_sequences)):
        raise ResultSchemaError("candidate commit event sequences must be ordered")


def _validate_incumbent_declarations(
    declarations: list[Any], commits: list[Dict[str, Any]]
) -> None:
    committed = {commit["candidate_id"] for commit in commits}
    expected_previous = None
    event_sequences = []
    for expected_index, declaration in enumerate(declarations, start=1):
        if not isinstance(declaration, dict):
            raise ResultSchemaError("incumbent declarations must be objects")
        if declaration.get("declaration_index") != expected_index:
            raise ResultSchemaError("incumbent declaration indices must be contiguous")
        event_seq = declaration.get("event_seq")
        if (
            isinstance(event_seq, bool)
            or not isinstance(event_seq, int)
            or event_seq < 1
        ):
            raise ResultSchemaError(
                "incumbent declaration event_seq must be a positive integer"
            )
        candidate_id = declaration.get("candidate_id")
        if candidate_id not in committed:
            raise ResultSchemaError(
                "incumbent declaration must reference a committed Candidate"
            )
        if (
            not isinstance(declaration.get("timestamp"), str)
            or not declaration["timestamp"].strip()
        ):
            raise ResultSchemaError("incumbent declaration timestamp must be a string")
        if declaration.get("previous_candidate_id") != expected_previous:
            raise ResultSchemaError("incumbent declaration history is not chained")
        source_leases = declaration.get("source_leases")
        if (
            not isinstance(source_leases, list)
            or not source_leases
            or any(
                not isinstance(lease_id, str) or not lease_id
                for lease_id in source_leases
            )
        ):
            raise ResultSchemaError(
                "incumbent declaration source_leases must be a non-empty string array"
            )
        expected_source_leases = []
        for commit in commits:
            if (
                commit["candidate_id"] == candidate_id
                and commit["event_seq"] < event_seq
                and commit["source_lease"] not in expected_source_leases
            ):
                expected_source_leases.append(commit["source_lease"])
        if source_leases != expected_source_leases:
            raise ResultSchemaError(
                "incumbent declaration source Leases do not match prior commits"
            )
        leases_used = declaration.get("leases_used")
        if (
            isinstance(leases_used, bool)
            or not isinstance(leases_used, int)
            or leases_used < 0
        ):
            raise ResultSchemaError(
                "incumbent declaration leases_used must be non-negative"
            )
        elapsed = declaration.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or elapsed < 0
        ):
            raise ResultSchemaError(
                "incumbent declaration elapsed_seconds must be finite and non-negative"
            )
        expected_previous = candidate_id
        event_sequences.append(event_seq)
    if event_sequences != sorted(set(event_sequences)):
        raise ResultSchemaError("incumbent declaration events must be ordered")
    commit_event_sequences = [commit["event_seq"] for commit in commits]
    if len(set(event_sequences + commit_event_sequences)) != len(
        event_sequences + commit_event_sequences
    ):
        raise ResultSchemaError(
            "Candidate and incumbent event sequences must be unique"
        )


def _validate_analysis_binding(
    trajectory: list[Dict[str, Any]],
    commits: list[Dict[str, Any]],
    declarations: list[Dict[str, Any]],
    *,
    require_complete: bool,
) -> None:
    if require_complete and len(trajectory) != len(commits):
        raise ResultSchemaError(
            "a finalized ANALYSIS trajectory must cover every Candidate commit"
        )
    if len(trajectory) > len(commits):
        raise ResultSchemaError("ANALYSIS trajectory exceeds Candidate inventory")
    declarations_by_candidate: Dict[str, list[int]] = {}
    for declaration in declarations:
        declarations_by_candidate.setdefault(declaration["candidate_id"], []).append(
            declaration["event_seq"]
        )
    for point, commit in zip(trajectory, commits):
        for field in (
            "commit_index",
            "event_seq",
            "candidate_id",
            "source_lease",
            "timestamp",
            "elapsed_seconds",
        ):
            if point.get(field) != commit.get(field):
                raise ResultSchemaError(
                    f"ANALYSIS trajectory is not bound to candidate commit {field}"
                )
        expected_events = declarations_by_candidate.get(commit["candidate_id"], [])
        if point.get("declaration_event_seqs") != expected_events:
            raise ResultSchemaError(
                "ANALYSIS trajectory declaration evidence does not match history"
            )
        if point.get("declared_incumbent") != bool(expected_events):
            raise ResultSchemaError(
                "ANALYSIS trajectory incumbent flag does not match history"
            )
