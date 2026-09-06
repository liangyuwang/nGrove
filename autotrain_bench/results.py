"""Versioned episode-result schema and metric normalization."""

from __future__ import annotations

import math
from typing import Any, Dict

from .artifact_identity import (
    CANDIDATE_TREE_IDENTITY_ALGORITHM,
    FILE_IDENTITY_ALGORITHM,
    HIDDEN_BUNDLE_IDENTITY_ALGORITHM,
)
from .config import AGENT_EGRESS_POLICIES, AGENT_RUNNERS
from .worker_protocol import (
    CostContract,
    PerformanceScoreContract,
    ResourceAllocationReceipt,
    ResourceEnvelope,
    assess_resource_allocation,
)


RESULT_SCHEMA_VERSION = 13
_VALID_STATUSES = {
    "created",
    "active",
    "closed",
    "dnf",
    "finalized",
    "infrastructure_failure",
    "operator_stopped",
}
_PENDING_OUTCOME_STATUSES = {"created", "active", "closed"}
_FAILED_OUTCOME_STATUSES = {
    "dnf",
    "finalized",
    "infrastructure_failure",
}


class ResultSchemaError(ValueError):
    """An episode state cannot be represented by the requested result schema."""


def _validate_cost(value: Any, contract: CostContract, field: str) -> None:
    try:
        contract.validate_measurement(value)
    except (TypeError, ValueError) as exc:
        raise ResultSchemaError(f"invalid {field}: {exc}") from exc


def normalize_result_report(payload: Any) -> Dict[str, Any]:
    """Return a current report without mutating the persisted episode state."""

    if not isinstance(payload, dict):
        raise ResultSchemaError("episode state must be an object")
    version = payload.get("result_schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ResultSchemaError("result_schema_version must be an integer")
    if version not in {12, RESULT_SCHEMA_VERSION}:
        raise ResultSchemaError(f"unsupported result schema version: {version}")
    if version == 12 and (
        payload.get("status") == "operator_stopped"
        or payload.get("outcome") == "aborted"
        or "operator_limits" in payload
    ):
        raise ResultSchemaError("operator protection requires result schema version 13")
    result = dict(payload)
    result.setdefault("deployment_teardown_failures", [])
    result.pop("pending_result", None)
    _validate_current_result(result)
    return result


def _validate_artifact_identity_algorithms(result: Dict[str, Any]) -> None:
    expected = {
        "candidate_tree": CANDIDATE_TREE_IDENTITY_ALGORITHM,
        "file": FILE_IDENTITY_ALGORITHM,
        "hidden_evaluation": HIDDEN_BUNDLE_IDENTITY_ALGORITHM,
    }
    algorithms = result.get("artifact_identity_algorithms")
    if not isinstance(algorithms, dict) or set(algorithms) != set(expected):
        raise ResultSchemaError(
            "artifact_identity_algorithms must identify candidate_tree, file, "
            "and hidden_evaluation"
        )
    if algorithms != expected:
        raise ResultSchemaError(
            "artifact identity algorithms do not match the active protocol"
        )

    verification = result.get("formal_release_verification")
    if verification is None:
        return
    if not isinstance(verification, dict):
        raise ResultSchemaError("formal_release_verification must be an object")
    receipt_algorithms = verification.get("artifact_identity_algorithms")
    expected_receipt = {
        "file": FILE_IDENTITY_ALGORITHM,
        "hidden_evaluation": HIDDEN_BUNDLE_IDENTITY_ALGORITHM,
    }
    if not isinstance(receipt_algorithms, dict) or set(receipt_algorithms) != set(
        expected_receipt
    ):
        raise ResultSchemaError(
            "formal release verification must identify artifact algorithms"
        )
    if receipt_algorithms != expected_receipt:
        raise ResultSchemaError(
            "formal release artifact algorithms do not match the active protocol"
        )


def _performance_score_contract(result: Dict[str, Any]) -> PerformanceScoreContract:
    value = result.get("performance_score_contract")
    try:
        return PerformanceScoreContract.from_manifest(value)
    except (TypeError, ValueError) as exc:
        raise ResultSchemaError(f"invalid performance score contract: {exc}") from exc


def _cost_contract(result: Dict[str, Any]) -> CostContract:
    value = result.get("cost_contract")
    try:
        return CostContract.from_manifest(value)
    except (TypeError, ValueError) as exc:
        raise ResultSchemaError(f"invalid cost contract: {exc}") from exc


def _validate_performance_measurement(
    performance_score: Any,
    raw_metrics: Any,
    contract: PerformanceScoreContract,
    field: str,
) -> None:
    try:
        contract.validate_measurement(performance_score, raw_metrics)
    except (TypeError, ValueError) as exc:
        raise ResultSchemaError(f"invalid {field}: {exc}") from exc


def _validate_workload_resolutions(
    value: Any,
    *,
    leaderboard_eligible: bool,
    release_verification: Any,
) -> None:
    roles = {"training", "candidate", "evaluator"}
    if not isinstance(value, dict) or set(value) != roles:
        raise ResultSchemaError(
            "workload_resolutions must contain training, candidate, and evaluator"
        )
    for role in sorted(roles):
        evidence = value[role]
        if evidence is None:
            if leaderboard_eligible:
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
        if leaderboard_eligible and acceptance == "unverified":
            raise ResultSchemaError(
                "a leaderboard-eligible result cannot use an unverified workload"
            )


def _validate_resource_allocations(result: Dict[str, Any]) -> None:
    envelope_value = result.get("resource_envelope")
    allocations = result.get("resource_allocations")
    try:
        envelope = ResourceEnvelope.from_manifest(envelope_value)
    except (TypeError, ValueError) as exc:
        raise ResultSchemaError(f"invalid resource envelope: {exc}") from exc
    if not isinstance(allocations, list):
        raise ResultSchemaError("resource_allocations must be an array")

    episode_id = result.get("episode_id")
    workers = result.get("workers")
    research = workers.get("research") if isinstance(workers, dict) else None
    worker_id = research.get("worker_id") if isinstance(research, dict) else None
    accepted_receipts = []
    seen_receipts = set()
    seen_leases = set()
    for record in allocations:
        if not isinstance(record, dict) or set(record) != {
            "receipt",
            "assessment",
            "accepted",
        }:
            raise ResultSchemaError(
                "resource allocation records must contain receipt, assessment, and accepted"
            )
        try:
            receipt = ResourceAllocationReceipt.from_manifest(record["receipt"])
            assessment = assess_resource_allocation(
                envelope,
                receipt,
                episode_id=episode_id,
                lease_id=receipt.lease_id,
                worker_id=worker_id,
            )
        except (TypeError, ValueError) as exc:
            raise ResultSchemaError(
                f"invalid resource allocation receipt: {exc}"
            ) from exc
        if receipt.receipt_id in seen_receipts or receipt.lease_id in seen_leases:
            raise ResultSchemaError("resource allocation receipts must be unique")
        seen_receipts.add(receipt.receipt_id)
        seen_leases.add(receipt.lease_id)
        if record["assessment"] != assessment.public_manifest():
            raise ResultSchemaError(
                "resource allocation assessment does not match its evidence"
            )
        if not isinstance(record["accepted"], bool):
            raise ResultSchemaError("resource allocation accepted must be boolean")
        if record["accepted"] is not assessment.satisfied:
            raise ResultSchemaError(
                "resource allocation acceptance does not match its assessment"
            )
        if record["accepted"]:
            accepted_receipts.append(receipt)

    leases_used = result.get("leases_used", 0)
    if (
        isinstance(leases_used, bool)
        or not isinstance(leases_used, int)
        or leases_used < 0
    ):
        raise ResultSchemaError("leases_used must be a non-negative integer")
    if len(accepted_receipts) != leases_used:
        raise ResultSchemaError(
            "accepted resource allocations must match the used Lease count"
        )
    if result.get("outcome") == "success" and len(accepted_receipts) != len(
        allocations
    ):
        raise ResultSchemaError(
            "a successful result cannot contain a rejected resource allocation"
        )
    active = result.get("active_lease")
    if isinstance(active, dict):
        receipt_id = active.get("resource_allocation_receipt_id")
        if receipt_id not in {receipt.receipt_id for receipt in accepted_receipts}:
            raise ResultSchemaError(
                "an active Lease must reference its accepted resource allocation"
            )
    if result.get("leaderboard_eligible") is True:
        if not accepted_receipts or len(accepted_receipts) != len(allocations):
            raise ResultSchemaError(
                "a leaderboard-eligible result requires accepted allocation evidence"
            )
        if any(
            requirement.minimum_enforcement.value != "enforced"
            for requirement in envelope.requirements
        ):
            raise ResultSchemaError(
                "a leaderboard-eligible resource envelope must require enforcement"
            )
        challenge_release = result.get("challenge_release")
        if (
            not isinstance(challenge_release, dict)
            or challenge_release.get("resource_envelope") != envelope.public_manifest()
        ):
            raise ResultSchemaError(
                "leaderboard resource evidence must match the frozen release"
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


def _validate_current_result(result: Dict[str, Any]) -> None:
    if any(
        field in result
        for field in ("test_nll", "test_perplexity", "test_serving_metrics")
    ):
        raise ResultSchemaError(
            "current results may not contain legacy TEST metric fields"
        )
    _validate_agent_system(result)
    performance_contract = _performance_score_contract(result)
    cost_contract = _cost_contract(result)
    _validate_artifact_identity_algorithms(result)
    _validate_workload_resolutions(
        result.get("workload_resolutions"),
        leaderboard_eligible=result.get("leaderboard_eligible") is True,
        release_verification=result.get("formal_release_verification"),
    )
    _validate_resource_allocations(result)
    status = result.get("status")
    if status not in _VALID_STATUSES:
        raise ResultSchemaError(f"invalid result status: {status!r}")
    operator_limits = result.get("operator_limits")
    if operator_limits is not None:
        if not isinstance(operator_limits, dict) or set(operator_limits) != {
            "max_research_seconds",
            "clock",
        }:
            raise ResultSchemaError("invalid operator_limits")
        limit = operator_limits["max_research_seconds"]
        if limit is not None and (
            isinstance(limit, bool)
            or not isinstance(limit, (int, float))
            or not math.isfinite(limit)
            or limit <= 0
            or result.get("track") != "B"
        ):
            raise ResultSchemaError("invalid Track B operator research time limit")
        if operator_limits["clock"] != "monotonic_since_task_reveal_without_refunds":
            raise ResultSchemaError("invalid operator research clock")
    analysis_complete = result.get("analysis_complete")
    analysis_status = result.get("analysis_status")
    analysis_failure = result.get("analysis_failure")
    analysis_identity_verified = result.get("analysis_identity_verified")
    analysis_trajectory = result.get("analysis_trajectory")
    research_sealed_at = result.get("research_sealed_at")
    test_cost = result.get("test_cost")
    if analysis_complete is not None and not isinstance(analysis_complete, bool):
        raise ResultSchemaError("analysis_complete must be null or boolean")
    if analysis_trajectory is not None and not isinstance(analysis_trajectory, list):
        raise ResultSchemaError("analysis_trajectory must be an array")
    if isinstance(analysis_trajectory, list):
        _validate_analysis_trajectory(
            analysis_trajectory,
            performance_contract=performance_contract,
            cost_contract=cost_contract,
        )
        _validate_analysis_summary_counts(result, analysis_trajectory)
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
        raise ResultSchemaError("only incomplete ANALYSIS may contain a failure code")
    if analysis_identity_verified is not None and not isinstance(
        analysis_identity_verified, bool
    ):
        raise ResultSchemaError("analysis_identity_verified must be null or boolean")
    if not isinstance(analysis_trajectory, list):
        raise ResultSchemaError("current results require an ANALYSIS trajectory array")
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
    if outcome not in {None, "success", "failed", "aborted"}:
        raise ResultSchemaError(f"invalid result outcome: {outcome!r}")
    if (status == "operator_stopped") != (outcome == "aborted"):
        raise ResultSchemaError("only an operator stop may have an aborted outcome")
    if outcome is None:
        if status not in _PENDING_OUTCOME_STATUSES:
            raise ResultSchemaError(
                f"status {status!r} requires a success or failed outcome"
            )
        if (
            result.get("test_performance_score") is not None
            or result.get("test_raw_metrics") is not None
        ):
            raise ResultSchemaError("a pending result may not contain TEST metrics")
        if test_cost is not None:
            raise ResultSchemaError("a pending result may not contain TEST cost")
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
        _validate_performance_measurement(
            result.get("test_performance_score"),
            result.get("test_raw_metrics"),
            performance_contract,
            "TEST performance measurement",
        )
        if result.get("failure") is not None:
            raise ResultSchemaError("a successful result may not contain a failure")
        if result.get("test_status") != "evaluated":
            raise ResultSchemaError("a successful result requires evaluated TEST")
        _validate_cost(test_cost, cost_contract, "TEST cost")
        if not any(
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
            result.get("test_performance_score") is not None
            or result.get("test_raw_metrics") is not None
        ):
            raise ResultSchemaError("a failed result may not contain TEST metrics")
        if test_cost is not None:
            raise ResultSchemaError("a failed result may not contain TEST cost")
        failure = result.get("failure")
        if not isinstance(failure, str) or not failure.strip():
            raise ResultSchemaError("a failed result must contain a failure reason")
        if status == "finalized" and result.get("test_status") != "failed":
            raise ResultSchemaError("a failed finalized result requires failed TEST")
    elif outcome == "aborted":
        if (
            not operator_limits
            or operator_limits["max_research_seconds"] is None
            or result.get("failure") != "operator_time_limit_reached"
            or result.get("closure_reason") != "operator_time_limit_reached"
            or not isinstance(research_sealed_at, str)
            or not research_sealed_at
            or result.get("leaderboard_eligible") is not False
            or candidate_id is not None
            or test_status is not None
            or analysis_status is not None
            or result.get("test_performance_score") is not None
            or result.get("test_raw_metrics") is not None
            or test_cost is not None
        ):
            raise ResultSchemaError("an operator stop cannot be a benchmark result")


def _validate_analysis_summary_counts(
    result: Dict[str, Any], trajectory: list[Dict[str, Any]]
) -> None:
    """Bind ANALYSIS summary fields to the trajectory."""

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


def _validate_analysis_trajectory(
    trajectory: list[Any],
    *,
    performance_contract: PerformanceScoreContract,
    cost_contract: CostContract,
) -> None:
    """Validate the auditable postmortem measurement attached to a result."""

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
        if any(
            field in point
            for field in (
                "analysis_nll",
                "analysis_perplexity",
                "analysis_serving_metrics",
            )
        ):
            raise ResultSchemaError(
                "current ANALYSIS points may not contain legacy metric fields"
            )
        if status == "evaluated":
            _validate_performance_measurement(
                point.get("analysis_performance_score"),
                point.get("analysis_raw_metrics"),
                performance_contract,
                "ANALYSIS performance measurement",
            )
            if point.get("analysis_error") is not None:
                raise ResultSchemaError(
                    "an evaluated ANALYSIS point may not contain an error"
                )
            cost = point.get("analysis_cost")
            if cost is not None:
                _validate_cost(cost, cost_contract, "ANALYSIS trajectory cost")
            else:
                raise ResultSchemaError("an evaluated ANALYSIS point requires cost")
        elif status in {"failed", "infrastructure_failure", "not_run"}:
            if (
                point.get("analysis_performance_score") is not None
                or point.get("analysis_raw_metrics") is not None
            ):
                raise ResultSchemaError(
                    "an unmeasured ANALYSIS point may not contain metrics"
                )
            if point.get("analysis_cost") is not None:
                raise ResultSchemaError(
                    "an unmeasured ANALYSIS point may not contain cost"
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
