"""Validation of optional benchmark results and historical comparison records."""

from .artifact_identity import FILE_IDENTITY_ALGORITHM, HIDDEN_BUNDLE_IDENTITY_ALGORITHM
from .gateway_execution import (
    GatewayArtifactIdentity,
    GatewayExecutionRequest,
    GatewayExecutionReceipt,
    gateway_eligibility_violations,
)
from .worker_protocol import (
    ConformanceReceipt,
    ResourceEnvelope,
    WORKER_CONFORMANCE_SUITE_VERSION,
)
from .results import ResultSchemaError


_FIELDS = {
    "leaderboard_eligible",
    "leaderboard_eligibility_violations",
    "comparison_key",
    "benchmark_release",
    "formal_release_verification",
    "formal_release_verified",
    "policy_compliant",
    "policy_violations",
}
_WORKER_FIELDS = {"trust_satisfied", "conformance_satisfied", "leaderboard_eligible"}
_EXTENSION_FIELDS = {
    "release",
    "comparison_key",
    "formal_release_verification",
    "policy_compliant",
    "policy_violations",
    "leaderboard_eligible",
    "leaderboard_eligibility_violations",
    "workers",
}


def validate_result_extensions(result):
    legacy = result["result_schema_version"] < 17
    if legacy:
        if "benchmark" in result:
            raise ResultSchemaError(
                "benchmark extension requires result schema version 17"
            )
        view = result
    else:
        # Reject misplaced fields instead of silently dropping evidence.
        if _FIELDS.intersection(result):
            raise ResultSchemaError(
                "comparison fields belong inside the benchmark extension"
            )
        for name in ("agent_system", "evaluation_system"):
            component = result.get(name)
            if component is None:
                continue
            if not isinstance(component, dict):
                raise ResultSchemaError("execution evidence must be an object")
            if _FIELDS.intersection(component) or _FIELDS.intersection(
                component.get("execution_evidence") or {}
            ):
                raise ResultSchemaError(
                    "execution evidence must contain facts, not comparison judgments"
                )
        if "benchmark_manifest_sha256" in result.get("artifact_identities", {}):
            raise ResultSchemaError(
                "benchmark identity belongs inside the benchmark extension"
            )
        for worker in result.get("workers", {}).values():
            if "conformance" in worker or _WORKER_FIELDS.intersection(
                worker.get("assessment", {})
            ):
                raise ResultSchemaError(
                    "Worker certification belongs inside the benchmark extension"
                )
        if "benchmark" not in result:
            for role, evidence in result["workload_resolutions"].items():
                if evidence is None:
                    continue
                if result["result_schema_version"] >= 18:
                    if "acceptance" in evidence:
                        raise ResultSchemaError(
                            "workload acceptance requires a benchmark extension"
                        )
                else:
                    # Version 17 emitted identity judgments for ordinary runs.
                    # Preserve their validation without admitting organizer claims.
                    acceptance = _validate_workload_acceptance(role, evidence)
                    if acceptance == "organizer_receipt":
                        raise ResultSchemaError(
                            "organizer acceptance requires a benchmark extension"
                        )
            return
        extension = result["benchmark"]
        if not isinstance(extension, dict) or set(extension) != _EXTENSION_FIELDS:
            raise ResultSchemaError("invalid benchmark extension")
        release = extension["release"]
        if (
            not isinstance(release, dict)
            or not isinstance(release.get("benchmark_id"), str)
            or not release["benchmark_id"].strip()
        ):
            raise ResultSchemaError("benchmark extension requires a release")
        for field in ("leaderboard_eligible", "policy_compliant"):
            if not isinstance(extension[field], bool):
                raise ResultSchemaError(f"benchmark {field} must be boolean")
        for field in ("policy_violations", "leaderboard_eligibility_violations"):
            if not isinstance(extension[field], list) or any(
                not isinstance(item, str) or not item for item in extension[field]
            ):
                raise ResultSchemaError(f"benchmark {field} must be an array of codes")
        if extension["policy_compliant"] != (not extension["policy_violations"]):
            raise ResultSchemaError(
                "benchmark compliance disagrees with its violations"
            )
        if extension["policy_violations"] != result.get("execution_issues", []):
            raise ResultSchemaError(
                "benchmark compliance disagrees with execution evidence"
            )
        _validate_workers(result, extension)
        if extension["leaderboard_eligible"] and (
            not extension["policy_compliant"]
            or extension["leaderboard_eligibility_violations"]
            or release.get("leaderboard_eligible") is not True
            or release.get("release_stage") != "official"
        ):
            raise ResultSchemaError("benchmark eligibility disagrees with its evidence")
        view = {**result, **extension, "benchmark_release": release}
    if (
        view.get("outcome") == "aborted"
        and view.get("leaderboard_eligible") is not False
    ):
        raise ResultSchemaError("an operator stop cannot be an evaluation result")
    _validate_verification(view)
    _validate_workloads(view)
    _validate_resources(view)
    if view["result_schema_version"] >= 14:
        _validate_evaluation(view)
    if view["result_schema_version"] >= 15:
        _validate_gateway(view)
    from .benchmark import validate_benchmark_binding

    try:
        validate_benchmark_binding(view)
    except (TypeError, ValueError) as exc:
        raise ResultSchemaError(str(exc)) from exc


def _validate_workers(result, extension):
    workers = extension["workers"]
    if not isinstance(workers, dict) or set(workers) != {"research", "evaluation"}:
        raise ResultSchemaError("benchmark requires both Worker assessments")
    for role, record in workers.items():
        if not isinstance(record, dict) or set(record) != {"assessment", "conformance"}:
            raise ResultSchemaError("invalid benchmark Worker assessment")
        descriptor = result["workers"].get(role, {})
        receipt = record["conformance"]
        try:
            receipt = ConformanceReceipt(**receipt) if receipt is not None else None
        except (TypeError, ValueError) as exc:
            raise ResultSchemaError(
                f"invalid benchmark Worker conformance: {exc}"
            ) from exc
        trust = descriptor.get("trust_level") == "certified"
        conformance = bool(
            receipt
            and receipt.worker_id == descriptor.get("worker_id")
            and receipt.suite_version == WORKER_CONFORMANCE_SUITE_VERSION
            and receipt.passed
        )
        capability = descriptor.get("assessment", {})
        eligible = (
            capability.get("capabilities_satisfied") is True and trust and conformance
        )
        expected = {
            **capability,
            "trust_satisfied": trust,
            "conformance_satisfied": conformance,
            "leaderboard_eligible": eligible,
        }
        if record["assessment"] != expected:
            raise ResultSchemaError(
                "benchmark Worker assessment disagrees with execution evidence"
            )
        if extension["leaderboard_eligible"] and not eligible:
            raise ResultSchemaError("benchmark eligibility requires qualified Workers")


def _validate_verification(result):
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


def _validate_workload_acceptance(role, evidence):
    """Validate the identity judgment shared by legacy and benchmark records."""
    acceptance = evidence.get("acceptance")
    if not isinstance(acceptance, str) or acceptance not in {
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
            evidence["requested_identity"].rsplit(marker, 1)[1]
            if marker in evidence["requested_identity"]
            else None
        )
        if (
            requested_digest != evidence["effective_sha256"]
            or evidence["effective_format"] != "oci"
            or evidence.get("transformation_sha256") is not None
        ):
            raise ResultSchemaError(
                f"workload resolution {role} is not a canonical identity"
            )

    return acceptance


def _validate_workloads(result):
    release_verification = result.get("formal_release_verification")
    for role, evidence in result["workload_resolutions"].items():
        if evidence is None:
            if result.get("leaderboard_eligible") is True:
                raise ResultSchemaError(
                    "a leaderboard-eligible result requires every workload resolution"
                )
            continue
        acceptance = _validate_workload_acceptance(role, evidence)
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
        if result.get("leaderboard_eligible") is True and acceptance == "unverified":
            raise ResultSchemaError(
                "a leaderboard-eligible result cannot use an unverified workload"
            )


def _validate_resources(result):
    envelope = ResourceEnvelope.from_manifest(result["resource_envelope"])
    allocations = result["resource_allocations"]
    accepted_receipts = [record for record in allocations if record["accepted"]]
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
        benchmark_release = result.get("benchmark_release")
        if (
            not isinstance(benchmark_release, dict)
            or benchmark_release.get("resource_envelope") != envelope.public_manifest()
        ):
            raise ResultSchemaError(
                "leaderboard resource evidence must match the frozen release"
            )


def _validate_evaluation(result):
    value = result["evaluation_resource_envelope"]
    envelope = ResourceEnvelope.from_manifest(value) if value is not None else None
    measurements = {
        (record["split"], record["candidate_id"]): record
        for record in result["evaluation_resource_allocations"]
    }
    if result.get("leaderboard_eligible") is True:
        if (
            envelope is None
            or not envelope.requirements
            or any(
                item.minimum_enforcement.value != "enforced"
                for item in envelope.requirements
            )
        ):
            raise ResultSchemaError(
                "leaderboard evaluation resources must require enforcement"
            )
        test_record = measurements.get(("test", result.get("final_candidate_id")))
        if test_record is None or not test_record["accepted"]:
            raise ResultSchemaError(
                "leaderboard cost requires accepted TEST resource evidence"
            )
        release = result.get("benchmark_release")
        try:
            frozen = ResourceEnvelope.from_manifest(
                release.get("evaluation_resource_envelope")
                if isinstance(release, dict)
                else None
            )
        except (TypeError, ValueError) as exc:
            raise ResultSchemaError(
                "invalid frozen evaluation resource envelope"
            ) from exc
        if frozen.sha256 != envelope.sha256:
            raise ResultSchemaError(
                "leaderboard evaluation resources must match the frozen release"
            )


def _validate_gateway(result):
    record = result.get("gateway_execution")
    if record is None:
        return
    request = GatewayExecutionRequest.from_manifest(record["request"])
    receipt = (
        GatewayExecutionReceipt.from_manifest(record["receipt"])
        if record["receipt"] is not None
        else None
    )
    try:
        verification = result.get("formal_release_verification")
        expected = (
            verification.get("inference_gateway_artifact")
            if isinstance(verification, dict)
            else None
        )
        artifact = (
            GatewayArtifactIdentity.from_manifest(expected)
            if expected is not None
            else None
        )
    except (TypeError, ValueError) as exc:
        raise ResultSchemaError(f"invalid Gateway execution evidence: {exc}") from exc
    if result.get("leaderboard_eligible") is True:
        if (
            request.core_revision is None
            or not isinstance(verification, dict)
            or request.core_revision != verification.get("core_revision")
        ):
            raise ResultSchemaError("Gateway Core revision does not match the release")
        violations = gateway_eligibility_violations(receipt, artifact)
        if violations:
            raise ResultSchemaError(
                "leaderboard-eligible Gateway evidence is incomplete: "
                + ", ".join(violations)
            )
