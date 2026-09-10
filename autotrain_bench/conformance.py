"""Core-owned diagnostic probes for selected Worker Protocol semantics.

These probes deliberately do not issue :class:`ConformanceReceipt`. They
exercise resolution and evaluation resource evidence, while the active Worker
Protocol also includes Lease admission/execution/termination, physical resource
enforcement, isolation, and teardown guarantees. Treating this partial
coverage as complete protocol conformance would create false formal evidence.

Run a probe only against a dedicated, disposable Worker deployment. The
trusted Operator supplies the release-owned identities and a deployment-level
teardown callback. The callback always runs, including when a Worker raises an
unexpected exception. Platform-specific setup, watchdogs, and destruction
remain outside Core.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from .config import TrainingResources
from .worker_protocol import (
    WORKER_PROTOCOL_VERSION,
    CandidateEvaluationResult,
    EvaluationResourceFailure,
    EvaluationTaskSpec,
    HiddenDatasetIdentityMismatch,
    HiddenDatasetRequest,
    HiddenDatasetResolutionFailure,
    ResolvedHiddenDataset,
    ResolvedWorkload,
    ResourceEnvelope,
    WorkerDescriptor,
    WorkerFailure,
    WorkerTeardownFailure,
    WorkloadRequest,
    WorkloadResolutionFailure,
    assess_evaluation_resources,
)


RESOLUTION_PROBE_VERSION = f"worker-v{WORKER_PROTOCOL_VERSION}-resolution-probe-v1"
EVALUATION_PROBE_VERSION = f"worker-v{WORKER_PROTOCOL_VERSION}-evaluation-probe-v1"

PASSED = "passed"
FAILED = "failed"
SKIPPED = "skipped"


def _corrupt_sha256(digest: str) -> str:
    """Return a different, still well-formed SHA-256 digest."""

    replacement = "0" if digest[-1] != "0" else "1"
    return digest[:-1] + replacement


@dataclass(frozen=True)
class ConformanceCheck:
    """One probe observation with public and trusted detail separated."""

    name: str
    status: str
    code: str = ""
    private_detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in {PASSED, FAILED, SKIPPED}:
            raise ValueError("conformance check status is invalid")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("conformance check name must not be empty")
        if not isinstance(self.code, str) or not isinstance(self.private_detail, str):
            raise TypeError("conformance check code and detail must be strings")

    @property
    def satisfied(self) -> bool:
        return self.status == PASSED

    def public_manifest(self) -> Dict[str, Any]:
        payload = {"name": self.name, "status": self.status}
        if self.code:
            payload["code"] = self.code
        return payload

    def private_manifest(self) -> Dict[str, Any]:
        payload = self.public_manifest()
        if self.private_detail:
            payload["detail"] = self.private_detail
        return payload


@dataclass(frozen=True)
class WorkerProbeReport:
    """A limited diagnostic report that is never formal conformance evidence."""

    probe_version: str
    run_id: str
    worker_id: str
    checks: Tuple[ConformanceCheck, ...] = field(default_factory=tuple)

    @property
    def failed(self) -> Tuple[ConformanceCheck, ...]:
        return tuple(item for item in self.checks if item.status == FAILED)

    @property
    def skipped(self) -> Tuple[ConformanceCheck, ...]:
        return tuple(item for item in self.checks if item.status == SKIPPED)

    @property
    def passed(self) -> bool:
        """Whether this probe passed, not whether the Worker is conformant."""

        return bool(self.checks) and all(item.satisfied for item in self.checks)

    def public_manifest(self) -> Dict[str, Any]:
        return {
            "probe_version": self.probe_version,
            "run_id": self.run_id,
            "worker_id": self.worker_id,
            "passed": self.passed,
            "formal_conformance": False,
            "checks": [item.public_manifest() for item in self.checks],
        }

    def private_manifest(self) -> Dict[str, Any]:
        payload = self.public_manifest()
        payload["checks"] = [item.private_manifest() for item in self.checks]
        return payload


@dataclass(frozen=True)
class ResolutionProbeReport(WorkerProbeReport):
    """Workload and hidden-data resolution observations."""


@dataclass(frozen=True)
class EvaluationProbeReport(WorkerProbeReport):
    """Evaluation observations; operation evidence is trusted-Operator-only."""

    task_id: str = ""
    evaluations: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def private_manifest(self) -> Dict[str, Any]:
        payload = super().private_manifest()
        payload["task_id"] = self.task_id
        payload["evaluations"] = list(self.evaluations)
        return payload


class _Recorder:
    def __init__(self) -> None:
        self.checks: List[ConformanceCheck] = []

    def record(
        self,
        name: str,
        status: str,
        *,
        code: str = "",
        private_detail: str = "",
    ) -> None:
        self.checks.append(ConformanceCheck(name, status, code, private_detail))

    def expect(
        self,
        name: str,
        condition: bool,
        *,
        code: str,
        private_detail: str = "",
    ) -> bool:
        self.record(
            name,
            PASSED if condition else FAILED,
            code="" if condition else code,
            private_detail="" if condition else private_detail,
        )
        return condition


def _record_unexpected(log: _Recorder, name: str, exc: Exception) -> None:
    log.record(
        name,
        FAILED,
        code="unexpected_worker_exception",
        private_detail=f"{type(exc).__name__}: {exc}",
    )


def verify_hidden_dataset_resolution(
    worker: Any,
    *,
    episode_id: str,
    task: EvaluationTaskSpec,
    hidden_dataset_identities: Mapping[str, str],
    timeout_seconds: float,
    recorder: _Recorder,
) -> None:
    """Probe release-identity binding without prescribing a storage mechanism."""

    log = recorder
    for name, expected_sha256 in sorted(hidden_dataset_identities.items()):
        prefix = f"hidden_dataset[{name}]"
        request = HiddenDatasetRequest(name, expected_sha256)
        try:
            value = worker.resolve_hidden_dataset(
                episode_id=episode_id,
                task=task,
                hidden_dataset=request,
                timeout_seconds=timeout_seconds,
            )
        except HiddenDatasetIdentityMismatch as exc:
            log.record(
                f"{prefix}.resolves_expected_identity",
                FAILED,
                code="expected_identity_rejected",
                private_detail=exc.receipt.sha256,
            )
            continue
        except HiddenDatasetResolutionFailure as exc:
            log.record(
                f"{prefix}.resolves_expected_identity",
                FAILED,
                code="resolution_failed",
                private_detail=str(exc),
            )
            continue
        except Exception as exc:  # noqa: BLE001 - the probe records Worker failure
            _record_unexpected(log, f"{prefix}.resolves_expected_identity", exc)
            continue

        if not log.expect(
            f"{prefix}.returns_resolved_type",
            isinstance(value, ResolvedHiddenDataset),
            code="invalid_resolution_type",
            private_detail=f"returned {type(value).__name__}",
        ):
            continue
        log.expect(
            f"{prefix}.binds_release_identity",
            value.name == name
            and value.sha256 == expected_sha256
            and value.identity_algorithm == request.identity_algorithm,
            code="resolution_identity_mismatch",
            private_detail=(
                f"name={value.name!r}, sha256={value.sha256}, "
                f"identity_algorithm={value.identity_algorithm!r}"
            ),
        )

        wrong_request = HiddenDatasetRequest(name, _corrupt_sha256(expected_sha256))
        try:
            wrong_result = worker.resolve_hidden_dataset(
                episode_id=episode_id,
                task=task,
                hidden_dataset=wrong_request,
                timeout_seconds=timeout_seconds,
            )
        except HiddenDatasetIdentityMismatch as exc:
            log.expect(
                f"{prefix}.rejects_wrong_identity",
                exc.receipt.name == name
                and exc.receipt.sha256 == expected_sha256
                and exc.receipt.identity_algorithm == wrong_request.identity_algorithm,
                code="invalid_mismatch_evidence",
                private_detail=repr(exc.receipt),
            )
        except HiddenDatasetResolutionFailure as exc:
            log.record(
                f"{prefix}.rejects_wrong_identity",
                FAILED,
                code="mismatch_evidence_missing",
                private_detail=f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - the probe records Worker failure
            _record_unexpected(log, f"{prefix}.rejects_wrong_identity", exc)
        else:
            log.record(
                f"{prefix}.rejects_wrong_identity",
                FAILED,
                code="wrong_identity_accepted",
                private_detail=f"returned {wrong_result!r}",
            )


def verify_workload_resolution(
    worker: Any,
    *,
    episode_id: str,
    workloads: Sequence[WorkloadRequest],
    timeout_seconds: float,
    recorder: _Recorder,
) -> None:
    """Probe that workload resolution remains bound to the release request."""

    log = recorder
    for request in workloads:
        prefix = f"workload[{request.role}]"
        try:
            value = worker.resolve_workload(
                episode_id=episode_id,
                workload=request,
                timeout_seconds=timeout_seconds,
            )
        except WorkloadResolutionFailure as exc:
            log.record(
                f"{prefix}.resolves",
                FAILED,
                code="resolution_failed",
                private_detail=str(exc),
            )
            continue
        except Exception as exc:  # noqa: BLE001 - the probe records Worker failure
            _record_unexpected(log, f"{prefix}.resolves", exc)
            continue
        if not log.expect(
            f"{prefix}.returns_resolved_type",
            isinstance(value, ResolvedWorkload),
            code="invalid_resolution_type",
            private_detail=f"returned {type(value).__name__}",
        ):
            continue
        log.expect(
            f"{prefix}.binds_release_request",
            value.role == request.role
            and value.requested_identity == request.requested_identity,
            code="resolution_request_mismatch",
            private_detail=(
                f"role={value.role!r}, requested_identity="
                f"{value.requested_identity!r}"
            ),
        )


def probe_worker_resolution(
    worker: Any,
    *,
    task: EvaluationTaskSpec,
    workloads: Sequence[WorkloadRequest],
    hidden_dataset_identities: Mapping[str, str],
    teardown: Callable[[], object],
    timeout_seconds: float = 60.0,
) -> ResolutionProbeReport:
    """Run the limited resolution probe in a disposable Worker deployment.

    ``timeout_seconds`` is passed only as a positive protocol deadline. This
    synchronous probe cannot safely preempt a wedged Worker; the Operator must
    run the whole probe behind a deployment-level watchdog.
    """

    if not isinstance(task, EvaluationTaskSpec):
        raise TypeError("task must be an EvaluationTaskSpec")
    if not workloads:
        raise ValueError("at least one release-owned workload is required")
    if any(not isinstance(item, WorkloadRequest) for item in workloads):
        raise TypeError("workloads must contain WorkloadRequest values")
    if not hidden_dataset_identities:
        raise ValueError("release-owned hidden dataset identities are required")
    for name, digest in hidden_dataset_identities.items():
        HiddenDatasetRequest(name, digest)
    if not callable(teardown):
        raise TypeError("teardown must be callable")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")

    run_id = "conformance:" + uuid.uuid4().hex
    log = _Recorder()
    worker_id = "unknown"
    try:
        try:
            descriptor = worker.describe_capabilities()
        except Exception as exc:  # noqa: BLE001 - the probe records Worker failure
            _record_unexpected(log, "descriptor.available", exc)
        else:
            if log.expect(
                "descriptor.is_worker_descriptor",
                isinstance(descriptor, WorkerDescriptor),
                code="invalid_descriptor_type",
                private_detail=f"returned {type(descriptor).__name__}",
            ):
                worker_id = descriptor.worker_id

        verify_hidden_dataset_resolution(
            worker,
            episode_id=run_id,
            task=task,
            hidden_dataset_identities=hidden_dataset_identities,
            timeout_seconds=timeout_seconds,
            recorder=log,
        )
        verify_workload_resolution(
            worker,
            episode_id=run_id,
            workloads=workloads,
            timeout_seconds=timeout_seconds,
            recorder=log,
        )
    finally:
        try:
            teardown_result = teardown()
        except Exception as exc:  # noqa: BLE001 - any teardown failure is evidence
            log.record(
                "deployment.teardown_confirmed",
                FAILED,
                code="deployment_teardown_failed",
                private_detail=f"{type(exc).__name__}: {exc}",
            )
        else:
            if teardown_result is False:
                log.record(
                    "deployment.teardown_confirmed",
                    FAILED,
                    code="deployment_teardown_incomplete",
                )
            else:
                log.record("deployment.teardown_confirmed", PASSED)

    return ResolutionProbeReport(
        probe_version=RESOLUTION_PROBE_VERSION,
        run_id=run_id,
        worker_id=worker_id,
        checks=tuple(log.checks),
    )


class _ProbeDeadline(TimeoutError):
    pass


class _ProbeEvidenceError(ValueError):
    pass


def probe_worker_evaluation(
    worker: Any,
    *,
    task: EvaluationTaskSpec,
    candidate_dir: Path,
    output_root: Path,
    resources: TrainingResources,
    hidden_dataset_identities: Mapping[str, str],
    resource_envelope: ResourceEnvelope,
    rejected_resource_envelope: ResourceEnvelope,
    teardown: Callable[[], bool],
    timeout_seconds: float = 60.0,
) -> EvaluationProbeReport:
    """Exercise evaluation receipts and typed resource rejection on fixtures.

    The Operator supplies a known-good immutable Candidate, public synthetic
    evaluation fixtures, one supported contract and one deliberately unsupported
    contract. Never attach this probe to a live episode or real hidden data.
    There are at most four evaluations: accepted TEST/ANALYSIS followed by
    rejected TEST/ANALYSIS. Each invocation shares one deadline with its workload
    and hidden-data resolution; an unexpected failure stops subsequent cases.

    ``teardown`` must clean the dedicated deployment and return exactly True
    only after confirming owned workloads and private materializations absent.
    It runs after every started probe, including exceptions and interrupts.
    An outer deployment watchdog must bound this synchronous probe and cleanup;
    the probe can reject late replies, but cannot preempt a wedged Worker.
    """

    if not isinstance(task, EvaluationTaskSpec):
        raise TypeError("task must be an EvaluationTaskSpec")
    if not isinstance(resources, TrainingResources):
        raise TypeError("resources must be TrainingResources")
    resources = resources.validate()
    if set(hidden_dataset_identities) != {"test", "analysis"}:
        raise ValueError("public fixture identities for TEST and ANALYSIS are required")
    requests = {
        split: HiddenDatasetRequest(split, digest)
        for split, digest in hidden_dataset_identities.items()
    }
    if any(request.expected_sha256 is None for request in requests.values()):
        raise ValueError("public fixture identities must have explicit SHA-256 values")
    for envelope in (resource_envelope, rejected_resource_envelope):
        if not isinstance(envelope, ResourceEnvelope):
            raise TypeError("probe contracts must be ResourceEnvelope values")
        if not envelope.requirements:
            raise ValueError("probe contracts require normative resource properties")
    if resource_envelope.sha256 == rejected_resource_envelope.sha256:
        raise ValueError("accepted and rejected resource contracts must differ")
    if not callable(teardown):
        raise TypeError("teardown must be callable")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be finite and positive")
    candidate_dir = Path(candidate_dir).resolve()
    output_root = Path(output_root).resolve()
    if not candidate_dir.is_dir() or not output_root.is_dir():
        raise ValueError("candidate_dir and output_root must be existing directories")
    if candidate_dir == output_root or candidate_dir in output_root.parents:
        raise ValueError("probe output must be outside the immutable Candidate")

    workload_requests = (
        WorkloadRequest("candidate", task.candidate_image),
        WorkloadRequest("evaluator", task.evaluator_image),
    )
    run_id = "evaluation-probe:" + uuid.uuid4().hex
    run_dir = output_root / run_id.split(":", 1)[1]
    log = _Recorder()
    worker_id = "unknown"
    records: List[Mapping[str, Any]] = []
    receipt_ids: set[str] = set()
    resolved_identities: Dict[str, Dict[str, Any]] = {}
    cases = (
        ("accepted_test", "test", resource_envelope, False),
        ("accepted_analysis", "analysis", resource_envelope, False),
        ("rejected_test", "test", rejected_resource_envelope, True),
        ("rejected_analysis", "analysis", rejected_resource_envelope, True),
    )
    completed = 0
    stage = "descriptor.available"
    try:
        descriptor = worker.describe_capabilities()
        if not isinstance(descriptor, WorkerDescriptor):
            raise _ProbeEvidenceError("Worker returned an invalid descriptor type")
        worker_id = descriptor.worker_id
        if (
            descriptor.protocol_version != WORKER_PROTOCOL_VERSION
            or "evaluation" not in descriptor.roles
        ):
            raise _ProbeEvidenceError(
                "Worker does not declare the current evaluation role"
            )
        log.record(stage, PASSED)
        stage = "output.scope_created"
        run_dir.mkdir(exist_ok=False)
        log.record(stage, PASSED)
        for case, split, envelope, expect_rejection in cases:
            prefix = f"evaluation[{case}]"
            deadline = time.monotonic() + timeout_seconds

            def remaining_seconds() -> float:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _ProbeDeadline("evaluation probe operation deadline reached")
                return remaining

            def call(method: Callable[..., Any], **kwargs: Any) -> Any:
                try:
                    value = method(timeout_seconds=remaining_seconds(), **kwargs)
                except WorkerTeardownFailure:
                    # Preserve an unknown workload boundary even if its reply is late.
                    raise
                except Exception:
                    # A late error is not timely resource rejection either.
                    remaining_seconds()
                    raise
                remaining_seconds()
                return value

            evaluation_id = "evaluation:" + uuid.uuid4().hex
            record: Dict[str, Any] = {
                "case": case,
                "evaluation_id": evaluation_id,
                "split": split,
                "expects_resource_rejection": expect_rejection,
                "passed": False,
                "resource_envelope": envelope.public_manifest(),
                "hidden_dataset_request": {
                    "name": split,
                    "sha256": requests[split].expected_sha256,
                    "identity_algorithm": requests[split].identity_algorithm,
                },
                "workloads": {},
            }
            records.append(record)
            workloads: Dict[str, ResolvedWorkload] = {}
            for request in workload_requests:
                stage = f"{prefix}.{request.role}_resolution"
                resolved = call(
                    worker.resolve_workload, episode_id=run_id, workload=request
                )
                if not isinstance(resolved, ResolvedWorkload) or (
                    resolved.role != request.role
                    or resolved.requested_identity != request.requested_identity
                ):
                    raise _ProbeEvidenceError(
                        "resolved workload does not bind its request"
                    )
                evidence = resolved.evidence_manifest()
                record["workloads"][request.role] = evidence
                previous = resolved_identities.setdefault(request.role, evidence)
                if previous != evidence:
                    raise _ProbeEvidenceError("effective workload identity changed")
                workloads[request.role] = resolved
                log.record(stage, PASSED)
            stage = f"{prefix}.hidden_resolution"
            hidden_request = requests[split]
            hidden = call(
                worker.resolve_hidden_dataset,
                episode_id=run_id,
                task=task,
                hidden_dataset=hidden_request,
            )
            if not isinstance(hidden, ResolvedHiddenDataset) or (
                hidden.name != split
                or hidden.sha256 != hidden_request.expected_sha256
                or hidden.identity_algorithm != hidden_request.identity_algorithm
            ):
                raise _ProbeEvidenceError(
                    "resolved hidden dataset does not bind its fixture"
                )
            record["hidden_dataset"] = {
                "name": hidden.name,
                "sha256": hidden.sha256,
                "identity_algorithm": hidden.identity_algorithm,
            }
            log.record(stage, PASSED)
            stage = (
                f"{prefix}.resource_rejection"
                if expect_rejection
                else f"{prefix}.result"
            )
            try:
                result = call(
                    worker.evaluate_candidate,
                    episode_id=run_id,
                    task=task,
                    resources=resources,
                    candidate_dir=candidate_dir,
                    hidden_dataset=hidden,
                    output_dir=run_dir / case,
                    candidate_workload=workloads["candidate"],
                    evaluator_workload=workloads["evaluator"],
                    evaluation_id=evaluation_id,
                    resource_envelope=envelope,
                )
            except EvaluationResourceFailure as exc:
                if not expect_rejection:
                    raise
                record["resource_rejection"] = str(exc)
                record["passed"] = True
                log.record(stage, PASSED)
                completed += 1
                continue
            if expect_rejection:
                raise _ProbeEvidenceError(
                    "Worker accepted the deliberately rejected contract"
                )
            if not isinstance(result, CandidateEvaluationResult):
                raise _ProbeEvidenceError(
                    "Worker returned an invalid evaluation result type"
                )
            record["performance"] = result.performance_manifest(task.performance_score)
            record["cost"] = result.cost_manifest(task.cost)
            log.record(stage, PASSED)
            stage = f"{prefix}.resource_receipt"
            receipt = result.resource_receipt
            if receipt is not None:
                record["resource_receipt"] = receipt.public_manifest()
            assessment = assess_evaluation_resources(
                envelope,
                receipt,
                episode_id=run_id,
                evaluation_id=evaluation_id,
                worker_id=worker_id,
            )
            record["assessment"] = assessment.public_manifest()
            if not assessment.satisfied:
                raise _ProbeEvidenceError(
                    "evaluation resource contract is not satisfied"
                )
            if receipt.receipt_id in receipt_ids:
                raise _ProbeEvidenceError("evaluation resource receipt_id was reused")
            receipt_ids.add(receipt.receipt_id)
            remaining_seconds()
            log.record(stage, PASSED)
            record["passed"] = True
            completed += 1
    except Exception as exc:  # noqa: BLE001 - retain failure and always clean the deployment
        if isinstance(exc, _ProbeDeadline):
            code = "operation_deadline_exceeded"
        elif isinstance(exc, WorkerTeardownFailure):
            code = "worker_teardown_unconfirmed"
        elif isinstance(exc, EvaluationResourceFailure):
            code = "unexpected_resource_rejection"
        elif isinstance(exc, WorkerFailure):
            code = "worker_failure"
        elif isinstance(exc, (TypeError, ValueError)):
            code = "invalid_worker_evidence"
        else:
            code = "unexpected_worker_exception"
        log.record(
            stage, FAILED, code=code, private_detail=f"{type(exc).__name__}: {exc}"
        )
        for case, _, _, _ in cases[completed:]:
            log.record(
                f"evaluation[{case}].completed",
                SKIPPED,
                code="probe_stopped_after_failure",
            )
    finally:
        try:
            confirmed = teardown()
        except Exception as exc:  # noqa: BLE001 - cleanup failures remain evidence
            log.record(
                "deployment.teardown_confirmed",
                FAILED,
                code="deployment_teardown_failed",
                private_detail=f"{type(exc).__name__}: {exc}",
            )
        else:
            log.expect(
                "deployment.teardown_confirmed",
                confirmed is True,
                code="deployment_teardown_unconfirmed",
            )
    return EvaluationProbeReport(
        probe_version=EVALUATION_PROBE_VERSION,
        run_id=run_id,
        worker_id=worker_id,
        checks=tuple(log.checks),
        task_id=task.task_id,
        evaluations=tuple(records),
    )
