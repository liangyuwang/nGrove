"""Core-owned diagnostic probes for Worker Protocol resolution semantics.

These probes deliberately do not issue :class:`ConformanceReceipt`. They
exercise only workload and hidden-data resolution, while the active Worker
Protocol also includes Lease admission/execution/termination, evaluation,
resource evidence, isolation, and teardown guarantees. Treating this partial
coverage as complete protocol conformance would create false formal evidence.

Run a probe only against a dedicated, disposable Worker deployment. The
trusted Operator supplies the release-owned identities and a deployment-level
teardown callback. The callback always runs, including when a Worker raises an
unexpected exception. Platform-specific setup, watchdogs, and destruction
remain outside Core.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from .worker_protocol import (
    WORKER_PROTOCOL_VERSION,
    EvaluationTaskSpec,
    HiddenDatasetIdentityMismatch,
    HiddenDatasetRequest,
    HiddenDatasetResolutionFailure,
    ResolvedHiddenDataset,
    ResolvedWorkload,
    WorkerDescriptor,
    WorkloadRequest,
    WorkloadResolutionFailure,
)


RESOLUTION_PROBE_VERSION = f"worker-v{WORKER_PROTOCOL_VERSION}-resolution-probe-v1"

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
class ResolutionProbeReport:
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
