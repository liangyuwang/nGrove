"""Core-owned diagnostic probes for selected Worker Protocol semantics.

These probes deliberately do not issue :class:`ConformanceReceipt`. They
exercise resolution, Lease lifecycle (including optional independently observed
command timeouts) and evaluation resource evidence, while
the active Worker Protocol also includes physical resource enforcement,
isolation, full deadline/cancellation and recovery guarantees. The separate
``deadline_probe`` checks idle Lease expiry through a deployment observer,
optionally with an independently observed Worker pause or termination.
Treating this partial coverage as complete protocol conformance would create
false formal evidence.

Run a probe only against a dedicated, disposable Worker deployment. The
trusted Operator supplies the release-owned identities and a deployment-level
teardown callback. The callback always runs, including when a Worker raises an
unexpected exception. Platform-specific setup, watchdogs, and destruction
remain outside Core.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import TrainingResources
from .worker_protocol import (
    WORKER_PROTOCOL_VERSION,
    CandidateEvaluationResult,
    CommandResult,
    EvaluationResourceFailure,
    EvaluationTaskSpec,
    HiddenDatasetIdentityMismatch,
    HiddenDatasetRequest,
    HiddenDatasetResolutionFailure,
    LeaseState,
    ResearchTaskSpec,
    ResolvedHiddenDataset,
    ResolvedWorkload,
    ResourceAllocationReceipt,
    ResourceEnvelope,
    WorkerDescriptor,
    WorkerFailure,
    WorkerTeardownFailure,
    WorkloadRequest,
    WorkloadResolutionFailure,
    assess_evaluation_resources,
    assess_resource_allocation,
)


RESOLUTION_PROBE_VERSION = f"worker-v{WORKER_PROTOCOL_VERSION}-resolution-probe-v1"
EVALUATION_PROBE_VERSION = f"worker-v{WORKER_PROTOCOL_VERSION}-evaluation-probe-v1"
LEASE_PROBE_VERSION = f"worker-v{WORKER_PROTOCOL_VERSION}-lease-probe-v1"
LEASE_COMMAND_TIMEOUT_PROBE_VERSION = (
    f"worker-v{WORKER_PROTOCOL_VERSION}-lease-command-timeout-probe-v1"
)

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


@dataclass(frozen=True)
class LeaseProbeReport(WorkerProbeReport):
    """Research observations, with workload and command details kept private."""

    task_id: str = ""
    leases: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def private_manifest(self) -> Dict[str, Any]:
        payload = super().private_manifest()
        payload["task_id"] = self.task_id
        payload["leases"] = list(self.leases)
        return payload


@dataclass(frozen=True)
class CommandTimeoutObservation:
    """Independent command-family observation, not Worker return/status evidence.

    Evidence must bind the one exact argv invocation in this fresh Lease scope.
    ``running=False`` confirms absence of every command-owned descendant, not
    merely the local client. A source unable to establish that must raise.
    """

    episode_id: str
    lease_id: str
    worker_id: str
    command_sha256: str
    observation_id: str
    source: str
    running: bool
    start_evidence_id: Optional[str] = None
    timeout_evidence_id: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("episode_id", "lease_id", "worker_id", "observation_id", "source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"observation {name} must be a non-empty string")
        if (
            not isinstance(self.command_sha256, str)
            or len(self.command_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.command_sha256)
        ):
            raise ValueError("command_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.running, bool):
            raise TypeError("observation running must be a boolean")
        for name in ("start_evidence_id", "timeout_evidence_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string when supplied")

    def private_manifest(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class CommandTimeoutCase:
    """One public fixture command that keeps working past its requested timeout."""

    command: Sequence[str]
    observe: Callable[..., CommandTimeoutObservation]
    timeout_seconds: float = 2.0
    termination_grace_seconds: float = 1.0
    observation_timeout_seconds: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.command, (str, bytes)) or not isinstance(
            self.command, Sequence
        ):
            raise TypeError("timeout command must be an argument sequence")
        if not self.command or any(
            not isinstance(arg, str) or not arg for arg in self.command
        ):
            raise ValueError("timeout command arguments must be non-empty strings")
        object.__setattr__(self, "command", tuple(self.command))
        if not callable(self.observe):
            raise TypeError("timeout command observer must be callable")
        for name in (
            "timeout_seconds",
            "termination_grace_seconds",
            "observation_timeout_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.timeout_seconds + self.termination_grace_seconds):
            raise ValueError("command timeout and grace must have a finite sum")

    @property
    def command_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.command, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


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


class _CommandTimeoutFailure(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def probe_worker_leases(
    worker: Any,
    *,
    task: ResearchTaskSpec,
    workspace: Path,
    output_root: Path,
    resources: TrainingResources,
    resource_envelope: ResourceEnvelope,
    rejected_resource_envelope: ResourceEnvelope,
    command: Sequence[str],
    expected_stdout: str,
    teardown: Callable[[], bool],
    duration_seconds: float = 30.0,
    timeout_seconds: float = 60.0,
    cleanup_timeout_seconds: float = 30.0,
    command_timeout_case: Optional[CommandTimeoutCase] = None,
) -> LeaseProbeReport:
    """Exercise two accepted Leases and one rejected admission on public fixtures.

    The deployment supplies a disposable workspace and a harmless, deterministic
    command known to succeed in its fixture workload. No Agent or hidden data is
    involved. Successful calls must bind resource receipts, execute the command,
    stop, tolerate repeated termination, and no longer execute successfully.

    Each case shares an operation deadline. Lease duration is a separate Worker
    grant, not a startup timeout. The v16 start/state/stop methods have no timeout
    argument; this probe rejects late returns but cannot preempt a wedged call.
    An independent deployment watchdog must bound the entire diagnostic.
    Every attempted start is registered for compensation before calling Worker
    code. Unconfirmed Leases get one cleanup attempt under a shared, separate
    allowance, followed by mandatory deployment teardown even after interrupts.

    An optional public command timeout case runs once on the first accepted
    Lease. An independent observer must confirm a clean scope, actual command
    start, command-timeout enforcement and descendant absence before cleanup.
    """

    if not isinstance(task, ResearchTaskSpec):
        raise TypeError("task must be a ResearchTaskSpec")
    if not isinstance(resources, TrainingResources):
        raise TypeError("resources must be TrainingResources")
    resources = resources.validate()
    for envelope in (resource_envelope, rejected_resource_envelope):
        if not isinstance(envelope, ResourceEnvelope):
            raise TypeError("probe contracts must be ResourceEnvelope values")
        if not envelope.requirements:
            raise ValueError("probe contracts require normative resource properties")
    if resource_envelope.sha256 == rejected_resource_envelope.sha256:
        raise ValueError("accepted and rejected resource contracts must differ")
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        raise TypeError("command must be an argument sequence")
    if not command or any(not isinstance(arg, str) or not arg for arg in command):
        raise ValueError("command arguments must be non-empty strings")
    command = tuple(command)
    if command_timeout_case is not None:
        if not isinstance(command_timeout_case, CommandTimeoutCase):
            raise TypeError("command_timeout_case must be a CommandTimeoutCase")
        if command_timeout_case.command == command:
            raise ValueError("normal and timeout fixture commands must differ")
    if not isinstance(expected_stdout, str) or not expected_stdout:
        raise ValueError("expected_stdout must be a non-empty fixture marker")
    if not callable(teardown):
        raise TypeError("teardown must be callable")
    for name, value in (
        ("duration_seconds", duration_seconds),
        ("timeout_seconds", timeout_seconds),
        ("cleanup_timeout_seconds", cleanup_timeout_seconds),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    workspace = Path(workspace).resolve()
    output_root = Path(output_root).resolve()
    if not workspace.is_dir() or not output_root.is_dir():
        raise ValueError("workspace and output_root must be existing directories")
    for fixture in (
        workspace,
        Path(task.train_dir).resolve(),
        Path(task.public_api_dir).resolve(),
    ):
        if (
            fixture == output_root
            or fixture in output_root.parents
            or output_root in fixture.parents
        ):
            raise ValueError("probe inputs and output must be separate directory trees")
    workload_request = WorkloadRequest("training", task.training_image)
    run_id = "lease-probe:" + uuid.uuid4().hex
    run_dir = output_root / run_id.split(":", 1)[1]
    cases = (
        ("accepted_first", resource_envelope, False),
        ("accepted_second", resource_envelope, False),
        ("rejected", rejected_resource_envelope, True),
    )
    log = _Recorder()
    worker_id = "unknown"
    records: List[Mapping[str, Any]] = []
    pending: Dict[str, Dict[str, Any]] = {}
    receipt_ids: set[str] = set()
    resolved_identity = None
    completed = 0
    stage = "descriptor.available"

    def remaining(deadline: float) -> float:
        allowance = deadline - time.monotonic()
        if allowance <= 0:
            raise _ProbeDeadline("Lease probe operation deadline reached")
        return allowance

    def call(deadline: float, method: Callable[..., Any], **kwargs: Any) -> Any:
        remaining(deadline)
        try:
            value = method(**kwargs)
        except WorkerTeardownFailure:
            raise  # Unconfirmed workload ownership remains explicit even if late.
        except Exception:
            remaining(deadline)
            raise
        remaining(deadline)
        return value

    def state(deadline: float, lease_id: str) -> LeaseState:
        value = call(deadline, worker.lease_state, episode_id=run_id, lease_id=lease_id)
        if not isinstance(value, LeaseState):
            raise _ProbeEvidenceError("Worker returned an invalid Lease state")
        return value

    def stop(deadline: float, lease_id: str) -> None:
        result = call(
            deadline, worker.terminate_lease, episode_id=run_id, lease_id=lease_id
        )
        if result is not None or state(deadline, lease_id) is not LeaseState.STOPPED:
            raise _ProbeEvidenceError("Worker did not confirm a stopped Lease")

    def command_result(value: Any) -> Dict[str, Any]:
        if not isinstance(value, CommandResult):
            raise _ProbeEvidenceError("Worker returned an invalid command result")
        return {
            "exit_code": value.exit_code,
            "timed_out": value.timed_out,
            "stdout": value.stdout[:4096],
            "stderr": value.stderr[:4096],
            "stdout_truncated": value.stdout_truncated or len(value.stdout) > 4096,
            "stderr_truncated": value.stderr_truncated or len(value.stderr) > 4096,
        }

    def check_command_timeout(
        record: Dict[str, Any], lease_id: str, deadline: float, lease_deadline: float
    ) -> None:
        fixture = command_timeout_case
        assert fixture is not None
        evidence: Dict[str, Any] = {
            "command": list(fixture.command),
            "command_sha256": fixture.command_sha256,
            "timeout_seconds": fixture.timeout_seconds,
            "termination_grace_seconds": fixture.termination_grace_seconds,
            "observation_timeout_seconds": fixture.observation_timeout_seconds,
            "case_deadline_monotonic": deadline,
            "lease_earliest_expiry_monotonic": lease_deadline,
            "attempted": False,
            "passed": False,
            "observations": [],
        }
        record["command_timeout"] = evidence
        source = None

        def observe(until: float, phase: str) -> CommandTimeoutObservation:
            nonlocal source
            observation_id = "observation:" + uuid.uuid4().hex
            item: Dict[str, Any] = {
                "phase": phase,
                "observation_id": observation_id,
                "started_monotonic": time.monotonic(),
                "accepted": False,
            }
            evidence["observations"].append(item)
            observation_deadline = min(
                until, time.monotonic() + fixture.observation_timeout_seconds
            )

            def fetch(**kwargs: Any) -> Any:
                try:
                    value = fixture.observe(**kwargs)
                finally:
                    item["finished_monotonic"] = time.monotonic()
                if isinstance(value, CommandTimeoutObservation):
                    item["observation"] = value.private_manifest()
                return value

            value = call(
                observation_deadline,
                fetch,
                episode_id=run_id,
                lease_id=lease_id,
                worker_id=worker_id,
                argv=fixture.command,
                command_sha256=fixture.command_sha256,
                observation_id=observation_id,
                timeout_seconds=remaining(observation_deadline),
            )
            if not isinstance(value, CommandTimeoutObservation):
                raise _ProbeEvidenceError("invalid command timeout observation type")
            if (
                value.episode_id,
                value.lease_id,
                value.worker_id,
                value.command_sha256,
                value.observation_id,
            ) != (run_id, lease_id, worker_id, fixture.command_sha256, observation_id):
                raise _CommandTimeoutFailure(
                    "command_observation_binding_mismatch",
                    "stale or misbound command observation",
                )
            if source is not None and source != value.source:
                raise _CommandTimeoutFailure(
                    "command_observation_source_changed",
                    "command observer changed identity",
                )
            source = value.source
            item["accepted"] = True
            return value

        before = observe(min(deadline, lease_deadline), "before_command")
        if (
            before.running
            or before.start_evidence_id is not None
            or before.timeout_evidence_id is not None
        ):
            raise _CommandTimeoutFailure(
                "command_scope_not_clean",
                "timeout fixture has pre-existing work or events",
            )
        began = time.monotonic()
        command_deadline = (
            began + fixture.timeout_seconds + fixture.termination_grace_seconds
        )
        if command_deadline >= min(deadline, lease_deadline):
            raise _CommandTimeoutFailure(
                "command_timeout_allowance_insufficient",
                "command timeout and observation must fit before case and Lease expiry",
            )
        evidence["attempted"] = True
        evidence["started_monotonic"] = began
        evidence["deadline_monotonic"] = command_deadline

        def execute() -> Any:
            try:
                result = worker.exec_lease(
                    episode_id=run_id,
                    lease_id=lease_id,
                    argv=fixture.command,
                    timeout_seconds=fixture.timeout_seconds,
                )
            finally:
                evidence["returned_monotonic"] = time.monotonic()
            if isinstance(result, CommandResult):
                evidence["result"] = command_result(result)
            return result

        result = call(command_deadline, execute)
        if not isinstance(result, CommandResult):
            raise _ProbeEvidenceError("invalid timeout command result type")
        if not result.timed_out or result.exit_code == 0:
            raise _CommandTimeoutFailure(
                "command_timeout_not_reported",
                "fixture command did not report a failed timeout",
            )
        if evidence["returned_monotonic"] < began + fixture.timeout_seconds:
            raise _CommandTimeoutFailure(
                "command_timeout_reported_early",
                "command timeout reported before its grant elapsed",
            )
        after = observe(command_deadline, "after_command")
        if after.start_evidence_id is None or after.timeout_evidence_id is None:
            raise _CommandTimeoutFailure(
                "command_timeout_cause_unconfirmed",
                "independent command start and timeout evidence are required",
            )
        if after.running:
            raise _CommandTimeoutFailure(
                "command_workloads_still_running",
                "command or descendants survived the timeout return",
            )
        evidence["passed"] = True

    try:
        descriptor = worker.describe_capabilities()
        if not isinstance(descriptor, WorkerDescriptor):
            raise _ProbeEvidenceError("Worker returned an invalid descriptor type")
        worker_id = descriptor.worker_id
        if (
            descriptor.protocol_version != WORKER_PROTOCOL_VERSION
            or "research" not in descriptor.roles
        ):
            raise _ProbeEvidenceError(
                "Worker does not declare the current research role"
            )
        log.record(stage, PASSED)
        stage = "output.scope_created"
        run_dir.mkdir(exist_ok=False)
        log.record(stage, PASSED)
        for case, envelope, expect_rejection in cases:
            prefix = f"lease[{case}]"
            deadline = time.monotonic() + timeout_seconds
            lease_id = "lease:" + uuid.uuid4().hex
            record: Dict[str, Any] = {
                "case": case,
                "lease_id": lease_id,
                "passed": False,
                "start_attempted": False,
                "termination_confirmed": False,
                "expects_resource_rejection": expect_rejection,
                "resource_envelope": envelope.public_manifest(),
                "duration_seconds": duration_seconds,
                "command": list(command),
            }
            records.append(record)
            stage = f"{prefix}.workload_resolution"
            workload = call(
                deadline,
                worker.resolve_workload,
                episode_id=run_id,
                workload=workload_request,
                timeout_seconds=remaining(deadline),
            )
            if not isinstance(workload, ResolvedWorkload) or (
                workload.role != "training"
                or workload.requested_identity != task.training_image
            ):
                raise _ProbeEvidenceError("resolved workload does not bind its request")
            identity = workload.evidence_manifest()
            record["workload"] = identity
            if resolved_identity is not None and resolved_identity != identity:
                raise _ProbeEvidenceError("effective workload identity changed")
            resolved_identity = identity
            log.record(stage, PASSED)
            stage = f"{prefix}.admission"
            record["start_attempted"] = True
            pending[lease_id] = record
            admission_started = time.monotonic()
            try:
                receipt = call(
                    deadline,
                    worker.start_lease,
                    episode_id=run_id,
                    lease_id=lease_id,
                    duration_seconds=duration_seconds,
                    task=task,
                    workspace=workspace,
                    resources=resources,
                    resource_envelope=envelope,
                    lease_dir=run_dir / case,
                    workload=workload,
                )
            except WorkerTeardownFailure:
                raise
            except WorkerFailure as exc:
                if not expect_rejection:
                    raise
                record["admission_rejection"] = str(exc)
                # Ordinary failure promises no unaccounted workload. Observe
                # this before compensation could hide a leaked creation.
                if state(deadline, lease_id) is not LeaseState.STOPPED:
                    raise _ProbeEvidenceError(
                        "rejected admission left a live Lease"
                    ) from exc
            else:
                if isinstance(receipt, ResourceAllocationReceipt):
                    record["resource_receipt"] = receipt.public_manifest()
                if expect_rejection:
                    raise _ProbeEvidenceError("Worker accepted the rejected contract")
                stage = f"{prefix}.resource_receipt"
                assessment = assess_resource_allocation(
                    envelope,
                    receipt,
                    episode_id=run_id,
                    lease_id=lease_id,
                    worker_id=worker_id,
                )
                record["assessment"] = assessment.public_manifest()
                if not assessment.satisfied or receipt.receipt_id in receipt_ids:
                    raise _ProbeEvidenceError(
                        "invalid resource allocation or reused receipt ID"
                    )
                receipt_ids.add(receipt.receipt_id)
                log.record(stage, PASSED)
                stage = f"{prefix}.running"
                if state(deadline, lease_id) is not LeaseState.RUNNING:
                    raise _ProbeEvidenceError("admitted Lease is not running")
                log.record(stage, PASSED)
                stage = f"{prefix}.command"
                result = call(
                    deadline,
                    worker.exec_lease,
                    episode_id=run_id,
                    lease_id=lease_id,
                    argv=command,
                    timeout_seconds=remaining(deadline),
                )
                record["command_result"] = command_result(result)
                if (
                    result.exit_code != 0
                    or result.timed_out
                    or result.stdout_truncated
                    or result.stderr_truncated
                    or result.stdout != expected_stdout
                ):
                    raise _ProbeEvidenceError(
                        "fixture command did not succeed as expected"
                    )
                if case == "accepted_first" and command_timeout_case is not None:
                    log.record(stage, PASSED)
                    stage = f"{prefix}.command_timeout"
                    check_command_timeout(
                        record, lease_id, deadline, admission_started + duration_seconds
                    )
            log.record(stage, PASSED)
            stage = f"{prefix}.termination"
            stop(deadline, lease_id)
            log.record(stage, PASSED)
            stage = f"{prefix}.repeated_termination"
            stop(deadline, lease_id)
            log.record(stage, PASSED)
            if not expect_rejection:
                stage = f"{prefix}.post_stop_command"
                try:
                    result = call(
                        deadline,
                        worker.exec_lease,
                        episode_id=run_id,
                        lease_id=lease_id,
                        argv=command,
                        timeout_seconds=remaining(deadline),
                    )
                except WorkerTeardownFailure:
                    raise
                except WorkerFailure as exc:
                    record["post_stop_rejection"] = str(exc)
                else:
                    record["post_stop_result"] = command_result(result)
                    if (
                        result.exit_code == 0
                        or result.timed_out
                        or result.stdout
                        or result.stdout_truncated
                        or result.stderr_truncated
                    ):
                        raise _ProbeEvidenceError(
                            "stopped Lease did not reject the fixture command"
                        )
                if state(deadline, lease_id) is not LeaseState.STOPPED:
                    raise _ProbeEvidenceError("post-stop command revived a Lease")
                log.record(stage, PASSED)
            record["termination_confirmed"] = True
            pending.pop(lease_id)
            record["passed"] = True
            completed += 1
    except Exception as exc:  # noqa: BLE001 - diagnostic failure always compensates
        code = (
            exc.code
            if isinstance(exc, _CommandTimeoutFailure)
            else "operation_deadline_exceeded"
            if isinstance(exc, _ProbeDeadline)
            else "worker_teardown_unconfirmed"
            if isinstance(exc, WorkerTeardownFailure)
            else "worker_failure"
            if isinstance(exc, WorkerFailure)
            else "invalid_worker_evidence"
            if isinstance(exc, (TypeError, ValueError))
            else "unexpected_worker_exception"
        )
        log.record(
            stage, FAILED, code=code, private_detail=f"{type(exc).__name__}: {exc}"
        )
        for case, _, _ in cases[completed:]:
            log.record(
                f"lease[{case}].completed", SKIPPED, code="probe_stopped_after_failure"
            )
    finally:
        try:
            cleanup_deadline = time.monotonic() + cleanup_timeout_seconds
            for lease_id, record in pending.items():
                try:
                    stop(cleanup_deadline, lease_id)
                except Exception as exc:  # noqa: BLE001 - retain cleanup failure
                    log.record(
                        f"lease[{record['case']}].compensation",
                        FAILED,
                        code="lease_cleanup_unconfirmed",
                        private_detail=f"{type(exc).__name__}: {exc}",
                    )
                else:
                    record["termination_confirmed"] = True
                    log.record(f"lease[{record['case']}].compensation", PASSED)
        finally:
            try:
                confirmed = teardown()
            except Exception as exc:  # noqa: BLE001 - outer cleanup is independent
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
    return LeaseProbeReport(
        probe_version=(
            LEASE_COMMAND_TIMEOUT_PROBE_VERSION
            if command_timeout_case is not None
            else LEASE_PROBE_VERSION
        ),
        run_id=run_id,
        worker_id=worker_id,
        checks=tuple(log.checks),
        task_id=task.task_id,
        leases=tuple(records),
    )


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
