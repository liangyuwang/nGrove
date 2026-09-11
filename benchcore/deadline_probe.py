"""Idle Lease deadline diagnostic using an independent deployment observer.

Core never polls the Worker's state or sends commands to make an idle Lease
expire. Platform observation, deadline enforcement and recovery stay with the
trusted deployment. This diagnostic cannot issue formal conformance evidence.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .config import TrainingResources
from .conformance import (
    FAILED,
    PASSED,
    SKIPPED,
    WorkerProbeReport,
    _ProbeDeadline,
    _ProbeEvidenceError,
    _Recorder,
)
from .worker_protocol import (
    WORKER_PROTOCOL_VERSION,
    LeaseState,
    ResearchTaskSpec,
    ResolvedWorkload,
    ResourceAllocationReceipt,
    ResourceEnvelope,
    WorkerDescriptor,
    WorkerFailure,
    WorkerTeardownFailure,
    WorkloadRequest,
    assess_resource_allocation,
)


LEASE_DEADLINE_PROBE_VERSION = (
    f"worker-v{WORKER_PROTOCOL_VERSION}-idle-deadline-probe-v1"
)
WORKER_FAULT_DEADLINE_PROBE_VERSION = (
    f"worker-v{WORKER_PROTOCOL_VERSION}-worker-fault-deadline-probe-v1"
)
MAX_DEADLINE_OBSERVATIONS = 4096


class WorkerFaultKind(str, Enum):
    PAUSED = "paused"
    TERMINATED = "terminated"


@dataclass(frozen=True)
class WorkerFaultEvidence:
    """Independent confirmation that the injected Worker fault is still active.

    All control execution for this Worker's fixture scope must have remained
    paused or terminated since injection, including automatic replacements.
    An unreachable client or a failed status request does not establish either
    condition. The enclosing observation binds the episode, Lease and Worker.
    """

    fault_id: str
    kind: WorkerFaultKind
    evidence_id: str

    def __post_init__(self) -> None:
        for name in ("fault_id", "evidence_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"fault {name} must be a non-empty string")
        if not isinstance(self.kind, WorkerFaultKind):
            raise TypeError("fault kind must be a WorkerFaultKind")

    def private_manifest(self) -> Dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "kind": self.kind.value,
            "evidence_id": self.evidence_id,
        }


@dataclass(frozen=True)
class LeaseDeadlineObservation:
    """Fresh, read-only observation from the deployment's independent controller.

    The source must inspect actual owned workloads, without invoking Worker
    methods or triggering cleanup. Deadline evidence identifies the controller's
    event that caused termination; a scheduled deadline alone is not that event.
    These are trusted diagnostic inputs, not remotely authenticated receipts.
    """

    episode_id: str
    lease_id: str
    worker_id: str
    observation_id: str
    source: str
    state: LeaseState
    deadline_triggered: bool
    deadline_evidence_id: Optional[str] = None
    worker_fault: Optional[WorkerFaultEvidence] = None

    def __post_init__(self) -> None:
        for name in ("episode_id", "lease_id", "worker_id", "observation_id", "source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"observation {name} must be a non-empty string")
        if not isinstance(self.state, LeaseState):
            raise TypeError("observation state must be a LeaseState")
        if not isinstance(self.deadline_triggered, bool):
            raise TypeError("deadline_triggered must be a boolean")
        if self.deadline_triggered:
            if (
                not isinstance(self.deadline_evidence_id, str)
                or not self.deadline_evidence_id.strip()
            ):
                raise ValueError("a triggered deadline requires an evidence ID")
        elif self.deadline_evidence_id is not None:
            raise ValueError("an untriggered deadline cannot carry a termination event")
        if self.worker_fault is not None and not isinstance(
            self.worker_fault, WorkerFaultEvidence
        ):
            raise TypeError("worker_fault must be WorkerFaultEvidence")

    def private_manifest(self) -> Dict[str, Any]:
        payload = {
            "episode_id": self.episode_id,
            "lease_id": self.lease_id,
            "worker_id": self.worker_id,
            "observation_id": self.observation_id,
            "source": self.source,
            "state": self.state.value,
            "deadline_triggered": self.deadline_triggered,
            "deadline_evidence_id": self.deadline_evidence_id,
        }
        if self.worker_fault is not None:
            payload["worker_fault"] = self.worker_fault.private_manifest()
        return payload


@dataclass(frozen=True)
class LeaseDeadlineProbeReport(WorkerProbeReport):
    task_id: str = ""
    lease: Mapping[str, Any] = field(default_factory=dict)
    observations: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def private_manifest(self) -> Dict[str, Any]:
        payload = super().private_manifest()
        payload["task_id"] = self.task_id
        payload["lease"] = dict(self.lease)
        payload["observations"] = list(self.observations)
        return payload


class _DeadlineCheckFailure(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def probe_worker_lease_deadline(
    worker: Any,
    *,
    task: ResearchTaskSpec,
    workspace: Path,
    output_root: Path,
    resources: TrainingResources,
    resource_envelope: ResourceEnvelope,
    observe: Callable[..., LeaseDeadlineObservation],
    teardown: Callable[[], bool],
    worker_fault: Optional[WorkerFaultKind] = None,
    inject_fault: Optional[Callable[..., None]] = None,
    fault_timeout_seconds: float = 1.0,
    duration_seconds: float = 2.0,
    startup_timeout_seconds: float = 60.0,
    termination_grace_seconds: float = 1.0,
    observation_timeout_seconds: float = 1.0,
    poll_interval_seconds: float = 0.1,
    cleanup_timeout_seconds: float = 30.0,
) -> LeaseDeadlineProbeReport:
    """Observe one idle Lease through expiry before any Worker cleanup call.

    The trusted deployment supplies public fixtures and a read-only observer
    independent of Worker code. Each observation binds a fresh request ID and
    the exact episode/Lease/Worker. It must report both workload absence and
    independent deadline-trigger evidence to pass. Observation latency counts
    inside the termination allowance. A deployment watchdog must bound wedged
    synchronous calls, observer calls and final teardown.

    Optionally inject a deployment-owned Worker pause or termination after the
    live observation. Independent observations must confirm the fault remains
    active through expiry. No Worker method is called after injection begins,
    including on injection failure; the external teardown owns recovery.
    """

    if not isinstance(task, ResearchTaskSpec):
        raise TypeError("task must be a ResearchTaskSpec")
    if not isinstance(resources, TrainingResources):
        raise TypeError("resources must be TrainingResources")
    resources = resources.validate()
    if not isinstance(resource_envelope, ResourceEnvelope):
        raise TypeError("resource_envelope must be a ResourceEnvelope")
    if not resource_envelope.requirements:
        raise ValueError("the diagnostic requires normative resource properties")
    if not callable(observe) or not callable(teardown):
        raise TypeError("observe and teardown must be callable")
    if worker_fault is None:
        if inject_fault is not None:
            raise ValueError("inject_fault requires a selected worker_fault")
    elif not isinstance(worker_fault, WorkerFaultKind) or not callable(inject_fault):
        raise TypeError("worker_fault requires a typed kind and callable inject_fault")
    for name, value in (
        ("duration_seconds", duration_seconds),
        ("startup_timeout_seconds", startup_timeout_seconds),
        ("termination_grace_seconds", termination_grace_seconds),
        ("observation_timeout_seconds", observation_timeout_seconds),
        ("poll_interval_seconds", poll_interval_seconds),
        ("cleanup_timeout_seconds", cleanup_timeout_seconds),
        ("fault_timeout_seconds", fault_timeout_seconds),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    window_seconds = duration_seconds + termination_grace_seconds
    if (
        not math.isfinite(window_seconds)
        or poll_interval_seconds
        < window_seconds / (MAX_DEADLINE_OBSERVATIONS - (3 if worker_fault else 2))
        or poll_interval_seconds > termination_grace_seconds
    ):
        raise ValueError(
            "poll interval must fit the termination grace and observation limit"
        )
    workspace, output_root = Path(workspace).resolve(), Path(output_root).resolve()
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
    request = WorkloadRequest("training", task.training_image)
    run_id = "deadline-probe:" + uuid.uuid4().hex
    lease_id = "lease:" + uuid.uuid4().hex
    run_dir = output_root / run_id.split(":", 1)[1]
    log = _Recorder()
    worker_id = "unknown"
    record: Dict[str, Any] = {
        "lease_id": lease_id,
        "start_attempted": False,
        "deadline_passed": False,
        "cleanup_confirmed": False,
        "duration_seconds": duration_seconds,
        "termination_grace_seconds": termination_grace_seconds,
        "startup_timeout_seconds": startup_timeout_seconds,
        "observation_timeout_seconds": observation_timeout_seconds,
        "poll_interval_seconds": poll_interval_seconds,
        "cleanup_timeout_seconds": cleanup_timeout_seconds,
        "resource_envelope": resource_envelope.public_manifest(),
    }
    fault_record: Dict[str, Any] = (
        {
            "fault_id": "fault:" + uuid.uuid4().hex,
            "kind": worker_fault.value,
            "timeout_seconds": fault_timeout_seconds,
            "attempted": False,
            "accepted": False,
        }
        if worker_fault is not None
        else {}
    )
    if fault_record:
        record["worker_fault"] = fault_record
    expiry_check = (
        "deadline.expiry_during_worker_fault"
        if worker_fault is not None
        else "deadline.idle_expiry"
    )
    observations = []
    source = None
    stage = "descriptor.available"
    deadline_started = False

    def remaining(deadline: float) -> float:
        value = deadline - time.monotonic()
        if value <= 0:
            raise _ProbeDeadline("deadline diagnostic allowance exhausted")
        return value

    def call(deadline: float, method: Callable[..., Any], **kwargs: Any) -> Any:
        remaining(deadline)
        try:
            value = method(**kwargs)
        except WorkerTeardownFailure:
            raise
        except Exception:
            remaining(deadline)
            raise
        remaining(deadline)
        return value

    def observation(deadline: float, phase: str) -> LeaseDeadlineObservation:
        nonlocal source
        if len(observations) >= MAX_DEADLINE_OBSERVATIONS:
            raise _DeadlineCheckFailure(
                "observation_limit_reached", "too many observations"
            )
        observation_id = "observation:" + uuid.uuid4().hex
        began = time.monotonic()
        item: Dict[str, Any] = {
            "phase": phase,
            "requested_observation_id": observation_id,
            "started_monotonic": began,
            "accepted": False,
        }
        observations.append(item)
        limit = min(deadline, began + observation_timeout_seconds)

        def fetch(**kwargs: Any) -> Any:
            try:
                value = observe(**kwargs)
            finally:
                item["finished_monotonic"] = time.monotonic()
            if isinstance(value, LeaseDeadlineObservation):
                item["observation"] = value.private_manifest()
            return value

        value = call(
            limit,
            fetch,
            episode_id=run_id,
            lease_id=lease_id,
            worker_id=worker_id,
            observation_id=observation_id,
            timeout_seconds=remaining(limit),
        )
        if not isinstance(value, LeaseDeadlineObservation):
            raise _ProbeEvidenceError("observer returned an invalid observation type")
        if (
            value.episode_id != run_id
            or value.lease_id != lease_id
            or value.worker_id != worker_id
            or value.observation_id != observation_id
        ):
            raise _DeadlineCheckFailure(
                "observation_binding_mismatch",
                "observer returned stale or misbound evidence",
            )
        if source is not None and source != value.source:
            raise _DeadlineCheckFailure(
                "observation_source_changed", "observer identity changed"
            )
        source = value.source
        item["accepted"] = True
        return value

    def require_fault(value: LeaseDeadlineObservation) -> None:
        evidence = value.worker_fault
        if evidence is None:
            raise _DeadlineCheckFailure(
                "worker_fault_unconfirmed",
                "Worker fault is not independently confirmed",
            )
        if (
            evidence.fault_id != fault_record["fault_id"]
            or evidence.kind != worker_fault
        ):
            raise _DeadlineCheckFailure(
                "worker_fault_binding_mismatch",
                "observation does not bind the injected fault",
            )

    def confirm_absence(deadline: float) -> None:
        if observation(deadline, "cleanup").state is not LeaseState.STOPPED:
            raise _ProbeEvidenceError("owned workload absence is unconfirmed")
        record["cleanup_confirmed"] = True
        log.record("lease.cleanup_confirmed", PASSED)

    def cleanup_failure(exc: Exception) -> None:
        log.record(
            "lease.cleanup_confirmed",
            FAILED,
            code="lease_cleanup_unconfirmed",
            private_detail=f"{type(exc).__name__}: {exc}",
        )

    def final_teardown(deadline: Optional[float] = None) -> None:
        try:
            confirmed = teardown() if deadline is None else call(deadline, teardown)
        except Exception as exc:  # noqa: BLE001 - deployment owns this final boundary
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

    try:
        descriptor = worker.describe_capabilities()
        if not isinstance(descriptor, WorkerDescriptor):
            raise _ProbeEvidenceError("Worker returned an invalid descriptor")
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
        run_dir.mkdir(mode=0o700, exist_ok=False)
        log.record(stage, PASSED)
        startup_deadline = time.monotonic() + startup_timeout_seconds
        stage = "lease.workload_resolution"
        workload = call(
            startup_deadline,
            worker.resolve_workload,
            episode_id=run_id,
            workload=request,
            timeout_seconds=remaining(startup_deadline),
        )
        if not isinstance(workload, ResolvedWorkload) or (
            workload.role != "training"
            or workload.requested_identity != task.training_image
        ):
            raise _ProbeEvidenceError("resolved workload does not bind its request")
        record["workload"] = workload.evidence_manifest()
        log.record(stage, PASSED)
        stage = "lease.admission"
        record["start_attempted"] = True
        record["admission_started_monotonic"] = time.monotonic()
        receipt = call(
            startup_deadline,
            worker.start_lease,
            episode_id=run_id,
            lease_id=lease_id,
            duration_seconds=duration_seconds,
            task=task,
            workspace=workspace,
            resources=resources,
            resource_envelope=resource_envelope,
            lease_dir=run_dir / "lease",
            workload=workload,
        )
        acknowledged = time.monotonic()
        record["admission_returned_monotonic"] = acknowledged
        if isinstance(receipt, ResourceAllocationReceipt):
            record["resource_receipt"] = receipt.public_manifest()
        assessment = assess_resource_allocation(
            resource_envelope,
            receipt,
            episode_id=run_id,
            lease_id=lease_id,
            worker_id=worker_id,
        )
        record["assessment"] = assessment.public_manifest()
        if not assessment.satisfied:
            raise _ProbeEvidenceError("Lease resource envelope is not satisfied")
        log.record(stage, PASSED)
        # v16 has no cross-clock atomic activation acknowledgement. The return
        # time is a conservative latest activation bound, not the true start.
        latest_stop = acknowledged + window_seconds
        earliest_expiry = record["admission_started_monotonic"] + duration_seconds
        record["latest_stop_monotonic"] = latest_stop
        stage = "deadline.initial_running"
        initial = observation(latest_stop, "initial")
        if (
            initial.state is not LeaseState.RUNNING
            or initial.deadline_triggered
            or initial.worker_fault is not None
        ):
            raise _DeadlineCheckFailure(
                "initial_running_not_observed",
                "fixture was not observed running before expiry",
            )
        log.record(stage, PASSED)
        if worker_fault is not None:
            stage = "deadline.worker_fault_injected"
            # Injection and its live/fault confirmation must finish before even
            # the earliest possible expiry, without extending the original grant.
            fault_deadline = min(
                earliest_expiry, time.monotonic() + fault_timeout_seconds
            )
            fault_allowance = remaining(fault_deadline)
            fault_record["attempted"] = True
            fault_record["started_monotonic"] = time.monotonic()
            try:
                injected = call(
                    fault_deadline,
                    inject_fault,
                    episode_id=run_id,
                    lease_id=lease_id,
                    worker_id=worker_id,
                    fault_id=fault_record["fault_id"],
                    kind=worker_fault,
                    timeout_seconds=fault_allowance,
                )
            finally:
                fault_record["finished_monotonic"] = time.monotonic()
            if injected is not None:
                raise _ProbeEvidenceError("fault injection must return None")
            after_fault = observation(fault_deadline, "post_fault")
            require_fault(after_fault)
            if (
                after_fault.state is not LeaseState.RUNNING
                or after_fault.deadline_triggered
            ):
                raise _DeadlineCheckFailure(
                    "running_after_fault_not_observed",
                    "a live Lease was not observed under fault before expiry",
                )
            fault_record["accepted"] = True
            log.record(stage, PASSED)
        stage = expiry_check
        deadline_started = True
        while True:
            allowance = latest_stop - time.monotonic()
            if allowance <= 0:
                raise _DeadlineCheckFailure(
                    "idle_lease_deadline_exceeded",
                    "idle workload absence was not confirmed in time",
                )
            time.sleep(min(poll_interval_seconds, allowance))
            if time.monotonic() >= latest_stop:
                raise _DeadlineCheckFailure(
                    "idle_lease_deadline_exceeded",
                    "idle workload absence was not confirmed in time",
                )
            value = observation(latest_stop, "idle")
            if worker_fault is not None:
                require_fault(value)
            if value.deadline_triggered and time.monotonic() < earliest_expiry:
                raise _DeadlineCheckFailure(
                    "deadline_triggered_early",
                    "deadline triggered before the earliest possible grant expiry",
                )
            if value.state is LeaseState.STOPPED:
                if not value.deadline_triggered:
                    raise _DeadlineCheckFailure(
                        "deadline_cause_unconfirmed",
                        "workload stopped without independent deadline evidence",
                    )
                record["deadline_passed"] = True
                log.record(stage, PASSED)
                break
            if value.state not in {LeaseState.RUNNING, LeaseState.STOPPING}:
                raise _DeadlineCheckFailure(
                    "workload_state_unconfirmed",
                    "observer cannot confirm the workload state",
                )
    except Exception as exc:  # noqa: BLE001 - retain diagnostic failure, always clean up
        code = (
            exc.code
            if isinstance(exc, _DeadlineCheckFailure)
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
        if not deadline_started:
            log.record(expiry_check, SKIPPED, code="probe_stopped_after_failure")
    finally:
        if fault_record.get("attempted"):
            # Even a failed/late injection can leave the Worker unavailable.
            # Never risk a synchronous call into it on this recovery path.
            cleanup_deadline = time.monotonic() + cleanup_timeout_seconds
            try:
                final_teardown(cleanup_deadline)
            finally:
                try:
                    confirm_absence(cleanup_deadline)
                except Exception as exc:  # noqa: BLE001 - preserve cleanup uncertainty
                    cleanup_failure(exc)
        else:
            try:
                if record["start_attempted"]:
                    try:
                        cleanup_deadline = time.monotonic() + cleanup_timeout_seconds
                        stopped = call(
                            cleanup_deadline,
                            worker.terminate_lease,
                            episode_id=run_id,
                            lease_id=lease_id,
                        )
                        if stopped is not None:
                            raise _ProbeEvidenceError(
                                "termination returned an invalid result"
                            )
                        confirm_absence(cleanup_deadline)
                    except Exception as exc:  # noqa: BLE001 - callback cleanup still runs
                        cleanup_failure(exc)
            finally:
                final_teardown()
    return LeaseDeadlineProbeReport(
        probe_version=(
            WORKER_FAULT_DEADLINE_PROBE_VERSION
            if worker_fault is not None
            else LEASE_DEADLINE_PROBE_VERSION
        ),
        run_id=run_id,
        worker_id=worker_id,
        checks=tuple(log.checks),
        task_id=task.task_id,
        lease=record,
        observations=tuple(observations),
    )
