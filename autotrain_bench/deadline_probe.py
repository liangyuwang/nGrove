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
MAX_DEADLINE_OBSERVATIONS = 4096


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

    def private_manifest(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "lease_id": self.lease_id,
            "worker_id": self.worker_id,
            "observation_id": self.observation_id,
            "source": self.source,
            "state": self.state.value,
            "deadline_triggered": self.deadline_triggered,
            "deadline_evidence_id": self.deadline_evidence_id,
        }


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
    for name, value in (
        ("duration_seconds", duration_seconds),
        ("startup_timeout_seconds", startup_timeout_seconds),
        ("termination_grace_seconds", termination_grace_seconds),
        ("observation_timeout_seconds", observation_timeout_seconds),
        ("poll_interval_seconds", poll_interval_seconds),
        ("cleanup_timeout_seconds", cleanup_timeout_seconds),
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
        or poll_interval_seconds < window_seconds / (MAX_DEADLINE_OBSERVATIONS - 2)
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
        if initial.state is not LeaseState.RUNNING or initial.deadline_triggered:
            raise _DeadlineCheckFailure(
                "initial_running_not_observed",
                "fixture was not observed running before expiry",
            )
        log.record(stage, PASSED)
        stage = "deadline.idle_expiry"
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
            log.record(
                "deadline.idle_expiry", SKIPPED, code="probe_stopped_after_failure"
            )
    finally:
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
                    if (
                        observation(cleanup_deadline, "cleanup").state
                        is not LeaseState.STOPPED
                    ):
                        raise _ProbeEvidenceError(
                            "owned workload absence is unconfirmed"
                        )
                except Exception as exc:  # noqa: BLE001 - callback cleanup still runs
                    log.record(
                        "lease.cleanup_confirmed",
                        FAILED,
                        code="lease_cleanup_unconfirmed",
                        private_detail=f"{type(exc).__name__}: {exc}",
                    )
                else:
                    record["cleanup_confirmed"] = True
                    log.record("lease.cleanup_confirmed", PASSED)
        finally:
            try:
                confirmed = teardown()
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
    return LeaseDeadlineProbeReport(
        probe_version=LEASE_DEADLINE_PROBE_VERSION,
        run_id=run_id,
        worker_id=worker_id,
        checks=tuple(log.checks),
        task_id=task.task_id,
        lease=record,
        observations=tuple(observations),
    )
