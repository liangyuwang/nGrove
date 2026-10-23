# Lease deadline diagnostic

`envcore.deadline_probe.probe_worker_lease_deadline()` checks one idle
Research Lease through an independent deployment observer, optionally while its
Worker control execution is paused or terminated. This extends the
[Lease lifecycle diagnostic](worker-lease-probe.md), which explicitly stops its
Leases and therefore cannot establish autonomous expiry. It exercises idle expiry and Worker-loss behavior.

## Independent observation

The trusted deployment supplies a read-only `observe()` callback. It must inspect
actual owned workloads and the independent controller's deadline events. It must
not call `Worker.lease_state()`, execute a command inside the Lease, stop a
workload, or trigger reconciliation that enforces the deadline. Merely labelling
a Worker-derived status as independent is insufficient.

Core starts a single public fixture Lease, validates its resource receipt and
observes it running before expiry. It then makes only observer calls and waits.
There are no Worker status, command or termination calls during this idle window.
Without fault injection, only after the diagnostic has passed or failed does it
call Worker termination for cleanup. A workload that stops only when queried or
manually terminated fails the expiry check, even if cleanup subsequently succeeds.

The callback receives keyword arguments `episode_id`, `lease_id`, `worker_id`,
`observation_id`, and `timeout_seconds`. It returns a `LeaseDeadlineObservation`
with the same four IDs plus:

| Field | Meaning |
| --- | --- |
| `source` | Stable identity of the independent observation boundary |
| `state` | Typed `LeaseState`; `STOPPED` means all workloads owned by this Lease are confirmed absent or stopped |
| `deadline_triggered` | Boolean indicating that the controller triggered this Lease's deadline termination |
| `deadline_evidence_id` | Nonempty reference to that controller event when triggered; otherwise `None` |
| `worker_fault` | Optional typed `WorkerFaultEvidence` confirming the selected fault remains active; required after injection in a fault case |

The deployment retains the underlying observations and event evidence. The
event must bind this Lease and its requested grant, not an unrelated job limit,
Operator cancellation, scheduled-but-untriggered deadline or prior execution.
Workload absence alone cannot prove why it stopped. Core requires both `STOPPED`
and deadline-trigger evidence; unexplained exits fail the check. Unknown states
and missing evidence do not become successful expiry observations.

Each observer call gets a fresh observation ID. Replayed or misbound snapshots,
changed source identities and late replies are rejected. The callback must
actually take a fresh observation; copying an old value and replacing its ID
does not satisfy this contract. These are trusted diagnostic inputs, not signed
receipts or a new remote attestation protocol.

## Inputs and invocation

Prepare an authorized disposable Research Worker with public synthetic TRAIN,
public API files, a separate fixture workspace, prepared workload artifacts, and
a supported normative resource envelope. No Agent, participant workspace or
hidden evaluation data is involved. The output tree must be separate from all
fixture input trees. No platform setup or observer implementation is bundled
into Core.

The deployment's trusted launcher can then call:

```python
from envcore.deadline_probe import probe_worker_lease_deadline

report = probe_worker_lease_deadline(
    research_worker,
    task=public_fixture_task,
    workspace=fixture_workspace,
    output_root=diagnostic_output_root,
    resources=resources,
    resource_envelope=supported_envelope,
    observe=independent_observer,
    teardown=cleanup_and_confirm,
    duration_seconds=2,
    startup_timeout_seconds=60,
    termination_grace_seconds=1,
    observation_timeout_seconds=1,
    poll_interval_seconds=0.1,
    cleanup_timeout_seconds=30,
)
```

There is at most one Lease start attempt. Resolution and admission share the
startup allowance. Registering the semantic Lease before the call preserves
cleanup responsibility if startup creates work and then fails, is interrupted,
or returns too late.

## Worker pause or termination

Select **one** authorized fault per fresh diagnostic scope with
`worker_fault=WorkerFaultKind.PAUSED` or `WorkerFaultKind.TERMINATED`, and supply
an `inject_fault()` callback. These parameters must be supplied together. The
default remains the ordinary idle case. The fault version reports
`worker-v17-worker-fault-deadline-probe-v1`; the ordinary idle version retains
`worker-v17-idle-deadline-probe-v1`.

The callback receives `episode_id`, `lease_id`, `worker_id`, a fresh `fault_id`,
the typed `kind`, and `timeout_seconds`, and must return `None`. The deployment
maps these semantic identities to the exact disposable Worker control scope.
Its implementation owns the platform pause/termination mechanism; Core supplies
no signals, runtime commands, host names or platform handles.

Inject only after the initial live observation. Pause or terminate **the Worker
execution that handles this fixture's lifecycle calls**, including its internal
deadline enforcement and replacement Workers. The independently supervised
deadline controller, observer, diagnostic caller and external recovery path must
survive that fault in the frozen deployment. Do not move or add a controller just
for the diagnostic and claim guarantees for the original deployment. The injector must not terminate
the Lease workload, trigger its deadline, change the grant, or resume/restart
Worker control before the expiry decision. An unavailable client connection or
failed Worker status request is not evidence that control execution stopped.

For this mode the observer returns `WorkerFaultEvidence(fault_id, kind,
evidence_id)` in `LeaseDeadlineObservation.worker_fault`. The evidence must bind
the exact fault request and attest that the selected fault has remained active
without interruption since injection. Retain independent control-state and
replacement/fencing evidence in the deployment. Missing or uncertain evidence,
a different fault kind/ID, or observed revival fails the case. Merely repeating
the injector's acknowledgement or copying a previous snapshot cannot establish
continued control loss.

The probe requires a second live Lease observation under the fault, with no
deadline event yet. Injection **and** this confirmation share
`fault_timeout_seconds` (default 1 second), capped at the earliest possible Lease
expiry. Both must finish before that bound; injection cannot extend the original
grant or its termination grace. Every subsequent expiry observation, including
the final `STOPPED` observation, must still carry matching active-fault evidence.
A fault that immediately stops the workload does not pass this deadline case.

For example, add these arguments to the invocation above:

```python
from envcore.deadline_probe import WorkerFaultKind

# Additional keyword arguments for one disposable, authorized pause case:
fault_options = dict(
    worker_fault=WorkerFaultKind.PAUSED,
    inject_fault=pause_owned_worker_control,
    fault_timeout_seconds=1,
)
```

The caller passes `**fault_options` to `probe_worker_lease_deadline()`. Run the
termination case separately only when it is also in the approved diagnostic
scope; the probe does not automatically escalate a pause to termination or retry.

## Clocks and cleanup

Worker v17 has no atomic cross-clock activation receipt. The successful
`start_lease()` return is used as a **conservative latest activation bound**.
Confirmation must return by that time plus `duration_seconds` and the explicit
termination grace. This is an upper-bound diagnostic, not proof of exact
activation timing or exact resource consumption. A triggered deadline observed
before admission-call start plus the grant is provably premature and fails.
The fixture must first be observed running with no triggered deadline; a fixture
already stopped at the initial observation cannot establish this test case.

The grace includes controller termination and observer latency. Each callback
gets the smaller of its own observation allowance and the remaining window;
a late response cannot retroactively pass the check. Select fixture duration,
grace and polling interval for the deployment's expected latency. The interval
must fit the grace and the report's 4096-observation limit. Core does not silently
extend the grant or the observation window to obtain a successful result.

Without an attempted fault injection, cleanup uses a separate shared allowance:
one Worker termination attempt, then an independent absence observation.
Regardless of that outcome, the mandatory bounded deployment `teardown()` callback
runs and must return exactly `True` to
confirm final cleanup. It owns remaining workloads and private runtime storage.
Interrupts propagate after cleanup. Retain diagnostic artifacts separately from
disposable runtime storage under the agreed retention policy.

Once injection is **attempted**, including a partial failure, late return or
interrupt, Core never calls another Worker method. Calling the possibly paused
or dead Worker for compensation could wedge the diagnostic. Instead, the
mandatory external `teardown()` callback runs once and an independent observation
then confirms Lease workload absence. The callback and that observation share
`cleanup_timeout_seconds`; a late callback reply cannot establish timely cleanup.
The callback retains its zero-argument interface and must impose its own bounded
mechanism within that allowance. It must fence remaining Worker control, owned
workloads, controller activity and private runtime storage, and return exactly
`True`. It may resolve the injected fault during cleanup; the final observation
need not claim the fault remains active after teardown. Preserve the observer's
ability to confirm absence after private runtime storage is reclaimed.

The callback's acknowledgement and independently observed Lease absence are
separate checks. Neither replaces the other; failed or unconfirmed final cleanup
fails the diagnostic even when independent expiry passed. If injection was never
attempted, the original Worker compensation followed by external teardown applies.

Run the entire diagnostic behind an independent deployment watchdog. Synchronous
Worker methods, observers and callbacks can wedge; checking elapsed time does
not preempt them. Include preparation, the single Lease request, startup, idle
observation, compensation, reporting and outer teardown in the resource/time
authorization. This helper does not authorize an allocation or further retries.

## Report and limits

The public Operator summary contains stable check codes and the observed outcome. The private report retains the Lease request,
workload and allocation evidence, caller monotonic timestamps, observer requests
and returned observations, including unaccepted late replies when available.
An observation's `accepted` flag means its timeliness and identity bindings
passed. `lease.deadline_passed` separately records the idle-expiry decision;
`lease.cleanup_confirmed` records subsequent cleanup. A failed expiry cannot be
repaired by successful cleanup, and failed cleanup prevents the overall probe
from passing even if expiry was observed correctly.

Fault reports also retain the selected kind, fresh fault request ID, injection
attempt/return timestamps and independent fault evidence. The
`lease.worker_fault.accepted` flag records successful pre-expiry injection and
live-state confirmation; it does not replace `lease.deadline_passed`. The latter
requires fault confirmation on all subsequent expiry observations too. Private
fault evidence never appears in the public check summary.

Regression coverage includes deterministic failed-injection/revival/evidence and
cleanup cases, plus real local control/workload/controller processes. The process
fixtures cover pause and termination with an independent controller, and reject
the case where pausing Worker control disables deadline enforcement. Their
allocation receipts are fixtures; they are not a real v17 Worker deployment run.

Passing does not prove observer honesty, the whole deployment's independence,
exact Worker-local activation timing, command cancellation, allocation/host loss
recovery, physical isolation. A fault case covers only its
selected fault and frozen fixture; the ordinary idle case covers no Worker fault.
Those require independently reviewed deployment evidence and the remaining
cases in the [acceptance matrix](worker-acceptance-matrix.md).
