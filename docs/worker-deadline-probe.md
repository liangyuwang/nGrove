# Idle Lease deadline diagnostic

`autotrain_bench.deadline_probe.probe_worker_lease_deadline()` checks one idle
Research Lease through an independent deployment observer. This extends the
[Lease lifecycle diagnostic](worker-lease-probe.md), which explicitly stops its
Leases and therefore cannot establish autonomous expiry. It contributes to
original issues 4/8/10 without changing Worker Protocol v16 or formal eligibility.

## Independent observation

The trusted deployment supplies a read-only `observe()` callback. It must inspect
actual owned workloads and the independent controller's deadline events. It must
not call `Worker.lease_state()`, execute a command inside the Lease, stop a
workload, or trigger reconciliation that enforces the deadline. Merely labelling
a Worker-derived status as independent is insufficient.

Core starts a single public fixture Lease, validates its resource receipt and
observes it running before expiry. It then makes only observer calls and waits.
There are no Worker status, command or termination calls during this idle window.
Only after the diagnostic has passed or failed does it call Worker termination
for cleanup. A workload that stops only when queried or manually terminated fails
the expiry check, even if cleanup subsequently succeeds.

The callback receives keyword arguments `episode_id`, `lease_id`, `worker_id`,
`observation_id`, and `timeout_seconds`. It returns a `LeaseDeadlineObservation`
with the same four IDs plus:

| Field | Meaning |
| --- | --- |
| `source` | Stable identity of the independent observation boundary |
| `state` | Typed `LeaseState`; `STOPPED` means all workloads owned by this Lease are confirmed absent or stopped |
| `deadline_triggered` | Boolean indicating that the controller triggered this Lease's deadline termination |
| `deadline_evidence_id` | Nonempty reference to that controller event when triggered; otherwise `None` |

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
from autotrain_bench.deadline_probe import probe_worker_lease_deadline

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

## Clocks and cleanup

Worker v16 has no atomic cross-clock activation receipt. The successful
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

Cleanup uses a separate shared allowance: one Worker termination attempt, then
an independent absence observation. Regardless of that outcome, the mandatory
bounded deployment `teardown()` callback runs and must return exactly `True` to
confirm final cleanup. It owns remaining workloads and private runtime storage.
Interrupts propagate after cleanup. Retain diagnostic artifacts separately from
disposable runtime storage under the agreed retention policy.

Run the entire diagnostic behind an independent deployment watchdog. Synchronous
Worker methods, observers and callbacks can wedge; checking elapsed time does
not preempt them. Include preparation, the single Lease request, startup, idle
observation, compensation, reporting and outer teardown in the resource/time
authorization. This helper does not authorize an allocation or further retries.

## Report and limits

The public Operator summary contains stable check codes and always states
`formal_conformance: false`. The private report retains the Lease request,
workload and allocation evidence, caller monotonic timestamps, observer requests
and returned observations, including unaccepted late replies when available.
An observation's `accepted` flag means its timeliness and identity bindings
passed. `lease.deadline_passed` separately records the idle-expiry decision;
`lease.cleanup_confirmed` records subsequent cleanup. A failed expiry cannot be
repaired by successful cleanup, and failed cleanup prevents the overall probe
from passing even if expiry was observed correctly.

Passing does not prove observer honesty, the whole deployment's independence,
exact Worker-local activation timing, command cancellation, paused/killed Worker
behavior, allocation/host loss recovery, physical isolation or full conformance.
Those require independently reviewed deployment evidence and the remaining
cases in the [acceptance matrix](worker-acceptance-matrix.md). No capability,
trust, compliance or leaderboard flag is upgraded by this report.
