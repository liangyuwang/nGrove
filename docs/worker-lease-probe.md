# Research Worker Lease diagnostic

`autotrain_bench.conformance.probe_worker_leases()` exercises the supplied
Research Worker through Worker Protocol v16. It adds reusable Lease lifecycle
coverage toward original issue 10, alongside the resolution and
[evaluation resource](worker-evaluation-probe.md) diagnostics. It does not
create a Tested Agent, run hidden evaluation, select an incumbent, publish a
benchmark result, or issue a formal `ConformanceReceipt`.

## Disposable inputs

The trusted deployment prepares a public synthetic `ResearchTaskSpec`, a
dedicated fixture workspace, an existing trusted output root, and the fixture's
runtime artifacts. Keep the output tree separate from the workspace, TRAIN and
public API trees; the probe rejects overlapping paths before Worker calls or
writes. It creates a fresh child output directory, semantic episode scope and
Lease IDs for every invocation. Do not use a live episode, a participant
workspace, real hidden data, or production resources outside the approved scope.

Supply one supported `ResourceEnvelope` and a different envelope containing a
normative condition that the deployment is known to reject. Both need normative
properties. Property names, available resources and enforcement mechanisms
remain release/deployment-owned. The diagnostic invents none of them.

Supply a harmless deterministic fixture command and its exact nonempty stdout
marker. It must work in the selected workload, terminate promptly, and tolerate
being attempted again after its Lease stops. It must not train a model or launch
unowned background work. Core does not assume a shell, interpreter, filesystem
mount layout or platform command. Deployment-specific fixture preparation and
independent observation remain in the trusted launcher.

Supply a bounded `teardown()` callback that stops and inspects all owned
workloads, disposes of private runtime storage and returns **exactly `True`**
only after absence is confirmed. Other return values and exceptions fail the
cleanup check. Preserve the diagnostic workspace/output and report separately
according to the approved evidence-retention policy.

## Invocation and checks

After preparing and authorizing those inputs, the deployment can call:

```python
from autotrain_bench.conformance import probe_worker_leases

report = probe_worker_leases(
    research_worker,
    task=public_fixture_task,
    workspace=fixture_workspace,
    output_root=diagnostic_output_root,
    resources=resources,
    resource_envelope=supported_envelope,
    rejected_resource_envelope=unsupported_envelope,
    command=fixture_command,
    expected_stdout=fixture_stdout,
    teardown=cleanup_and_confirm,
    duration_seconds=30,
    timeout_seconds=60,
    cleanup_timeout_seconds=30,
)
```

There are at most **three start attempts**: two accepted Leases, then one
deliberately rejected admission. This permits checking receipt-ID reuse across
otherwise valid allocations. The probe checks:

- Current protocol/research role, requested workload binding and stable
  effective workload identity across cases.
- Successful admission receipts bound to the episode, Lease, Worker and exact
  resource request; effective properties and enforcement satisfying the envelope.
- `RUNNING` state, valid command results, exact expected stdout, no timeout and
  no truncated command streams.
- Confirmed `STOPPED` state after termination and after repeated termination.
- A post-stop command that fails, followed by another `STOPPED` observation.
  A normal `WorkerFailure` or a nonzero command result with no stdout, timeout
  or truncated streams is accepted as the diagnostic rejection signal.
- A rejected admission that raises `WorkerFailure` and is already `STOPPED`
  **before** compensation. `WorkerTeardownFailure` fails the case; cleanup cannot
  hide a workload that an ordinary failure incorrectly left running.

An unexpected failure stops later cases and marks unfinished cases skipped.
Each attempted start is registered for compensation before the Worker call,
including creation that later raises or returns too late. Unconfirmed Leases
receive a separate cleanup attempt. Deployment teardown always follows a
started diagnostic, even if compensation fails or execution is interrupted.
Interrupts propagate after cleanup; durable interruption/recovery evidence
remains the launcher's responsibility.

## Clocks and limits

`duration_seconds` is the requested Worker Lease grant. It is not the startup
timeout or the whole diagnostic budget. Choose it to cover the fixture command
and checks after preparation. Each case shares `timeout_seconds` across
resolution, admission, commands, state observations and normal termination.
Resolution and command calls receive only the remaining allowance. Late success
and late ordinary failure both fail the case; a late teardown failure retains
its unconfirmed-boundary classification.

Fault compensation uses one shared `cleanup_timeout_seconds` allowance, separate
from the failed operation. The example permits three 30-second Lease requests,
up to three 60-second cooperative case allowances and 30 seconds of compensation,
plus separately bounded preparation, descriptor, reporting and deployment
cleanup. A rejected request may still create work in a faulty implementation;
include that possibility in the resource authorization.

Run the entire diagnostic behind the deployment's independent watchdog. The
synchronous v16 `start_lease`, `lease_state` and `terminate_lease` methods have no
timeout argument. Checking elapsed time can reject a late reply but cannot
preempt a wedged method or callback. No thread left behind by a local timeout is
treated as terminated. The separate [idle deadline diagnostic](worker-deadline-probe.md)
checks expiry through an independent deployment observer. Command cancellation,
launcher loss and recovery still require separate deployment tests.

## Evidence and remaining coverage

`public_manifest()` reports stable check codes and always sets
`formal_conformance: false`. `private_manifest()` additionally retains attempted
Lease requests, resolutions, returned resource receipts, assessments, command
diagnostics and cleanup observations. Command streams in that report are capped
at 4096 characters with explicit truncation flags. A compensated failure remains
`passed: false`, even when its later `termination_confirmed` becomes true.

Record exact Core, Worker source/dependency, runtime, fixture and resource
identities alongside the private report. Temporary Workers created during
preparation use the same interface; Core requires no site adapter.

A passing report verifies these protocol observations. It does not independently
prove physical resource enforcement, absence of a workload hidden by the Worker,
the cause of resource rejection, or that a failed post-stop command executed no
instructions. It also does not establish cross-UID access, hidden-data isolation,
Candidate immutability, independent deadlines, crash recovery or certification.
Use adversarial fixtures and independent deployment observations for those
guarantees, following the [acceptance matrix](worker-acceptance-matrix.md).
