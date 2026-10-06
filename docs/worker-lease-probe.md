# Research Worker Lease diagnostic

`envcore.conformance.probe_worker_leases()` exercises the supplied
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
from envcore.conformance import probe_worker_leases

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

## Optional command timeout case

The default probe checks a successful command and post-stop rejection. Supply a
`CommandTimeoutCase` to additionally check one deliberately long-running public
fixture command on the **first accepted Lease**, after its normal command and
before any termination call. The timeout command must differ from the normal
fixture command, keep owned work running beyond its requested timeout, and never
launch work outside the disposable scope. Include a descendant that outlives its
parent where the deployment needs to validate command-family termination.

```python
from envcore.conformance import CommandTimeoutCase

timeout_case = CommandTimeoutCase(
    command=public_timeout_fixture_command,
    observe=independent_command_observer,
    timeout_seconds=2,
    termination_grace_seconds=1,
    observation_timeout_seconds=1,
)
```

Pass `command_timeout_case=timeout_case` to `probe_worker_leases()` above. This
adds one command and two observations, with no additional Lease starts. Include
that command, its resources and possible failed cleanup in the authorized fixture
scope. The resulting probe version is
`worker-v16-lease-command-timeout-probe-v1`; omitting the case retains
`worker-v16-lease-probe-v1`. Both remain non-formal diagnostics.

The independent, read-only observer receives `episode_id`, `lease_id`,
`worker_id`, the exact `argv`, `command_sha256`, a fresh `observation_id`, and
`timeout_seconds`. It returns a typed `CommandTimeoutObservation` with the same
five identity fields (all except `argv` and the allowance), plus:

| Field | Meaning |
| --- | --- |
| `source` | Stable independent observation boundary |
| `running` | Whether any workload belonging to this exact command invocation is still running; `False` confirms all command-owned descendants stopped |
| `start_evidence_id` | Reference to independently retained evidence that this command actually started, or `None` before dispatch |
| `timeout_evidence_id` | Reference to the enforcement event for this command's requested timeout, or `None` before that event |

The observer must inspect actual owned workloads and retained execution/enforcement
events, without calling Worker status, executing another command, or initiating
termination/reconciliation. It must raise if workload state cannot be determined.
An exited launcher/client or command parent is insufficient when descendants
survive. Event evidence must bind this episode, Lease, exact argument vector and
timeout grant; Lease expiry, unrelated exit or Operator cleanup cannot substitute
for command timeout enforcement. Worker return flags alone cannot supply these
observations. Retain the underlying evidence in the trusted deployment.

Worker v16 supplies no command operation ID. This fixture uses a fresh episode
and Lease, a distinct timeout argument vector, and exactly one such invocation to
establish the observed scope. `command_sha256` hashes the UTF-8 JSON argument array
with ASCII escaping and separators `(',', ':')`; the observer receives it directly.
The first observation must confirm no running command, prior start or timeout
event. Fresh request IDs, all scope bindings and stable source identity are
checked again on the second observation; replayed or misbound responses fail.

The command must return a typed `CommandResult` with `timed_out=True` and a
nonzero exit code. An ordinary failure, successful exit or premature timeout
claim cannot pass this controlled case. Immediately after that return, **before
another Worker method or cleanup**, the probe takes one fresh observation. It
requires command-start evidence, command-timeout evidence and no surviving work.
This establishes absence at that observation, not an exact remote stop timestamp.
A running or unknown result fails; polling or subsequent cleanup cannot repair it.

Command execution, return and the post-return observation share the command grant
plus `termination_grace_seconds`. Each observation is additionally capped by
`observation_timeout_seconds`. The entire command window must fit before both
the case deadline and admission-call start plus the Lease grant. This conservative
Lease bound avoids crediting Lease expiry as command timeout enforcement; it is
not an atomic activation receipt. Slow startup, a slow normal command or initial
observation can leave insufficient time, in which case dispatch is rejected
without silently shortening the requested timeout or extending any budget.

Private `leases[0].command_timeout` records the request, attempted/passed flags,
command timestamps, bounded returned streams, and observations. Late replies are
retained when available but cannot pass. Command timeout failure stops later cases
and preserves normal compensation/external teardown. Conversely, later cleanup
failure cannot erase an accepted timeout observation, but still fails the overall
diagnostic. Synchronous wedges require the existing independent outer watchdog;
this is not an asynchronous cancellation or crash-recovery implementation.

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
checks expiry through an independent deployment observer, optionally while the
Worker is paused or terminated by a deployment-owned injector. The optional
command timeout case covers returned timeouts and independent descendant checks;
wedged calls, external cancellation, launcher loss and recovery still require
separate deployment tests.

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
