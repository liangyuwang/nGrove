# Evaluation Worker diagnostic probe

External Worker testing distinguishes a successful preview from acceptance of its
Worker's failure paths. Worker v16 also requires evaluation resource receipts
that an older temporary deployment may not implement. The resolution probe
cannot detect a Worker that resolves inputs correctly but omits, misbinds or
reuses evaluation receipts.

`benchcore.conformance.probe_worker_evaluation()` provides a reusable,
platform-independent diagnostic for that gap. It calls the actual supplied
Evaluation Worker interface. It does not create an Agent, launch a benchmark
episode, select an incumbent, publish a benchmark result or issue a
`ConformanceReceipt`. Its report always carries `formal_conformance: false`.

This probe supplies part of R08-E in the
[acceptance evidence matrix](worker-acceptance-matrix.md). Supervisor failure
attribution/publication, actual allocation inspection, independent deadline
enforcement and crash recovery have separate acceptance criteria there.

## Prepare a disposable deployment

Use the authorized integration-test working root and resource allowance. Supply:

- A dedicated Evaluation Worker configured with **public synthetic fixtures**
  for both TEST and ANALYSIS, and the fixtures' independently computed
  `hidden-bundle-v1` SHA-256 identities. Do not use organizer hidden data or a
  Worker attached to a live episode. No Agent or Research workloads may be active
  in this diagnostic deployment.
- An `EvaluationTaskSpec` for those fixtures, prepared Candidate/evaluator
  images, and a known-good immutable fixture Candidate in its own directory.
  This is operator test data, never a reference Agent or a participant workspace.
- A `TrainingResources` launch selection, one supported `ResourceEnvelope`,
  and a different envelope with a normative condition the deployment is known
  to reject. Both envelopes need normative properties. Core invents no property
  names, hardware requirements or rejection conditions for the deployment.
- An existing trusted `output_root` outside the fixture Candidate. The probe
  creates a unique child directory and distinct per-call output paths. Retain
  diagnostic evidence separately from private runtime storage.
- A bounded `teardown()` callback that stops and inspects owned workloads,
  removes private materializations, and returns **exactly `True`** only after
  absence is confirmed. `None`, other truthy values, `False`, and exceptions
  cannot establish cleanup. The older resolution probe's callback convention
  remains unchanged.

Record the Core revision, Worker source/dependency identity, runtime identity,
fixture Candidate identity, fixture task manifest and requested resources beside
the report. In particular, preserve temporary Worker source created during
preparation so that these diagnostics can be repeated against that deployment.

## Invoke through the deployment launcher

After the deployment prepares the inputs above, its trusted launcher can call:

```python
from benchcore.conformance import probe_worker_evaluation

report = probe_worker_evaluation(
    evaluation_worker,
    task=fixture_task,
    candidate_dir=fixture_candidate,
    output_root=probe_output_root,
    resources=resources,
    hidden_dataset_identities=fixture_identities,
    resource_envelope=supported_envelope,
    rejected_resource_envelope=unsupported_envelope,
    teardown=cleanup_and_confirm,
    timeout_seconds=30.0,
)
```

There are at most four evaluation calls, in order: supported TEST, supported
ANALYSIS, rejected TEST, rejected ANALYSIS. Each receives a new evaluation ID
inside one unique probe scope. Workload resolution, hidden-data resolution and
evaluation share the same per-call monotonic deadline; they never receive fresh
full timeouts after earlier work. The example allows at most 120 seconds of
cooperative evaluation operations, plus separately bounded setup, descriptor,
reporting and cleanup. Include all these costs in the diagnostic authorization.

The probe checks requested workload and fixture identity bindings, stable
effective workload identities, valid performance/cost records, resource receipt
episode/evaluation/Worker/request bindings, effective requirements, enforcement
levels, and receipt-ID uniqueness. A supported case must return satisfactory
evidence. A rejected case must raise `EvaluationResourceFailure` from evaluation;
ordinary `WorkerFailure`, invalid returns, successful evaluations and
`WorkerTeardownFailure` fail that case. Late success and late resource rejection
also fail. A teardown failure retains its distinct unconfirmed-boundary code
even when reported late.

Any unexpected failure stops subsequent cases and marks the unfinished cases
skipped. Deployment teardown runs after a started probe whether it succeeds,
fails or is interrupted. Invalid caller inputs fail before Worker calls or output
creation. Interrupts propagate after cleanup; the deployment remains responsible
for durable interruption records.

Run the whole probe behind the deployment's independent watchdog. The in-process
interface cannot forcibly preempt a Worker or cleanup callback that never
returns. SIGKILL, host loss and allocation loss still need the deployment's
out-of-process ownership inspection and recovery path.

## Read and retain the report

`report.public_manifest()` contains the probe version, scope, Worker ID, aggregate
status, and stable check/failure codes. It omits measurements, fixture hashes,
resource details, workload identities and exception text. It is an Operator
summary, not an Agent-visible research feedback channel.

`report.private_manifest()` also contains trusted exception details and each
attempted invocation's request, resolved identities, measurements and resource
evidence when available. Each invocation records its own `passed` flag; a
measurement retained from a failed receipt check is not an accepted result.
Store this report and the output directory in trusted diagnostic storage.

A passing probe means these calls and the explicit deployment cleanup check
passed in that run. It does not independently observe physical CPU/GPU/storage
enforcement, prove that rejected requests never launched a workload, certify
hidden-data isolation, or test Worker kill/restart recovery. Resource rejection
classification alone does not prove the underlying rejection cause. Use separate
deployment observations and adversarial tests for those guarantees. Local mock
tests verify the probe's decisions; they do not replace running it against an
authorized real Worker. No capability, trust or leaderboard flag is upgraded.
