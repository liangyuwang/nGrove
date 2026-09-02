# Worker Protocol boundary

AutoTrain-Bench Core owns benchmark decisions.  A Worker deployment owns
execution and enforcement.  The boundary between them is expressed in
benchmark terms and must not expose a scheduler, container runtime, or cloud
API.

```text
AutoTrain-Bench Coordinator
  authoritative benchmark state
              │
              │ Worker Protocol
              ▼
Research Worker      Evaluation Worker
  enforcement          hidden evaluation
       │                       │
       └──── immutable artifact┘
```

This is distinct from the Agent-facing control API. In the bundled reference
deployment, the Tested Agent sends semantic requests such as `lease_start` and
`lease_commit` through a local file mailbox. The mailbox is one Agent-control
transport; it is not the Coordinator-to-Worker Protocol and is not a required
Worker implementation mechanism.

## Repository boundary

The core repository owns:

- Track A/B state and scoring rules;
- Agent and Worker protocol schemas;
- semantic capability and trust requirements;
- canonical Candidate validation and artifact identity;
- conformance specifications;
- leaderboard eligibility decisions.

Platform packages own:

- resource allocation and placement;
- image preparation;
- process, container, Pod, or job lifecycle;
- GPU, filesystem, and network enforcement mechanisms;
- platform logs and environment receipts;
- site-specific security hardening.

Platform packages are distributed separately.  Examples of possible package
names are `autotrain-worker-docker`, `autotrain-worker-orix`, and
`autotrain-worker-kubernetes`; these names are examples, not core backends.
Core must never grow platform selection branches or import their SDKs.

## Policy and enforcement

The Coordinator decides:

- the selected track and budget;
- whether another lease is allowed;
- whether a VAL result is valid;
- which Candidate is the declared incumbent;
- when TEST is requested;
- whether the result is leaderboard eligible.

The Worker enforces:

- local lease and episode deadlines;
- the maximum resource envelope;
- TRAIN/workspace/output visibility;
- hidden-dataset isolation;
- network restrictions;
- termination and the guarantee that a stopped producer can no longer mutate
  selected output.

A Worker may retain a small local enforcement envelope so a Coordinator outage
cannot extend a deadline.  It must not duplicate incumbent selection, scoring,
or other benchmark policy.

In v0, the Coordinator materializes and hashes the immutable Candidate snapshot
from a shared path after the Worker has stopped the Lease. Worker-side capture
and transfer are future protocol capabilities for deployments without shared
paths; Core still owns Candidate acceptance and canonical identity.

## Capabilities, conformance, and trust

`autotrain_bench.worker_protocol` describes capabilities by semantic name and
an enforcement level:

```text
UNSUPPORTED < UNVERIFIED < BEST_EFFORT < ENFORCED
```

Mechanisms may be included as evidence, but requirements never name a platform
mechanism.  For example, Core requires `training_network_isolation`; it does
not require a particular network command or policy object.

Three independent checks govern formal eligibility:

```text
required capabilities satisfied
AND conformance receipt passed
AND deployment trust is certified
```

Passing conformance does not certify a user-controlled deployment.  Likewise,
an organizer-controlled deployment is not eligible when required technical
capabilities are missing.  Missing or insufficient capabilities are recorded
explicitly and never trigger a fallback.

## Research and evaluation roles

Worker Protocol v2 models ResearchWorker and EvaluationWorker as separate
interfaces because their trust domains differ:

- ResearchWorker sees TRAIN, the Agent workspace, and training outputs, but
  never hidden VAL/TEST;
- EvaluationWorker sees an immutable Candidate and the selected hidden split,
  but does not need the research workspace.

The Python protocol enforces that separation in its argument types.
`ResearchWorker.start_lease` receives a `ResearchTaskSpec` containing only the
task identity, training image, TRAIN path, and generated public helper API.
`EvaluationWorker` receives an
`EvaluationTaskSpec` with no dataset paths plus exactly one `HiddenSplitRef`
selected by the Supervisor. A complete `TaskConfig` never crosses either role
boundary.

Version 2 is a breaking interface revision in the current unreleased
development snapshot; it does not imply a package release version. Version 1
Workers are rejected rather than adapted implicitly. External Worker packages
must update `start_lease` to accept `ResearchTaskSpec`, update
`evaluate_candidate` to accept `EvaluationTaskSpec` plus `HiddenSplitRef`, and
return the new stdout/stderr truncation flags in `CommandResult` when needed.

A local development Worker may implement both roles in one process. A future
formal deployment may place them on separate services connected only through
an immutable artifact reference. The current version 2 Python interface does
not yet define that remote artifact transport.

Worker operation failures and teardown failures have different episode
semantics. `WorkerFailure` reports an operation that did not produce a usable
result and may be returned only after the Worker has confirmed that the failed
operation left no unaccounted workload boundary. `WorkerTeardownFailure`
reports that termination or removal remains unconfirmed. The latter is always
an infrastructure failure: evaluation must not continue and the Coordinator
records a stable teardown code while retaining deployment-specific details only
in its trusted event stream. In particular, `start_lease()` may fail after the
platform has created a workload but before the caller receives an
acknowledgement. A Worker must retain enough private state for
`terminate_lease(episode_id, lease_id)` to be safely retried in that case. The
Coordinator persists only that semantic Lease ID; container, scheduler, or
other platform handles remain Worker-owned.

The same ownership rule applies to evaluation: `evaluate_candidate()` must not
return a metric until its isolated Candidate/Evaluator boundary has been torn
down. If teardown cannot be confirmed, it raises `WorkerTeardownFailure` and
retains whatever private handles its deployment needs for operator-level retry;
those handles are not added to the Core protocol.

## Workload identity in v0

The current task fields `training_image`, `candidate_image`,
`evaluator_image`, and `inference_gateway_image` are release-selected OCI-style
workload references. They are not Docker container IDs. Leaderboard-eligible
releases require digest-pinned values, so their semantic requirement is an
immutable, release-owned workload identity rather than use of the Docker
Engine. Development tasks may still use mutable references and remain
ineligible.

A non-Docker Worker may consume the OCI artifact directly. If it executes a
derived SIF, VM image, or platform artifact, the mapping must be reproducible,
identified, recorded, and accepted by the organizer. A Worker may not silently
replace the workload environment. Version 2 does not yet define the normalized
resolution receipt needed to make such a translated workload formally
eligible.

## Current migration stage

The phase-one interface is an in-process Python protocol.  The Supervisor uses
semantic lease operations (`start_lease`, `exec_lease`, `lease_state`, and
`terminate_lease`) and semantic evaluation (`evaluate_candidate`).  It neither
stores container identifiers nor asks how those effects are implemented.

Its `Path` arguments are local deployment handles. They assume that the Worker
implementation can access or materialize the Coordinator-named TRAIN,
workspace, Lease output, Candidate, hidden split, public API, and evaluation
locations. They must not be treated as a stable serializable remote protocol.
A shared-filesystem Worker can implement version 2 directly; a topology without
shared paths needs future artifact-reference and materialization semantics.

Version 2 also passes the requested `TrainingResources` to Worker operations
but does not define a normalized effective-allocation receipt;
`start_lease()` returns `None`. Environment inventory is useful diagnostic
metadata, not proof that every release-normative resource property was granted
and enforced. A future receipt is described in Phase C4 of the development
plan.

`CommandResult` streams are capped at 1 MiB each by the local Worker and again
by the Supervisor before they enter the mailbox or `commands.jsonl`. Concrete
Workers must drain excess output while retaining only the bounded prefix and
must report the truncation flags.

The pre-existing Docker runtime remains temporarily as the bundled local
reference implementation so the runnable preview does not regress.  It is a
compatibility implementation, not the extension point for more platforms.  A
later extraction moves it to a separately distributed Worker package without
changing Supervisor semantics.  Its deadline guard currently shares the
Coordinator process, so it reports that capability as `BEST_EFFORT`; this is an
explicit development limitation rather than a silent fallback.

Deliberately deferred work includes:

1. canonical remote ArtifactRef transport;
2. an atomic `activate_episode` operation with Worker-local monotonic timing;
3. asynchronous operations, idempotent request IDs, and event streams;
4. Unix-socket and remote mTLS transports;
5. deployment certification;
6. normalized resource-allocation evidence.

Those features should be added to the Worker protocol and external Worker SDK,
not by adding platform code to AutoTrain-Bench Core.

The staged plan and the trigger for each deferred abstraction are recorded in
[Thin Core development plan](thin-core-development-plan.md).
