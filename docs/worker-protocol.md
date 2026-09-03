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
names are `autotrain-worker-docker`, `autotrain-worker-apptainer`, and
`autotrain-worker-kubernetes`; these names are examples, not core backends.
Core must never grow platform selection branches or import their SDKs.

## Policy and enforcement

The Coordinator decides:

- the selected track and budget;
- whether another lease is allowed;
- which Candidate is the declared incumbent;
- when research is irreversibly sealed;
- when postmortem ANALYSIS and TEST are allowed;
- which frozen Candidate inventory is selected for ANALYSIS;
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

Evaluation additionally requires `hidden_dataset_identity_verification`. This
means the Worker verifies the effective hidden bytes before Candidate startup
and binds the subsequent evaluation to that resolution. It does not require
Core to read those bytes or prescribe where they came from.

Three independent checks govern formal eligibility:

```text
required capabilities satisfied
AND conformance receipt passed
AND deployment trust is certified
```

Passing conformance does not certify a user-controlled deployment.  Likewise,
an organizer-controlled deployment is not eligible when required technical
capabilities are missing.  Missing or insufficient capabilities are recorded
explicitly and never trigger a fallback. A receipt is conformant only for the
suite version associated with the active Worker Protocol; a passing v4 receipt
cannot certify a v5 Worker boundary.

## Research and evaluation roles

Worker Protocol v5 models ResearchWorker and EvaluationWorker as separate
interfaces because their trust domains differ:

- ResearchWorker sees TRAIN, the Agent workspace, and training outputs, but
  never hidden ANALYSIS/TEST or their resolving credentials;
- EvaluationWorker sees an immutable Candidate and one selected hidden dataset,
  but does not need the research workspace and may be invoked only after Core's
  research-sealing preconditions hold.

The Python protocol enforces that separation in its argument types.
`ResearchWorker.start_lease` receives a `ResearchTaskSpec` containing only the
task identity, training image, TRAIN path, and generated public helper API.
`EvaluationWorker` receives an `EvaluationTaskSpec` with no dataset paths. Once
the Supervisor has confirmed research sealing and teardown, it sends a
`HiddenDatasetRequest` containing only the logical split name and an optional
release-frozen expected SHA-256 digest. The Worker privately resolves that
request and returns a `ResolvedHiddenDataset` containing the effective digest
and an opaque resolution ID. Core validates the receipt before passing the
opaque resolution back for Candidate evaluation. A complete `TaskConfig` never
crosses either role boundary.

Version 5 is a breaking interface revision in the current unreleased
development snapshot; it does not imply a package release version. Older
Workers are rejected rather than adapted implicitly. External Worker packages
must implement path-free hidden-data resolution, accept TEST before ANALYSIS,
bind evaluation to the resolved identity, and preserve the precise
prefix/target/artifact visibility fields in `CandidateChannelRequirements`
without assuming a particular transport.

A local development Worker may implement both roles in one process. A future
formal deployment may place them on separate services connected only through
an immutable artifact reference. The current version 5 Python interface does
not yet define that remote artifact transport.

Worker operation failures and teardown failures have different episode
semantics. `WorkerFailure` reports an operation that did not produce a usable
result and may be returned only after the Worker has confirmed that the failed
operation left no unaccounted workload boundary. `WorkerTeardownFailure`
reports that termination or removal remains unconfirmed. The latter is always
deployment-integrity evidence: further evaluation must not continue and the
Coordinator records a stable teardown code while retaining deployment-specific
details only in its trusted event stream. Before TEST it terminates the episode;
during optional post-TEST ANALYSIS it retains the completed TEST measurement but
marks the run non-compliant and leaderboard-ineligible. In particular,
`start_lease()` may fail after the
platform has created a workload but before the caller receives an
acknowledgement. A Worker must retain enough private state for
`terminate_lease(episode_id, lease_id)` to be safely retried in that case. The
Coordinator persists only that semantic Lease ID; container, scheduler, or
other platform handles remain Worker-owned.

`resolve_hidden_dataset()` has the same trusted failure discipline. A
`HiddenDatasetResolutionFailure` means resolution failed and the Worker has no
unconfirmed materialization boundary. An unexpected exception means Worker
state is unknown and is treated as deployment-integrity evidence. Resolution
must verify the effective bytes against the request before untrusted Candidate
code starts. A successful resolution may place bytes in trusted Worker-private
storage, but it must not leave an unaccounted Candidate/Evaluator workload;
per-evaluation mounts and untrusted workloads remain owned by
`evaluate_candidate()` and its teardown guarantee. A digest mismatch returns no
metric. Paths, source URLs, credentials, mount handles, and cache identifiers
remain Worker-private; only the opaque resolution ID and content digest cross
the protocol.

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
replace the workload environment. Version 4 does not yet define the normalized
resolution receipt needed to make such a translated workload formally
eligible.

## Hidden-evaluation lifecycle

Candidate commit is a research operation, but hidden scoring is not. Commit
stops the Lease and freezes the artifact; it returns no organizer-owned quality
metric. The Agent independently decides whether and how to validate using
release-permitted information.

The Coordinator durably enters `closed`/research-sealed state before hidden
evaluation. It then requires confirmed Agent termination, no active or pending
Research Lease, successful Agent-control cleanup, and no unresolved deployment
teardown failure. Only then may it invoke the Evaluation Worker. ANALYSIS scores
the release-selected frozen Candidate history and never changes the incumbent;
TEST scores only the final Agent-declared incumbent. TEST runs first and its
decision is persisted before diagnostic ANALYSIS begins. ANALYSIS completeness
does not gate the TEST outcome or leaderboard eligibility; separate deployment
integrity evidence still fails closed if an evaluation boundary cannot be
confirmed. A failed hidden-data resolution cannot reopen research.

Hidden-data resolution is path-free at the protocol boundary. A deployment may
use an already isolated organizer store, keep the bytes physically absent until
the call, mount storage only into the Evaluation Worker, or decrypt with
credentials unavailable to the research trust domain. The bundled local Worker
receives its private name-to-path mapping from the CLI composition root; those
paths are not fields of `HiddenDatasetRequest` or `ResolvedHiddenDataset`. Core
defines the ordering and frozen identity; the Worker defines and attests the
materialization mechanism.

## Candidate channel

`CandidateChannelRequirements` fixes the release-owned application protocol and
security properties. The Candidate receives only the authorized token prefix
needed by `next_token_logits_v1`; it never receives the hidden dataset artifact,
the current target token, future tokens, or egress. Only the allowed
request/response semantics cross the boundary, and teardown is confirmed before
a metric returns. The requirements deliberately contain no URL, port, socket
path, network name, or scheduler handle.

Task evaluator modules export `evaluate(data_path, vocab_size, predict,
batch_size=...)`. The `predict` callable is the deployment seam. The bundled
Docker Worker injects an HTTP/TCP adapter and private network; an external
Worker may inject a Unix-socket, proxy, shared-memory, scheduler-native, or VM
channel adapter. None of those mechanisms changes the Core protocol or the
release-owned Candidate ABI.

## Current migration stage

The phase-one interface is an in-process Python protocol. The Supervisor uses
semantic lease operations (`start_lease`, `exec_lease`, `lease_state`, and
`terminate_lease`), Worker-owned hidden resolution (`resolve_hidden_dataset`),
and semantic evaluation (`evaluate_candidate`). It neither stores container
identifiers nor asks how those effects are implemented.

Its `Path` arguments are local deployment handles. They assume that the Worker
implementation can access or materialize the Coordinator-named TRAIN,
workspace, Lease output, Candidate, public API, and evaluation locations. They
must not be treated as a stable serializable remote protocol. Hidden datasets
are the exception: their semantic request and resolution contain no `Path`.
A shared-filesystem Worker can implement version 5 directly; a topology without
shared Candidate or workspace paths still needs future artifact-reference and
materialization semantics.

Version 5 also passes the requested `TrainingResources` to Worker operations
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
3. asynchronous operations and idempotent operation IDs so slow Worker calls do
   not block deadline enforcement;
4. remote authenticated Worker transports where a real topology requires one;
5. deployment certification;
6. normalized resource-allocation evidence distinguishing requested,
   scheduler-allocated, workload-visible, and enforced resources;
7. workload-resolution evidence binding a release-owned OCI reference to a
   verified derived SIF, VM image, or other immutable artifact.

Those features should be added to the Worker protocol and external Worker SDK,
not by adding platform code to AutoTrain-Bench Core.

The staged plan and the trigger for each deferred abstraction are recorded in
[Thin Core development plan](thin-core-development-plan.md).
