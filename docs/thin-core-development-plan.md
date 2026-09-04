# Thin Core development plan

## Purpose

AutoTrain-Bench Core defines what an episode is allowed to do. It does not
define where workloads run, how an Operator reaches a host, or which platform
mechanism enforces the contract.

The reusable idea is **sandbox-style workload semantics**: fixed identity,
bounded visibility and resources, revocable execution, immutable artifacts,
and auditable receipts. Docker is one bundled mechanism for those semantics,
not the idea that Core standardizes.

This document records the incremental path from the bundled v0 local Docker
prototype to independently distributed Worker deployments. The migration is
deliberately driven by a real external Worker instead of an up-front framework
for every scheduler, runtime, and network topology.

The target dependency direction is:

```text
Trusted Operator or site launcher
              |
              | selects and composes trusted components
              v
Coordinator / semantic Supervisor
  |                         |
  | Agent API semantics     | Worker Protocol semantics
  v                         v
Agent control transport     Research / Evaluation Workers
  |                         |
  | local implementation    | deployment implementation
  v                         v
File mailbox                Docker, Apptainer, VM, scheduler, or site runtime
```

The Agent control transport and the Coordinator-to-Worker Protocol are
different boundaries. A deployment may use a shared file mailbox for one and a
remote Worker service for the other, or replace either independently.

## Invariants throughout the migration

- The Coordinator remains authoritative for Track A/B budgets, Lease
  admission, Candidate commit timing, incumbent selection, irreversible
  research sealing, postmortem ANALYSIS/TEST lifecycle, results, and
  leaderboard eligibility. It provides no organizer feedback during research.
- A Challenge Release defines every resource property that is normative for
  comparison or eligibility. Core validates the requested requirements and
  the semantic evidence selected by that release. A Worker deployment chooses
  the platform mechanism, enforces the allocation, and reports effective
  evidence.
- Core decides when a Candidate must become immutable and defines its
  acceptance and canonical identity. A Worker guarantees that the producing
  workload can no longer mutate the selected output; who materializes the
  immutable snapshot depends on the protocol stage described below.
- The Research Worker never receives hidden ANALYSIS or TEST data or the
  credentials needed to resolve them.
- The Evaluation Worker receives one selected hidden dataset and an immutable
  Candidate, but not the research workspace, and may act only after research
  teardown is confirmed.
- The Tested Agent never receives Supervisor state, Worker privilege, hidden
  data, evaluator diagnostics, or an unrestricted platform control channel.
- No security-sensitive local transport extraction may replace the existing
  symlink, file-type, size, atomic-publication, quarantine, or cleanup checks
  with weaker filesystem operations.
- Core does not gain backend selection branches or imports of scheduler,
  container-runtime, SSH, cloud, or site SDKs.
- A Worker must not silently replace a release-owned workload environment.

Resource properties are not a universal fixed checklist. A release classifies
each relevant property as normative, observational, or deliberately
unspecified. CPU, memory, storage, accelerator topology, and other properties
become Core requirements only when the release says that they affect its
budget, isolation, comparison, or eligibility. Device selection, scheduler
requests, cgroups, filesystem quotas, and other enforcement mechanisms remain
Worker-owned.

Worker Protocol v10 retains the v7 implementation of this contract for Training
Lease admission.
It retains `TrainingResources` only as a bundled v0 compatibility input while
passing the release-owned, property-extensible `ResourceEnvelope` and requiring
an operation-bound allocation receipt. Resource vocabulary and enforcement
mechanisms remain Worker/release-owned rather than becoming a Core checklist.

## Phase A: state the v0 boundary accurately

Status: complete.

The v0 support statement is:

- The bundled `autotrain-bench prepare` and `autotrain-bench start` commands
  are a single-host local Docker reference composition.
- `autotrain_bench.worker_protocol` version 10 is a synchronous in-process
  Python contract. Its remaining TRAIN/workspace/Candidate/output `Path` values
  are local deployment handles, not a serializable remote wire format. Hidden
  dataset requests and resolutions are path-free.
- The bundled Agent-facing control transport is a local file mailbox. Mailbox
  filesystem operations and their operating-system support are transport
  properties, not benchmark semantics.
- `training_image`, `candidate_image`, `evaluator_image`, and
  `inference_gateway_image` are release-selected OCI-style workload references
  in v0. They are not Docker container IDs; leaderboard-eligible releases
  require digest-pinned values.
- Core has no native remote artifact handoff, deployment discovery, scheduler
  integration, or non-Docker formal execution path yet.
- The current local Docker and `host_cli` paths are development/preview paths;
  no official leaderboard challenge has been released.
- The v0 Docker GPU preflight verifies that enough matching devices exist, but
  its count-based Docker grant does not yet prove that those exact devices were
  selected. GPU profile matching is therefore a known reference-Worker
  limitation, not a Core allocation mechanism.
- The local mailbox bounds the live entries and regular-file bytes it will
  process, but the bundled bind-mounted storage does not yet provide a hard
  filesystem quota. Pre-inspection disk containment is a known local Worker
  deployment limitation, not a reason for Core to traverse hostile trees or
  define site storage policy. The bundled launcher records this limitation and
  therefore does not claim policy compliance.
- After the v0 Worker stops a Lease, the Coordinator validates, copies, hashes,
  and atomically stores the Candidate from a shared path. Version 6 has no
  Worker-side capture operation or remote `ArtifactRef`.

Phase A originally changed documentation only. The later postmortem-evaluation
migration deliberately advanced the Agent, Worker, task, challenge, and result
schemas; it still did not claim support for a new deployment.

## Phase B: expose the existing seams

Status: complete for the in-repository v0 boundary.

Phase B is an internal boundary refactor. It must preserve the intended v0
Agent API, Worker Protocol version, result schema, episode semantics, and
Docker reference behavior. Characterization is not permission to preserve a
known security or failure-classification defect.

### B1. Characterize behavior and freeze intended invariants

Status: complete in the Core test suite.

Before moving code, preserve tests for:

- every Agent request and response;
- malformed mailbox files, symlinks, directories, FIFOs, output conflicts,
  size limits, Unicode normalization, and cleanup;
- episode terminal states and failure classification;
- Task reveal, Lease accounting, Candidate commit, research sealing,
  ANALYSIS/TEST, and incumbent;
- Worker capability, trust, conformance, and eligibility records;
- Docker reference mount, network, deadline, and evaluation boundaries.

Known mailbox gaps must be represented as desired acceptance tests before or
alongside extraction rather than frozen as compatibility behavior:

- bound accepted request/response entries and bytes independently from the
  Worker's hard filesystem quota;
- treat inbox, outbox, and rejected storage as one Agent-control surface;
- clean or destroy all three areas only after the Agent has stopped;
- never follow an Agent-created link during inspection or cleanup;
- record incomplete cleanup instead of ignoring it, and classify the failure
  according to whether it came from Agent input or trusted deployment teardown;
- require the Worker deployment to contain pre-scan disk exhaustion and
  cross-UID teardown with a quota-bound disposable storage mechanism.

### B2. Separate the local file mailbox driver

Status: complete. `LocalFileMailbox` implements the deployment transport seam;
the semantic Supervisor can be constructed and tested without it.

Move request discovery, safe filesystem reads, atomic response publication,
quarantine, and cleanup behind an explicitly local transport component. The
transport invokes the existing semantic request handler; it does not decide
Lease or evaluation policy.

Failure ownership must remain explicit:

- malformed Agent-controlled entries are rejected or become Agent policy
  failures;
- valid requests that violate benchmark state return a protocol error;
- failure of trusted transport or Coordinator persistence becomes an
  infrastructure failure.
- raw trusted exception details remain in private Coordinator events; Agent
  responses and public status expose only stable failure codes;
- deployment teardown failures remain explicitly recorded even when the
  episode already ended as DNF or infrastructure failure; a later successful
  Worker termination is reconciled through Coordinator state rather than only
  through a runtime-local handle;
- a failed `start_lease()` is treated as an uncertain creation until the Worker
  confirms termination; Core retains the semantic Lease ID for teardown retry,
  while the Worker retains every platform-specific resource handle;
- hidden evaluation cannot begin until Agent stop metadata, Research Worker
  teardown, and transport cleanup are complete. TEST output is then staged
  briefly while the episode is `closed`, and the Coordinator publishes exactly
  one terminal result. Cleanup is rejected until termination has been recorded.

The initial extraction should move proven code rather than redesign the wire
format.

### B3. Separate Agent API publication

Status: complete. The local launcher publishes an explicit read-only allowlist
before constructing the Supervisor; the Supervisor only receives the resulting
public directory handle.

Core continues to own the versioned Agent request/response schema and the
minimal client implementation. A local bootstrap component owns copying that
client into an episode and making it read-only. The Worker deployment owns how
the resulting bundle is mounted, installed, or otherwise exposed to a
workload.

### B4. Make the Docker path an explicit composition root

Status: complete for the in-repository v0 boundary. The bundled CLI remains the
explicit local Docker composition root, while architecture tests prevent the
Supervisor and Worker Protocol from importing it or the local mailbox driver.

The bundled local launcher may continue to instantiate `DockerRuntime` and
manage Agent, inference-gateway, network, and log lifecycles. Those operations
must not move into the semantic Supervisor or Worker Protocol.

External Worker packages use their own launchers to construct the Coordinator,
Workers, Agent execution, and transport. Core does not add a backend name map
or platform conditionals.

### B5. Separate Agent mechanism from policy evidence

Status: complete as a deliberately narrow v0 seam. `AgentConfig` describes the
participant selection and launch parameters. A trusted launcher separately
supplies `AgentExecutionEvidence`, including its evidence source, mechanism,
overall semantic enforcement level, deployment trust, immutable workload
identity, and explicit limitations. The Supervisor assesses those fields and
does not infer compliance merely from `container` or `host_cli`. In v0, a
container evidence identity must exactly match the selected Agent image.

The strings `container` and `host_cli` describe bundled v0 launch mechanisms;
they are not proof of isolation. The Coordinator must eventually base Agent
policy compliance on trusted semantic evidence supplied by the deployment.

Phase B should establish that seam without attempting to define a complete
cross-platform attestation format. Until sufficient trusted evidence exists,
the run remains ineligible rather than receiving a capability fallback.

This aggregate v0 evidence is not the final per-capability attestation format.
The bundled local launcher is development-trust only. A real external Worker
prototype must determine whether Phase C1 needs finer-grained evidence,
signatures, or conformance semantics.

### B6. Enforce lifecycle failure invariants

Status: complete for the bundled v0 composition.

Every effectful boundary follows one failure model:

| Boundary | Before the call | Success | Unknown or failed teardown |
| --- | --- | --- | --- |
| Research Lease start | Persist the semantic Lease ID as a cleanup intent | Publish the active Lease and clear the intent | Keep the intent and retry `terminate_lease` during deployment teardown |
| Local Agent, Gateway, or network start | Register the deterministic local name for cleanup | Keep it registered until confirmed removal | Attempt every independent cleanup and record a stable teardown code |
| Local isolated evaluation | Register Candidate, Evaluator, and network handles inside the Worker | Remove all three before returning the metric | Retain Worker-private handles and retry them from the local deployment composition |
| Candidate snapshot | Use an episode-private staging directory | Atomically install the content-addressed snapshot | Remove staging before propagating the infrastructure failure |
| Terminal result publication | Persist the primary TEST decision before diagnostic ANALYSIS; keep public `outcome: null` until ANALYSIS has a complete/incomplete disposition; append a non-terminal publication-prepared record | Atomically replace authoritative `state.json` with a reference to that prepared record | Retain TEST across diagnostic failures, record deployment integrity separately, leave any interrupted preparation orphaned, and keep audit sequence numbers monotonic |

Additional invariants enforced before every state write are:

- Agent-control cleanup cannot complete before trusted Agent termination;
- an active episode cannot record a terminated Agent;
- only `closed` may contain a staged TEST result;
- a finalized result has no active or pending Worker Lease and has completed
  Agent termination and control cleanup, and references its audit preparation
  record;
- a finalized result has an explicit complete or incomplete postmortem ANALYSIS
  disposition, and ANALYSIS never changes the Agent-declared incumbent or gates
  eligibility merely because diagnostic points are missing;
- failed outcomes never retain TEST metrics;
- platform resource handles never enter Core state.

`state.json`, not the append-only event stream, is the result commit point. A
publication-prepared event is evidence of an attempted commit, not evidence of
a terminal result by itself. Consumers accept a terminal result only when the
authoritative finalized state references that event. Event sequence numbers are
never decremented or reused, because an append cannot be rolled back after it
becomes visible. If the atomic state replacement fails, an in-process retry
reuses the same prepared record instead of appending a second terminal attempt.

These are lifecycle semantics, not Docker semantics. A different Worker may
use a scheduler allocation, VM, process sandbox, or another mechanism, but it
must retain its private handles until the semantic cleanup operation is
confirmed.

The bundled v0 Worker can retry evaluation teardown while its composition
process remains alive; those random local evaluation handles are not a durable
crash-recovery protocol. A production external Worker must persist or
reconstruct its own handles across the failure modes it claims to tolerate.
Core must not solve that deployment requirement by storing container, job, or
network identifiers.

### Phase B exit criteria

- The semantic request handler can be tested without creating a mailbox.
- The local file transport can be tested without Docker.
- Docker-specific process, network, mount, and log calls live only in the
  bundled reference composition/runtime.
- An external launcher can supply Research and Evaluation Worker
  implementations without modifying the Core CLI or adding a backend branch.
- Platform mechanisms remain outside Core even when a semantic protocol
  revision is required.

## External validation deployment

Status: completed outside this repository.

An independently distributed shared-filesystem Worker was exercised on a real
scheduler/runtime deployment without changing Core. It implemented the
semantic Lease lifecycle successfully. The site commands, configuration, and
resource handles remain outside this repository.

The external Operator obtains the allocation and stages the repository and
artifacts. The Worker consumes that authorized environment; Core does not call
`ssh`, `sbatch`, `srun`, or `apptainer`.

The prototype deliberately retained the then-current local/shared-filesystem
`Path` contract:

- TRAIN, workspace, Lease output, Candidate snapshots, hidden datasets, and
  evaluation output are materialized on the shared filesystem;
- the Research and Evaluation implementations map semantic Worker operations
  to the site runtime;
- OCI workloads are consumed directly or mapped to prebuilt SIF artifacts;
- every OCI-to-SIF mapping is recorded explicitly and the deployment remains
  leaderboard-ineligible unless that mapping is organizer-approved;
- Worker-local deadline and termination behavior are exercised rather than
  simulated by Core.

The observation log demonstrated four follow-up needs: workload-resolution
evidence, resource-allocation evidence, non-blocking Worker operations, and a
Worker-owned Candidate/Evaluator channel mechanism. It also supplied negative
evidence: shared paths were sufficient, so remote `ArtifactRef` work remains
deferred. No scheduler or runtime mechanism is promoted into Core.

That external run predates Worker Protocol v5 through v9. Its evidence still
validates the thin-Core boundary, but its old conformance result cannot certify
the path-free hidden-data, workload-resolution, or resource-receipt contracts.
The out-of-tree Worker must implement `resolve_hidden_dataset()` and
`resolve_workload()`, bind the returned resolutions to execution, return a v7
allocation receipt from `start_lease()`, and rerun the matching conformance
suite before a new deployment claim is made. The in-repository resolution probe
is useful for that migration but is intentionally too narrow to issue the
complete formal receipt.

## Postmortem hidden evaluation migration

Status: implemented in the development schemas.

- Agent Protocol v4 makes Candidate commit return artifact acceptance only.
- Task schema v2 and Challenge schema v2 replaced legacy VAL with hidden ANALYSIS and a
  release-frozen Candidate limit, selection rule, and result-visibility policy.
- Task schema v3 and Worker Protocol v10 replace progressive all-prefix scoring
  with balanced, non-overlapping contexts that each contribute one permanently
  hidden next-token target. The release freezes the context-length set, and the
  Candidate receives no target feedback.
- Research is durably sealed before Agent/Research teardown; hidden evaluation
  is rejected until teardown and Agent-control cleanup are confirmed.
- The final Agent-declared incumbent is scored on TEST first. The TEST decision
  is persisted before selected historical Candidates are attempted on ANALYSIS.
- Result schema v8 retains v7 workload-resolution evidence and adds the
  release resource envelope plus per-Lease allocation receipts. It also records the frozen Candidate
  inventory, ordered incumbent declarations, ANALYSIS trajectory, explicit
  analysis disposition, and trajectory-derived selected/evaluated point
  counts. Every trajectory point is
  bound back to its commit and declaration event evidence. Read-only migration
  preserves old VAL evidence as legacy evidence and never calls it ANALYSIS; v4
  postmortem results remain readable with an explicit limitation instead of an
  invented declaration history, while v5 counts are derived without inventing
  new measurement evidence.
- Ordinary ANALYSIS incompleteness never erases TEST or gates leaderboard
  eligibility. A post-TEST evaluation teardown failure is recorded independently
  as a deployment-integrity violation. TEST remains staged if a cleanup retry
  succeeds; teardown that remains unconfirmed blocks publication and becomes an
  infrastructure failure. Terminal publication happens only after all
  postmortem workload and private-materialization cleanup attempts finish.
- Worker Protocol v10 retains the transport-neutral Candidate application
  contract and a Worker-owned hidden-data resolution seam. Core supplies only a
  logical split name and release-frozen expected digest; the Worker returns an
  effective digest plus opaque resolution ID before Candidate execution. The
  task evaluator exports `evaluate(..., predict, context_lengths=...)`; the bundled Docker
  Worker supplies its HTTP adapter and a private local-path resolver, while
  external Workers may supply other isolated mechanisms.

## Phase C: add only demonstrated semantic capabilities

Status: split by independently observed triggers.

### C1. Agent execution evidence

Trigger: a policy-compliant non-Docker Agent sandbox cannot express its
identity, visibility, network, termination, and trust evidence through the v0
configuration.

Possible result: a semantic Agent execution descriptor or receipt assessed
independently from its runtime mechanism.

External result: not fired. The tested Agent could be sandboxed, but its
inference channel could not be mediated by the registered Gateway. A different
participant with a compliant inference path is needed before redesigning C1.

Result schema version 8 still pins `agent_system` to the bundled v0 runner
vocabulary, so this phase also owns that structure. `_validate_agent_system()`
in `autotrain_bench/results.py` rejects a `runner` outside `AGENT_RUNNERS` and
keys its "a host runner declares no egress policy" rule on the literal
`host_cli`; `AgentConfig.network` derives `host_managed` from the same literal.
Those checks are correct while Core owns a closed runner set, but a result
recording an unregistered runner would become unreadable rather than merely
unverified. This phase must re-express the `agent_system` invariants over
semantic fields instead of mechanism names, which changes the persisted
structure and therefore requires a result schema revision with a read-only
migration. Worker Protocol v10 resolves only Worker-owned training, Candidate,
and Evaluator workloads. The v0 Agent evidence binds the Agent image but not a
separate inference Gateway image, so formal runs using registered inference now
fail closed with `inference_gateway_workload_evidence_missing` until C1 defines
trusted evidence for that deployment boundary.

### C2. Remote immutable artifacts

Trigger: the Coordinator and a Worker cannot safely share a `Path`, or the
Coordinator cannot locally snapshot a stopped Lease output.

Possible result: an `ArtifactRef`, Worker-side atomic Candidate capture,
materialization verification, and content-identity receipt. Core continues to
own Candidate acceptance and canonical identity rules. The remote Worker owns
atomic capture and transport enforcement. This replaces, rather than
retroactively describes, the v0 Coordinator-side shared-path snapshot.

External result: not fired. The tested deployment safely shared every required
path and could use Coordinator-side Candidate capture.

### C3. Workload resolution evidence

Status: implemented in Worker Protocol v6 and result schema v7.

Trigger: a deployment executes a SIF, VM image, or proprietary workload derived
from the release-owned OCI reference.

Possible result: a receipt binding the requested canonical workload, effective
artifact identity, transformation provenance, and organizer acceptance.

External result: fired. Worker Protocol v6 now separates `WorkloadRequest` from
`ResolvedWorkload`, passes the opaque resolution into the operation that may
launch it, and records the requested identity, effective format and SHA-256,
optional transformation-provenance SHA-256, Worker identity, and acceptance
status. An unchanged digest-pinned OCI artifact is accepted by identity. A
derived SIF, VM image, or other artifact is rejected for formal use unless its
evidence exactly matches a `WorkloadResolutionAcceptance` supplied by the
trusted organizer `ReleaseVerificationReceipt`. No runtime, conversion, cache,
or scheduler mechanism was added to Core.

#### C3 resolution closure

Status: implemented in Worker Protocol v8, retained by v9, and the bundled
Worker. The asynchronous transport work below remains deliberately deferred.

1. **Hidden-data identity is bound to the bytes actually evaluated.** The
   bundled Worker copies its configured source into Worker-private read-only
   storage, hashes that exact snapshot, and mounts only the snapshot. Regression
   coverage mutates the source after resolution and proves that evaluation
   still consumes the receipted bytes. Snapshot cleanup is part of deployment
   teardown.
2. **Every primary hidden-resolution path is terminal.** If a Worker raises
   `HiddenDatasetIdentityMismatch` with a receipt that does not actually differ
   from the release or previously resolved identity, the Coordinator treats it
   as invalid Worker evidence and reaches an explicit primary failure terminal.
3. **Workload-resolution failure classes are preserved.** The bundled
   `docker image inspect` path translates a confirmed, side-effect-free failure
   into `WorkloadResolutionFailure`; only genuinely unknown resource state uses
   `WorkerTeardownFailure`.
4. **Bound resolution liveness.** Version 8 gives workload and hidden-data
   resolution explicit deadlines. TEST resolution and execution share one
   evaluator-operation deadline, so slow resolution cannot obtain a fresh full
   evaluation allowance. The bundled Worker checks filesystem-copy progress
   and bounds runtime inspection. Because the current interface remains a
   synchronous in-process call, Core still cannot forcibly preempt a wedged or
   nonconforming Worker; asynchronous operation IDs and cancellation remain a
   separately triggered future seam rather than a platform branch.

### C4. Resource allocation evidence

Status: implemented in Worker Protocol v7, retained by v9, and result schema v8
for Training Lease admission.

Trigger: a release makes one or more resource properties normative, but the
Worker cannot return authenticated evidence that the effective allocation and
enforcement matched those requested properties.

Possible result: a versioned allocation receipt containing the requested
normative requirements, effective values, enforcement level, and Worker
identity needed by Core's comparison and eligibility checks. The schema must
remain property-extensible; it must not turn one platform's CPU, accelerator,
topology, cgroup, or storage vocabulary into a universal Core checklist.

External result: fired. Requested resources, scheduler allocation,
workload-visible resources, and the enforcing layer differed. Version 7
distinguishes those views and allows a trusted allocator—not necessarily the
sandbox runtime—to supply evidence. `ResourceEnvelope` contains only
release-classified normative and observational property names. A versioned
`ResourceAllocationReceipt` binds its request digest, episode, Lease, and
Worker, records allocator-visible and workload-visible values separately, and
names the enforcement level and authority. Version 8 makes the effective
workload-visible value normative when available, retains allocator allocation
as separate evidence and fallback, and adds `at_most` for release-defined
resource ceilings. Core fails closed, with compensating termination, before a
mismatched allocation can become an active Lease.

The schema fixes no CPU, accelerator, topology, scheduler, cgroup, or storage
property list. The existing v0 NVIDIA flags generate a compatibility envelope.
During this schema transition, an explicit `resource_contract` is a full
envelope: it must restate the legacy profile requirements consistently and may
add other semantic property names without a platform branch in Core. It cannot
silently erase an H100 count/model/memory requirement while the same runtime
profile still declares one. Authenticity still depends on the trusted Worker
deployment and invocation channel. Remote cryptographic transport and
deployment certification remain separate future work.

The bundled Docker Worker reports count-based GPU selection as `best_effort`.
That honestly satisfies current preview envelopes but not an `enforced` formal
GPU envelope. Exact Docker GPU binding remains a reference-Worker issue rather
than part of the C4 Core schema. Evaluation-operation allocation evidence is
also deferred until a release demonstrates that resources outside Training
Leases are independently normative.

### C5. Trusted conformance evidence

Status: evidence boundary implemented in Worker Protocol v9 and retained by
v10; complete formal conformance suite remains future organizer work.

Trigger: the first in-repository resolution probe demonstrated that a partial
black-box check could otherwise be mistaken for complete Worker conformance,
and that a receipt returned in `WorkerDescriptor` could not prove who issued it
or which Worker was tested.

Result: Worker self-description no longer carries conformance. A trusted
Operator composition supplies a versioned `ConformanceReceipt` independently;
the receipt binds the assessed Worker ID, issuing authority, organizer-owned
report SHA-256, suite version, and decision. Core checks that binding separately
from capability claims and deployment trust. The diagnostic resolution probe
requires release-owned identities, a unique run scope, a mandatory deployment
teardown callback, and public/private diagnostic separation, but cannot issue a
formal receipt because it does not exercise the complete v9 contract.

The future formal suite belongs to organizer tooling and the external Worker
SDK. Platform-specific deployment setup, outer watchdogs, destructive cleanup,
and remote receipt authentication stay outside Core.

### Other independently triggered work

- atomic episode activation with Worker-local monotonic timing;
- asynchronous operations and cancellation (trigger fired because a wedged or
  nonconforming synchronous Worker can block Coordinator request and deadline
  processing). The future Core contract should cover durable operation IDs,
  absolute deadlines, status/reconciliation, idempotent retry, and cancellation
  outcomes. The Worker deployment remains responsible for the actual sandbox,
  scheduler-job, VM, or other termination mechanism; no platform handles or
  platform branches belong in Core;
- event streaming;
- a remote authenticated Worker transport;
- deployment certification.

None of these requires SSH, scheduler, cloud, or object-storage implementation
inside Core.

## Versioning rule

Phase A documentation and a behavior-preserving Phase B refactor did not change
the then-current protocol. The postmortem hidden-evaluation contract advanced
Worker Protocol to version 4; Worker-owned, path-free hidden-data resolution
advanced it to version 5; operation-bound workload resolution evidence advanced
it to version 6 and result schema version 7; operation-bound Training Lease
resource evidence advanced it to version 7 and result schema version 8;
required workload and hidden-data resolution deadlines advanced the Worker
Protocol to version 8 without changing the result schema; trusted,
Worker-bound conformance evidence advanced it to version 9; the single-target,
non-overlapping Candidate evaluation contract advanced it to version 10 and
task schema version 3.

A new protocol revision is required when an implementation must change a
required method, argument, return value, ordering guarantee, or failure
semantic. Optional negotiated evidence may instead advance its own schema or
the conformance-suite version, provided an existing Worker can still implement
the same required contract without ambiguity.

The Python package remains an unreleased development package until the project
chooses a release boundary; Worker Protocol revision and package version are
separate decisions.

## Features that stay outside Core

- SSH authentication and remote host management;
- scheduler allocation and job submission;
- Docker, Apptainer, Kubernetes, VM, and cloud API calls;
- platform device selection and GPU binding;
- OCI conversion, image caches, and site registries;
- Candidate transfer and object-storage mechanics;
- site filesystem, quota, logging, and observability configuration;
- physical placement of the Coordinator, Agent, Research Worker, and
  Evaluation Worker.

Core must define the schema and validation semantics for evidence that affects
comparison, policy, or eligibility, and may define optional observational
metadata. The Operator or Worker deployment implements the mechanism and
produces the evidence.
