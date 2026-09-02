# Thin Core development plan

## Purpose

AutoTrain-Bench Core defines what an episode is allowed to do. It does not
define where workloads run, how an Operator reaches a host, or which platform
mechanism enforces the contract.

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
  admission, Candidate commit timing, VAL feedback, incumbent selection, TEST
  lifecycle, results, and leaderboard eligibility.
- A Challenge Release defines every resource property that is normative for
  comparison or eligibility. Core validates the requested requirements and
  the semantic evidence selected by that release. A Worker deployment chooses
  the platform mechanism, enforces the allocation, and reports effective
  evidence.
- Core decides when a Candidate must become immutable and defines its
  acceptance and canonical identity. A Worker guarantees that the producing
  workload can no longer mutate the selected output; who materializes the
  immutable snapshot depends on the protocol stage described below.
- The Research Worker never receives hidden VAL or TEST data.
- The Evaluation Worker receives one selected hidden split and an immutable
  Candidate, but not the research workspace.
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

## Phase A: state the v0 boundary accurately

Status: complete.

The v0 support statement is:

- The bundled `autotrain-bench prepare` and `autotrain-bench start` commands
  are a single-host local Docker reference composition.
- `autotrain_bench.worker_protocol` version 2 is a synchronous in-process
  Python contract. Its `Path` values are local deployment handles, not a
  serializable remote wire format.
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
  define site storage policy.
- After the v0 Worker stops a Lease, the Coordinator validates, copies, hashes,
  and atomically stores the Candidate from a shared path. Version 2 has no
  Worker-side capture operation or remote `ArtifactRef`.

Phase A changes documentation only. It does not rename schema fields, change
Worker Protocol version 2, or claim support for a new deployment.

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
- Task reveal, Lease accounting, Candidate commit, VAL/TEST, and incumbent;
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
does not infer compliance merely from `container` or `host_cli`.

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

### Phase B exit criteria

- The semantic request handler can be tested without creating a mailbox.
- The local file transport can be tested without Docker.
- Docker-specific process, network, mount, and log calls live only in the
  bundled reference composition/runtime.
- An external launcher can supply Research and Evaluation Worker
  implementations without modifying the Core CLI or adding a backend branch.
- Existing v0 tasks and tests run without a Worker Protocol revision.

## External validation deployment

Status: planned outside this repository.

After Phase B, build one development-only Worker package using a real
non-Docker runtime. The preferred first target is a shared-filesystem Slurm
allocation with Apptainer, optionally preceded by a same-host Apptainer smoke
deployment.

The external Operator obtains the allocation and stages the repository and
artifacts. The Worker consumes that authorized environment; Core does not call
`ssh`, `sbatch`, `srun`, or `apptainer`.

The first preview deliberately retains the version 2 `Path` contract:

- TRAIN, workspace, Lease output, Candidate snapshots, hidden splits, and
  evaluation output are materialized on the shared filesystem;
- the Research and Evaluation implementations map semantic Worker operations
  to the site runtime;
- OCI workloads are consumed directly or mapped to prebuilt SIF artifacts;
- every OCI-to-SIF mapping is recorded explicitly and the deployment remains
  leaderboard-ineligible unless that mapping is organizer-approved;
- Worker-local deadline and termination behavior are exercised rather than
  simulated by Core.

The deployment should keep an observation log containing each place where the
version 2 local contract forced a platform-specific workaround. Those observed
constraints, not hypothetical topologies, determine the next protocol work.

## Phase C: add only demonstrated semantic capabilities

Status: deferred and split into independent decisions.

### C1. Agent execution evidence

Trigger: a policy-compliant non-Docker Agent sandbox cannot express its
identity, visibility, network, termination, and trust evidence through the v0
configuration.

Possible result: a semantic Agent execution descriptor or receipt assessed
independently from its runtime mechanism.

### C2. Remote immutable artifacts

Trigger: the Coordinator and a Worker cannot safely share a `Path`, or the
Coordinator cannot locally snapshot a stopped Lease output.

Possible result: an `ArtifactRef`, Worker-side atomic Candidate capture,
materialization verification, and content-identity receipt. Core continues to
own Candidate acceptance and canonical identity rules. The remote Worker owns
atomic capture and transport enforcement. This replaces, rather than
retroactively describes, the v0 Coordinator-side shared-path snapshot.

### C3. Workload resolution evidence

Trigger: a deployment executes a SIF, VM image, or proprietary workload derived
from the release-owned OCI reference.

Possible result: a receipt binding the requested canonical workload, effective
artifact identity, transformation provenance, and organizer acceptance.

### Other independently triggered work

- atomic episode activation with Worker-local monotonic timing;
- asynchronous operations and idempotent operation IDs;
- event streaming;
- a remote authenticated Worker transport;
- deployment certification.

None of these requires SSH, scheduler, cloud, or object-storage implementation
inside Core.

## Versioning rule

Phase A documentation and a behavior-preserving Phase B refactor do not change
Worker Protocol version 2.

A new protocol revision is required when an implementation must change a
required method, argument, return value, ordering guarantee, or failure
semantic. Optional negotiated evidence may instead advance its own schema or
the conformance-suite version, provided a version 2 Worker can still implement
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
