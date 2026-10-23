# Worker Protocol boundary

The Python namespace is `envcore.worker_protocol`.
Run-owned environment configurations use this same admission and evidence
contract. They do not add a platform backend or weaken Worker requirements.
See [configurable environments](environments.md) for resource selection and freezing.

Core owns research lifecycle and evaluation decisions. A Worker deployment owns
execution and enforcement.  The boundary between them is expressed in
research protocol terms and must not expose a scheduler, container runtime, or cloud
API.

```text
Coordinator
  authoritative episode state
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
- validation and recording of trusted TRAIN verification evidence;
- execution diagnostics and failure evidence.

Platform packages own:

- resource allocation and placement;
- image preparation;
- process, container, Pod, or job lifecycle;
- GPU, filesystem, and network enforcement mechanisms;
- platform logs and environment receipts;
- site-specific security hardening.

Platform implementations live outside Core, either as separately distributed
packages or temporary deployments created during authorized preparation. Keep
their source and dependency identities with run evidence. Examples of package
names are `ngrove-worker-docker`, `ngrove-worker-apptainer`, and
`ngrove-worker-kubernetes`; these names are examples, not core backends.
Core must never grow platform selection branches or import their SDKs.
See [Worker integration](worker-integration.md) for trusted composition,
startup verification, supervision, cleanup, and resource-planning guidance.

The trusted launcher or Worker verifies the public TRAIN view before task
reveal. The launcher supplies that evidence to `Supervisor`, which checks its
verified status and binding to the current root manifest without scanning the
dataset files. The Worker must keep that verified view immutable and read-only
during research. The bundled local launcher uses the shared file verifier;
other deployments choose their verification mechanism. These checks do not
read hidden data or replace Worker enforcement; see
[TRAIN integrity](training-data.md).

## Policy and enforcement

The Coordinator decides:

- the selected track and budget;
- whether another lease is allowed;
- which Candidate is the declared incumbent;
- when research is irreversibly sealed;
- when postmortem ANALYSIS and TEST are allowed;
- which frozen Candidate inventory is selected for ANALYSIS;
- whether the result has complete evaluation and execution evidence.

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
or other research policy.

In v0, the Coordinator materializes and hashes the immutable Candidate snapshot
from a shared path after the Worker has stopped the Lease. Worker-side capture
and transfer are future protocol capabilities for deployments without shared
paths; Core still owns Candidate acceptance and canonical identity.

## Capabilities and execution evidence

`envcore.worker_protocol` describes capabilities by semantic name and
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
It also requires `candidate_prediction_storage_quota` and
`candidate_prediction_artifact_lifecycle`: untrusted output storage must be
hard-bounded during execution and then removed according to the release-owned
ephemeral lifecycle.

`assess_worker` checks the required capabilities and records missing or
insufficient enforcement. `AgentExecutionEvidence` records the mechanism,
workload identity, enforcement level, provenance and concrete limitations.
The run retains those facts in `execution_issues` and the detailed evidence.

`envcore.conformance` provides resolution,
[Research Lease](worker-lease-probe.md), and
[evaluation resource](worker-evaluation-probe.md) diagnostics. The Lease probe
checks admission receipts, fixture command execution, stopped-state observations,
repeated termination and rejected admission, with failed-start compensation.
Its optional command timeout case adds independent command-start, timeout-event
and descendant-absence observations before cleanup. A returned timeout flag alone
cannot pass it; command return and observation share an explicit termination grace
within the original case/Lease windows.
The evaluation probe calls
the supplied Evaluation Worker on public fixtures, checks per-invocation receipts
and typed resource rejection, rejects late replies, and requires explicit
deployment cleanup confirmation. The probes use unique operation scopes, separate
public failure codes from trusted details, and invoke deployment teardown.
The separate [idle Lease deadline diagnostic](worker-deadline-probe.md) uses an
independent deployment observer to verify workload absence and a bound deadline
event before any Worker cleanup call. Its observer must never enforce expiry
as a side effect of inspection. The optional Worker pause/termination case uses
deployment fault injection and independently verifies that the fault remains
active through expiry. After injection is attempted, cleanup uses only external
recovery and observation, with no further call into the potentially lost Worker.
Physical enforcement, isolation and recovery require additional deployment
checks beyond these scoped observations.

## Research and evaluation roles

Worker Protocol v17 models ResearchWorker and EvaluationWorker as separate
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
`HiddenDatasetRequest` containing only the logical split name, the active
hidden-bundle identity algorithm, and an optional release-frozen expected
SHA-256 digest. The Worker privately resolves that request and returns a
`ResolvedHiddenDataset` containing the same algorithm, the effective digest,
and an opaque resolution ID. Core validates the receipt before passing the
opaque resolution back for Candidate evaluation. A complete `TaskConfig` never
crosses either role boundary.

Before either role launches release-owned code, Core sends a `WorkloadRequest`
containing a semantic role and the canonical release identity. The Worker
returns `ResolvedWorkload`: effective format and SHA-256, optional immutable
transformation provenance, and an opaque resolution ID. Core passes that exact
resolution into `start_lease()` or `evaluate_candidate()`. A digest-pinned OCI
identity is accepted when the effective OCI digest is unchanged. Any derived
artifact records its effective identity and transformation provenance.

Version 15 is a breaking interface revision in the current unreleased
development snapshot; it does not imply a package release version. Older
Workers are rejected rather than adapted implicitly. Version 15 retains the v6
workload-resolution rules, the v7 requirement that a successful `start_lease()`
return a `ResourceAllocationReceipt`, and the v8 mandatory resolution
deadlines, as well as v10 single-target semantics, the v11 offline two-artifact file contract, and the v12 hidden-bundle
identity and prediction-storage rules. It also retains the v13 generic
performance-score contract and the v14 release-scoped cost contract. External
Worker packages must implement workload and hidden-data resolution, bind
operations to resolved identities and resource requests, accept TEST before
ANALYSIS, return valid performance and cost measurements, and preserve the
precise context/target/artifact visibility fields in
`CandidateChannelRequirements` and the offline file ABI without assuming a
particular storage or sandbox implementation. Version 15 separates the bounded
runtime prediction allocation from strict final artifact validation, as
specified under [Candidate channel](#candidate-channel).

A local development Worker may implement both roles in one process. A distributed
deployment may place them on separate services connected only through
an immutable artifact reference. The current version 17 Python interface does
not yet define that remote artifact transport.

Worker operation failures and teardown failures have different episode
semantics. `WorkerFailure` reports an operation that did not produce a usable
result and may be returned only after the Worker has confirmed that the failed
operation left no unaccounted workload boundary. `WorkerTeardownFailure`
reports that termination or removal remains unconfirmed. The latter is always
deployment-integrity evidence: further evaluation must not continue and the
Coordinator records a stable teardown code while retaining deployment-specific
details only in its trusted event stream. Before TEST it terminates the episode.
During optional post-TEST ANALYSIS it retains the staged TEST decision when a
cleanup retry succeeds, while recording the execution boundary failure; cleanup that remains unconfirmed blocks terminal result
publication. In particular,
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
the protocol together with the release-owned identity-algorithm identifier.

Both `resolve_workload()` and `resolve_hidden_dataset()` receive a positive,
finite `timeout_seconds`. A conforming Worker must stop resolution by that
deadline and return no authorization for a late result. For primary TEST, the
Coordinator passes only the remaining evaluator-operation allowance through
workload resolution, hidden-data resolution, and `evaluate_candidate()`; these
are not additive timeout grants.

### Known liveness limit: cooperative deadlines

Worker Protocol v17 is a synchronous, in-process Python interface. Its timeout
arguments are semantic deadlines and an execution requirement, but they are
cooperative rather than a Core-side preemption mechanism. Core can reject a
result returned after its deadline; while a call has not returned, however,
Core cannot safely interrupt arbitrary Worker code or continue processing other
deadline-sensitive operations in that process. A wedged or nonconforming
Worker can therefore block the Coordinator. This is a deployment/infrastructure
failure, not a participant failure, although detecting and terminating it may
currently require an outer Operator or deployment watchdog.

The bundled launcher guards Track B research with a private operator wall-time
limit, enabled by default for `lm-pretrain` (24 hours) and `smoke_lm-pretrain`
(30 minutes). Operators may override or disable it before launch; disabling
prints a warning about potentially extreme token consumption. These settings
and notices never enter the Agent view. Its separate local thread/client can revoke
workloads without acquiring the Coordinator lock, but still shares the
Coordinator process and cannot preempt arbitrary Worker code. It is best-effort
operator protection, not a new Worker deadline guarantee or a Track B scoring
budget. The container CLI now additionally uses an independent parent watchdog
for heartbeat loss, private research limits, active Lease durations and
instrumented Worker-operation timeouts. It fences the local process group,
then uses the Worker journal to recover resources; it does not resume the
episode or preempt an individual Core call safely. See
[Worker recovery](worker-recovery.md) for tolerances, scope and reports.
The semantic Worker deadline contract is unchanged by this local supervision.
External launchers own equivalent protection
in their execution environment.

A future platform-neutral seam should define asynchronous operation IDs,
absolute deadlines, status/reconciliation, idempotent retry, and cancellation
semantics. Core should define what cancellation means and how its result affects
the episode; the Worker deployment should implement the mechanism, such as
terminating a sandbox, cancelling a scheduler job, or deleting a workload.
Platform handles and platform-specific cancellation branches must not enter
Core. Merely moving the current call to a thread would not provide safe forced
cancellation and is not considered a solution.

The same ownership rule applies to evaluation: `evaluate_candidate()` must not
return a metric until its isolated Candidate/Evaluator boundary has been torn
down. If teardown cannot be confirmed, it raises `WorkerTeardownFailure` and
retains whatever private handles its deployment needs for operator-level retry;
those handles are not added to the Core protocol.

## Workload identity and resolution

The current task fields `training_image`, `candidate_image`,
`evaluator_image`, and `inference_gateway_image` are release-selected OCI-style
workload references. A digest-pinned reference identifies immutable workload
content; a mutable reference requires the Worker to record the resolved identity.
These references describe content independently of the runtime that executes it.

A non-Docker Worker may consume the OCI artifact directly. If it executes a
derived SIF, VM image, or platform artifact, the mapping must be reproducible,
identified and recorded. A Worker may not silently
replace the workload environment. Version 6 defines that semantic evidence and
operation binding without defining how a platform converts or stores the
artifact.

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
does not gate the TEST outcome; separate deployment
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

`analysis_sha256` and `test_sha256` use the normative algorithm identifier
`envcore-hidden-bundle-v1-sha256`. Its SHA-256 input is the following
byte string, with no separators other than those shown:

```text
"envcore-hidden-bundle-v1\0"
|| for name in ("contexts.json", "targets.json"):
     u64be(len(utf8(name))) || utf8(name)
     || u64be(len(file_bytes)) || file_bytes
```

The bundle directory must not be a symlink and must contain exactly those two
regular files. Lengths are unsigned 64-bit big-endian integers. The shared
`envcore.artifact_identity.hidden_bundle_sha256()` implementation is
normative for Python Workers. The organizer release receipt declares
`file: sha256` and
`hidden_evaluation: envcore-hidden-bundle-v1-sha256`; the current result
schema records the latter under
`artifact_identity_algorithms.hidden_evaluation`.

## Candidate channel

`CandidateChannelRequirements` fixes the release-owned application protocol and
security properties. Under `offline_next_token_logits_v3`, one resolved hidden
split consists of exactly two mutually bound artifacts. `contexts.json`
contains the dataset identity, vocabulary, declared context lengths, example
IDs, and input token IDs. `targets.json` contains matching example IDs,
next-token targets, protected source spans, and the SHA-256 of the exact context
artifact. The Worker validates both before untrusted code starts, mounts only
the context artifact read-only into the Candidate, and retains targets for the
trusted scorer. Protected spans may not overlap.

The Agent-visible task manifest declares the Candidate entrypoint at
`evaluation.candidate_boundary.entrypoint`; the Worker channel manifest carries
the same declaration. The Worker imports `candidate.py` from the artifact root
as a Python module, then synchronously calls
`run_offline(candidate_dir, contexts_path, output_dir, device)` with four
positional arguments. The first three are absolute `pathlib.Path` objects:
the read-only Candidate directory, the read-only target-free `contexts.json`,
and isolated writable prediction storage. The fourth is the selected evaluation
device string (`cpu` or `cuda`). Module-level code runs during import with
`__name__ != '__main__'`; `sys.argv` is not part of the Candidate ABI. The return
value is ignored; predictions must be written before the function returns.

The Candidate owns input
reading, model lifetime, ordering, batching, and parallel scheduling, and emits
exactly `predictions.json` plus a row-major little-endian float32 `logits.f32`.
There is one vocabulary-sized row per context in request order. A trusted
scorer later validates the output and joins it to hidden targets. No target,
score, or feedback crosses into Candidate execution. The requirements contain
no URL, port, socket path, network name, or scheduler handle; the bundled
Docker Worker implements the contract with isolated filesystem mounts and
network-disabled processes.

The bundled Docker Worker verifies the bind-mount owner's UID/GID with
disposable private input/output files and explicitly selects that identity for
every role, overriding image `USER` defaults. Its private HOME directories and
writable output roots belong to that identity; hidden snapshots retain `0400`
permissions. Required image executables must support the selected numeric
identity. UID mapping, file ownership, and permission recovery after confirmed
termination remain deployment responsibilities; no UID fields or Docker
identity rules enter the semantic Core/Worker protocol. Failed permission
verification blocks launch rather than widening access.

Prediction storage is an untrusted resource boundary. Worker Protocol v15
separates runtime resource protection from exact final artifact validation.
Before Candidate startup, the Worker supplies an independent output allocation:

```text
artifact_max_bytes = examples * vocab_size * 4 + 65536
runtime_max_bytes = round_up(max(16 MiB, 2 * artifact_max_bytes), 1 MiB)
runtime_max_inodes = 128
```

The runtime byte limit bounds allocated filesystem storage, including
open-but-unlinked allocations; it does not bound sparse logical file lengths
or cumulative bytes written. The inode limit includes the filesystem root and
bounds file/directory metadata; implementation overhead must also remain
bounded. Temporary files and directories are allowed within this allocation.
The final artifact must still contain exactly `logits.f32` with the exact
request-derived length and `predictions.json` of at most 65536 bytes, with the
specified regular-file, schema, dtype, order and finite-value checks. Temporary
entries must be removed before Candidate exit. Sparse or oversized files never
relax final validation.

The output allocation must not consume the capacity reserved for episode logs
and results. A polling watcher or final validation alone does not enforce the
runtime limit. A Worker must refuse Candidate startup if the bounded allocation
cannot be established; it cannot substitute an ordinary writable directory.
This guarantee remains `candidate_prediction_storage_quota`. Its v14 meaning
(exact logical byte ceiling and two entries throughout execution) is retired;
v14 descriptors/receipts are not evidence for the v15 contract. The public
boundary changes in Agent Protocol v12 and is bound to Task schema v7; the
offline v3 artifact JSON shapes are unchanged. Earlier task schemas are rejected.
The task manifest version changes its release fingerprint;
measurements using the old and new boundary must not be compared as one release.

Prediction artifacts use the release-owned lifecycle
`ephemeral_until_evaluation_teardown`. After scoring or failure, and only after
confirmed Candidate/scorer teardown, the Worker deletes the prediction
workspace. It must retain and register the workspace for cleanup retry if
workload teardown is unconfirmed; it may not delete a path that untrusted code
might still mutate. This guarantee is reported as
`candidate_prediction_artifact_lifecycle`. A future release that retains
predictions for audit must define a different lifecycle with an immutable
identity, storage budget, and retention period.

The bundled Docker Worker uses a separate Linux Docker local-driver `tmpfs`
volume with `size` and `nr_inodes` limits. It verifies its options, capacity and
empty state before Candidate startup. A trusted, network-disabled, read-only
holder keeps it mounted between Candidate exit and scoring; the holder sees
only prediction storage. Candidate receives the writable volume, while the
scorer receives it read-only. Image-declared extra volumes are rejected in all
evaluation images, even when private operator storage protection is disabled.
After confirmed Candidate/scorer termination, the Worker removes the holder
and volume and verifies removal, retaining failed cleanup handles for retry.
The per-evaluation private `<split>.storage.json` records limits and observations.

This implementation requires Linux Docker local-driver tmpfs support and shared
Worker/Docker paths for the other inputs. Unsupported mounting fails closed.
Output pages consume execution-host memory (and potentially swap), so deployment
memory/swap provisioning must include the declared output capacity. The bounded
volume protects the episode log filesystem from prediction allocation exhaustion;
it does not reserve host memory or guarantee log persistence during host OOM,
shared backing-device exhaustion, or runtime failure. Normal exception/retry
cleanup is implemented. The bundled launcher also journals episode resources
for [cleanup after Worker process loss](worker-recovery.md), with exclusive
ownership, identity checks and conservative handling of unknown creations.
Recovery does not resume or publish an episode. The Worker remains
development-trusted with best-effort deadlines; this
capability describes the enforced execution boundary.

Separately, the trusted local launcher can require a deployment-provided,
capacity-bounded filesystem for the entire episode. Its private operator limit
applies to either Track and does not replace the prediction quota above. The
Worker verifies storage in its execution environment and records observations
in its private environment manifest as `operator_storage_protection`. This
does not add provisioning commands, public budgets or filesystem handles to
the semantic Worker protocol. A remote Worker must admit and enforce its own
storage allocations; operator-host disk observations cannot stand in for that
evidence. See the README's private operator storage protection section for the
bundled launcher's admission requirements and scope.

The v3 JSON shapes are exact (additional fields are rejected):

```json
{
  "schema_version": 2,
  "candidate_protocol": "offline_next_token_logits_v3",
  "dataset_id": "release-owned-id",
  "vocab_size": 50257,
  "context_lengths": [2],
  "examples": [
    {"example_id": "example-00000000", "input_ids": [50256, 123]}
  ]
}
```

```json
{
  "schema_version": 2,
  "candidate_protocol": "offline_next_token_logits_v3",
  "dataset_id": "release-owned-id",
  "contexts_sha256": "<sha256-of-exact-contexts.json-bytes>",
  "examples": [
    {
      "example_id": "example-00000000",
      "target": 456,
      "source_id": "private-source-id",
      "source_start": 0,
      "source_end": 2
    }
  ]
}
```

The output manifest is also exact:

```json
{
  "schema_version": 1,
  "candidate_protocol": "offline_next_token_logits_v3",
  "dataset_id": "release-owned-id",
  "examples": 1,
  "vocab_size": 50257,
  "dtype": "float32-le",
  "logits_file": "logits.f32"
}
```

The input contract is published at
`evaluation.candidate_boundary.context_manifest`, with exact root and example
field sets, JSON types, constants, and bounds. The root has `schema_version`,
`candidate_protocol`, `dataset_id`, `vocab_size`, `context_lengths`, and
`examples`; each example has only `example_id` and `input_ids`. `input_ids` is
an integer array in oldest-to-newest order, with length in `context_lengths`
and token values in `[0, vocab_size)`. IDs are opaque, nonempty strings of at
most 512 characters without NUL; example IDs are unique within the request.
The request contains no targets, split labels, or protected source metadata.

The research-visible task manifest publishes `evaluation.max_context_tokens`
as an input bound and `evaluation.context_policy` as the variable-length
declaration. Participants choose how to use context within their resource
budget. No sample is guaranteed to reach that bound. Exact sampling lengths,
proportions, and construction parameters stay in trusted evaluation
configuration, frozen before research begins. `EvaluationTaskSpec.context_lengths`
remains trusted Worker input and every length must fit `context_length`, the
trusted configuration's input bound. The `context_lengths` values inside
`contexts.json` describe that request and are visible to the frozen Candidate
only after research sealing. Publishing their schema does not publish their
values during research. This distinction is independent of the Worker sandbox
or storage mechanism.

The output contract is present in the Agent-visible public task manifest at
`evaluation.candidate_boundary.prediction_manifest`, including the exact field
set, JSON types, constants, request-derived values, and the prohibition on
additional fields. Its `logits_artifact` declares a headerless, row-major,
little-endian IEEE-754 float32 matrix: row `i` matches `examples[i]`, column `j`
matches token ID `j`, and all values must be finite. JSON member order and
whitespace are not significant. The adjacent `public_helper_api` entry names
the optional generated request reader and prediction writer. The reader's
declared return type is a dictionary with the parsed context object's keys
and values unchanged, referencing `context_manifest` for the full structure.
Both declarations are shared with the Worker channel manifest.
Post-seal hidden-evaluation failures retain their fixed
public failure codes and trusted-only details.

The target rows must align one-for-one with context rows and IDs. Each protected
span has `source_end - source_start == len(input_ids)`, spans for one source may
not overlap, every declared context length must occur, and `logits.f32` must be
exactly `examples * vocab_size * 4` bytes. The trusted validator rejects an
oversized manifest before parsing it.
Candidate-visible dataset and example identifiers are opaque correlation keys;
they must not encode whether the invocation is ANALYSIS or TEST or reveal
trusted source metadata.

`evaluate_candidate()` returns `CandidateEvaluationResult`, carrying a generic
`performance_score`, release-declared `raw_metrics`, and trusted cost evidence. The
score contract fixes a finite `[0, 100]` range, `maximize` direction, exact raw
metric names, a definition ID, and `comparison_scope: task_release`. The
task evaluator owns the mapping from task metrics into this interface; Core and
the Worker protocol do not assume NLL, perplexity, accuracy, or another
particular formula. The current `uniform_normalized_log_score_v1` definition is
implemented by the frozen task evaluator. It requires finite non-negative
`mean_nll` and computes
`100 * max(0, min(1, 1 - mean_nll / log(vocab_size)))`. Uniform prediction and
worse therefore map to 0; the raw NLL retains their distinction. Perplexity is
not required, so overflow in that derived display value cannot invalidate a
finite score.

The independent cost contract defines `time_per_task` in
`seconds_per_task`, with `minimize` direction, as
`candidate_wall_seconds / task_count`. The task manifest freezes what one task
means; `lm-pretrain` uses one next-token prediction. This is an amortized
workload cost rather than per-request latency because the Candidate controls
batching and scheduling. `task_count` comes from the trusted, release-required
context set; incomplete prediction output fails before a cost is returned.
The same cost object retains `candidate_wall_seconds`, `task_count`,
`input_tokens`, `prediction_bytes`, `tasks_per_second`, and
`input_tokens_per_second` as auditable submetrics. The Candidate wall clock
starts immediately before Candidate invocation and stops after process exit
and output flush, so it includes sandbox startup, Candidate import/model load,
context-file reading, participant-controlled scheduling/inference, and
prediction writing. Hidden resolution, trusted validation, target joining, and
score computation are outside that clock. Teardown is confirmed before the
operation returns. Cost comparisons require the same task release, hidden
dataset identity, and effective evaluation resources. Matching resource descriptions do not prove
equivalent cache or contending-workload conditions.

Worker v16 additionally receives `evaluation_id` and `resource_envelope` on each
evaluation. When the latter is non-null, `CandidateEvaluationResult` must carry
an `EvaluationResourceReceipt` binding that ID, episode, Worker and request hash.
Core validates effective normative values and enforcement before accepting the
score, and records/revalidates the evidence in the result. The evaluation
resource envelope is separate from Training requirements. A resource
rejection is an infrastructure failure; a later ANALYSIS failure preserves an
accepted TEST decision. See [evaluation resource contracts](evaluation-resources.md)
for comparison limits, protocol migration and Docker support.

## Current migration stage

The phase-one interface is an in-process Python protocol. The Supervisor uses
semantic workload resolution (`resolve_workload`), lease operations
(`start_lease`, `exec_lease`, `lease_state`, and `terminate_lease`), Worker-owned
hidden resolution (`resolve_hidden_dataset`), and semantic evaluation
(`evaluate_candidate`). It neither stores container
identifiers nor asks how those effects are implemented.

Its `Path` arguments are local deployment handles. They assume that the Worker
implementation can access or materialize the Coordinator-named TRAIN,
workspace, Lease output, Candidate, public API, and evaluation locations. They
must not be treated as a stable serializable remote protocol. Hidden datasets
are the exception: their semantic request and resolution contain no `Path`.
A shared-filesystem Worker can implement version 14 directly; a topology without
shared Candidate or workspace paths still needs future artifact-reference and
materialization semantics.

Version 13 retains the release-selected `ResourceEnvelope` passed into
`start_lease()` in addition to the bundled v0 `TrainingResources` compatibility
input. The
envelope contains only properties that the release classifies as normative or
observational; an absent property remains deliberately unspecified. Property
names are extensible, while the generic relations (`exact`, `at_least`,
`at_most`, and `contains`) and minimum enforcement level give Core enough
semantics to check evidence without knowing a scheduler or sandbox vocabulary.

A successful startup returns a versioned `ResourceAllocationReceipt` bound to
the request digest, episode ID, Lease ID, and Worker ID. Each property keeps
allocator-visible and workload-visible observations separate and names the
authority responsible for enforcement. Normative requirements apply to the
effective workload-visible value when it is observable, with allocator evidence
as the fallback. A release that independently constrains allocator capacity
names that semantic property explicitly. Core checks that effective value, the
minimum enforcement level, missing and unexpected properties, and the operation
bindings before admitting the Lease. A rejected receipt causes
compensating Lease termination; the workload never becomes an active,
budgeted Lease. Receipt authenticity comes from the trusted Worker deployment
and its invocation channel—v14 does not pretend that an in-process Python object
is a remote cryptographic transport.

During the v0 manifest transition, an explicit `resource_contract` is the full
semantic envelope and must include requirements consistent with the profile's
legacy `host_count` and `accelerator` compatibility fields. It may add arbitrary
release-defined properties, but it cannot erase or contradict those legacy
requirements. A future schema may remove that duplicate compatibility
projection; until then the cross-check prevents two conflicting resource
sources of truth.

The bundled Docker Worker selects full NVIDIA GPUs matching the requested
count, model and minimum memory, then freezes their UUIDs for that Worker
instance. Selection is deterministic in UUID order, independent of inventory
row order. Training and CUDA Candidate launches use the same explicit UUID
grant through Docker's `--gpus` device parameter, including its CSV quoting for
multiple devices; see [NVIDIA's device-selection documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#gpu-enumeration).
No physical device identifiers are added to the semantic resource envelope.

Before returning a Training Lease receipt, the Worker queries the idle
Training container and requires the visible UUID set, model and memory to match
the selection and requested profile. A mismatch fails startup and triggers the
existing Lease cleanup; unconfirmed removal retains the retryable handle.
Successful checks write private `gpu-binding.json` evidence in the Lease
directory. The trusted environment manifest records
`selected_nvidia_gpu_uuids` separately from its full host inventory.

CUDA evaluation rechecks that the frozen devices remain eligible before
starting Candidate code, using the remaining evaluation allowance. It never
substitutes another matching device. A new GPU requirement needs a new Worker
instance. Inventory queries are bounded; malformed/duplicate UUIDs and invalid
memory evidence are rejected. This implementation selects full GPUs only;
MIG-enabled devices are excluded instead of treating a partition as the
profile's full GPU. It does not change MIG mode or allocate accelerators.

GPU resource evidence remains `best_effort`: command-boundary regressions and
inventory checks do not establish denial of access to every ungranted device,
exclusive use, a remote allocator's ownership, or the independent Evaluation
allocation contract. The local deployment must already be authorized to use
the selectable host devices. Scheduler-specific ownership and grants remain
external Worker responsibilities. A real heterogeneous-host and negative-access
acceptance run is still required before strengthening these guarantees; the
bundled Worker reports development trust and best-effort GPU enforcement.

`CommandResult` streams are capped at 1 MiB each by the local Worker and again
by the Supervisor before they enter the mailbox or `commands.jsonl`. Concrete
Workers must drain excess output while retaining only the bounded prefix and
must report the truncation flags.

The pre-existing Docker runtime remains temporarily as the bundled local
reference implementation so the runnable preview does not regress.  It is a
compatibility implementation, not the extension point for more platforms.  A
later extraction moves it to a separately distributed Worker package without
changing Supervisor semantics. Its container CLI supplies an independent
process watchdog, but the synchronous protocol, grace-based escalation and
bundled development deployment still report `BEST_EFFORT` deadline evidence. Direct
runtime integrations do not inherit the CLI's process supervision.

Deliberately deferred work includes:

1. canonical remote ArtifactRef transport;
2. an atomic `activate_episode` operation with Worker-local monotonic timing;
3. asynchronous operations and idempotent operation IDs so slow Worker calls do
   not block deadline enforcement;
4. remote authenticated Worker transports where a real topology requires one;

Those features should be added to the Worker protocol and external Worker SDK,
not by adding platform code to Core.

The staged plan and the trigger for each deferred abstraction are recorded in
[Thin Core development plan](thin-core-development-plan.md).
