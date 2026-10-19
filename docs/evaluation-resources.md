# Evaluation resources and cost comparability

`time_per_task` is the trusted Candidate invocation wall time divided by the
frozen task count. A successful measurement is not proof that another invocation
had equivalent CPU, I/O, cache or contention conditions. External Worker preview
testing identified this gap; Training Lease evidence and host inventory do not
establish the effective environment of a later evaluation.

Research comparisons must define evaluation conditions as well as Training
resources, keep effective conditions consistent, and disclose differences.

<a id="release-contract"></a>

## Evaluation resource contracts

A user-owned environment can declare `evaluation_resource_contract` independently
of its training `resource_contract`. The resolved `FrozenEnvironment` binds both
contracts and the research budget before task reveal; see
[environment configuration](environments.md).

Benchmark schema v3 adds an optional `evaluation_resource_contract` to each
runtime profile. It is independent of the Training `resource_contract` and uses
the same property-extensible `ResourceEnvelope`. Core does not impose a universal
CPU, filesystem, scheduler or cache vocabulary. The release owner must define the
meaning of each property and select which conditions are normative and which are
only observed. An absent property is unspecified.

For example, a **derived local preview** profile could add this envelope (the
values are illustrative, not defaults or official requirements):

```json
"evaluation_resource_contract": {
  "schema_version": 1,
  "normative": [
    {"name": "cpu.quota_cores", "relation": "exact", "value": 1.5,
     "minimum_enforcement": "best_effort"},
    {"name": "memory.limit_bytes", "relation": "exact", "value": 2147483648,
     "minimum_enforcement": "best_effort"}
  ],
  "observational": ["cache.policy", "storage.io_policy", "host.exclusivity"]
}
```

The frozen environment records the effective evaluation envelope. Trusted
launchers pass it to the Supervisor through the environment or explicitly as
`evaluation_resource_envelope=ResourceEnvelope(...)`. When a benchmark release
also fixes an evaluation envelope, conflicting execution and release contracts
are rejected. The contract is bound by its canonical SHA-256 and environment
identity; benchmark comparisons additionally bind the selected release identity.
Public profile names alone are insufficient.

Existing Benchmark v2 profiles remain readable without evaluation contracts. V2
rejects this new field, and older Cores reject v3, avoiding silent omission of a
selected contract. The current preview profiles leave CPU, host-memory, cache,
I/O and exclusivity conditions unspecified; their GPU declarations do not
establish a verified cost environment.

## Startup preflight

The local `start` launcher passes the selected evaluation envelope to
`DockerRuntime.preflight_evaluation_resources()` after TRAIN verification and
before storage admission, episode/workspace creation, image preparation or task
reveal. This read-only check rejects normative conditions the local Worker cannot
satisfy, including unsupported cache/exclusivity properties, stronger enforcement
than it provides, missing CPU/memory limit support, and unavailable GPU selections.
It uses the same property checks as evaluation, with a shared 20-second default
budget for resource queries, including initial GPU selection.

Preflight does not launch workloads, read hidden data, issue allocation receipts,
or establish permanent admission. An absent envelope adds no resource probes;
unavailable observational properties remain allowed. The selected contract still
reaches the Supervisor unchanged, and every evaluation repeats the checks and
validates its actual receipt. A successful preflight cannot authorize a changed
backend, a substitute GPU, or an unsupported later invocation.

`prepare --task` has no resolved run environment, so its readiness
output does not certify an evaluation contract. External launchers should perform
equivalent Worker-side checks before spending a research budget. This local
preflight does not add a remote operation or change Worker Protocol v16.

After preparation, external deployments can use the
[evaluation Worker diagnostic](worker-evaluation-probe.md) on public fixtures to
exercise actual protocol calls, receipt binding and typed rejection. This is a
separately authorized disposable test, not an automatic step in `start` or a
formal conformance receipt.

## Worker v16 and result v14

Each `evaluate_candidate()` call receives `evaluation_id` (a fresh Core-issued
opaque ID) and `resource_envelope` (`ResourceEnvelope` or `None`). Core associates
that ID with the episode, Candidate and split. The Worker must apply normative
constraints before starting the Candidate, collect actual allocation evidence,
confirm teardown, and return `CandidateEvaluationResult(resource_receipt=...)`.
If there is a contract, the receipt has this shape:

```python
EvaluationResourceReceipt(
    episode_id=episode_id,
    evaluation_id=evaluation_id,
    worker_id=worker.describe_capabilities().worker_id,
    resource_request_sha256=resource_envelope.sha256,
    receipt_id=f"evaluation-allocation:{unique_receipt_id}",
    properties=tuple(actual_property_evidence),
)
```

Each `ResourcePropertyEvidence` separately reports allocator and workload views,
their evidence authorities, and the enforcement level/authority. Unknown views
must be `unavailable`, never copied from the requested value. Normative checks
use the workload-visible value when observed, otherwise allocator evidence.
Observations do not substitute for enforcement. Receipt authenticity relies on
the trusted Worker and invocation channel; these objects are not signed remote
attestations.

Core rejects missing receipts, wrong episode/evaluation/Worker/request bindings,
reused receipt IDs, unexpected properties, mismatched effective values, and weak
enforcement. Training receipts use a different operation field and cannot be
replayed. A resource rejection is an infrastructure failure, not poor Candidate
performance. TEST failure produces no score; an ANALYSIS resource failure marks
diagnostics incomplete while preserving an already accepted TEST result.

Result v14 records `evaluation_resource_envelope` and
`evaluation_resource_allocations` separately from Training allocations. Each
record includes the evaluation ID, Candidate, split, receipt, recomputed
assessment and `accepted`. The report reader revalidates the evidence and checks
that each measured cost has a matching evaluation record. Missing contracts use
`null`, with `receipt: null`, `assessment: null`, `accepted: false`; this is an
unspecified preview, not a verified cost environment. V12/v13 reports retain their
original semantics without fabricated retrospective receipts.

Formal releases require a nonempty enforced evaluation contract, matching the
frozen release, and accepted TEST resource evidence before eligibility can become
true. This is additional to existing identities, hidden data, trust, policy and
conformance checks. **Acceptance verifies only the declared envelope.** Core
cannot establish that the organizer selected every property needed for a fair
comparison. A weakly specified contract does not justify a broad comparability
claim.

## Bundled Docker Worker

The local Worker supports `exact` normative limits for:

- `cpu.quota_cores`: aggregate CPU time ceiling in cores, represented by Docker
  `--cpus`. This does not allocate particular cores or guarantee CPU availability.
- `memory.limit_bytes`: container memory ceiling in integer bytes (minimum 6 MiB).
  Docker `--memory` and `--memory-swap` are both set to that value, requesting no
  swap allowance. This is not a reservation of physical RAM.

The Worker checks backend support before Candidate launch and checks the stopped
Candidate's Docker configuration before accepting the measurement. It also records
selected host/GPU allocation properties (`host.count`, `accelerator.count`, vendor,
model, memory per device, and optional UUIDs) when requested. CPU evaluation
requests no GPUs even if Training had a GPU grant; CUDA requests the frozen UUID selection.
The recorded GPU device request is checked against that selection.

These are **best-effort configuration/allocation observations**. Workload-visible
views remain unavailable, and no new enforced/formal guarantee is claimed. A
contract requiring stronger enforcement or unsupported normative properties is
rejected; unavailable observational properties are explicitly recorded. The
backend settings follow [Docker's resource limit semantics](https://docs.docker.com/engine/containers/resource_constraints/).

Trusted resource preparation and post-exit inspection are excluded from Candidate
wall time. The existing clock still includes Candidate sandbox startup, imports,
model loading, context reads, inference, output writes and process exit. Scoring,
resource verification and teardown do not extend that clock. The Worker does not
perform a Candidate warmup, drop host caches, or alter the TEST-before-ANALYSIS
ordering. Image, model, filesystem and accelerator cache states, I/O limits and
resource contention remain uncontrolled unless a deployment explicitly supports
the release's chosen contract.

Before comparing costs externally, the organizer still needs to freeze suitable
conditions, implement and validate their enforcement in the authorized Worker,
and record every invocation's evidence. Local command-boundary tests are not real
Docker/GPU admission or formal conformance evidence. Existing external preview
measurements retain their original comparison limits.
