# Research-controlled trajectory branching

`ngrove.branching` is an optional layer around independent research executions.
A target Agent requests checkpoints, forks, checkouts, merges and stops. A trusted
monitor applies those requests using a deployment-owned handler. The module stores
immutable state references, a shared history DAG, durable requests, resource
reservations and clock transitions. It does not choose research paths or embed a
scheduler, model provider, cloud API or container backend.

The existing `envcore.Supervisor`, Worker Protocol and task/benchmark semantics
are unchanged. The new interface is available as Python APIs and the trusted
`ngrove branches` CLI (`python -m ngrove.branching` also works).

## What is executable today

The module implements concurrent request admission, atomic monitor claims,
checkpoint ancestry, shared/copy restore intents, explicit component merges,
READY accounting, cancellation, fenced cleanup recovery and session sealing.
The optional `ArtifactStore` captures and restores quiescent regular-file trees
and explicit JSON context exports, verifies content identities, deduplicates
objects and supports physical copies into independent stores. A projection of
existing Core recording evidence is available through `episode_segment()`.

Run the self-contained exercise from a checkout:

```sh
PYTHONPATH=. python3 examples/branching/demo.py
```

It uses actual isolated workspace directories and JSON-state snapshots to run a
four-branch lifecycle, sharing a prefix, copying a prefix, merging selected
components and sealing the session. It requires no model credentials, Docker or
accelerators. It is a synthetic runner fixture, not evidence that a proprietary
Agent supports session forks or that a physical resource limit is enforced.

A real deployment supplies its Agent transport, runner capture/restore methods,
allocation and execution handler, independent supervision and cleanup. There is
no bundled proprietary-model adapter or automatic plug-in to `ngrove start`.

## Roles and trust

- The target research system chooses the operation, historical snapshot,
  requested resources, context fidelity, and merge selections. Its controller
  runs as a resource-accounted branch too. The Operator bootstraps the initial
  branch before any target requests are possible.
- `ControlEndpoint(session, authenticated_branch_id)` binds a target identity
  outside the request body. The deployment authenticates the transport and
  revokes it at teardown. The endpoint itself is not an authentication service.
- `BranchClient(transport)` is the small target-facing client. Use
  `publish_client_bundle()` to publish only that client; never mount the session
  database, monitor credentials or full Python package into an Agent sandbox.
- A monitor/assistant claims requests and implements environment operations.
  Common handlers can be reused without an LLM. New platform mechanisms may be
  prepared online within the frozen contract. The monitor does not invent a
  research merge, choose promising branches, or relay private measurements.

The CLI is an **Operator interface**, including its `request --actor` test and
integration entry point. It is not safe to expose that command directly to an
Agent. The target transport calls a bound `ControlEndpoint`, never the CLI with
an Agent-supplied actor or database path.

## Frozen state contract and provider admission

Create a session with `SessionPolicy`, `RunnerCapabilities`, an initial state
manifest and the initial resource request. These settings are detached from
caller-owned dictionaries and fingerprinted at creation. Each component is:

```json
{
  "format": "runner-owned-context-format-v1",
  "sha256": "<64 lowercase hexadecimal characters identifying exported bytes>",
  "locator": "<opaque immutable adapter reference; private to the monitor>"
}
```

`agent_context` and `workspace` are mandatory. The Operator can require other
components such as model/optimizer weights, RNG state, data cursors, memory,
tools and environment configuration. A digest must identify an immutable export,
not merely the name of a mutable provider conversation. The runner must freeze,
verify and pin each component until all dependent snapshots are retired. A Core
episode's artifact-retention policy does not automatically pin it for this DAG.
Never capture the trusted episode directory, hidden datasets, Operator credentials
or private trajectory export as the Agent's workspace/context.

`RunnerCapabilities` identifies a runner and its independently obtained probe
record, supported component formats, context modes and prefix modes. Unsupported
formats or modes are rejected before branch admission. **No closed-source
provider capability is assumed.** Before registering a real runner, verify:

1. Export and restore include all state required by the selected experiment.
2. Restoring twice produces independent mutable continuations; neither changes
   the original immutable export or the other continuation.
3. `exact` means continuation of that runner's declared exported state; it does
   not promise deterministic model outputs or access to provider-internal state.
4. `replay` explicitly reconstructs a context with documented omissions. It is
   never an implicit fallback, and a replay-derived checkpoint cannot be
   relabelled exact by a later request.
5. A snapshot taken while a tool call is in flight must either include a
   deployment-supported consistent continuation or wait for a quiescent boundary.
   This module does not recreate arbitrary process stacks or live network calls.

Capability declarations are trusted admission evidence, not provider
certification. The fixture supports exact restoration of its JSON state only.

## Target operations

Queued research operations have an Agent-selected stable request ID. Reuse it after a transport
timeout; changing the arguments with that same ID is rejected. IDs are scoped to
the authenticated actor branch. The client accepts `request_id=`; a caller needing
crash-safe retries must persist its ID before the first transport call.
Session closure is idempotent for the same actor and final snapshot; changing that
final selection after requesting closure is rejected.

| Operation | Semantics |
| --- | --- |
| `checkpoint(branch)` | Quiesce all research execution in that branch, capture state and selected research-visible recording deltas, commit an immutable node, then release the original branch to continue. |
| `fork(snapshot, ...)` | Create a fresh branch from a saved node with independent mutable execution state. The source branch can continue. |
| `checkout(snapshot, ...)` | Create a fresh continuation at a selected historical node. It does not destroy later history or implicitly stop the current branch; the Agent can request a stop separately. |
| `merge(sources, selection, ...)` | Create a new node with multiple parents, then restore it as a new branch. The Agent must select a source for every state component explicitly. |
| `stop(branch)` | Request revocation/cleanup. Queued preparation can be cancelled without execution; claimed operations retain ownership until confirmed cleanup. |
| `history(snapshot)` | Expand the immutable DAG prefix once per node, retaining identities, parents, capture scope and records. Private component locators are excluded. |
| `snapshot(id)` | Read one node and its parent IDs to traverse a large history incrementally. Private component locators are excluded. |
| `status()`, `branch(id)`, `operation(id)` | Observe the public budget or a focused branch/operation status; private execution receipts and diagnostics are excluded. |
| `finish(snapshot)` | Select the final snapshot and request session-wide closure. Hidden evaluation is not enabled by this request alone. |

For example, in a runner with an authenticated transport:

```python
from ngrove.branching.client import BranchClient

client = BranchClient(transport)
operation = client.fork(
    snapshot_id,
    resources={"gpu": 1},
    context_mode="exact",          # must be supported by the admitted runner
    prefix_mode="shared",
    request_id=persisted_request_id,
)
```

A merge selects complete components; it does not automatically concatenate Agent
contexts, resolve code conflicts or merge optimizer states. To produce a new
combined component, the target can construct it within a research branch,
checkpoint it and select that result. The runner must validate that the selected
components form a usable state, reporting incompatibility rather than silently
rewriting the Agent's choices. Multiple parents make the history a DAG.

### Shared versus copied prefixes

Logical ancestry is retained in both modes. `shared` reuses immutable objects and
requires private writable state (for example materialization or copy-on-write).
`copy` asks the adapter to materialize an independent copy of the full selected
prefix and its reachable state. It is not a request to reset history or budgets.
The registry itself always references immutable nodes; the adapter receipts state
which physical mode was applied. `ArtifactStore.copy_component()` and the demo
show real local copying, including all nodes in the prefix. Other storage formats
and provider session copying remain adapter responsibilities.

`history()` produces a topological traversal with shared nodes emitted once. Its
default limit is 256 nodes and aggregate node metadata is bounded by the session's
`max_payload_bytes`. A `history_limit_use_snapshot` error directs callers to walk
parent IDs through `snapshot()`; no truncated history is silently substituted. It
preserves each node's own event order and does not claim a total causal order
between sibling branches. Checkpoints retain local evidence deltas; imports can
use `episode_segment(episode, after_seq=previous_boundary, through_seq=boundary)`.
That projection verifies recorded payload identities and includes only explicitly
Agent-visible exchanges before sealing. Missing capture remains explicit; handler
return/publication does not prove consumption. The complete private Core export
is not used as a branch prefix.

## Three clocks and resource admission

The frozen session policy defines a capacity vector, a global wall-time limit,
a total active-research limit, an ancestry-inclusive branch limit, and optional
resource-seconds limits. Resource names are deployment-defined portable quantities.
The module does not interpret them as GPUs, schedulers or inference vendors.

| Interval | Branch active time | Known allocation time |
| --- | --- | --- |
| Queued or provisioning before allocation acknowledgement | Stopped | None recorded yet |
| Allocation acknowledged, state restoring | Stopped | Charged |
| READY committed and execution permitted | Running | Charged |
| Stop/checkpoint requested but quiescence not confirmed | Still running | Charged |
| Quiescence confirmed, capture or cleanup in progress | Stopped | Charged until release acknowledgement |

The session wall clock continues throughout. A child waiting for resources does
not pause its parent. Checkpoint capture excludes only the mechanical state capture
operation; Agent-selected data preparation, compilation and experiments remain
research work. A successful checkpoint retains its allocation and rearms the
original branch before continuation. READY/continuation are recorded **before**
the monitor opens the execution gate; Agent acknowledgement is not the clock gate.
A duplicate receipt returns `already_applied: true`, never a second launch permit.

Ancestry time is the union of recorded per-branch research intervals. Shared
ancestors count once within a merged ancestry; divergent parents both contribute.
This is explicit accounting of inherited work, not the longest wall-clock path.
The global research ledger sums actual branch-active intervals once, including
failed work; forks/checkouts cannot reset it. Resource-seconds use trusted monitor
allocation/release acknowledgement intervals, including preparation and capture.
They are not a provider invoice or proof of physical enforcement. Missing/late
observations require deployment reconciliation; uncertain resources are never
silently declared free.

Capacity reservation and operation claiming happen atomically, so multiple
monitors cannot claim the same request or overbook the declared vector. Requests
that do not fit stay queued; independent fitting work can proceed. Physical
allocation evidence, actual device identities, storage limits and global hard
budget enforcement belong to the deployment. Admission is not enforcement.

The service uses a trusted wall clock that survives process restart. Backwards
clock movement is rejected. Deployments must provide synchronized clock semantics
and Worker-local monotonic deadline enforcement; a monitor LLM or a SQLite read
is not a deadline mechanism. Regular programmatic supervision calls `status()`
or `claim()`: exhausted budgets enqueue stops/revoke pending preparation while
retaining clocks and ownership until quiescence/cleanup are confirmed. Aggregate
budgets need deployment-wide enforcement across simultaneously running branches;
a per-branch permit alone cannot enforce a shared cumulative limit.

These are opt-in branching-session clocks. They do not silently redefine existing
Track A/B rules, grant new Lease budgets or make inherited model checkpoints
eligible under an unchanged task policy.

## Monitor protocol, uncertainty and recovery

`Monitor.run_once()` atomically claims one operation and invokes a trusted,
preconfigured `OperationHandler`. It neither polls through an LLM nor selects
executable code from an Agent request. A handler can return while an asynchronous
operation is outstanding; the operation stays claimed until explicit receipts.
The handler must not block supervision of other branches. Use separate dispatch
workers or asynchronous deployment operations when needed.

For preparation:

1. Claim: persist operation ID and fencing token; reserve the resource vector.
2. Allocate and report `allocated()` with a unique allocation identity.
3. Restore and verify the entire requested state, prefix mode and independent
   writes while the runner is still gated.
4. Call `ready()` with the snapshot, allocation and new episode identities.
5. Open execution only after a newly committed permit. Independently enforce
   the returned limits and the session's global contract.

For checkpoint: quiesce, report `quiesced()`, export/pin state, call
`complete(..., execution_held=True)`, then resume only on a newly committed permit.
For stop: terminate, report `quiesced()`, clean owned resources, report
`released()`, then `complete()`. Preparation failure and cancellation also require
quiescence and release before `fail_clean()`.

An exception or lost acknowledgement leaves `needs_recovery` or an outstanding
claim with its reservation. Automatic retry must not create another environment.
The Operator fences the old executor and uses `recover()` with that evidence to
rotate the token. Old tokens are rejected. Recovery is cleanup-only; a new run
requires a new target request. Tokens provide service-side fencing, while the
deployment must also prevent a stale executor from making platform side effects.
Timeouts alone do not prove termination or authorize resource reuse.

SQLite transactions commit state and its audit event together. Multiple processes
on one trusted local filesystem may open the same database; a distributed/NFS
control store is not claimed. The prototype uses bounded session-wide metadata
(maximum branches/requests and per-node metadata); large artifacts live outside
SQLite. The local directory format preserves regular-file bytes, names, executable
bits and empty directories; it rejects links/special files and does not claim to
preserve owners, extended attributes or live process state. Total storage capacity
and any garbage collection are deployment-owned.
Use focused branch/operation observations for routine polling instead of loading
the full history into an Assistant context. The private audit journal contains
deployment details and must not be sent to an Agent. Histories and status are
observations, not complete captures of Agent internals.

## Compose with existing episodes

A deployment handler can use existing Supervisor/Worker components for each
execution, or a runner exposing the same execution evidence. Bind distinct episode
IDs, workspaces, control surfaces and recovery scopes to branches. Begin execution
only after the branch permit, and make the effective environment/time contract
agree with that permit. `episode_segment()` reuses existing recording output;
runner exports provide the additional restorable context and state.

This package does not make an old Core episode resumable. It also does not make a
straight call to `ngrove start` a branching handler: that CLI starts its own clock
and performs its own teardown/evaluation lifecycle. An integration must coordinate
those boundaries and the outer session explicitly. It must use a task/research
contract permitting inherited state; the current LM preview's restrictions on
initial target-model checkpoints are not waived by a fork.

`finish()` records the target-selected final snapshot and closes admissions. Every
branch must be quiescent, all resources released and pending operations reconciled.
After all Agent control capabilities are revoked, the trusted launcher calls
`confirm_sealed()`. Only then may it proceed to hidden evaluation under the task's
existing immutable Candidate and evaluator contracts. A branch snapshot is not
itself a validated Candidate. No hidden evaluation is performed by this module.
Do not expose results from a completed child to other live branches or allow a
child launcher to bypass the session-wide teardown boundary.

## Operator commands

```sh
python -m ngrove branches --help
python -m ngrove branches create --database /trusted/session.db --config /trusted/session.json
python -m ngrove branches status --database /trusted/session.db
python -m ngrove branches claim --database /trusted/session.db --monitor monitor-1
python -m ngrove branches history --database /trusted/session.db --snapshot SNAPSHOT_ID
```

The create JSON contains `policy`, `runner`, `initial_state`, `resources` and
`context_mode`, matching the Python contracts. `receipt` applies explicit monitor
acknowledgements, and `recover` accepts independently obtained fencing evidence.
Keep claim outputs and receipt tokens private. Configuration/paths are trusted
Operator inputs, not commands the Agent can arbitrarily supply.
