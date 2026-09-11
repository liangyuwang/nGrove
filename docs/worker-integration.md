# Integrating an external Worker

This guide is for the trusted benchmark operator preparing a deployment outside
Core. It complements the [Worker Protocol](worker-protocol.md); it does not
select a platform, Tested Agent, task, or resource budget. The bundled
`prepare`/`start` commands compose the local Docker implementation. They are not
an external backend loader.

A Worker can be temporary source created during authorized run preparation.
Reuse a compatible implementation where possible; otherwise implement the
required boundary in the approved trusted working root. No additional source
repository is required. Keep platform code and dependencies outside Core, and
keep both outside the Tested Agent workspace. Before launch, preserve the
actual Worker source, dependency identities, launcher configuration, and hashes
alongside the Core revision and dirty-state evidence. A path or a revision
label alone does not freeze generated code. Later fixes belong to a new
deployment revision; do not rewrite a running or archived deployment.

## Composition and lifecycle

Worker Protocol v16 is synchronous Python. The trusted launcher constructs the
Supervisor with its Research Worker, Evaluation Worker, Agent execution
evidence, and Agent-control transport. Remaining `Path` arguments require a
deployment in which their consumers can resolve the intended files; they are
not remote artifact-transfer instructions. Hidden-data locations and resolving
credentials are private Evaluation Worker configuration.

Registered inference has a separate [Gateway execution evidence](gateway-execution.md)
interface. A trusted launcher obtains `Supervisor.gateway_execution_request`,
starts and inspects the Gateway, and records a `GatewayExecutionReceipt` before
`begin()`. Formal eligibility requires the complete running artifact to match
the organizer's release verification; a configured image string is insufficient.
The bundled development launcher still provides no certified Gateway receipt.

When updating a deployment from v15, `evaluate_candidate()` must accept the new
`evaluation_id` and `resource_envelope` keyword arguments. An explicit envelope
requires an operation-bound `EvaluationResourceReceipt` on the returned result.
It cannot reuse Training or another evaluation's receipt. The Supervisor reads
the evaluation envelope from the resolved challenge release; standalone trusted
launchers may supply `evaluation_resource_envelope` directly. Resolve unsupported
normative properties before authorizing an expensive run. See the
[evaluation resource contract and evidence example](evaluation-resources.md).

The bundled `start` now calls the local Worker's read-only evaluation-resource
preflight before episode setup and task reveal. External composition roots need
the equivalent deployment check; do not postpone a known unsupported contract
until the Agent has spent its research budget. Preflight produces no allocation
receipt and never replaces per-evaluation enforcement and evidence validation.

The following is a constructor sketch with inputs already resolved by the
trusted launcher, not a standalone launcher or a reference Tested Agent:

```python
from benchcore.supervisor import Supervisor

supervisor = Supervisor(
    config=task_config,
    agent=agent_config,
    resources=training_resources,
    training_data_verification=train_verification,
    resource_envelope=resource_envelope,
    track=track,
    episode_dir=episode_dir,
    agent_api_dir=public_agent_api,
    agent_execution=agent_execution_evidence,
    research_worker=research_worker,
    evaluation_worker=evaluation_worker,
    agent_control_transport=agent_control,
    research_worker_environment=research_environment,
    evaluation_worker_environment=evaluation_environment,
    comparison_key=comparison_key,
    challenge_release=challenge_release,
    release_verification=release_verification,
    research_worker_conformance=research_conformance,
    evaluation_worker_conformance=evaluation_conformance,
    worker_resolution_timeout_seconds=resolution_timeout,
    operator_max_research_seconds=operator_research_limit,
)
```

Resolve task, named Track configuration, runtime profile, and resource envelope
from the selected manifest before constructing these inputs. Supply verified
TRAIN evidence bound to the actual manifest; keep TRAIN read-only. Publish only
the generated Agent API into the participant view, not the Core package. Use
the release's comparison metadata and trusted receipts where applicable;
missing organizer verification or conformance remains missing evidence.
Self-description and diagnostic test results cannot manufacture those receipts.
The Supervisor writes trusted episode state and may prepare the selected
control transport during construction, so storage and ownership admission must
already be complete.

Use this sequence when implementing the composition layer:

| Stage | Core interaction | Deployment responsibility |
| --- | --- | --- |
| Admit | Construct `Supervisor` with frozen inputs | Verify boundaries, images, TRAIN, tools, resource authorization, and recovery readiness before task reveal |
| Start registered Gateway | Obtain `gateway_execution_request`, then `record_gateway_execution(receipt)` | Register cleanup before startup; attest the actual running artifact and fixed inference registration through the trusted launcher boundary |
| Reveal and launch | `begin()` immediately before the selected Agent starts | Start only the frozen Agent command and workspace, with the agreed inference path; record actual process startup separately from allocation |
| Research | Repeatedly call `enforce_operator_limit()`, `process_one_request()`, and `enforce_deadlines()` while active | Enforce deadlines independently of the request loop; if the Agent exits while Core is still active, call `agent_exited_without_finish()` |
| Close research | Let Agent requests and Core deadline/failure transitions close research | Stop admitting work; do not call `finish()` on behalf of an interrupted Agent or choose an incumbent |
| Stop research workloads | `terminate_worker_leases_for_teardown()` | Retry pending Lease teardown and stop all remaining Research/Agent/inference workloads and control capabilities; preserve errors |
| Record termination | `record_agent_execution(...)`, then `cleanup_agent_control()` | Confirm Agent termination first, finish bounded log draining, and record actual usage completeness; do not turn missing logs into zero usage |
| Evaluate | `evaluate_postmortem()` only when Core is still `closed` and teardown is confirmed | Resolve hidden data only now; Core stages TEST for the declared final incumbent first, then postmortem ANALYSIS |
| Clean evaluation | Report teardown errors through `deployment_teardown_failed(...)` | Finish all Evaluation Worker workloads, prediction storage, and private hidden materializations, including retrying retained handles after failures |
| Publish | `publish_postmortem_result()` only after successful required cleanup and when Core remains `closed` | Preserve Core's staged TEST decision and ANALYSIS disposition; refresh the final progress snapshot before its storage is detached |
| Retain/dispose | No new benchmark transition | Verify the authorized archive, then dispose of owned working storage; record this independently of workload cleanup and benchmark outcome |

`evaluate_postmortem()` does **not** publish a terminal result. Worker methods
must already confirm their own per-operation teardown before returning; the
composition layer additionally owns any retained resources and private
materializations. A deployment-specific `close()` is not a Worker Protocol
method. Supply that cleanup in the external implementation rather than assuming
Supervisor will discover it. `terminate_worker_leases_for_teardown()` returns
whether state changed, not a cleanup-success receipt; handle raised failures
and the retained active/pending Lease state. `cleanup_agent_control()` must
return true before evaluation.

When using `start_host_agent_process`, call `handle.stop()` and then pass
`handle.summarize_usage()` to `record_agent_execution(usage=...)`. The summary
combines retained stdout records with in-memory capture evidence for both
streams. Disk-full, quota, write/flush/close errors, missing EOF, and unfinished
draining must not make a surviving valid usage prefix look complete. The
file-only `summarize_host_agent_trace(path)` parser cannot detect those losses.
Stderr capture has its own status; stdout is the source of token accounting.
The log sink discards further output after a write failure while continuing to
drain the pipe, and records the first error without relying on another write
to the failed log file. Preserve this evidence in trusted state/archive when
storage is available. Loss of the launcher or its state storage still requires
external evidence; an in-memory flag is not a durable recovery receipt.

On a live launcher's infrastructure failure, use
`infrastructure_failed(code, private_detail=...)` to record the cause. For
unconfirmed termination/removal, use `deployment_teardown_failed(...)` and
retain owned handles. A failure writing Core evidence must not skip independent
cleanup actions or permit publication. Preserve both the direct failure and
cleanup failures. Unexpected process loss is handled by the deployment's
external recovery path, without reconstructing a Supervisor to resume or
finalize the interrupted episode.

## Agent environment and tool admission

Check the environment **inside the final execution view**, under the selected
identity and working directory. A runtime's host-side environment can contain
paths that are invalid inside the Agent view. Explicitly map `HOME`, temporary
directories (`TMPDIR`, `TMP`, `TEMP` when used), caches, configuration paths,
helper executables, and `PATH` entries to their intended visible locations.
Keep runtime-private state outside the participant view. For example, a host
runtime temporary directory must not leak through `TMPDIR` when the Agent has
a separate writable `/tmp`.

Verify creation, reading, atomic replacement, and removal of disposable files
in the permitted workspace and temporary directories. Check the Agent's actual
file and command tools too: a shell write succeeding does not prove that a
native file tool's path mapping works. Check required sandbox/helper processes
under the same mounts and permissions. Do not address a tool failure by exposing
trusted host directories, hidden data, or weakening the approved policy.

Version output, executable availability, and authentication status are useful
prechecks, but do not establish real session compatibility. Where invocation is
authorized, use the selected frozen Agent in a separate disposable diagnostic
workspace with a harmless task to exercise initialization and required tools.
Do not reveal the benchmark task, carry that session or its files into the
episode, or inject training code. Account for inference and resource costs
within the approved diagnostic scope. If that check cannot be performed, record
session compatibility as unverified; do not silently claim it passed or change
the Agent packaging after task reveal.

## Monitoring and cancellation

Keep three evidence sources distinct: Core's benchmark snapshot, current
execution-system observations, and the operator's preparation/progress record.
Use [Operator status](operator-status.md) for the first two. A monitor should
record check time, last actual progress, current work, and the next action and
owner. Allocation, successful process creation, a working Agent session, and
an active Training Lease are separate milestones. Never derive training
activity or a completion percentage solely from allocation or Lease counts.

Publish derived state from the authoritative `state.json`, using
`closure_reason` and `research_sealed_at`. Serialize periodic and final writes
so an older periodic snapshot cannot overwrite the final one. After result
publication, finish the final refresh and stop/join the snapshot publisher
before unmounting its source. Retain the authoritative final state as evidence;
a stale progress file must not override it. An external cancellation can leave
Core `active`: preserve that historical snapshot and report current termination
through a separate checked deployment observation.

Monitoring failures need independent handling. Bound each source query; a hung
allocator query must not prevent reading Core or launcher logs. An unreadable
progress file must not stop heartbeat production. For a file-based watchdog,
read directly and validate the record rather than relying on an earlier
existence check. Cache only the last valid heartbeat and operation deadlines;
transient I/O/parse failures do not reset startup time, renew a deadline, or
cause an immediate kill. Reject invalid or regressing heartbeats, keep checking
cached deadlines, and escalate when the configured loss/deadline tolerance is
actually exhausted. Record read failures and the precise trigger. A healthy
heartbeat alone must not authorize a Worker operation to run indefinitely.

All Worker operations and cleanup attempts need finite allowances. An outer
watchdog must remain able to act when the synchronous Coordinator call blocks;
a second thread that needs the same lock is insufficient. Register owned
resources and creation intents durably before dispatch, outside participant
writable storage and its exhaustion budget. Retain enough identity evidence
to reconcile a creation whose acknowledgement was lost. Fence workload
creation before recovery takes ownership.

Use a shared recovery deadline as well as per-command timeouts: repeated
inspection/removal failures must not each restart the full recovery allowance.
Stop dispatching new cleanup work when it expires, preserve unresolved handles
and prerequisite ordering, and report incomplete cleanup for a later authorized
retry. State which local I/O or external operations the deployment cannot
preempt. Preserve durable cleanup progress so retries do not repeatedly spend
their allowance on already confirmed historical removals while live workloads
wait. Uncertain or unpersisted outcomes remain pending. The bundled Docker
recovery implements this shared allowance. Its CLI and
`DockerRuntime.recover_resources()` also isolate journal access, cleanup and
report persistence in independently supervised processes. Low-level
`DockerRecovery.recover()` remains cooperative and needs an outer supervisor
when used directly. The bundled mechanism does not supervise an external
Worker's arbitrary cleanup implementation.

Handle supported termination signals with a bounded shutdown path: stop new
work, attempt cooperative teardown within its allowance, then fence the owned
execution boundary and verify absence. Record the received signal, trigger,
attempts, and unresolved resources if storage remains available. Signal handling
and `finally` cannot survive forced kill, host loss, or every allocator shutdown.
A parent in the same allocation can disappear with its child. Arrange a
separate deployment service or a later operator entry point that can inspect
and recover resources after the original launcher is gone.

That external entry point needs these operations; their commands and platform
mechanisms belong to the deployment:

| Operation | Required behavior/evidence |
| --- | --- |
| Inspect | Read the durable run/ownership record and actual resource state; report absent, present, or unconfirmed per resource without changing Core |
| Recover workloads | Exclude a live owner and in-flight creation, verify immutable identities and ownership, stop only owned workloads/control capabilities, and confirm absence; retry safely |
| Recover private storage | Proceed only after writers are confirmed stopped; reconcile owned mounts and hidden/prediction materializations, retaining failed handles for retry |
| Archive and dispose | Preserve the selected artifacts and logs, verify archive size/checksums, then remove only authorized working allocations; record archive location and disposal result |

A reused PID, a matching filename prefix, or an ended allocation is insufficient
ownership or teardown proof. Missing receipts remain unconfirmed. Cleanup
success does not create a benchmark result, re-open research, or authorize a
replacement job. The bundled [Docker recovery](worker-recovery.md) command covers
its own journaled resources; it does not recover arbitrary external Workers.

### Archive deadlines and recovery

Budget archiving against representative artifact bytes, file counts and storage
throughput, including inventory, copying/compression, full verification and
report persistence. A small fixture proves the path, not its capacity for a full
episode. Record a finite archive deadline and any phase limits before launch;
pass the remaining allowance to nested helpers instead of using unrelated
hard-coded timeouts. Include bounded helper termination and a declared outer
watchdog grace. Progress does not renew the deadline. Fit this work within the
authorized reservation or arrange a separate CPU-side recovery allowance.

Keep these observations separate in durable deployment evidence; they are not
new Core result fields:

| Observation | Evidence and consequence |
| --- | --- |
| Workload/writer termination | Confirm absence of owned Agent, Research, Evaluation and control workloads independently of archive errors or the launcher's aggregate exit status. An archive failure must not erase an established termination receipt; unknown termination still blocks storage reclamation. |
| Archive helper termination | Confirm that timed-out copy, compression and verification helpers have stopped before releasing storage they use. Workload termination alone does not prove those helpers are gone. |
| Archive verification | Freeze the selected artifact inventory after its writers stop; verify completeness, sizes and checksums by reading the archive. A readable partial archive or successful copy command alone is insufficient. Publish a completion record atomically only after the verified archive and its manifest are durable. |
| Working-storage disposition | Record retained pending archive verification, disposed, or cleanup failed, with the reason and owned handles. Intentional retention after an archive failure is distinct from a failed removal attempt. Dispose only within the authorized retention scope and after all storage users have stopped. |

An archive timeout after a successful TEST remains an archive failure; preserve
the TEST measurement, ANALYSIS disposition and independent termination evidence.
Do not derive a `writers_stopped` condition from an empty global error list.
Record genuine teardown failures separately, and retain the source artifacts
when archive verification fails.

Provide an entry point outside the original launcher/allocation that can use
durable ownership records and retained artifacts to retry archiving on CPU.
After excluding a live owner and confirming termination, recover through a
read-only source view or a stable snapshot where supported. Use the same
bounded helper supervision and verification requirements as normal archiving;
preserve an existing verified archive until its replacement is verified. Record
the recovered archive location, verification evidence and authorized disposal
separately from the original failure. Missing source or termination evidence
remains unresolved. This recovery does not require restarting an Agent,
allocating accelerators or creating a new episode, and must preserve the saved
Core state rather than resume or finalize it.

## Staged verification and resource clocks

Use focused checks before a full run. Retain evidence against the exact Worker,
runtime, Agent, and Core identities; repeat affected checks after changes.
Unchanged unrelated checks do not need to be repeated on every ordinary run.

Use the [acceptance evidence matrix](worker-acceptance-matrix.md) to select the
remaining cases for the exact deployment. It separates current local regression
coverage from historical external Worker observations, defines failure and
cleanup criteria, and lists the evidence needed for each result. A completed
matrix or passing probe alone does not close its pending deployment cases.

| Stage | Minimum evidence before progressing |
| --- | --- |
| Interface | Protocol/version compatibility, bound workload/resource receipts, hidden resolution rules, and required failure types; the resolution diagnostic alone is not full conformance |
| Boundary and recovery | Required role visibility, permissions, resource/device limits, immutable capture, deadline enforcement, and owned-resource cleanup, using disposable fixtures without hidden evaluation content |
| Agent startup | The selected runtime view and actual required tools initialize successfully, or the remaining preview limitation is explicit |
| Short complete episode | An agreed development lane within the approved diagnostic scope reaches Lease, commit, Agent-declared incumbent, research sealing, confirmed teardown, TEST, ANALYSIS disposition, final publication and final status refresh |
| Full selected Track | Run the already agreed lane and budget, retaining real enforcement and completion evidence; never shorten its manifest in place and call that the full Track |

The [Research Lease diagnostic](worker-lease-probe.md) checks two supported
allocations and a rejected admission, resource receipt binding, fixture command
execution, repeated termination and failed-start compensation. Its command is
supplied by the disposable deployment; it creates no Agent or hidden evaluation.
An optional `CommandTimeoutCase` adds one deliberately long-running fixture
command on the first Lease. Supply an independent command-family observer that
can detect surviving descendants and bind actual timeout enforcement, including
when a client or parent has already exited. The case adds no Lease starts and
requires no platform commands in Core; its false-timeout and cleanup evidence
remains in the private diagnostic report.

The [idle deadline diagnostic](worker-deadline-probe.md) observes one live Lease
through expiry using the deployment's independent read-only observer and
controller event evidence. Supply that observer separately from Worker state
polling; the probe cannot validate independent expiry by asking the Worker to
perform its own deferred cleanup.
An explicitly selected pause/termination case adds a deployment-owned injector
and continuous fault evidence from the observer. Fault injection and external
recovery need their own authorized scope. Once injection starts, the probe makes
no further Worker calls, including during failed-injection cleanup.

The [evaluation Worker diagnostic](worker-evaluation-probe.md) supplies a reusable
interface check for supported TEST/ANALYSIS resource receipts and deliberately
unsupported contracts. Run it only on public disposable fixtures, with the
deployment's independent watchdog and explicit cleanup confirmation. It accepts
temporary external Worker implementations through the same protocol; no site
adapter or formal certification is bundled into Core.

Include focused regressions for the integration changes: runtime-private host
paths absent from Agent temporary-directory variables, real file-tool operation, short live log
visibility, final progress after publication, transient monitor failures,
blocked Worker calls, cancellation, and independent recovery after parent loss.
Include log-storage failures and incomplete draining: the Agent must not block
on a pipe whose destination filled, and retained usage must reflect capture
loss even when a truncation marker could not be written.
Use disposable owned resources with bounded costs. Unit simulations establish
logic; real-runtime tests establish the observed deployment behavior. A normal
successful episode alone does not certify cancellation, hard quota exhaustion,
or host-loss cleanup. Full adversarial conformance remains an organizer-owned
assessment, not a claim inferred from these preview checks.

Record resource clocks explicitly before submission:

| Clock | Start and meaning |
| --- | --- |
| Allocation | When resources are granted; includes any preparation or idle time performed inside that allocation |
| Track budget | Core's selected Track accounting and Worker Lease limits; not a percentage of research completion |
| Private research protection | When research starts; an Operator wall-time bound separate from the Track score and hidden from the Tested Agent |
| Evaluation and cleanup | Their own bounded operations; require capacity/time within the reservation or a separately authorized allocation |

Equal allocation and research limits can expire at different times because
their clocks start differently. Plan preparation, evaluation, cleanup grace,
and artifact retention within the approved resource ceiling; ask for any
additional material budget before consuming it. Preserve intentional GPU
reservation between Leases while reporting both reservation and Lease time.
When protection or cancellation ends a run, retain its actual closure/outcome
rather than reporting normal completion.
