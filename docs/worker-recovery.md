# Docker Worker cleanup after process loss

The bundled container launcher records episode-owned runtime resources in a
private Worker journal. Recovery reclaims resources; it does not resume an
episode, retry training/evaluation, change budgets, reopen research, or publish
results. When authorized to perform recovery, the Repository Assistant acting
as trusted operator may invoke the same recovery command that a deployment
supervisor invokes. The Tested Agent receives neither the journal nor recovery
authority.

## Launch and recover

After readiness checks complete, `start --agent-runner container` enables
recording before the first episode workload is created. The default directory
is `~/.envcore/workers/<episode-path-id>`, on the actual Worker host.
The launcher prints the resolved directory and records it in its private
Worker environment manifest. Override it with `--worker-state-dir PATH`.
Use dedicated, trusted local storage outside the episode's bounded filesystem
and the Agent workspace. This keeps participant disk exhaustion from filling
the journal. Journal size is limited to 4 MiB; inability to persist an intent
blocks new runtime creation. Admission reserves space for every record's full
lifecycle, including identities not yet returned by Docker and the recovery
phase. A create is rejected before dispatch if this reservation would exceed
the limit; already admitted records retain room for completion and cleanup.

After the Worker process has stopped, run in the **same Worker environment**:

```sh
envcore worker-recover --state-dir /worker/private/run-state
```

Each recovery pass has a shared 120-second allowance. Override it with
`worker-recover --timeout-seconds SECONDS`; direct Python callers can pass
`timeout_seconds` to `DockerRuntime.recover_resources()`. These entry points
execute recovery in a fresh interpreter and dedicated process group. The
supervisor starts its monotonic timer before helper launch; journal acquisition,
daemon checks, journal writes, private-storage deletion and journal close all
run inside the supervised process. The watchdog's subsequent report writer uses
another supervised process with only the remainder of the same allowance.
Each Docker command also retains its cooperative timeout of at most 20 seconds
and the algorithm's remaining allowance. More resources or failed commands do
not renew the parent's timer.

The recovery helper retries only a busy ownership lock, at short intervals
within that allowance, so a fenced runtime client can finish exiting and release
its inherited descriptor. Lock waiting and journal acquisition spend the same
cooperative deadline as subsequent cleanup; they do not grant a fresh command
budget. Ordinary `DockerRecovery(...)` opens still fail immediately with
`WorkerOwnershipBusy` when the lock is held. Permission errors, damaged or
replaced journals, and owner/directory mismatches are not retried. The helper
checks the journal and registered identities after acquisition as before.
A lock held through expiry leaves recovery incomplete and preserves the journal,
lock file and unconfirmed resources.

On detecting expiry, no further runtime commands or cleanup stages are started.
The report records `deadline_exceeded: true`, the configured
`timeout_seconds`, and `cleanup_complete: false`. A late successful reply is
not an absence confirmation; retained ownership records allow a later pass to
verify what actually happened. Holders and private storage remain protected
until prerequisite workload cleanup succeeds. A retry receives a new allowance
and verifies the retained records; it does not resume the episode.

The parent reads only bounded private socket messages and never opens the
journal or writes report files itself. At completion or expiry it signals its
own unreaped helper process group before waiting, preventing PID/PGID reuse.
Termination confirmation has a separate allowance of at most five seconds;
the parent returns an incomplete result instead of waiting indefinitely if
termination or fencing remains unconfirmed. It never unlocks another process's
journal. Every Docker recovery client inherits the ownership lock, so an
uninterruptible client still blocks a competing recovery pass after helper
loss. `process_supervision` records the attempt ID, helper exit code,
termination/fencing confirmation, received operator signal and failures.
Unavailable episode metadata remains null when journal acquisition failed.

The recovery and report subprocesses inherit no Agent/control channel and have
their stdout/stderr disconnected from the caller; only the bounded private
response channel is used. Recovery supervision requires POSIX, a main-thread
caller and exclusive child reaping. Direct `DockerRecovery.recover()` is the
cooperative primitive used **inside** that boundary and does not install its
own process supervisor. Custom launchers calling it directly need equivalent
supervision, as do custom recovery callbacks supplied to `supervise()`.

A resource whose absence was confirmed and whose complete `removed` record was
persisted is finished work. Recovery checks the journal ownership and daemon
binding on every pass, then processes only unresolved resource instances.
Historical removals do not spend the next pass's command allowance, including
when recovery is reopened in a new process. Workload containers still precede
prediction holders, networks, volumes and private storage. A deletion with a
lost acknowledgement, or a RAM-only `removed` state whose persistence failed,
still requires verification. Legitimate name reuse creates a new intent/token
and must pass the full ownership and identity checks again. These durable
confirmations rely on creation remaining fenced and on resources not being
recreated or relabelled outside the journaled lifecycle.

Blocked recovery-storage syscalls no longer trap the waiting supervisor. This
does not guarantee that a kernel will immediately terminate an uninterruptible
task or that unavailable storage will be reclaimed. Unconfirmed resources and
held locks remain for external recovery. The supervisor's own OS scheduling,
trusted interpreter/source availability and the caller's final console/output
delivery are outside the recovery-storage guarantee.

Exit code 0 means all registered resources were confirmed absent and private
temporary materializations were removed. Exit code 2 means recovery failed or
remains unconfirmed; the JSON report lists unresolved resources. Repeating the
command is supported. A recovered journal cannot be used to launch an episode.
Use a new episode/state directory for a new, separately authorized attempt.

The journal directory, its ownership lock and Docker labels are part of the
recovery boundary. Do not copy/move the journal, delete or replace its lock,
modify its records, or remove/change resource ownership externally while a
Worker is active. A copied/replaced directory, wrong UID, unsafe permissions,
malformed journal or different Docker daemon is rejected. Local storage must
provide reliable POSIX `flock`, atomic rename and `fsync`; a shared network
filesystem or a journal copied to another host is not a distributed lock.

## Guarantees and failure behavior

- An exclusive process-lifetime lock prevents a live Worker and a recovery
  process from owning the same session concurrently. Short resource-creation
  clients inherit this lock, so Worker death alone cannot permit takeover while
  such a client is still dispatching a create request.
- Each create intent is atomically persisted and synced before dispatch.
  Containers, networks and output volumes carry a Worker owner label and a
  unique resource token. Recovery verifies those labels and immutable
  container/network IDs; volume identity also includes its creation timestamp.
  The timestamp's JSON representation is bounded to 128 bytes so identity
  capture fits its reservation; larger backend metadata is rejected.
  It never performs a global prune or deletes resources by prefix alone.
- Recoverable `run` operations use `create`, durable identity capture, then
  `start`. Long-running attached/wait clients do not inherit the lock and only
  address the recorded container ID. They cannot prevent recovery from stopping
  that container. Foreground output/exit status and temporary-container cleanup
  remain available to the existing runtime callers.
- Every workload/control container must be confirmed gone before prediction
  holders, networks, output volumes or hidden snapshots are reclaimed. A failed
  termination retains downstream cleanup handles and private data for retry.
  Deletion identity is persisted before deletion, so losing its acknowledgement
  does not prevent a later absence check from completing cleanup.
- A journal capacity rejection leaves the proposed new intent unregistered.
  Containers with previously persisted, verified identities can still be
  stopped when the journal becomes full or read-only, including if recovery
  cannot save its phase. Failed cleanup acknowledgements remain errors and
  retain downstream storage for retry after journal writes are restored.
  Newly discovered identities still require persistence before deletion.
  An uncertain write (for example, a failed directory sync after rename) retains
  its pending records and blocks new creation until persistence is resolved;
  recovery never silently discards a possibly committed intent.
- An intent with no confirmed creation identity is **not** resolved by an empty
  inventory: its request might still be in flight. If the owned resource appears
  later, retry can identify and reclaim it. If uncertainty persists, deployment
  investigation/fencing is required; there is no force-success switch.
- Images declaring implicit `VOLUME` storage are rejected for recoverable
  workloads, including when private operator storage protection is disabled.
  Such anonymous resources cannot silently escape the journal.

Successful cleanup preserves Core state, existing TEST decisions, result files,
logs, workspaces and captured/training artifacts. It removes only registered
runtime resources and the journal's private temporary subtree (including hidden
snapshots). Episode filesystem allocations and retained artifacts remain the
deployment's retention/disposal responsibility. Historical Core state may still
show the interrupted phase; cleanup success is not benchmark completion.

The trusted [Operator status](operator-status.md) view keeps the Core snapshot,
current deployment execution, workload cleanup and storage disposal separate.
External recovery tools can report their checked observations through that
optional format without rewriting interrupted Core state.

## Supported scope

This implementation covers one local POSIX Docker Worker session after journal
activation: Agent and inference Gateway containers, Training containers,
Candidate/scorer containers, prediction holders/volumes, created networks and
Worker-private hidden snapshots. Readiness probes completed before activation,
standalone `prepare`, `host_cli`, external scheduler allocations and resources
created through a custom Worker are not certified by this journal. Direct
DockerRuntime integrations must call `enable_recovery` before their workloads
and retain ownership until all Worker calls stop.
Custom deployments must supply their own owned-resource inspection and
recovery entry point; see [Worker integration](worker-integration.md) for its
required evidence, cancellation handling, and verification stages.
The `worker-recover` command does not archive episode storage. External
deployments own [archive deadlines and CPU-side recovery](worker-integration.md#archive-deadlines-and-recovery),
including separate workload/helper termination, archive verification and
authorized source disposal.

The container CLI now runs its trusted launcher in a dedicated process group
under an independent parent watchdog. Once the journal is registered, the
parent automatically invokes recovery after normal completion, process loss,
or a watchdog trip. Manual `worker-recover` remains the retry path. Direct
Python calls to `command_start`/`DockerRuntime` do not install this parent;
external launchers must arrange their own supervision.

## Independent local watchdog

The private channel carries heartbeats and operation allowances, never Agent
requests or credentials. Docker clients do not inherit it. The watchdog
monitors the private Track B research limit, active Training Lease durations,
declared resolution/execution/evaluation timeouts, and local Docker control
calls. Disabling the private research limit does not disable process-loss or
Worker-operation protection. Normal cooperative timeout handling is attempted
first; the parent can terminate the whole Worker if it does not finish in time.
If the supervision channel breaks, the Worker refuses new runtime work while
still allowing inspection and best-effort removal; a lost watchdog acknowledgement
must not itself keep an existing workload alive. Notification failures during
nested cleanup calls are reported after the complete removal/absence check;
they do not interrupt an intermediate daemon or ownership inspection, or mask
an actual runtime cleanup failure.

The launcher prints these private settings before the first episode workload:

| Setting | Default | Override |
| --- | --- | --- |
| Missing heartbeat tolerance | 10 seconds | `--worker-watchdog-heartbeat-seconds` |
| Deadline grace for cooperative handling | 5 seconds | `--worker-watchdog-grace-seconds` |
| Control/startup operation without a declared timeout | 120 seconds | `--worker-watchdog-control-seconds` |
| Shared recovery allowance after fencing | 120 seconds | `--worker-watchdog-recovery-seconds` |

These are deployment protection settings, not comparison budgets. They do not
appear in the Agent view. A heartbeat must arrive more often than the configured
tolerance (the client sends every 0.5 seconds). The five-second grace does not
authorize a late benchmark result. It permits ordinary timeout/cleanup handling
before escalating to process termination.

The parent retains its direct child without reaping it until it has signalled
the dedicated process group. This prevents PID/PGID reuse from redirecting a
termination signal. A completed launcher reports its exit status and remains
the group leader until this fence, so leftover Docker clients are also stopped
on the normal path. It flushes stdout and stderr before reporting completion,
including when output is piped or redirected; flush failure cannot report success.
No process identity from a journal or participant is used
for signalling. Only after this fence does recovery attempt the exclusive
journal lock; a held lock still prevents takeover. Recovery verifies the
registered Worker owner and directory identity as well as Docker identities.
Sending the group signal and reaping its leader do not confirm that every
inherited descriptor has closed; bounded lock acquisition runs inside the
supervised recovery helper, never in the watchdog parent.

The parent records the first `SIGTERM` or `SIGINT` as a cancellation request
without raising from the signal handler. It stops supervision and fences its
owned process group even if the signal arrived while the child was being
created. Both handlers remain installed through fencing, journal recovery and
report writing, so repeated signals do not abandon cleanup. The previous
handlers are restored before `supervise()` returns. Nested recovery/report
supervision propagates cancellation back to this handler after its helper is
fenced. Manual recovery likewise defers signals through bounded cleanup and
returns exit code 2 after cancellation. Forced kill or host loss still requires
the separate recovery path described below.

`watchdog-report.json` records the trigger, process exit, completed launcher
status when available, the configured `recovery_timeout_seconds`, and cleanup
result in the private Worker directory.
Report persistence must finish within the remaining recovery allowance. If it
times out, fails, or is skipped because recovery exhausted the allowance or its
process termination is unconfirmed, the watchdog returns exit code 2 and
explicitly says that the current report was not persisted. Any retained file
may describe an earlier attempt; the printed recovery attempt ID can be matched
against `cleanup.process_supervision.attempt_id` in a successfully persisted
report. A failed writer does not cause a synchronous fallback write to the same
storage. Successfully persisted reports contain `report_persisted: true`.
`operator_signal` records the first observed cancellation signal number, or
null when none was received before the report snapshot. Cancellation during
recovery gives exit code 2 even if the launcher had already reported success;
an earlier deadline or watchdog failure remains the primary trigger when
present. Cleanup evidence remains separate from the saved benchmark result.
An abnormal stop or unconfirmed cleanup gives CLI exit code 2. A normally
completed launcher retains its exit status only if recovery succeeds. Existing
Core state, research sealing and TEST decisions are preserved; the watchdog
does not finalize, publish or resume an interrupted episode.

Local regressions deliver real `SIGTERM` and `SIGINT` signals during process
creation, active supervision, fencing and recovery. They verify child reaping,
handler restoration, cleanup reports and preservation of saved Core state with
the test Docker backend. These tests do not certify a live Docker daemon,
allocator-wide cancellation, device isolation, or node-loss recovery.
Recovery regressions also cover cumulative command timeouts, late deletion
responses, elapsed filesystem work, retained handles and retry. Equal-budget
retries with the same simulated per-command cost must advance past persisted
removals through workload and storage cleanup; RAM-only removals and reused
names must retain their verification requirements. A disposable
real subprocess verifies runtime timeout propagation; the resource backend in
these tests remains simulated.
Ownership regressions exercise delayed release of a real inherited lock through
the supervised recovery entry point, persistent contention, non-retryable
identity/storage failures, and shared lock-wait/cleanup/report allowances.
Fresh-process regressions stall journal reading, saving, private-tree removal,
journal close and report writing until killed. They verify bounded return,
ownership-lock retention by surviving runtime clients, retry from durable
progress, cancellation, malformed/missing replies and unchanged Core results.
These fixtures simulate the blocked filesystem operation; they do not certify
real uninterruptible kernel I/O or a particular cluster/storage deployment.

This covers one local POSIX execution host and its bundled Docker Worker, not
a distributed lease/fencing protocol. It does not supply asynchronous Core
operations, preempt arbitrary uninstrumented Python code while its heartbeat
is healthy, turn Track A accounting into a fixed wall-clock cap, or cover
pre-journal readiness resources. Runtime outages, uninterruptible host I/O,
storage loss or unreliable locking can still prevent cleanup. Loss of the
watchdog/host itself requires an external service or later manual recovery;
distributed fencing, power-loss certification and automatic episode
continuation remain separate capabilities. Development trust, best-effort
deadline evidence and leaderboard ineligibility remain unchanged.
