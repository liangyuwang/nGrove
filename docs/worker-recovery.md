# Docker Worker cleanup after process loss

The bundled container launcher records episode-owned runtime resources in a
private Worker journal. Recovery reclaims resources; it does not resume an
episode, retry training/evaluation, change budgets, reopen research, or publish
results. The trusted operator Agent may invoke the same recovery command that
a deployment supervisor invokes. The Tested Agent receives neither the journal
nor recovery authority.

## Launch and recover

After readiness checks complete, `start --agent-runner container` enables
recording before the first episode workload is created. The default directory
is `~/.autotrain-bench/workers/<episode-path-id>`, on the actual Worker host.
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
autotrain-bench worker-recover --state-dir /worker/private/run-state
```

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

`watchdog-report.json` records the trigger, process exit, completed launcher
status when available, and cleanup result in the private Worker directory.
An abnormal stop or unconfirmed cleanup gives CLI exit code 2. A normally
completed launcher retains its exit status only if recovery succeeds. Existing
Core state, research sealing and TEST decisions are preserved; the watchdog
does not finalize, publish or resume an interrupted episode.

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
