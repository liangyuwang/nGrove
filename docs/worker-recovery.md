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

## Supported scope

This implementation covers one local POSIX Docker Worker session after journal
activation: Agent and inference Gateway containers, Training containers,
Candidate/scorer containers, prediction holders/volumes, created networks and
Worker-private hidden snapshots. Readiness probes completed before activation,
standalone `prepare`, `host_cli`, external scheduler allocations and resources
created through a custom Worker are not certified by this journal. Direct
DockerRuntime integrations must call `enable_recovery` before their workloads
and retain ownership until all Worker calls stop.

Recovery does not run automatically after process loss. A trusted operator or
deployment service invokes it. Runtime outages, a hung creation client, storage
loss or unreliable locking may leave recovery unconfirmed. An independent
watchdog, distributed fencing, power-loss certification and automatic episode
continuation remain separate capabilities. Development trust, best-effort
deadlines and leaderboard ineligibility remain unchanged.
