# Trusted Operator status

`ngrove status` presents execution observations alongside the last
validated Core snapshot. It is a read-only Operator command. It does not
control workloads, query a scheduler, recover resources, or publish an episode.
Its module is excluded from the generated Agent API, and it does not change
the Agent, Worker, or result protocols.

```sh
ngrove status --episode /trusted/runs/episode
ngrove status --episode /trusted/runs/episode \
  --observation /trusted/monitor/observation.json
ngrove status --episode /trusted/runs/episode \
  --observation /trusted/monitor/observation.json --json
```

The existing `report --episode ...` continues to return the full Core audit
report in its original JSON format. Both commands are trusted Operator views,
not publication-safe or participant-facing exports.

## Sources and interpretation

The view reads `state.json` directly and validates it using the existing result
schema. It does not infer final results from `progress.json`, a launcher marker,
or `RESULT_PUBLICATION_PREPARED`. Only the authoritative state can establish a
terminal episode outcome. An old finalized state remains a valid historical
result; its age does not undo success or ANALYSIS completion.

Core file modification time describes that snapshot, not process liveness or
the last research advance. A long Worker call may leave the snapshot unchanged
while execution continues. Conversely, a recent copy of an `active` snapshot
does not prove that any workload is still running. Lease and postmortem phase
fields are explicitly labeled as snapshot facts. Lease counts are starts,
not a research completion percentage or an estimate of actual GPU activity.

Current execution requires an optional, trusted deployment observation with
the matching episode ID and a recent `observed_at`. The default maximum age is
60 seconds; `--max-age-seconds SECONDS` sets a finite positive threshold suited
to the deployment's checking frequency. This is a display threshold, not a
deadline or a watchdog action. Future-dated observations are not used to assert
execution or cleanup. Deployment clocks must be synchronized with the reader.
The observation file's mtime is recorded but never refreshes `observed_at`.

| Evidence | Display interpretation |
| --- | --- |
| Core `active`, no fresh bound observation | Current execution unknown; retain the Core snapshot |
| Old Core `active`, fresh deployment `running` | Execution running; Core snapshot stale |
| Nonterminal Core, fresh deployment `terminated` | Execution interrupted; Core has no terminal outcome |
| Core `finalized`, old progress still `closed` | Retain the finalized outcome and actual ANALYSIS disposition |
| Terminal Core, deployment still `running` | Retain the outcome and separately show the still-running execution boundary |
| Missing, unreadable, malformed, mismatched, or stale observation | Current execution and deployment cleanup unknown; retain valid Core facts |
| Missing or invalid Core snapshot | Core unavailable; an observation is unbound until the episode identity can be checked |

Sources are read independently. A failed read is shown with its source and
error and is not classified as an Agent failure, a watchdog timeout, or a
successful cleanup. Each invocation reads again; it neither caches failed
observations nor updates historical episode files. The command returns zero
when it produces a view, including an unknown/interrupted view; this exit code
does not mean episode success. Automation must inspect the structured
`summary`, `execution_state`, source availability/freshness and Core outcome.

## Deployment observation format

An external monitor translates its execution system's actual observations into
this small, platform-independent format. It must be stored outside the Agent
workspace and remain writable only by the trusted deployment/operator.

```json
{
  "schema_version": 1,
  "episode_id": "episode:example",
  "source": "deployment monitor",
  "observed_at": "2026-09-10T12:00:00+00:00",
  "execution_state": "terminated",
  "workload_cleanup": "unconfirmed",
  "storage_cleanup": "confirmed",
  "detail": "Execution ended; storage archive verified and working volume removed."
}
```

The required fields are `schema_version` (integer 1), `episode_id`, `source`,
`observed_at` (ISO 8601 with timezone), and `execution_state`. Execution values
are `queued`, `running`, `terminated`, or `unknown`. This describes the
deployment's episode execution boundary, not an assertion that training is
active. A scheduler allocation alone cannot prove that the Agent initialized
successfully or that a GPU is training.

Optional `workload_cleanup` and `storage_cleanup` each accept `confirmed`,
`unconfirmed`, or `unknown`, defaulting to `unknown`. They are independent:
confirmed storage disposal does not establish workload termination, and an
ended allocation does not establish artifact archival or storage recovery.
Core Agent-control cleanup and pending Lease teardown facts are shown
separately. Stale observation values remain visible as historical observations,
while the current cleanup fields become unknown. Historical teardown failures
remain visible even if a later deployment observation confirms cleanup.
Optional `detail` supplies a concise
trusted explanation. Unknown fields are rejected to catch mapping mistakes.

Observation schema version 1 belongs only to this Operator view. It is neither
a `ConformanceReceipt` nor execution attestation, and never changes policy
compliance or leaderboard eligibility.

## Producer responsibilities

- Set `observed_at` to the actual check time. Copying an old snapshot or retrying
  a failed check must not make old execution/cleanup evidence fresh. Publish a
  coherent observation; use `unknown` for facts that could not be checked.
- Keep source-specific monitoring failures independent. A scheduler query
  failure must not prevent reading episode evidence, and vice versa. Record
  the failure rather than inventing a new execution state.
- Write through a temporary file followed by atomic replacement. A consumer
  can still see a transient read failure and must treat it as unavailable
  evidence, not as authorization to terminate a workload.
- After `publish_postmortem_result()` succeeds, refresh any deployment-owned
  progress summary from the final authoritative state before stopping its
  snapshot publisher or unmounting episode storage. The actual Core field
  names are `closure_reason` and `research_sealed_at`, not `research_closure`.
  Serialize periodic and final writes so an older in-flight snapshot cannot
  overwrite the final refresh; join the publisher before detaching storage.
- After external cancellation, preserve the interrupted Core state. Record
  deployment termination and cleanup evidence separately; do not manufacture
  a finalized result, commit a leftover checkpoint, or restart research.

Existing site-specific `watch.json`, `progress.json`, and scheduler records are
not implicitly compatible with this schema. Their producers must supply the
mapping in the external deployment. Core does not add site-specific parsers or
silently interpret an arbitrary monitor file as authoritative evidence.
See [Worker integration](worker-integration.md) for the publisher's place in
the launch/cleanup sequence and focused deployment verification.
