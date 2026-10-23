# Research recording

Every run records research evidence on the trusted Operator side. The existing
`events.jsonl` is the event journal; large request/response values are stored by
SHA-256 under `recording/payloads/`. Candidate bundles remain in `candidates/`.
This records observable interactions, not a complete Agent state or a replayable
session. It does not implement branching, parallel scheduling, or checkout.

## Role in the RSI project

The project's [scope](research-scope.md) includes improving how research systems
find better research paths. Recording is the implemented interface connecting
autonomous research experience to future meta-optimization methods. It makes
observable actions, response and execution evidence, artifact history, and
optional runner observations available to later consumers.

Those consumers may study or improve research strategies, tools, memory or model
parameters. Their update methods and recursive orchestration remain future work
within project scope. The current interface captures and exports evidence; it
does not select an update method or establish that self-improvement occurred.
Consumers must retain capture gaps, evidence provenance and the separation
between research-visible feedback and private postmortem measurements described
below.

## Launch configuration

Pass `--recording recording.json` to `ngrove start`. Omission uses these defaults:

```json
{
  "schema_version": 1,
  "capture_payloads": true,
  "max_payload_bytes": 4194304,
  "max_total_payload_bytes": 67108864,
  "artifacts": {
    "candidates": "all",
    "lease_outputs": "all"
  }
}
```

The configuration is resolved before launch and frozen with its own fingerprint
in the private result's `recording` field. It is independent of the compute/time
configuration. Record both fingerprints when
comparing experiments. Trusted launchers may pass a `RecordingConfig` directly
to `Supervisor`.

Payload limits apply to the new content-addressed JSON store, including adapter
observations. Deduplicated objects count once. They do not limit the authoritative
event journal, existing command/Agent logs, Candidate bundles, or total episode
storage. The existing Operator storage limit remains the overall storage control.
Disabling payload capture preserves events and payload identity/size, with an
explicit `not_captured` status; existing audit and command logging still operate.
Failed payload writes are removed before a later capture can retry. If cleanup
fails, the full attempted payload size remains reserved against the total limit
until cleanup succeeds; the payload remains marked `capture_failed`.

## Artifact retention

| Setting | Effect after result finalization |
| --- | --- |
| `candidates: all` | Keep every unique committed Candidate bundle; the default. |
| `candidates: final` | Keep only the final declared Candidate bundle. |
| `candidates: metadata` | Remove committed bundles, preserving identities, event history and evaluation results. |
| `lease_outputs: all` | Keep original Lease output directories; the default. |
| `lease_outputs: discard_after_finalize` | Remove stopped Lease output directories, including original copies of committed bundles and uncommitted output. Keep Lease command logs. |

For a small archival footprint, select `candidates: final` or `metadata` together
with `lease_outputs: discard_after_finalize`. Otherwise original model files may
still exist in Lease outputs. Candidate policies apply to entire bundles,
including their code, configuration and weights; they do not classify files by
extension. A metadata-only bundle cannot later be loaded for a new experiment.
Workspace files, raw Agent logs, inference logs and evaluation-owned storage are
outside these retention policies. Uncommitted checkpoints are preserved only as
Lease output when that output is retained; they are not automatically registered
as independent artifacts.

All committed Candidates remain available until TEST and the selected ANALYSIS
work have completed and the terminal result has been persisted. Retention does
not run on live, aborted or interrupted episodes. It never changes selection,
evaluation or result validity. Removal intent, completion and failures are
recorded independently. A failed removal retains an explicit failure status;
trusted launchers can retry `supervisor.finalize_recording()` after finalization.
An unresolved removal intent is not proof that content was removed.
If deletion succeeds but its completion record fails, a retry on the same
Supervisor can confirm that the target is absent. It appends
`ARTIFACT_ABSENCE_CONFIRMED`, linked to the persisted removal intent, and marks
the artifact `absence_confirmed`. This resolves the current retention failure
without claiming how or when the original deletion completed; earlier events
remain unchanged. Missing targets without a prior persisted removal intent are
not reconciled this way. A failed confirmation write remains retryable, and
repeated successful retries do not add duplicate events.
Recorded deployment teardown failures defer automatic removal even when a TEST
result could be finalized. Such episodes retain their artifacts for Operator
recovery; historical failure records are not silently cleared to enable deletion.

## Events and links

Every Core event has an Episode-scoped stable `event_id`, a source, sequence,
timestamp and elapsed time. `seq` is journal order, not proof of causal dependence.
`phase` distinguishes observations before and after research sealing; it does
not imply that a measurement was available to the Agent.

- `ACTION_STARTED` records a request at the semantic handler boundary.
- `ACTION_RETURNED` records the handler's success/error response. Delivery is
  unknown at this boundary.
- `CONTROL_RESPONSE` records the response a participating transport actually
  published or failed to publish. The local mailbox records its bounded response,
  including a `response_too_large` replacement. Publication does not establish
  consumption: `consumption` remains `unknown`.
- `LEASE_COMMAND_STARTED`, `LEASE_COMMAND_FINISHED` and `LEASE_COMMAND_FAILED`
  share an `execution_id`; they also retain any surrounding `action_id`.
- `CANDIDATE_COMMITTED` associates the commit with an immutable bundle ID,
  source Lease and storage location. Repeated commits remain separate events
  even when they reference the same artifact.
- Retention events update the artifact's recorded storage disposition without
  deleting the original commit evidence.

`links` contains explicit `{relation, event_id}` references. Multiple inputs and
forward references are allowed. Request identity, execution identity and artifact
identity are separate; there is no single global current-experiment pointer.
Concurrent observations receive independent action IDs and serialized journal
positions. This permits later graph views and shared-history references without
implying that the current execution scheduler supports parallel Leases.

Payload references carry `status`, and when encoding succeeded, `sha256` and
`size_bytes`. A saved reference additionally has a relative path. The digest
identifies the canonical JSON value, not the transport's whitespace/byte encoding.
Size omission, total-store omission, capture failure and invalid JSON are explicit.
Worker stdout/stderr truncation flags remain separate from recording-store limits.

## Runner observations

Core cannot observe arbitrary Agent-internal tools or every workspace edit.
A trusted runner can supply additional evidence through:

```python
event_id = supervisor.record_observation(
    source="my-runner",
    source_event_id="tool-call-42",
    kind="code_change",
    data={"file": "train.py", "diff": "..."},
    links=({"relation": "responds_to", "event_id": earlier_event_id},),
)
```

This hook is not an Agent API operation and is excluded from the public API
bundle. It accepts JSON evidence, not paths to read or commands to execute.
Observations are labeled `adapter_reported`: the adapter is responsible for
source identity, capture coverage, accurate relationships, and keeping hidden
information outside Agent feedback. Source event IDs are retained for association;
the journal does not infer dependencies or deduplicate adapter reports by them.
There is no mandatory experiment-registration protocol for the Agent.

## Read-only export

```sh
ngrove trajectory --episode /trusted/runs/episode
```

The command validates the Core result and recording configuration, reads events,
verifies payload identities, and returns events, payloads, artifacts, commits,
incumbent declarations, usage evidence and a separate postmortem section.
Candidate `event_seq` values reference the journal's `seq`; `candidate_id` binds
commits to ANALYSIS results. The reader reports malformed/gapped events, open
actions/executions, unresolved references, missing/corrupt payloads and missing
Candidate bundles. Artifact metadata distinguishes recorded retention from
observed file presence. Reading does not rehash large Candidate bundles.
The reader also checks the commit, incumbent and result-publication event
references recorded in state, including missing journal tails. State-confirmed
Candidates remain in the export when their events are missing; absent storage
evidence is marked `unknown`, and their actual file presence is still checked.

An empty `issues` array means no listed inconsistency was detected, not that all
Agent behavior was captured. The export declares its scope; runner observations
are optional, and inference usage does not imply full prompts or responses.
A live export is an observation of available files, not an atomic snapshot.

The complete export is private Operator evidence. Core events may contain private
infrastructure details, and the postmortem section may contain hidden evaluation
measurements. `agent_exchange` marks captured request/response values; it does not
authorize exporting the complete journal to an Agent. ANALYSIS/TEST remain absent
from online feedback until the existing isolation and sealing rules permit their
Operator-side evaluation. Future training-data or participant-facing exports must
select the appropriate information boundary explicitly.
