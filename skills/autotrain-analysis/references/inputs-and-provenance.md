# Inputs, Admission, and Provenance

Read this reference when locating experiments or validating historical data.

## Method source and project discovery

- Author: `liangyuwang` (Liangyu Wang).
- Primary source: [Research Dynamics and RSI Diagnostics, pinned revision](https://github.com/liangyuwang/AutoTrain-Bench/blob/811d6a559270e4b4ea838aba1224dc330b62fb97/docs/research-dynamics.md). Commit `811d6a559270e4b4ea838aba1224dc330b62fb97`, timestamp `2026-09-17T01:50:42+08:00`.
- Interface baseline recorded in the document: `1868b7e7c9de2d38174f58614b65c0ce18f605a1`, checked on 2026-09-16. The new package, CLI, schemas, and tests are proposals.
- Supporting specifications: the same revision's `docs/benchmark-reference.md`, especially ANALYSIS, Intermediate Incumbents and Research Curves, Evaluation Axes, and Failures; also `benchcore/results.py` and `artifact_identity.py`.
- This skill operationalizes those specifications. Local navigation, configuration, and output-field suggestions are not schemas already implemented by the source document.

Locate the AutoTrain project from the user's working directory or an explicitly supplied path. Do not assume a particular machine, username, remote host, or deployment layout.

| Artifact | Discovery and use |
| --- | --- |
| Benchcore checkout | Read its applicable `AGENTS.md`, `docs/research-dynamics.md`, and specifications matching the experiment revision. A local mirror may not be a Git checkout or the latest version. |
| Episode directories or archive | Locate `state.json` and `events.jsonl` within the user's stated scope. Different directories may represent different campaigns; check each identity. |
| Viewer selection manifest / extracted JSON | Use selection, generation time, and provenance for discovery. These do not replace admission or complete event records. |
| Existing analyses | Useful navigation aids, but do not inherit their conclusions or assume old scripts satisfy this design. |
| Output directory | Use a dedicated user-selected directory or a new `analysis/<date>-<topic>/` directory outside all source episodes and snapshots. Preserve earlier reports. |

This repository contains this skill under `skills/autotrain-analysis/`, a local dashboard under `tools/visualization/`, and an invented worked example under `examples/research-dynamics/`. The example uses a demo-only schema; it is not a production episode loader or evidence about actual Agents. The dashboard's submitted-Candidate plots do not implement this skill's complete incumbent/oracle and admission analysis.

Do not automatically run host-specific refresh, publishing, or experiment-launch scripts. Check Git status before updating documentation; a fast-forward pull is appropriate when synchronization is authorized and safe. Record documentation and experiment revisions separately rather than upgrading historical experiments to a new protocol.

## Admission requires lifecycle evidence and stable files

Record `quality_analysis`, `failure_summary`, or `rejected` for each input, with file/event references supporting the decision.

**Quality analysis** requires irreversibly sealed research, stopped Agent and Research Worker workloads, confirmed control-channel and relevant workload cleanup, no active/pending Lease, and completed result writes, or a consistent snapshot preserving that evidence.

**Failure/abort summaries** may admit certain `dnf` or infrastructure-failure records without `research_sealed_at`, but still require stopped workloads, confirmed relevant cleanup, and stable inputs. Preserve absent sealing fields; do not invent records or unavailable quality values.

**Reject admission** for active episodes, ongoing result writes, unconfirmed termination/cleanup, or inconsistent state and events. List the reason and missing evidence rather than claiming completed quality analysis. One rejected input does not prevent analysis of independent admissible inputs.

These fields and events are inspection entry points, not individually sufficient evidence:

- `research_sealed_at` and `RESEARCH_SEALED`; `closed` means research is sealed while evaluation or publication may still be underway.
- `agent_system.termination_reason`, `agent_control.cleanup_complete`, and `AGENT_CONTROL_CLEANED`, together with deployment-retained termination/cleanup receipts. Core records alone do not prove every external workload has stopped.
- Active Leases, pending startup/teardown, `pending_lease_teardowns`, `deployment_teardown_failures`, and related events, interpreted under the actual source revision.
- `state.json` is the result commit point. When `result_publication` exists, check its publication ID / `prepared_event_seq` against `RESULT_PUBLICATION_PREPARED`. An unreferenced prepared event is not a final result; determine whether the binding is required for that version.
- `operator_stopped` or a failure status can be persisted before cleanup; a status alone cannot establish admission.

Snapshots preserve source episode identity, revision, extraction method, cutoff, termination/cleanup evidence, and freezing/publication evidence. Detect changes to files during reads and validate cross-file references. Matching hashes on two reads or stable modification times demonstrate byte stability during observation, not workload termination. Do not recursively scan or hash hidden-data directories.

## Sources and bindings

| Source | Use |
| --- | --- |
| `state.json` | Identities, resources, results, `candidate_commits`, `incumbent_declarations`, `analysis_trajectory`, and coverage |
| `events.jsonl` | Authoritative ordering and resource lifecycle; event `seq` corresponds to result `event_seq` |
| `inference/inference.jsonl` | Usage, models, and timing when present; do not assume complete prompts or responses |
| Agent/Lease logs and source changes | Optional process evidence. Preserve truncation, timestamp reliability, and exact sources. Read source text without importing or executing a Candidate. |

Commits typically contain `commit_index, candidate_id, event_seq, source_lease, timestamp, elapsed_seconds`. Declarations additionally include `leases_used` and the previous incumbent. A declaration must reference the same Candidate already committed at that point; ANALYSIS must bind to that Candidate and its events.

The same artifact may be committed repeatedly and the same incumbent declared repeatedly. Preserve history while separately reporting commit count, declaration count, effective incumbent changes, and unique-artifact count. Duplicates are not independent quality samples. Never silently overwrite conflicting measurements for the same identity.

`analysis_complete=true` means selected points received a disposition, not necessarily a score. `analysis_selected_points` / `analysis_evaluated_points` count commit-trajectory entries. Also report successful-evaluation coverage over unique Candidates and incumbent-interval coverage.

## Version compatibility

At the source revision, `normalize_result_report` accepts schema v12–v15 but also requires identity algorithms to match the active constants. An accepted version number alone does not establish compatibility.

The Benchcore rename changed Candidate-tree and hidden-bundle identifiers and hash-input prefixes. Source-revision identifiers include `benchcore-tree-v2-sha256` and `benchcore-hidden-bundle-v1-sha256`; historical `autotrain-*` labels cannot simply be replaced.

Record the original schema, identity algorithms, and Core revision before selecting matching-version read-only validation or an explicit adapter. Do not instantiate a Supervisor or enter launch/recovery paths to validate records. Compatibility logic interprets saved records without recomputing hidden digests, changing source files, or reevaluating Candidates. Record the adapter identity and supported combinations. Mark unsupported combinations `unsupported`; extracting a few fields is not equivalent to validation. Compatibility alone does not establish comparability.

## Usage coverage

Distinguish unconfigured collection, a missing file, an empty record, corruption, truncation, and supported zero calls. The source implementation's `summarize_inference_trace` can return zero calls and complete usage for a missing file; do not treat that alone as evidence of completeness.

Gateway and Agent traces may record the same calls. Specify source precedence and deduplicate with reliable association keys. Without such keys, report sources separately instead of summing them. Attribute all tokens of a call spanning a commit/declaration to its trusted completion time, without proportional allocation. When timestamps cannot align with events, report supported cumulative usage only, without invented pointwise token coordinates. Token counts across models do not imply equal computation.
