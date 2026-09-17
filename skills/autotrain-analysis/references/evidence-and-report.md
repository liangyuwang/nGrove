# Process Evidence and Reporting

Read this reference when interpreting RSI, comparing Agents, or producing a full report. These are suggested derived formats for this skill, not official Benchcore schemas.

## Three observation layers

1. **Research feedback:** training logs, participant-created validation, and experimental results the Agent actually saw. Record sources, protocols/data scope, and comparability groups. Do not join observations into one loss curve across incompatible proxy objectives or validation sets.
2. **Independent delivered quality:** saved official ANALYSIS for historical Candidates, providing postmortem held-out observations of research outputs. These values were never returned to the Agent while research remained active.
3. **Mechanisms and reuse:** changes to tools, strategies, code, memory, or context, and relations documenting subsequent use. A Candidate is an output, not the research system's complete state.

## Evidence records

Retain a stable ID, source file and line/event ID, Candidate/experiment association, observed change, subsequent-use location, associated outcome, and `observed / inferred / unknown` designation. Prefer code diffs, tool calls, and execution results. Self-reports do not establish that a change took effect.

Use composable tags rather than uncalibrated numeric levels:

| Tag | Required support |
| --- | --- |
| `reported_change` | A claim that a change occurred, explicitly labeled as a claim |
| `implemented_change` | An actual code/tool/strategy change; if only source is available, state whether execution adopted it |
| `reused_change` | A later call, execution, or decision uses that change |
| `recursive_reuse` | An earlier improvement participates in generating, testing, or selecting a later research mechanism, with event relations and sources |
| `observed_outcome` | An observed result bound to a Candidate/experiment, without automatically implying causal attribution |

Relations use `used_to_generate / used_to_test / used_to_select`, retaining source/target events, evidence references, and observed/inferred/unknown status. For example: improve a validation tool → use it to select a new research strategy → use that strategy to guide a later tool improvement. Repeated tool calls establish reuse, not necessarily recursive reuse.

Each `evidence.jsonl` line may contain one record. These examples illustrate fields only; replace placeholders with actual values in generated reports:

```json
{"record_type":"observation","id":"e1","episode_id":"...","sources":[{"file":"...","line_start":12,"line_end":18,"event_seq":null}],"candidate_id":null,"experiment_id":"...","change":"...","subsequent_use":null,"associated_outcome":null,"tags":["reported_change"],"certainty":"observed","limitations":["Only a self-report is observed; execution evidence is unavailable."]}
```

```json
{"record_type":"relation","id":"r1","source_id":"e1","target_id":"e2","relation":"used_to_test","sources":[{"file":"...","event_seq":42}],"certainty":"inferred","limitations":["The call is visible, but the decision rationale is incomplete."]}
```

Insufficient logs may support no relation records. Observing a claim is not observing its implementation.

## Strength of interpretation

Check whether feedback influenced later research, whether the change ran, whether gains are confounded by continued training/different models/failure repair, and whether earlier mechanisms informed later mechanisms. Allow supported in-context learning and memory reuse; do not speculate about parameter or KV-cache updates.

- **Observed:** actions, values, and relations directly supported by sources.
- **Inferred:** a plausible explanation of later behavior, with concurrent changes and confounders retained.
- **Unknown:** unrecorded state, unattributable usage, missing evaluations, or unestablished dependencies.

Do not output `RSI=true/false`, an aggregate capability score, or causal gains inferred from curves alone. First-Candidate timing, training volume, and fallback-submission habits differ; first-to-last improvement is not automatically learning capability. Missing internal updates do not imply absent self-improvement.

For stronger attribution, suggest controls that freeze particular research mechanisms while retaining task feedback, remove specific memory/feedback, or branch from the same state under equal total budgets. Specify what each control freezes and include improvement overhead. These are future designs, not authorization to resume or launch episodes. Do not ambiguously label different baselines as no RSI.

## Configuration and reproducibility

Save the actual `analysis-config.json` used:

| Setting | Contents |
| --- | --- |
| metric | Name, direction, unit, validity definition, and source |
| budget | Track, primary axis, requested domain, fixed checkpoints/event-cutoff rules |
| targets / cost_slices | User-supplied or predeclared thresholds; use empty lists when unset and skip corresponding metrics |
| missing_data / end_of_run | Unknowns are not zero-filled; incumbents are not substituted; specify holds after normal early finishes and no extension after abnormal termination |
| aggregation | A's integration interval; null for unspecified B aggregation, otherwise checkpoints/positive weights or explicit endpoint rules |
| usage | Source precedence, association keys/deduplication, completion-time attribution, and handling of unalignable timestamps |
| exploratory | Explicitly enabled features; no default fitting, smoothing, significance testing, or convergence models |

Choose checkpoints from the comparison design or budget contract, independently of submission habits. For a single trajectory without suitable fixed checkpoints, report event curves first rather than inventing thresholds to populate metrics.

Use an independent schema for `analysis.json`, such as `autotrain-analysis-report-v1`. Retain per episode:

- Input provenance, cutoff, admission and evidence; original result schema, identity algorithms, Core revision, and compatibility/evidence adapter identities.
- Analysis tool/script identity, hashes of reports/events/logs actually read, complete configuration and its hash. Hash only authorized files actually read; do not scan hidden data or unrelated directories.
- Unmodified official TEST/outcome/policy/eligibility, verified comparison groups, and reasons for incomparability.
- Each metric's value/status/reason, budget unit, domain, point count, unique-artifact count, coverage, and sources. Keep failed/aborted/missing runs in overall denominators and report metric-specific availability separately.
- All warnings, unsupported combinations, log truncation, and usage completeness. Unknown/invalid values are null plus a reason, never NaN or Infinity.

Identical inputs and configuration must produce identical metrics; non-metric metadata such as generation time is excluded. Keep any analysis scripts in the output directory and record the reproduction entry point.

Each `curves.csv` row represents a raw point, checkpoint, or step boundary. Suggested fields: `episode_id, track, axis, budget, event_cutoff, curve_type, candidate_id, source_event_seq, loss, time_per_task, status, reason, held_after_finish`. Add interval ends/raw metrics as needed and document units. Leave unknown numeric cells empty with a reason rather than writing zero.

## Reports and figures

Scale the report to the question rather than mechanically including every metric. A full report usually includes:

1. Main findings and strength of conclusions, separating observations, inferences, and unknowns.
2. Scope, run contracts, revisions/identities, admission/exclusion reasons, and all denominators.
3. Final TEST performance/cost and official status, with comparisons/Pareto plots only within compatible groups.
4. The three ANALYSIS curves, fixed-budget values, coverage, and reliably computable metrics.
5. Sourced feedback → change → reuse → outcome chains, including failures and ineffective changes.
6. Limitations, missing evidence, testable follow-up questions, and reproduction materials.

Use standard plotting libraries for static PNG/SVG figures. Draw incumbent quality as right-continuous steps, submitted observations as raw points, and the oracle as an explicitly postmortem reference. Do not connect unknown intervals into an apparently observed process. Use separate panels for Tracks and label budget units, metric direction, missingness, and held-after-finish regions. With one point, show the point and coverage rather than inventing a curve or curvature.

Align process annotations through trusted events; use an ordered evidence table when timing cannot be aligned. Label TEST and ANALYSIS separately. Separate research expenditure and Candidate serving cost into different plots/columns. Online feedback uses a separate panel with its protocol identified, never a combined training curve with hidden quality.

Link actual source files/lines and figure paths. Keep reports and evidence private under ANALYSIS visibility rules. Auditability does not require copying complete sensitive logs into the report.
