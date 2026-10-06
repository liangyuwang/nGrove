# AutoTrain-Bench: Research Dynamics and RSI Diagnostics

> **Status: v1 design proposal; not implemented.** This document specifies an
> independent, read-only postmortem analysis layer. The `bench_analysis/` package,
> analysis command, derived schemas, and acceptance tests are proposed additions.
> This documentation change does not implement them, change benchmark rules, or
> authorize an episode launch.
>
> Repository interfaces were checked against revision
> `1868b7e7c9de2d38174f58614b65c0ce18f605a1` on 2026-09-16. The sections below
> distinguish existing interfaces, proposed behavior, and statistical conventions
> that remain to be specified.

## 1. Development objective

The goal of v1 is to add an independent, read-only postmortem layer for research
dynamics and recursive self-improvement (RSI) diagnostics to existing episodes.

The central questions are:

> Does the Agent use experience gained during research and autonomously generated
> improvements to keep improving what it delivers within a limited budget?
> What evidence suggests that earlier improvements enhanced its ability to
> achieve later improvements?

The first version delivers reproducible curves, metrics, evidence links, and
data-quality reports. It does not prescribe a single RSI score or require a
complete account of internal update mechanisms before implementation can begin.

Metric interpretation and baseline calibration can develop through analysis of
real runs. They need not block engineering. This layer measures existing research
records and leaves Envcore's supervision and evaluation responsibilities intact.

## 2. Intuition: from training loss to a researcher's learning curve

### 2.1 Start with ordinary model training

In ordinary model training, a model uses training data to calculate loss and
update its parameters. Recording training loss at successive checkpoints gives
us a curve against training steps or compute budget.

The curve describes more than the final result: how loss falls, how much compute
the next improvement requires, how the improvement rate changes, and where
progress plateaus or reverses.

AutoTrain-Bench can use the same intuition: **treat the Tested Agent's autonomous
research as a higher-level learning process and examine how experience changes
the quality of its delivered results.**

### 2.2 Research interactions as learning experience

In ordinary training, data and feedback contribute to parameter updates. In
autonomous research, the Agent proposes ideas, runs experiments, reads results,
and uses that experience to guide subsequent research.

For example, a poor training configuration may prompt a change in direction.
An unreliable validation method may prompt an improvement to comparison tools.
Those tools and accumulated experience may then inform the next experiments.

This process can be written as:

```text
S_(k+1) = U(S_k, τ_k)
M_k = Readout(S_k)
```

Here, `S_k` is the research system's complete state at observation point k;
`τ_k` is the research interaction and experience between observations; `U` is
the update process; and `M_k` is the Candidate submitted at that point.

**An update need not be a visible parameter gradient.** It may be carried by
parameters, the harness, tools, research strategy, memory, or context state.
We first observe the process's inputs, artifacts, and subsequent performance
without prescribing its internal implementation.

At this outer, researcher level, the losses and experimental results the Agent
uses to guide later actions are learning signals. A value called validation loss
at the target-model level is still training feedback at the researcher level if
the Agent sees it and uses it to change its research.

A Candidate is an evaluable product of research, not a snapshot of the
researcher's complete state. Many internal updates may occur between commits.
Sequence numbers describe order; comparing improvement efficiency also requires
actual time, training resources, and inference usage.

### 2.3 Why a training curve also needs an independent validation curve

The next question is familiar from ordinary training: **if the signals used for
learning improve, do delivered results also improve under an independent
standard?**

Looking only at the Agent's own research losses measures performance against its
chosen feedback criteria. Agents and research stages may use different data
splits, evaluation methods, or proxy objectives. Those values cannot necessarily
be joined into one comparable curve.

Alongside research feedback, we therefore need observations that do not
participate in updates and use a consistent evaluation standard: independent
evaluation of Candidates from different research stages.

This resembles recording validation loss for model checkpoints. Here, however,
the evaluator measures the Candidates delivered by the research system. It
observes the research loop's external results rather than directly measuring a
loss on the researcher's own parameters.

```text
Research feedback → update the research system → produce later Candidates
Candidates at observation points → independent evaluation → validation loss curve
```

**The training side records the experience the system used; the validation side
measures the results associated with that experience and those updates.**
Validation values need not be returned to the Agent and supply no update gradient.

Research need not have a single differentiable training loss. Comparable online
losses can be retained as separate feedback curves. Other experimental results
and interactions remain learning signals; they must not be forced into one
scalar or joined across incompatible evaluation protocols.

### 2.4 Mapping this view to ANALYSIS

The repository's existing postmortem ANALYSIS serves this outer validation role.
After research is permanently sealed, it evaluates the historical Candidates
selected under the release's rules using the same hidden data and evaluation
protocol.

```text
L_val,k = L_ANALYSIS(M_k)
```

Aligning these measurements with commits and resource consumption produces a
postmortem held-out loss curve for the research system's outputs. The current
LM task can use `mean_nll` as its loss.

Together, this curve and the research-feedback trajectory describe how the Agent
used experience and whether those experiences and updates were accompanied by
independently measurable improvements in what it delivered.

The implementation retains the name ANALYSIS. Documentation explains its outer
validation role without conflating it with an Agent-created validation set.
ANALYSIS never enters the research feedback loop. TEST continues to evaluate
only the final declared incumbent; no historical TEST curve is reconstructed.

### 2.5 Implications for the analysis layer

The layer connects research experience, resource expenditure, and delivered
quality. Logs and traces preserve feedback and subsequent use; held-out curves
show changes in results; slopes, curvature, and convergence diagnostics describe
the rate and efficiency of those changes.

Research feedback and postmortem validation values must be stored and labeled
separately. Feedback records identify their source, evaluation protocol, and
scope of comparability. Postmortem measurements retain official evaluation
semantics. They must never be presented as feedback the Agent saw during research.

| Layer | Observations | Question it addresses |
| --- | --- | --- |
| Research feedback | Experimental results, logs, and validation metrics actually visible to the Agent | What experience did it receive? |
| Independent delivered quality | Saved official ANALYSIS results for historical Candidates | Did the submitted solutions improve? |
| Mechanism and reuse evidence | Changes to tools, strategies, memory, and their subsequent use | Did earlier changes participate in later improvements? |

Together, these observations support analysis of task-local self-improvement and
recursive dependencies. Curves describe end-to-end improvement; no particular
curvature is assumed to be a unique signature of RSI or a direct measurement of
internal researcher capability.

## 3. Boundaries that must remain unchanged

### 3.1 Input admission and consistency

Quality analysis accepts only episodes whose research is sealed, whose research
cleanup is confirmed, and whose results are no longer being written, or
consistent snapshots of those episodes. Reject active episodes. Copying an
active episode's files does not make it eligible for analysis.

A state string or file modification time alone does not establish stable input.
In the current implementation:

- `closed` means research is sealed; TEST, ANALYSIS, and result publication may
  still be pending or running.
- `operator_stopped` and failure states may be persisted before cleanup finishes.
- Ordinary `dnf` and some infrastructure-failure paths do not necessarily populate
  `research_sealed_at`.
- Termination records, control-channel cleanup, active or pending Leases, and
  result-publication evidence must be checked together.

Quality analysis and failure summaries therefore need distinct admission rules.
Failed or operator-stopped episodes may produce coverage and failure summaries
after workloads have stopped, relevant cleanup is confirmed, and inputs are
stable. Do not fabricate quality metrics. Preserve the absence of a formal
sealing field rather than inventing a sealing record. Unconfirmed termination or
cleanup fails admission; report that separately from missing quality measurements.

A consistent snapshot must preserve episode identity, provenance, its cutoff,
and evidence of termination, cleanup, and freezing. The analyzer checks the
consistency of the files and referenced events and detects changes during reads.
Stable modification times or two matching hashes alone do not prove workloads
have stopped. The specific snapshot-evidence format belongs to the analysis
layer's input contract and need not change the core result schema.

### 3.2 Read-only operation and visibility

Read only saved official evaluation results. Never rerun a Candidate, scorer, or
command found in a log, and never load hidden data. Identity checks use existing
identity records and trusted snapshot evidence; they must not read hidden content
to recompute its digest.

Do not change incumbents, TEST, leaderboards, resource accounting, visibility
policy, or original episode files. Write all derived outputs to a separate
directory outside the source episode or snapshot, with no overlap.

Derived reports inherit ANALYSIS's organizer-only or delayed-publication
restrictions. Generating a report does not authorize its publication.

Treat raw logs as untrusted and potentially sensitive. Keep them private by
default and do not send their contents to external models. Log content must not
cross execution, credential, privacy, or instruction boundaries. Paths, links,
and commands in logs are data, not authorization to access files or networks or
to execute code.

## 4. Existing interfaces and integration points

### 4.1 Existing data sources

The following interfaces were checked at the revision noted above. New modules
and commands described here are proposals, not existing functionality.

| Interface | Purpose |
| --- | --- |
| [`envcore/results.py::normalize_result_report`](../envcore/results.py) | Validate results against their declared schema and active artifact identity algorithms. The current core schema is v16. |
| [`envcore/cli.py::command_report`](../envcore/cli.py) | Reference for read-only reporting; preserve its output contract. |
| `state.json` | Read `candidate_commits`, `incumbent_declarations`, `analysis_trajectory`, identities, resources, status, and coverage fields. |
| `events.jsonl` | Authoritative event ordering and resource lifecycle. |
| [`envcore/supervisor.py::_analysis_trajectory`](../envcore/supervisor.py) | Existing links among commits, ANALYSIS metrics, costs, and declaration events. |
| `inference/inference.jsonl` | Call times, models, usage, and completeness when available; do not assume full prompts or responses are present. |
| `agent.log`, Lease logs, and other Operator-retained traces | Optional tool-call, experimental-feedback, code-change, and behavioral evidence. Host runners may also use `agent.stdout.log` and `agent.stderr.log`. |
| [`challenges/lm-pretrain/evaluator/evaluator.py`](../challenges/lm-pretrain/evaluator/evaluator.py) | The current loss is `analysis_raw_metrics.mean_nll`, with lower values better. The analyzer does not execute this scorer. |

Commits contain:

```text
commit_index
candidate_id
event_seq
source_lease
timestamp
elapsed_seconds
```

Declarations additionally include `leases_used`, the previous incumbent, and
other fields.

Bind metrics by `candidate_id` and order events by `event_seq`; do not infer
identity or ordering from log text or timestamps alone. The corresponding field
in the event stream is `seq`. Validate commit and declaration references on load.

The same artifact may be committed repeatedly, and the same Candidate may be
declared repeatedly. Preserve event history without treating these events as
independent quality samples.

`analysis_complete=true` does not mean every point was successfully evaluated.
Check each point's status. Existing `analysis_selected_points` and
`analysis_evaluated_points` count commit-trajectory entries; the analysis layer
should also report quality coverage over unique Candidates.

### 4.2 Independent package and result validation

Prefer a separate `bench_analysis/` package that reuses the result normalizer.
Supervisor, Worker, and Evaluator must not depend on the analysis package.

Use a separate CLI and independent schemas for derived files. This feature does
not require a core result-schema or Worker Protocol version change.

The normalizer validates the declared result schema and requires artifact
identity algorithms to match its active constants. The analysis layer should
use these checks without rewriting source records, rerunning Candidates or
reading hidden data. Schema validation alone does not make runs comparable;
retain the task, resource and artifact identities needed for each comparison.

## 5. Curves, budget coordinates, and missing data

### 5.1 Three curves

Let `C(b)` be the set of Candidates committed by budget b and `J(b)` the last
incumbent declared by that point. When events share a budget coordinate, use
`event_seq` to determine the cutoff.

The **submitted curve** displays each Candidate's:

```text
budget
ANALYSIS loss
Candidate serving cost
```

Preserve raw observations; improvement need not be monotonic.

The **incumbent curve** is:

```text
I(b) = L_ANALYSIS(J(b))
```

This is the primary delivered-quality curve. It measures the combined outcome
of producing solutions and recognizing and selecting among them. Regressions
are allowed.

The **observed oracle best curve** is:

```text
O(b) = min { L_ANALYSIS(M) : M ∈ C(b), M was successfully evaluated }
```

Label it as a postmortem reference, not the Agent's actual selection. Its
monotonicity follows from its definition and is not evidence of learning.
With missing evaluations, it need not identify the best of all committed
Candidates. If no Candidate has a successful evaluation, the oracle value is
unknown; do not output infinity as a metric.

### 5.2 Curve reconstruction

Reconstruct a right-continuous step function from declaration events:

- A Candidate affects the incumbent curve only after it is declared, not from
  its earlier commit time.
- Do not backfill past states using future declarations or future Candidates.
- Repeated declarations with no change do not add effective curve samples.
- Preserve the history of declarations at the same budget coordinate; a
  checkpoint uses the last declaration before its event cutoff.

Attaching a postmortem ANALYSIS value to a Candidate that was already committed
and declared at that point is the intended retrospective measurement. It does
not imply that the Agent knew that value during research.

### 5.3 Budget axes and two distinct costs

Track A uses the Supervisor's recorded `elapsed_seconds` as its primary axis.
Preserve the existing benchmark-owned overhead deductions; do not reconstruct
charged time from UTC timestamp differences.

Track B uses consumed Lease count. Leases that failed after consumption still
count. At checkpoint n, take the last declaration before Lease n+1 was consumed;
the final available point uses the research seal. Reconstruct this from events
and declarations' `leases_used`, not `commit_index` or the numeric suffix of a
Lease ID. Distinguish a startup that was never admitted from failure after a
Lease was consumed.

Time and tokens can be auxiliary axes. Candidate indices show research order
only.

A token axis must identify its accounting source and attribution rule. If a call
spans a commit or declaration, v1 attributes its tokens at call completion and
labels this limitation. Do not estimate proportional token consumption. If
completion times are untrusted or cannot be aligned with research events, do
not invent pointwise token coordinates; report only supported cumulative usage
or unknown values.

Name and report the two costs separately:

| Cost | Examples | Purpose |
| --- | --- | --- |
| Research expenditure | Agent inference usage, research time, training resources | Measure the budget and efficiency of producing subsequent results. |
| Candidate execution cost | ANALYSIS `time_per_task` and its submetrics | Describe the submitted system's quality–cost tradeoff. |

Candidate serving cost is not research-Agent token consumption. Avoid using an
unqualified “inference cost” for both in documentation or outputs.

### 5.4 Missing data and cutoffs

Define fixed checkpoints and analysis intervals in configuration before analysis.
Do not vary them with each Agent's submission habits.

Quality is unknown when no incumbent has been declared, the current incumbent
lacks ANALYSIS, or the data is corrupt:

- Do not fill it with zero.
- Do not silently substitute an earlier scorable Candidate.
- Carry a value forward only when no new declaration occurred and the existing
  incumbent's quality is known.

Order events with no budget difference by `event_seq`. Do not invent official
quality observations for unsubmitted experiments.

Whether to hold the final incumbent through the full budget after a normal early
finish must be an explicit setting, labeled `held-after-finish`. Such a hold is
not continued research. Do not extend abnormal terminations to the budget limit.

Postmortem checkpoints within one episode are not equivalent to separate runs
performed under different budgets.

## 6. Metrics: stable and exploratory layers

### 6.1 Stable v1 metrics

Report incumbent loss at fixed budgets, budget to reach predefined quality
targets, interval improvement rates, selection gaps, and coverage. The formulas
below assume a loss to minimize; other metric directions must be declared.

#### Budget to reach a fixed target

```text
H(ℓ) = inf { b : I(b) <= ℓ }
```

Compute this over a predefined observation domain. Record an unreached target as
`not_reached`, together with the observed budget. Never substitute the cutoff
budget for an attainment budget.

With incomplete coverage, distinguish `unknown` from `not_reached`. Do not
interpolate an earlier success. Distinguish three cases:

- The target is known to have been reached, with no earlier unknown incumbent
  interval that could conceal an earlier success: report the exact first
  attainment budget.
- The target is known to have been reached, but earlier incumbent quality is
  unknown: report the first **observed** attainment budget. Exact `H(ℓ)` is
  `null` with a reason. The observed success gives an upper bound on attainment
  budget, not an exact first attainment.
- No success has been observed: report `unknown` if a quality gap could conceal
  one. Use `not_reached` only when the evidence rules out attainment within the
  observation domain.

For example, if incumbent quality is unknown at budget 2 and known to meet the
target at budget 4, the supported claim is “reached by 4,” not “first reached at
4.” Before any incumbent is declared, there is no declared deliverable and no
known attainment; that period remains a coverage gap for average-quality metrics.

#### Mean incumbent loss over an interval

For a predefined continuous budget interval `[b0, B]`:

```text
mean_incumbent_loss = integral(I(b), b0, B) / (B - b0)
```

Integrate the step intervals exactly. Report a comparable mean only when quality
coverage spans the entire interval. Otherwise report the integral over known
intervals, coverage, and `insufficient_coverage`. Do not silently renormalize over
the remaining intervals or begin integration at the first submission.

If `b0=0` and there is initially no incumbent, the full-interval mean may be
unavailable. This is a coverage result, not a reason to move the analysis start
automatically. Require `B > b0`.

**Track B's discrete aggregation convention must be specified before that metric
is implemented.** Lease checkpoints are discrete and have no inherent time
interval length. One option is a weighted mean over fixed checkpoints:

```text
checkpoint_mean = sum(w_n * I(n)) / sum(w_n)
```

Freeze the checkpoint set and positive weights `w_n` in configuration. Report
the mean only with complete quality coverage over those checkpoints. An
alternative is to declare a step extension and integration rule on the Lease
axis, with explicit endpoint treatment and weight for the final checkpoint.
Do not present these different statistics as the same mean.

This proposal selects no implicit default. Until the convention is settled,
Track B can still report checkpoint values, target attainment, selection gaps,
and coverage. It must not silently inherit Track A's interpretation of an
integral.

#### Postmortem selection gap in the quality dimension

```text
loss_only_selection_gap(b) = I(b) - O(b)
```

Label this as a loss-only gap, not overall decision regret. A lower-loss
Candidate may serve more slowly, so retain Candidate serving cost and Pareto
information. If evaluations are missing, state that the oracle covers only
successfully evaluated Candidates. If either operand is unknown, do not report
a numeric gap.

For cost-constrained slices, freeze the threshold in configuration. If the
actual incumbent violates the cost constraint, mark that slice as not applicable;
do not select a different Candidate on the Agent's behalf. Record the slice's
eligible oracle set explicitly and exclude Candidates that violate its cost
constraint.

#### Metric scale

Use raw `mean_nll` as the primary loss. Do not use the clipped 0–100 score to
calculate convergence shape.

Future tasks declare metric name, direction, unit, and validity rules. Do not
hard-code LM assumptions in the generic analysis layer.

### 6.2 Exploratory v1 dynamics

Secant improvement rates and second divided differences for unequal budget
intervals may be implemented, explicitly labeled `exploratory`:

```text
r_i = (L_i - L_(i+1)) / (b_(i+1) - b_i)

a_i = 2 * (r_i - r_(i-1)) / (b_(i+1) - b_(i-1))
```

`r_i` is the interval's quality-improvement rate. `a_i` is a discrete proxy for
change in that rate, not an internal gradient. A positive `a_i` describes an
increasing improvement rate in this observation sequence; it does not establish
recursive self-improvement.

Identify whether the observations are submitted or incumbent points. Do not mix
the two.

- Do not take second derivatives directly on a forward-filled step-function grid.
- Do not treat repeated declarations as new samples.
- Preserve original points; do not select windows adaptively to produce
  “acceleration.”
- Budget coordinates must be strictly increasing for differences. Events at the
  same budget retain their order but cannot produce a rate by division by zero.
- Missing points are not valid observations. If an endpoint difference spans a
  gap, label the intervening process as unknown.

Every metric must include point count, unique-artifact count, budget span,
missingness, and time intervals. At least three valid points at distinct budget
coordinates are required for a second divided difference. This is only the
mathematical minimum, not evidence of statistical reliability.

Fitting, significance testing, and complex smoothing are not default v1 features.

### 6.3 Optional convergence models for later research

The following is an optional model interface, not an assumption about real data:

```text
e(b) = L(b) - L_star > 0
eta(b) = -d log(e(b)) / db
```

Under the assumption:

```text
e' = -eta * e
```

it follows that:

```text
L'' = (eta^2 - eta') * e
```

Thus raw loss may flatten even as normalized convergence efficiency eta
increases. Eta is a descriptive quantity under this model, not an independently
measured researcher capability.

Do not treat each run's lowest observed loss as the true `L_star`. Do not report
this metric without external justification for the lower limit. Lower-limit
selection, model families, sensitivity analysis, and sample requirements remain
subjects for later calibration.

Finite trajectories support empirical descriptions of convergence, not proofs
of asymptotic convergence or unbounded self-acceleration.

## 7. Trajectory evidence

v1 provides a unified timeline and a human-reviewable annotation interface.
Automated semantic analysis is not a prerequisite.

Each evidence record preserves:

```text
Source file and line number / event ID
Candidate / experiment association
Observed change
Subsequent use location
Associated outcome
observed / inferred / unknown designation
```

Prefer code changes, tool calls, and execution results over Agent self-reports.
Use composable evidence tags rather than uncalibrated numeric levels:

```text
reported_change
implemented_change
reused_change
recursive_reuse
observed_outcome
```

Recursive dependence asks:

> Did an earlier improvement participate in generating, testing, or selecting a
> later research mechanism?

Counting changes alone does not answer that question. For example:

```text
Improve a validation tool
    → use it to select research strategies
    → use the new strategy to guide the next tool improvement
```

Annotations should express relations between events, not just isolated labels.
Relations such as “used to generate,” “used to test,” and “used to select” can
record source and target events and supporting references. Each relation also
carries an observed / inferred / unknown designation. Its exact fields belong
to the proposed schema.

Repeated tool calls can establish reuse but do not alone establish recursive
improvement. That requires reviewable links to the generation, testing, or
selection of subsequent mechanisms. Evidence tags describe recorded behavior;
they do not automatically establish net benefit or causality.

Allow behavioral evidence of in-context learning or memory reuse without
requiring code modification. Mark inaccessible internal state as unknown; do
not infer how particular parameters or a KV cache changed. Insufficient evidence
does not imply zero RSI capability.

Display mechanism changes alongside subsequent convergence features. Temporal
order or a curvature change does not automatically establish causation. Do not
produce `RSI=true/false`.

## 8. Data quality, reproducibility, and comparison

### 8.1 Audit coverage before computing metrics

Distinguish:

- Candidate quality coverage from incumbent-interval coverage.
- Commit-event counts from unique-artifact counts.
- Usage completeness, log truncation, and timestamp reliability.
- The extent to which evidence can be linked to events and outcomes.

Saved logs do not imply complete system-state records. The current implementation
limits output and truncates logs; Gateway usage records are not full inference
transcripts. Additional traces retained by deployments can enter through adapters.

A missing file must not be interpreted as zero calls. Currently,
[`summarize_inference_trace`](../envcore/inference.py) can return zero calls
and `usage_complete=true` when the file does not exist. The analysis layer must
not treat that return value alone as proof of completeness. Distinguish an
unconfigured collection source, a missing file, an empty record, corruption,
truncation, and supported zero usage before assessing available usage evidence.

Do not double-count calls recorded by both the Gateway and Agent. Record source
precedence and deduplication rules explicitly. Without reliable association
keys, report the sources separately instead of adding them into an alleged
total. Aggregate usage cannot automatically be reconstructed into per-call or
per-Candidate consumption.

### 8.2 Reproducibility metadata

Derived reports preserve:

```text
Analysis schema version
Tool version / source identity
Input hashes
Complete configuration and its hash
Original result version and identity algorithms
Compatibility-adapter and evidence-adapter identities
Snapshot provenance, cutoff, and admission evidence
Event and quality coverage
All warnings
```

Hash the reports, events, traces, and configuration files actually read. Do not
recursively scan hidden data or unrelated files. Record declared release
identities separately from the scope of evidence verifiable in this analysis.

Represent unknown or invalid values as `null + reason`; do not output NaN or
Infinity as metrics. Identical inputs and configuration must produce identical
metrics. Non-metric metadata such as report generation time is excluded from
that equality requirement.

### 8.3 Cross-run comparison

Ordinary comparisons across Agents must check:

```text
comparison_key
Release / data / evaluator identities
Effective resource conditions
Metric definitions
Analysis configuration and coverage rules
```

Reject combined rankings when these conditions do not match. Budget-scaling
experiments treat budget as an explicit experimental variable and report it
separately.

Repeated episodes are needed to estimate between-run uncertainty. Candidates
within one episode are not independent replicates. Keep failed, aborted, and
missing runs in reporting denominators and explain their availability for each
metric separately.

Token counts across models do not directly represent equal computation. Similar
curve shapes do not remove the need to report initial Agent conditions, available
resources, submission strategies, and execution conditions.

## 9. Proposed deliverables and acceptance criteria

### 9.1 Package organization and outputs

Add a proposed `bench_analysis/` package with separate responsibilities for:

```text
Loading and validation
Curve reconstruction
Metric computation
Evidence adaptation
Report generation
```

Proposed additions include `tests/test_research_analysis.py` and synthetic
fixtures. Update this document, `docs/research-dynamics.md`, with implementation
status and usage instructions when the feature is built. The current deliverable
is this design document only.

Proposed command; it does not currently exist:

```bash
python -m bench_analysis analyze \
  --episode /path/to/sealed-episode \
  --config /path/to/analysis-config.json \
  --output /path/to/private-report
```

Use a separate output directory containing at least:

```text
analysis.json
curves.csv
evidence.jsonl
report.md
Static curve figures
```

Analysis configuration includes:

```text
Metric and direction
Budget axis
Analysis interval and checkpoints
Quality targets
Candidate cost slices
Missing-data and end-of-run rules
Continuous integration or discrete aggregation convention
Usage sources, deduplication, and time attribution
Exploratory-feature switches
```

Do not guess default quality thresholds. If no target is configured, do not
invent target-attainment metrics. When optional analyses lack required settings
or data, state why they are unavailable without preventing an independent basic
report.

### 9.2 Core acceptance scenarios

1. Step curves, target attainment, selection gaps, integrals, and unequal-interval
   divided differences match hand-calculated examples. A Track B mean, if
   implemented, also has examples specifying checkpoints, weights, and endpoint
   treatment.

2. A good undeclared Candidate changes only the oracle reference. Declaring an
   older Candidate can cause a genuine regression. Future information never
   backfills the past.

3. Repeated artifacts and declarations, identical timestamps, zero budget
   intervals, failed Leases, missing evaluations, and truncated logs are handled
   correctly. Never-admitted startups remain distinct from failures after Lease
   consumption.

4. `analysis_complete=true` with unscored points still produces coverage gaps.
   No incumbent, unreached targets, and unknown results remain distinct. A
   success observed after a gap is not presented as exact first attainment.
   Missing incumbents at the start of a fixed interval do not move its boundary.

5. Adding unchanged declarations leaves primary curves and metrics unchanged.
   Changes in valid submission timing may change the delivered-quality curve;
   explain the effect of the observation strategy.

6. Checkpoints from fixed training scripts and predetermined fast-convergence
   sequences may also improve. The system must not automatically classify them
   as RSI from that fact alone. Synthetic traces with verifiable recursive reuse
   show the corresponding evidence tags and relations; repeated calls alone do
   not become recursive-reuse evidence.

7. A single point or sparse trajectory can still produce a basic report without
   fabricated curvature significance. Incorrect identities and incompatible
   results are rejected for comparison. Relabeling old identity algorithms
   cannot bypass compatibility checks.

8. Identical inputs and configuration produce identical metrics. Verify that
   analysis does not modify source files, execute workloads or log commands,
   access the network or hidden data, or change TEST results. Links in logs are
   not followed.

9. Reject active episodes, ongoing result writes, unconfirmed cleanup, and
   inconsistent state and event records. Failure summaries neither invent
   sealing fields nor bypass termination and cleanup requirements.

10. Missing usage files remain distinct from known zero calls. Calls spanning
    commits are attributed at completion. Duplicate collection sources are not
    double-counted, and unalignable timestamps do not produce invented token
    axes. Research expenditure stays separate from Candidate serving cost.

Synthetic tests validate implementation and guard against specified
misinterpretations. They do not establish calibration of real RSI capability.

After new feature tests pass, run the existing results, supervisor, CLI, and
architecture regression tests. This applies to future implementation. The
current documentation change requires content, link, and diff checks only.

## 10. Implementation sequence and open research questions

### Phase 1: establish a reliable analysis pipeline

Implement loading and identity validation, coverage, the three curves, basic
metrics, and independent reports.

Define input admission and supported historical identity combinations before
implementation. Freeze the computation convention for every emitted metric.
If Track B mean aggregation remains unsettled, deliver the already-defined
discrete checkpoint metrics first.

### Phase 2: add dynamics and process evidence

Add exploratory divided differences, log timelines, evidence annotations, and
human review.

### Phase 3: cross-run analysis and calibration

Analyze multiple episodes, between-run uncertainty, optional convergence fits,
and associations between mechanism reuse and efficiency changes.

Later attribution studies may introduce:

- Baselines with fixed research mechanisms but normal task feedback.
- Baselines without memory or feedback.
- Continuations from the same state or equal-total-budget ablations.

Specify which capabilities each baseline freezes rather than calling different
baselines “no RSI.” Include self-improvement overhead in the budget. These are
future experimental designs, not permission for the analyzer to resume episodes.

Questions for calibration on real data that need not block the first two phases:

- Task-specific quality targets and analysis intervals.
- Effective sample requirements for more complex fits.
- Reviewer agreement on evidence tags and relations.
- Whether a unified summary score is useful.

The first version is complete when it can:

> Turn existing trajectories into trustworthy, reproducible, extensible research
> dynamics metrics and RSI process evidence.

It need not prove that every high-scoring Agent has RSI capability.

## Implementation references and methodological background

Repository facts refer to these files at the revision recorded above:

- [`envcore/supervisor.py`](../envcore/supervisor.py): `_elapsed`,
  `commit_lease`, `declare_incumbent`, `_analysis_trajectory`, sealing, cleanup,
  and result publication.
- [`envcore/results.py`](../envcore/results.py): result schema and binding
  validation.
- [`envcore/artifact_identity.py`](../envcore/artifact_identity.py): identity
  algorithms.
- [`envcore/inference.py`](../envcore/inference.py),
  [`envcore/host_agent.py`](../envcore/host_agent.py), and
  [`envcore/agent_gateway.py`](../envcore/agent_gateway.py): usage, logging,
  and their limitations.
- [`envcore/operator_status.py`](../envcore/operator_status.py): the distinction
  between Core state and independent execution observations.

For benchmark semantics, see the [full reference](benchmark-reference.md),
particularly [ANALYSIS](benchmark-reference.md#analysis),
[Intermediate Incumbents and Research Curves](benchmark-reference.md#intermediate-incumbents-and-research-curves),
and [Evaluation Axes](benchmark-reference.md#evaluation-axes).

Methodological background, not code dependencies:

- Hansen et al., [COCO: Performance Assessment](https://arxiv.org/abs/1605.03560):
  black-box anytime and fixed-target assessment.
- Zhang et al., [Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents](https://arxiv.org/abs/2505.22954):
  self-improvement and baseline comparisons.

The specific metrics, modules, command, and acceptance criteria here are a
proposed design for this repository. They are neither existing functionality nor
an established RSI measurement standard. The cited work does not constitute
calibration of this design's specific metrics.
