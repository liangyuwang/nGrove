# Curves, Budgets, and Metrics

Read this reference when computing or reviewing results. Formulas assume a loss to minimize; explicitly adapt and record other metric directions.

## Three curves

Let `C(b)` be the Candidates committed by budget b, `J(b)` the last incumbent declared by that budget/event cutoff, and `L(M)` the saved official ANALYSIS loss.

- **Submitted:** raw observations `(budget, candidate_id, L, serving_cost)` for each commit; monotonicity is not required.
- **Incumbent:** `I(b)=L(J(b))`, a right-continuous step function and the primary delivered-quality curve. Regressions are allowed.
- **Observed oracle best:** `O(b)=min L(M)` over successfully evaluated Candidates in `C(b)`; null if none exist. This is a postmortem reference with limited coverage, not the true optimum over all Candidates or the Agent's actual selection. Its monotonicity follows from its definition and does not establish learning.

Commit time cannot replace declaration time. A better undeclared Candidate changes only the oracle; declaring an older Candidate may regress the incumbent. Never backfill past states with future declarations. At equal budget coordinates, use event order to resolve cutoffs and preserve history. Unchanged repeated declarations do not add effective samples.

Attaching postmortem quality to a point where that Candidate was already committed/declared is the intended retrospective measurement. It does not mean the Agent saw that feedback then. TEST has only one point for the final declared incumbent.

## Budget axes and two costs

**Track A:** use Supervisor-recorded `elapsed_seconds`, preserving benchmark-owned overhead deductions. Do not reconstruct charged time from UTC differences. Form steps from declarations and truncate according to configuration and sealing evidence.

**Track B:** use consumed Lease count. Checkpoint n takes the last declaration before Lease n+1 was consumed; the final available point uses the research seal. Reconstruct from lifecycle events and declarations' `leases_used`, not commit indices or Lease-ID suffixes. A Lease consumed after successful startup still counts if the workload later fails; a never-admitted startup does not.

Time, training resources, and tokens can be auxiliary coordinates. Candidate index denotes order only. See the input reference for token attribution. Report separately:

- Research expenditure: Agent usage, research time, Lease time, and hardware resources.
- Candidate execution cost: `time_per_task=candidate_wall_seconds/task_count` and supporting measurements. This amortizes the full invocation rather than measuring single-request latency. The denominator is the release-required task count, not successful outputs. Cost includes startup, imports, loading, input reads, scheduling, inference, and output writes, but excludes trusted evaluator data materialization, validation, and scoring.

## Missingness and end-of-run rules

Quality is unknown without an incumbent, when the current incumbent lacks a score, or when data is corrupt. Carry forward only if no new declaration occurred and the existing incumbent's quality is known. Declaring an unscored Candidate creates a gap; do not retain the previous score. Do not fill with zero, interpolate, or substitute the oracle for the incumbent.

Freeze checkpoints, intervals, cost slices, and targets first. Holding the final incumbent through the full budget after a normal early finish requires an explicit setting and the label `held-after-finish`; it is not continued research. Do not extend abnormal terminations to the budget limit. Checkpoints within one episode are not independent runs at different budgets.

## Stable metrics

### Fixed-budget values and coverage

At each checkpoint, retain the actual incumbent, declaration event, budget cutoff, ANALYSIS loss/cost, status, and reason. Separately report submitted-point coverage, unique-Candidate coverage, known/requested interval duration for A, and known checkpoint proportion and weight coverage for B.

### Budget to reach a fixed target

`H(ell)=inf{b: I(b)<=ell}` applies only to preconfigured targets and observation domains. Do not invent default thresholds.

| Evidence | Result |
| --- | --- |
| Target reached with no earlier unknown incumbent interval that could hide success | Exact attainment budget |
| Target reached but earlier incumbent quality is unknown | First observed attainment; `H=null + reason`; the observation is an upper bound on attainment budget |
| No observed success and a gap could hide success | `unknown` |
| No observed success and evidence rules out attainment in the domain | `not_reached`, retaining observed budget separately |

Before the first declaration there is no declared deliverable and no known attainment; that period remains a coverage gap for means. Do not replace an unreached H with the cutoff budget or interpolate an earlier success.

### Means

A: integrate step intervals exactly over fixed `[b0,B]`, with `mean=integral(I)/(B-b0)`. Require `B>b0` and complete quality coverage. Otherwise report null / `insufficient_coverage`, plus known integral, known duration, and coverage. Do not renormalize over known intervals or move the start to the first submission.

B: **there is no implicit mean convention**. Predefine discrete checkpoints and positive weights before computing `checkpoint_mean=sum(w_n*I(n))/sum(w_n)`, requiring all specified points to be known. Alternatively, explicitly define a Lease-axis step extension, integration rule, and final-point weight. These are different statistics. Without a convention, omit the mean while still reporting checkpoints, attainment, and gaps.

### Loss-only selection gap

`gap(b)=I(b)-O(b)` is numeric only when both operands are known. State that the oracle covers successfully evaluated Candidates only. Retain both serving costs and Pareto information. Lower loss may require slower serving, so this is not overall decision regret.

Freeze cost constraints first. The constrained oracle includes only Candidates with known qualifying costs; list the eligible set. If the actual incumbent violates the constraint, mark `not_applicable`; if its cost is unknown, mark unknown. Do not choose another Candidate on the Agent's behalf.

## Exploratory improvement rates and curvature

Compute only when requested or enabled in configuration, and label `exploratory`. Use raw `mean_nll`, not clipped 0–100 scores, for convergence shape.

```text
r_i = (L_i - L_(i+1)) / (b_(i+1) - b_i)
a_i = 2 * (r_i - r_(i-1)) / (b_(i+1) - b_(i-1))
```

Identify submitted or incumbent observations and do not mix them. Coordinates must strictly increase. Retain equal-coordinate events without dividing by zero; if reducing them to one analysis point, predefine the cutoff selection rule and preserve raw observations. Do not differentiate a forward-filled step grid, count repeated declarations as new samples, or adaptively choose windows that appear to accelerate.

Attach point count, unique-artifact count, budget span, actual intervals, and missingness to each metric. Differences spanning a gap describe endpoint change only; the intervening process is unknown. Second divided differences require at least three valid points at distinct budgets. That mathematical minimum does not establish statistical reliability. Positive a does not establish RSI.

Fitting, smoothing, and significance testing are not defaults. Optional `eta=-d log(L-L_star)/db` requires external justification for `L_star`; a run's minimum observed loss is not its true lower bound. Finite trajectories do not prove asymptotic convergence or unlimited self-acceleration.

## Cross-run and final results

Check complete comparison keys, release/data/evaluator identities, effective resources, metrics, analysis configuration, and coverage rules. Do not combine incompatible rankings. Budget-scaling studies treat budget as an explicit variable; A/B differences are not RSI treatment effects.

Report final TEST score, higher being better, raw metrics with their declared directions, and `time_per_task`, lower being better. Under comparable conditions, report the Pareto non-dominated set without arbitrarily combining quality and cost into a scalar. Present ANALYSIS and TEST costs under their respective input conditions rather than mixing splits into one frontier.

Ordinary ANALYSIS point failures do not erase completed TEST. Boundary/cleanup failures may retain TEST while making a run non-compliant or ineligible. Preserve official outcome/policy/eligibility rather than upgrading them through analysis. ANALYSIS–TEST disagreement is a distribution diagnostic, not direct evidence of overfitting to ANALYSIS. Single-run ranking differences do not establish between-run confidence.

## Hand-calculated checks for new implementations

Verify these behaviors when writing or changing computation code; text-only edits do not require a new test suite.

1. A, fixed domain `[1,5]`: M1 is declared at b=1 with loss 5; M2 is committed at b=2 with loss 3 and declared at b=3; M1 is declared again at b=4. I is `[1,3):5, [3,4):3, [4,5]:5`, with mean `18/4=4.5`. O becomes 3 at b=2, gap at b=2 is 2, and target 3.5 is first reached at b=3. Unchanged repeated declarations leave metrics unchanged. Changing the domain to `[0,5]` makes the mean unavailable because of the initial gap.
2. The Candidate declared at b=2 has unknown quality; the new incumbent at b=4 has loss 3. For target 3.5, first observed attainment is 4 and exact H is null: only attainment by 4 is supported. Without an observed success but with that gap, the result is unknown rather than not_reached.
3. Raw points `(b,L)=(1,5),(3,4),(6,1)` give r=0.5,1 and a=0.2. Duplicate coordinates do not produce new derivatives. A fixed training program can also produce this curve; do not automatically label it RSI.
4. B checkpoints n=1,2,3 have losses 5,4,3. Preconfigured equal weights permit checkpoint_mean=4; any unknown point makes it unavailable. Without a convention, report points only. Each n ends at the next Lease-consumption event. Count failures after consumption, excluding never-admitted startups.
5. A better undeclared Candidate changes only the oracle. Failed points remain missing even when ANALYSIS is complete. Missing usage files do not imply zero calls. Failures/aborts remain in denominators. Incorrect identities reject the corresponding comparisons, and analysis leaves source files unchanged.
