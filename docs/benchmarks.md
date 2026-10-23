# Benchmark runs

Select benchmark functionality explicitly with `--benchmark` and `--track-config`.
The release supplies named budget and runtime presets. `--environment` can supply
run-owned resources and budgets; a comparison key exists only when the effective
conditions match the selected published lane.

## Optional result extension

Result schema v17, Agent Protocol v13 and Task schema v8 separate research
execution from benchmark qualification. `benchmark.json` owns
`leaderboard_eligible`; task manifests contain research and evaluation contracts.
The result's optional `benchmark` object contains:

- `release` and `comparison_key`;
- `formal_release_verification`;
- `policy_compliant` and `policy_violations`;
- `leaderboard_eligible` and `leaderboard_eligibility_violations`;
- Worker certification assessments and conformance receipts.

The Agent-visible extension contains an allowlisted release description, the
comparison key and qualification status. Private file identities, certification
receipts and organizer evidence remain in the trusted report. The `trajectory`
export includes the extension when present. Operator status displays comparison
and eligibility fields for benchmark runs.

External trusted launchers pass `BenchmarkContext` from `envcore.benchmark` to
`Supervisor(benchmark=...)`. The context accepts `release`, `comparison_key`,
`verification`, `research_conformance`, and `evaluation_conformance`; it is copied
before execution and owns the qualification state. The old separate Supervisor
comparison/receipt arguments are replaced by this explicit context.

Worker Protocol v17 uses task-scoped score and cost contracts (`task_release`).
Cost comparison requires matching hidden data and `effective_resources`; legacy
`runtime_profile` contracts are accepted and normalized when loading tasks.
Capability checks describe execution requirements; `assess_benchmark_worker`
adds certification and trust checks for benchmark evaluation. Earlier task schema
7 and result schemas v12–v16 remain readable. Historical task eligibility is
honored when resolving an official legacy release, while new task definitions
leave eligibility to the benchmark release.

## Formal qualification

A formal result requires an official frozen release, matching comparison
conditions, immutable workloads, organizer-owned hidden evaluation, trusted
release verification and certified conformant Research/Evaluation Workers.
Capabilities, trusted certification and deployment authority are separate checks.
A conformance receipt binds its suite version, Worker identity, authority and
report digest. A passing older suite does not certify a changed Worker protocol.
Worker diagnostics report their scoped observations; formal certification is
supplied independently by the organizer.

A derived workload requires organizer acceptance for formal use. Gateway execution
must match the registered inference policy and frozen trusted artifact. Missing
or inconsistent evidence is retained as eligibility violations. `host_cli` and
`unrestricted-dev` do not satisfy the formal isolation boundary. Task completion
and a measured TEST score alone do not confer leaderboard eligibility.

Ordinary missing ANALYSIS points do not revoke an otherwise valid TEST result.
An unconfirmed execution boundary or failed teardown does revoke eligibility,
while preserving the measured TEST result and failure evidence. Infrastructure
failures are excluded from benchmark scoring; retries require operator authority.

The H200 results in the README are non-compliant, leaderboard-ineligible
`host_cli` previews. Each configuration ran once, and two failed earlier attempts
were replaced by later attempts. They do not establish formal ranking or current
Worker certification.

# Leaderboards

The following publication conventions apply to organizer-provided leaderboards.
Separate Track A and Track B comparisons by their frozen conditions.

## Track A

Published as a Performance–Cost Pareto frontier at:

```text
final TEST performance score vs. TEST time per task
at fixed research wall-clock
```

Reported alongside:

```text
research-time curve
agent tokens
Training Leases
training compute
model size
Candidate cost submetrics
```

## Track B

Published as a Performance–Cost Pareto frontier at:

```text
final TEST performance score vs. TEST time per task
after at most N Training Leases
```

with fixed maximum duration per lease and no organizer feedback during research.

Reported alongside:

```text
postmortem ANALYSIS performance score and raw metrics by Candidate commit
final TEST performance score and raw metrics
agent tokens
agent reasoning between leases
training compute
model size
Candidate cost submetrics
```

Results from materially different runtime profiles or budget scales should not
be mixed into a single leaderboard comparison.

---

# Benchmark Releases and Comparison Lanes

A task fixes the corpus and evaluation semantics, but does not require one
hardware platform. A benchmark release publishes a deliberately finite map of
named track configurations and named runtime profiles. The preview publishes
provisional Track A/Track B budgets and a finite single-host hardware map; see
the [benchmark's supported preview lanes](../research_tasks/lm-pretrain/README.md#supported-preview-lanes)
or use `benchmark-info` for its current profile IDs and requirements.

Official benchmark comparisons will use a small set of published, frozen
reference environments. Runnable profiles may cover more hardware without
creating an official leaderboard for each configuration. Users may define
profiles in derived preview releases for their own hardware; they are
responsible for consistent resources, task/data identities, budgets and
evaluation rules across the Agents they compare, and for disclosing differences.
Custom runs remain subject to Worker enforcement and do not automatically
qualify for official rankings. See the
[reference environment and comparison policy](../research_tasks/README.md#reference-environments-and-custom-comparisons).

A run whose effective conditions match an explicitly selected benchmark lane
records the complete comparison key:

```text
benchmark_id + track + track_config + runtime_profile
```

Matching these fields is necessary for direct comparison, but also requires
the same frozen release and data identities and consistent effective resource
conditions. CPU, B300, TPU, and future platforms may execute the same benchmark
with compatible Workers and distinct runtime profiles. The runtime profile is
an episode-time allocation and eligibility input, not a field in `task.json`.

Inspect a release's supported maps with:

```bash
ngrove benchmark-info \
  --benchmark research_tasks/lm-pretrain/benchmark.json
```

With `--benchmark`, `--track-config` selects the Track and default budget;
`--runtime-profile` supplies a resource preset. An optional `--environment`
replaces the compute configuration and may override the time budget. Every path
uses the same frozen environment and execution lifecycle. A comparison key is
attached only when the effective conditions match the explicitly selected lane.
Use `--track` for a task-only launch; manual GPU flags cannot be mixed with a
benchmark or an environment file. See [environment configuration](environments.md).

The `lm-pretrain` TRAIN mount contains both raw FineWeb-Edu TRAIN documents with
source metadata and official GPT-2 token shards linked by document IDs and
token spans. The Agent may use either representation. Preview ANALYSIS/TEST rows are
excluded from both Agent-visible forms and remain evaluator-only.

---
