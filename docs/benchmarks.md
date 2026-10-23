# Benchmark runs

Select benchmark functionality explicitly with `--benchmark` and `--track-config`.
The release supplies named budget and runtime presets. `--environment` can supply
run-owned resources and budgets; a comparison key exists only when the effective
conditions match the selected published lane.

## Optional result extension

Result schema v18, Agent Protocol v13 and Task schema v8 separate research
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
`runtime_profile` contracts are accepted and normalized when loading schema 7
tasks. Schema 8 tasks use the current contract names.
Capability checks describe execution requirements; `assess_benchmark_worker`
adds certification and trust checks for benchmark evaluation. Earlier task schema
7 and result schemas v12–v17 remain readable. Historical task eligibility is
honored when resolving an official legacy release, while new task definitions
leave eligibility to the benchmark release.

Schema v17 reports retain their original optional `benchmark` extension and
workload-acceptance fields when read. Schema v18 writes qualification evidence
only when benchmark functionality is selected; historical reports are not
rewritten or reclassified.

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
the [benchmark's supported preview lanes](#lm-pretrain-preview-lanes)
or use `benchmark-info` for its current profile IDs and requirements.

Official benchmark comparisons will use a small set of published, frozen
reference environments. Runnable profiles may cover more hardware without
creating an official leaderboard for each configuration. Users may define
profiles in derived preview releases for their own hardware; they are
responsible for consistent resources, task/data identities, budgets and
evaluation rules across the Agents they compare, and for disclosing differences.
Custom runs remain subject to Worker enforcement and do not automatically
qualify for official rankings. See the
[reference environment and comparison policy](#reference-environments-and-custom-comparisons).

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

## Release verification details

Formal benchmark evaluation requires an `official` release with no provisional track or
runtime profiles, a complete comparison key, digest-pinned task and Agent
images, policy-compliant execution, and certified conformant Research and
Evaluation Workers. Formal deployments must also supply a trusted
`ReleaseVerificationReceipt` from the organizer-controlled Coordinator. The
receipt binds the accepted Core revision and the benchmark, task, policy,
evaluator, TRAIN-manifest, ANALYSIS, and TEST digests; hashes calculated only from
local files cannot self-promote a run. When a Worker executes a derived SIF, VM
image, or other artifact rather than the canonical digest-pinned OCI artifact,
the same trusted receipt must also contain the accepted workload-resolution
evidence. A Worker's own transformation claim is not organizer acceptance.
Missing evidence is recorded in
`benchmark.leaderboard_eligibility_violations`; it never silently promotes a preview.

A benchmark release also publishes finite `track_configs` and
`runtime_profiles` maps. The task remains hardware-independent; the runtime
profile selects episode resources and the leaderboard compares results only
when all four comparison-key fields match, within the same frozen release and
hidden-data identity. The names alone do not establish equal cost conditions:

```text
benchmark_id + track + track_config + runtime_profile
```

## Reference environments and custom comparisons

For research on hardware scaling, use run-owned
[environment configurations](environments.md). Select `--environment`
at launch to reuse a task and budget with custom hardware without editing this
benchmark release. Every run freezes an environment. If its effective conditions
match the explicitly selected Track/runtime preset, the benchmark application
can attach that lane's comparison key. Changed conditions retain the benchmark
identity but have no published comparison key and remain leaderboard-ineligible.

Official benchmark comparisons will use a small set of designated reference
environments, published and frozen with each release. Organizers are responsible
for verifying those environments and the conditions under which official results
are measured. No official reference environment has been finalized for the
current preview.

A reference environment covers Training and Candidate evaluation separately.
Its definition must specify the conditions relevant to comparison, including
GPU model, count, memory and interconnect topology; CPU model and allocated
cores; host-memory allocation and NUMA placement; GPU power limits; storage,
cache and sharing/isolation policies; and the task-owned base images and software
requirements. These details define one environment, rather than independent
leaderboard categories for every hardware combination. The official comparison
set does not expand automatically when a runnable profile is added.

Users may adapt the benchmark to their own hardware through a distinct runtime
profile in a derived preview release, selected with `--runtime-profile`. Keep
`leaderboard_eligible: false` and preserve the original release provenance;
changed resource requirements must not reuse an official profile identity.
When comparing Agents, the evaluation organizer is responsible for:

- providing the same benchmark task and data identities, Track budgets,
  task-owned starting environment, and evaluation and policy rules;
- providing consistent effective Training and evaluation resources, including
  CPU, memory, GPU connectivity and contention conditions, not just matching GPU
  product names;
- publishing the actual configuration, relevant run evidence and any differences
  that limit the comparison.

The Agents' own identities and initial software remain participant-owned and
recorded; fixing the available resources does not restrict how Agents explore
within the benchmark rules. Custom results are welcome for independent research
and community sharing, but do not automatically qualify for official rankings
or direct comparison with results from another environment.

User responsibility does not replace Worker enforcement. Existing resource,
budget, isolation and receipt checks still apply to the selected profile.
Passing those checks establishes only its declared requirements, not that every
condition affecting fairness has been specified. Matching comparison keys and
hardware names alone therefore do not establish equivalent conditions; see
[evaluation resource comparability](evaluation-resources.md).

## Runtime profile contracts

Benchmark schema v4 adds optional accelerator `model_match_mode: "exact"`.
The default remains case-sensitive substring matching (`"contains"`). Exact
matching compares the complete reported device name during selection and
verification, and projects to an `accelerator.model` resource requirement with
relation `exact` and one identical model string per allocated GPU. Use distinct
profiles for hardware variants; do not broaden a match to make a different
variant pass. Releases using exact matching require schema v4 so older Core
versions reject the manifest instead of silently applying substring matching.

Benchmark schema v3 and later profiles may declare `evaluation_resource_contract`,
independently of the Training `resource_contract`. Its normative and observational
properties use the existing extensible `ResourceEnvelope`. Formal releases must
declare a nonempty evaluation contract requiring `enforced` evidence; the release
owner selects the relevant CPU, memory, I/O, cache, and contention conditions.
An accepted envelope covers only its declared properties. Schema v2 remains
readable for existing previews and rejects the new field to prevent silent
omission by older Core versions. See [evaluation resources](evaluation-resources.md).

Agent-visible release metadata contains only the public release and selected
profile identifiers, release stage, and eligibility setting. `benchmark-info`
publishes the profile catalog without full task/benchmark file hashes. Those
hashes bind trusted release verification and remain in the organizer's episode
audit records, not the participant-facing manifest. New private release fields
are not implicitly public. The public task manifest separately declares the
selected budget and Training resources.

`state.json` and the `report` command retain complete organizer audit evidence.
They are not participant-facing or publication-safe exports: exposing a digest
of private configuration can enable enumeration, including from an earlier
episode when that configuration is reused. Public release identifiers must be
independent of private configuration fingerprints; formal verification still
checks the complete trusted file identities.

Runtime-profile preflight confirms that the execution environment satisfies
the hardware requirements declared by the selected profile, including host
count and accelerator vendor, model, count, and minimum memory.

Optional benchmark profile maps can be inspected without preparing task data:

```bash
ngrove benchmark-info \
  --benchmark research_tasks/lm-pretrain/benchmark.json
```


## LM pretrain preview lanes

For benchmark comparisons, `benchmark.json` supplies named Track/runtime presets
and comparison metadata. It remains `leaderboard_eligible: false`: the current
settings are provisional, and a verified release will replace the reproducible
local holdouts with organizer-owned hidden datasets.

The preview publishes two provisional budget presets:

- Track A: `preview-a-8h`;
- Track B: `preview-b-6x1h`.

Each hardware family has separate single-host profiles for **1, 2, 4, and 8
full GPUs**:

| Hardware | Runtime profile IDs | Minimum memory per GPU |
| --- | --- | --- |
| NVIDIA A100 SXM4 80GB | `a100-sxm4-80gb-x{1,2,4,8}` | 80,000 MiB |
| NVIDIA H100 SXM 80GB | `h100-sxm-80gb-x{1,2,4,8}` | 80,000 MiB |
| NVIDIA H200 SXM 141GB | `h200-sxm-141gb-x{1,2,4,8}` | 140,000 MiB |
| NVIDIA H200 NVL 141GB | `h200-nvl-141gb-x{1,2,4,8}` | 140,000 MiB |

This is a catalog of runnable configurations, not a set of official reference
environments or a commitment to a leaderboard for every profile. Official
reference environments remain to be selected and verified. Users can define
profiles for their own hardware and are responsible for fair conditions across
the Agents they compare; follow the
[reference environment and comparison policy](#reference-environments-and-custom-comparisons).

The braces abbreviate four concrete IDs; for example, use `h200-sxm-141gb-x2`.
H200 SXM and NVL have distinct profiles because their hardware specifications
differ ([NVIDIA H200 specifications](https://www.nvidia.com/en-us/data-center/h200/)).
They require exact device names `NVIDIA H200` and `NVIDIA H200 NVL`, respectively;
an unfamiliar reported name must be verified before adding a profile alias.
Memory thresholds are preflight lower bounds, not conversions of product labels.

The count is the episode allocation, not the host's total inventory. The Worker
must expose exactly the selected full GPUs, preserve their identities between
Leases and CUDA Candidate evaluation, and reject MIG-enabled devices. The
reference Worker can select a subset of a larger host. Multi-host execution and
other board variants require separate support and profiles.

Track budgets are unchanged across profiles: Track A remains eight wall-clock
hours, and Track B remains six one-hour Leases. They are not normalized to equal
GPU-hours. GPU count and model therefore remain part of the comparison key.

All profiles are provisional declarations, not evidence that every combination
has passed a real run. They currently declare GPU requirements without fixing
interconnect topology, CPU allocation or host-memory conditions. The expanded
catalog and GPU binding are covered by
hardware-free regression checks. Driver, pinned-image, CUDA execution, and
multi-GPU behavior still require the selected deployment's
[Worker integration checks](worker-integration.md) and an authorized
smoke episode. Host/interconnect calibration remains preview release work.

CPU, B300, TPU, and other platforms may run the same task when runtime support
exists, with effective conditions recorded in the frozen environment. Research
comparisons must identify varied and controlled conditions. For benchmark
comparisons, require the same frozen release and data identities, consistent
resource conditions, and this complete comparison key:

```text
benchmark_id + track + track_config + runtime_profile
```

Inspect the finite maps:

```bash
ngrove benchmark-info \
  --benchmark research_tasks/lm-pretrain/benchmark.json
```

The command below infers `task.json` from the benchmark manifest. The selected
task's [operator protections](../research_tasks/lm-pretrain/README.md#operator-protections)
also apply.

```bash
ngrove start \
  --benchmark research_tasks/lm-pretrain/benchmark.json \
  --track-config preview-b-6x1h \
  --runtime-profile h100-sxm-80gb-x4 \
  --episodes-root /path/to/ngrove-runs/episodes \
  --agent-runner container \
  --agent-image participant-agent@sha256:DIGEST \
  --agent-command "participant-agent run"
```

Profile-bound benchmark runs reject `--track`, `--training-gpus`, and
`--training-gpu-model`; those values come from the selected profiles so the
recorded comparison key cannot disagree with the actual episode.

## Smoke task preset integration

For optional preset-based integration, the same smoke task can exercise a
published accelerator profile without changing the task or using manual resource overrides:

```bash
ngrove start \
  --benchmark dev_tasks/smoke_lm-pretrain/benchmark.json \
  --track-config smoke-a-5m \
  --runtime-profile h100-sxm-80gb-x4 \
  --episodes-root /path/to/ngrove-runs/episodes \
  --agent-runner container \
  --agent-image participant-agent@sha256:DIGEST \
  --agent-command "participant-agent run"
```

## Evaluation resources

Benchmark schema v3 adds an optional `evaluation_resource_contract` to each runtime
profile. If a selected benchmark fixes an evaluation envelope, Core rejects a
conflicting run environment and records the frozen release identity. V2 profiles
remain readable without evaluation contracts and reject this newer field. Current
preview profiles leave CPU, host-memory, cache, I/O and exclusivity unspecified.

Formal qualification requires a nonempty enforced evaluation contract matching the
release, and accepted TEST allocation evidence. These checks contribute to the
optional result's `benchmark` extension alongside the other
[benchmark requirements](#formal-qualification).

## Registered inference qualification

For formal registered inference, the organizer supplies
`ReleaseVerificationReceipt.inference_gateway_artifact` as a
`GatewayArtifactIdentity(format, sha256)`. This is independent verification of
the complete executable artifact selected for that release. The format defines
the hashing procedure; Core compares exact format and digest values, without
loading a platform artifact or interpreting deployment handles.

The identity must cover every executable input: the runtime environment,
dependencies, entrypoint and any injected implementation code. A deployment
may define a canonical composite manifest when these are separate artifacts.
The organizer must verify that manifest and its components independently of
the launcher's claim. A configured image reference, a base-image digest, or a
label naming a Core revision is insufficient when other code is injected.
Changes to the executable bundle require a newly approved artifact identity.



Formal eligibility requires the observed artifact to match the organizer's
release artifact, `enforcement=ENFORCED`, `trust_level=CERTIFIED`, and no
limitations. Enforcement means the selected artifact and inference registration
remain enforced throughout execution. Certification must come from the
organizer's trusted launcher boundary; a mechanism name or self-description
does not establish it. Worker conformance and all other formal checks remain
independent requirements.

When benchmark functionality is selected, missing or insufficient evidence is
recorded in `benchmark.leaderboard_eligibility_violations` using these codes:

| Code | Meaning |
| --- | --- |
| `inference_gateway_workload_evidence_missing` | No actual execution receipt |
| `inference_gateway_release_artifact_missing` | No organizer-verified complete artifact |
| `inference_gateway_artifact_mismatch` | Observed format or digest differs from the approved artifact |
| `inference_gateway_execution_not_certified` | Insufficient enforcement, trust, or unresolved limitations |

These are eligibility findings. They do not blame a Candidate, erase an
otherwise accepted TEST measurement, or stand in for deployment teardown
failures. A malformed or misbound receipt is rejected at registration.

The bundled local launcher remains a development composition and does not
issue this receipt. In particular, its Gateway uses a base image plus a mounted
Core package; the image alone cannot certify the executable bundle or enforce
immutability of the host source. Formal inference therefore continues to fail
closed in that composition. Unit and fake-launcher integration tests establish
the Core contract; deployment certification and real execution acceptance
remain separate work.

## Environment overrides

Select a named budget with `--benchmark` and
`--track-config`. To also reuse a published hardware profile, use
`--benchmark ... --track-config ... --runtime-profile ...`. An optional
`--environment` replaces its compute configuration and overrides only explicitly
specified time budgets. The selected profile remains the comparison preset;
changing effective conditions does not claim that preset's comparison key.
Without `--runtime-profile`, a benchmark launch requires `--environment`.

The benchmark application can attach its release identity to the same run.
An explicitly selected Track/runtime preset receives a comparison key only when
its complete resolved budget and resource contracts match. An environment file
with identical effective conditions keeps that key. A different GPU count, time
budget or evaluation contract produces a custom experiment with a null key;
the source benchmark identity remains recorded. Formal eligibility still needs
an official release, trusted execution and all existing evidence requirements.

Benchmark resolution uses the same `resolve_environment_run` function as task-only
execution. The environment types and resolver are independent of benchmark catalogs.

## Certification development status

Original issues 10 and 13 remain deferred: the complete organizer-controlled
conformance suite, certification workflow, hidden release, submission and
publication infrastructure are incomplete. Gateway execution evidence supports
issue 12; real launcher evidence and certified deployment acceptance remain
separate from interface regressions.

The following migration history explains the independent receipt boundary.
Current trusted launchers supply these receipts through `BenchmarkContext`.

### C5. Trusted conformance evidence

Status: evidence boundary implemented in Worker Protocol v9 and retained by
v12; complete formal conformance suite remains future organizer work.

Trigger: the first in-repository resolution probe demonstrated that a partial
black-box check could otherwise be mistaken for complete Worker conformance,
and that a receipt returned in `WorkerDescriptor` could not prove who issued it
or which Worker was tested.

Result: Worker self-description no longer carries conformance. A trusted
Operator composition supplies a versioned `ConformanceReceipt` independently;
the receipt binds the assessed Worker ID, issuing authority, organizer-owned
report SHA-256, suite version, and decision. Core checks that binding separately
from capability claims and deployment trust. The diagnostic resolution probe
requires release-owned identities, a unique run scope, a mandatory deployment
teardown callback, and public/private diagnostic separation, but cannot issue a
formal receipt because it does not exercise the complete v9 contract.

A separate [evaluation resource diagnostic](worker-evaluation-probe.md) now
exercises v16 TEST/ANALYSIS invocation receipts and typed resource rejection on
public fixtures, sharing resolution/execution deadlines and requiring explicit
deployment cleanup confirmation. This gives external Worker preparation a
reusable check beyond resolution. It remains partial diagnostic coverage, not
physical enforcement acceptance or a formal conformance suite; historical
external preview runs are not retrospectively certified.

The future formal suite belongs to organizer tooling and the external Worker
SDK. Platform-specific deployment setup, outer watchdogs, destructive cleanup,
and remote receipt authentication stay outside Core.
Under the current closeout decision, the complete suite is deferred with formal
release work; the diagnostic increments below are delivered tools, not a queue
of further preview development stages.
