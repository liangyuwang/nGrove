# AutoTrain-Bench

**An open-ended benchmark for autonomous training research and recursive self-improvement.**

Given a fixed task, data, evaluation contract, and research budget, how good a
training system can an autonomous Agent discover?

**The entire training system is the search space.** We encourage exploration of
any participant-controlled component, including approaches the benchmark authors
have not anticipated. The Agent builds the system itself; there is no reference
`train.py` to tune.

> **Preview:** runnable development tasks and completed experimental runs.
> No frozen, leaderboard-eligible challenge has been released yet.

[Overview](#overview) · [Results](#results) · [Getting Started](#getting-started) ·
[Contributing](#contributing) · [Documentation](#documentation)

## Overview

Unless a release explicitly restricts it, the Agent is free to change any part
of its training system or task-local research process:

| Direction | Examples to explore |
| --- | --- |
| **Models** | Model scale, Transformers or other architectures, recurrence, sparsity, MoE |
| **Optimization** | Optimizers, objectives, initialization, learning-rate schedules, batch strategies |
| **Data** | Selection, filtering, mixtures, curricula, ordering, packing, training tokenization |
| **Systems** | Kernels, precision, compilation, memory management, parallelism, communication, data loading |
| **Research process** | Experiment design, profiling, tool creation, research memory, subagents, autonomous self-improvement |

**These are examples, not an exhaustive list.** New directions and combinations
are welcome. The fixed task, evaluation contract, resource budget, and isolation
rules define the boundary; the research strategy remains open.

**The initial system is fixed. Autonomous improvement is allowed.** After task
reveal, the Agent can improve its own software, tools, and research workflow
within the benchmark boundary. External human intervention is excluded.

A **Training Lease** gives the Agent bounded access to the training environment.
Two tracks ask complementary questions:

| Track | Budget | Research question |
| --- | --- | --- |
| **A: time-bounded research** | Total elapsed research time, including reasoning, coding, and training | How much can the Agent achieve within a fixed amount of time? |
| **B: lease-bounded research** | Number and maximum duration of Training Leases | How effectively can the Agent use a limited number of training opportunities? |

The Agent receives TRAIN and no organizer-owned validation scores during
research. It chooses its final model before research is sealed. Hidden TEST
then evaluates that model; postmortem ANALYSIS evaluates the committed history.

Results keep two axes separate: **higher TEST performance** and **lower serving
time per task**. Comparisons use the same release and comparison conditions.

Core defines the benchmark rules. Workers implement execution and isolation in
the user's environment. The repository includes a local Docker runtime;
cluster and other platform integrations live in external Workers.

Official benchmark comparisons will use a small set of published reference
environments. You can adapt runtime profiles for your own hardware and are
responsible for fair conditions across the Agents you compare. Custom runs do
not automatically enter official rankings; see the
[reference environment and comparison policy](challenges/README.md#reference-environments-and-custom-comparisons).

## Results

More results will be added after further testing. A dedicated website is planned
to present results in greater detail.

The `lm-pretrain` preview has completed full-data runs in both tracks, using
a public TRAIN pool of approximately **9.95 billion tokens**. Both runs completed
autonomous training, Candidate submission, research sealing, TEST, and postmortem
ANALYSIS.

These September 2026 runs used Codex CLI with `gpt-5.6-sol` at high reasoning
effort, on Core revision [`84f6a72`](https://github.com/liangyuwang/AutoTrain-Bench/commit/84f6a725c59f2c596acf8798e0df6aa4f62948eb).

| Track | Research budget | Training hardware | TEST score ↑ | TEST time per task ↓ |
| --- | --- | --- | --- | --- |
| A | 8 hours | 4 × H100 80 GB | 70.09 | 7.81 ms |
| B | 6 Leases × up to 1 hour | 1 × A100 80 GB, derived preview profile | 65.63 | 3.79 ms |

Scores use the preview's 0–100 performance metric; time per task measures
Candidate serving, not training. These are non-compliant, leaderboard-ineligible
`host_cli` previews with different hardware profiles and Agent CLI versions,
so they are not a controlled A/B comparison or a formal ranking. Historical
success also does not certify the current Worker revision; see the
[recorded evidence scope](docs/worker-acceptance-matrix.md#historical-evidence-catalog).

## Getting Started

We recommend using a **benchmark agent** to help prepare, launch, monitor, and
report your runs. It acts as the trusted benchmark operator; the **Tested Agent**
is the separate participant whose research ability is being measured. Give the
operator this checkout and ask it to follow [AGENTS.md](AGENTS.md) and the
[full reference](docs/benchmark-reference.md).

Choose a [benchmark challenge](challenges/README.md), such as
[LM pretrain](challenges/lm-pretrain/README.md), and select its published Track
configuration and a runtime profile matching your hardware. `lm-pretrain` is
currently a runnable preview challenge; it is not yet a frozen,
leaderboard-eligible release.

### **Evaluate one Tested Agent.**

Provide the Agent identity and command, an authorized environment, and a dedicated working root. For example:

> Run one episode of `challenges/lm-pretrain/challenge.json` with [published Track
> configuration] and [runtime profile matching my hardware]. My Tested Agent is
> [immutable Agent image], with command [autonomous command]. Use [authorized
> environment] and [working root outside the Core checkout], with an empty Agent
> workspace. Resolve missing prerequisites and any unapproved costs before
> launch, then monitor the episode through evaluation and report the result.
> Do not retry automatically.

Example values for this template are below. Replace the illustrative Agent
image, command, SSH alias, and path with your own. Use the image's actual digest;
the repository does not provide the example Agent.

| Template field | Example value |
| --- | --- |
| `[published Track configuration]` | `preview-a-8h` — Track A, eight hours of research |
| `[runtime profile matching my hardware]` | `h100-sxm-80gb-x1` — one NVIDIA H100 SXM 80 GB |
| `[immutable Agent image]` | `ghcr.io/your-org/agent-a@sha256:<real-digest-a>` |
| `[autonomous command]` | `agent-a run` — replace with your Agent's non-interactive command |
| `[authorized environment]` | `ssh bench-h100` — an existing authenticated SSH alias |
| `[working root outside the Core checkout]` | `/srv/autotrain-runs` on that host |

### **Evaluate multiple Agents or hardware configurations.**

Specify the combinations
to run, the concurrency limit, the total budget, and the retry policy:

> Evaluate [provided Agent identities and commands] using [challenge and Track
> configuration] across [provided environments and supported runtime profiles].
> Run [explicit Agent/hardware combinations], with at most [N] episodes in
> parallel and a total budget of [resource/time limits, including preparation,
> evaluation, and cleanup]. Use [dedicated working roots] and [retry policy].
> Reuse verified public TRAIN data and compatible cached images. Give every
> episode its own Agent workspace, outputs, and resource allocation. Report
> every result with its configuration, and compare Agents within matching
> benchmark conditions.

For example, evaluate two participant-provided Agents on two hardware profiles,
for four episodes. Agent A is the image and command above; Agent B could be
`ghcr.io/your-org/agent-b@sha256:<real-digest-b>` with command `agent-b run`,
again replaced with your actual Agent identity and command.

| Template field | Example value |
| --- | --- |
| `[provided Agent identities and commands]` | Agent A and Agent B, each with its frozen image digest and autonomous command |
| `[challenge and Track configuration]` | `challenges/lm-pretrain/challenge.json`, `preview-a-8h` |
| `[provided environments and supported runtime profiles]` | `ssh bench-a100`: `a100-sxm4-80gb-x1`; `ssh bench-h100`: `h100-sxm-80gb-x1` |
| `[explicit Agent/hardware combinations]` | A/A100, B/A100, A/H100, B/H100 |
| `[N]` | `2`, with at most one active episode per host |
| `[resource/time limits, including preparation, evaluation, and cleanup]` | Four episodes; at most 48 reserved GPU-hours and 36 hours elapsed for the whole batch, including all phases |
| `[dedicated working roots]` | `/srv/autotrain-runs` on each host, with a separate directory for every episode |
| `[retry policy]` | No automatic retries; retain failure evidence and continue with independent runs when their boundaries remain safe |

The operator checks that preparation, research, evaluation, and cleanup fit the
total limits before launch. In this example, compare A with B on A100 and on
H100 separately; keep the hardware profile attached to every result.

Once the run plan is agreed, the operator can carry out the authorized episodes
without asking again for each one. Parallel runs require independent resource
allocations and isolation. Batch scheduling belongs to the operator's deployment
launcher; Core supervises individual episodes.

**Prepare TRAIN once and reuse it.** Keep the verified public dataset in trusted
storage and expose it read-only to each episode. Remote environments can share
that storage or receive a verified copy once per storage domain. Each local
`start` checks the existing TRAIN files; it does not download or rebuild them.
Keep the data cache separate from disposable episode directories and retain it
for the agreed batch. Agent-produced preprocessing, checkpoints, and research
state remain episode-specific. Hidden ANALYSIS/TEST data follows its separate
evaluation lifecycle. See [TRAIN integrity](docs/training-data.md).

**Limited resources? Use the challenge's corresponding smoke task.** For
`lm-pretrain`, [Smoke LM pretrain](dev_tasks/smoke_lm-pretrain/README.md) offers a
small CPU-capable development run to check Agent integration and the episode
lifecycle. Choose this explicitly if you cannot allocate the challenge's data,
hardware, or time budget. A smoke task is a development counterpart, not the
benchmark challenge itself; its scores are not challenge results. The operator
should explain the resource gap and let you select smoke rather than silently
substituting it.

**Manual setup.** You can also operate the CLI directly.

You need **Python 3.9+**, **Docker** for the bundled local runtime, and **your own
Agent**. The repository does not ship a reference Agent. Match the selected
challenge's hardware requirements; NVIDIA GPU runs with Docker additionally
require the NVIDIA Container Toolkit.

From a checkout of this repository, install the CLI and inspect the challenge
configurations:

```bash
python3 -m pip install -e .
autotrain-bench challenge-info \
  --challenge challenges/lm-pretrain/challenge.json
```

Then follow the challenge's [data preparation](challenges/lm-pretrain/README.md#prepare-data)
and [run configuration](challenges/lm-pretrain/README.md#supported-preview-lanes) steps.
They cover the Agent command, resource settings, and a dedicated working root
outside the Core checkout.

- **Existing host CLI Agent:** [host-native preview setup](docs/benchmark-reference.md#bring-your-own-agent-host-native-cli-tools).
- **Your own execution environment:** [external Worker integration](docs/worker-integration.md).

## Contributing

Bring an Agent, an execution environment, or a contribution to the benchmark:

- **Evaluate your Agent.** Choose a benchmark challenge and use the
  [reproducibility and logging guidance](docs/benchmark-reference.md#reproducibility-and-logging)
  to record the run's configuration, results, and limitations.
- **Integrate a Worker.** Implement the [Worker Protocol](docs/worker-protocol.md)
  for your environment, using the existing [integration guide](docs/worker-integration.md)
  and [acceptance matrix](docs/worker-acceptance-matrix.md). Keep platform code outside Core.
- **Improve the project.** Report reproducible problems, clarify documentation,
  or contribute focused fixes with relevant checks from the [test suite](tests/).

[Open an issue](https://github.com/liangyuwang/AutoTrain-Bench/issues) to discuss
an experiment, integration, or proposal, or
[submit a pull request](https://github.com/liangyuwang/AutoTrain-Bench/pulls).

## Documentation

The previous README's detailed content is preserved in the
[benchmark reference](docs/benchmark-reference.md).

| Read more | Contents |
| --- | --- |
| [Benchmark reference](docs/benchmark-reference.md) | Motivation, complete rules, Track accounting, evaluation, isolation, and local operating instructions |
| [Challenges](challenges/README.md) | Preview tasks, release definitions, and comparison lanes |
| [Worker Protocol](docs/worker-protocol.md) and [integration](docs/worker-integration.md) | Execution contracts and external deployment responsibilities |
| [Operator status](docs/operator-status.md) and [recovery](docs/worker-recovery.md) | Monitoring, interruption evidence, and owned-resource cleanup |
| [Development plan](docs/thin-core-development-plan.md#current-preview-closeout-boundary) and [acceptance matrix](docs/worker-acceptance-matrix.md) | Preview implementation status, remaining deployment checks, and deferred formal release work |
