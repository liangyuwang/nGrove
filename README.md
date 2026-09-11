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

[Core idea](#the-core-idea) · [Results](#preview-results) · [Try it](#try-it) ·
[Get involved](#get-involved) · [Documentation](#documentation)

## The core idea

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

## Preview results

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

## Try it

Start with the small [Smoke LM pretrain task](dev_tasks/smoke_lm-pretrain/README.md) to exercise a
complete episode before preparing the larger LM preview.

You need **Python 3.9+**, **Docker** for the bundled local runtime, and **your own
Agent**. The repository does not ship a reference Agent. CPU smoke runs need no
GPU; NVIDIA GPU runs additionally require the NVIDIA Container Toolkit.

From a checkout of this repository, install the CLI and inspect the smoke
configurations:

```bash
python3 -m pip install -e .
autotrain-bench challenge-info \
  --challenge dev_tasks/smoke_lm-pretrain/challenge.json
```

Then follow the [data preparation](dev_tasks/smoke_lm-pretrain/README.md#prepare) and
[Agent integration](dev_tasks/smoke_lm-pretrain/README.md#run-an-agent-integration) steps.
They cover the Agent command, resource settings, and a dedicated working root
outside the Core checkout.

- **Local Docker run:** [quick start](docs/benchmark-reference.md#quick-start).
- **Existing host CLI Agent:** [host-native preview setup](docs/benchmark-reference.md#bring-your-own-agent-host-native-cli-tools).
- **Larger training experiment:** [LM pretrain preview](challenges/lm-pretrain/README.md).
- **Your own execution environment:** [external Worker integration](docs/worker-integration.md).

If an Agent is helping you operate the benchmark, it should follow
[AGENTS.md](AGENTS.md) and read the [full reference](docs/benchmark-reference.md)
before preparing or launching a run.

## Get involved

Bring an Agent, an execution environment, or a contribution to the benchmark:

- **Evaluate your Agent.** Start with the smoke task and use the
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
