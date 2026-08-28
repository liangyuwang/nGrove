# AutoTrain-Bench

**An open-ended benchmark for autonomous training research and recursive self-improvement.**

> **Status:** v0 local reference prototype
> **Project name:** AutoTrain-Bench
> Dataset, hardware, and concrete resource budgets are intentionally not fixed yet.

---

## Overview

AutoTrain-Bench is a benchmark for evaluating **autonomous systems that design, implement, run, analyze, and improve machine-learning training systems**.

The benchmark is inspired by training speedruns, but asks a broader question.

Instead of fixing a small model and asking:

> How quickly can this model reach a target validation loss?

AutoTrain-Bench asks:

> **Given a fixed task, fixed data, fixed training hardware, and a constrained opportunity to interact with the training environment, how good a model can an autonomous research system produce?**

The benchmark intentionally leaves most of the training stack open.

An AutoTrain system may explore:

* model size;
* model architecture;
* non-Transformer architectures;
* optimizers;
* learning-rate schedules;
* initialization;
* training objectives;
* data subsets;
* data mixtures;
* curriculum strategies;
* sequence lengths;
* batch sizes;
* precision formats;
* kernels;
* compilation;
* distributed strategies;
* communication;
* memory management;
* checkpointing;
* data loading;
* profiling;
* experiment strategy;
* research workflow;
* and other improvements not anticipated by the benchmark authors.

There is no reference `train.py` that participants are expected to tune.

The AutoTrain system is responsible for building the training system itself.

The primary objective is simple:

$$
\boxed{\text{Minimize hidden TEST loss}}
$$

under the resource constraints of the selected benchmark track.

---

# Local Reference Prototype

The repository now contains an executable, single-host Docker prototype of the benchmark protocol. It is intended to validate episode lifecycle, Training Lease accounting, hidden evaluation, and the agent-facing control surface before the first official benchmark instance is selected.

The prototype deliberately assumes that the user has already obtained a machine matching the task's hardware specification. AutoTrain-Bench verifies and consumes resources visible on that machine; it does **not** allocate machines or integrate with Slurm, Kubernetes, or a cloud scheduler.

## Quick start

Requirements:

* Python 3.9 or newer;
* Docker;
* NVIDIA Container Toolkit only when the selected task requires GPUs.

Install the local CLI:

```bash
python3 -m pip install -e .
```

Prepare the bundled CPU toy task and pull its fixed images:

```bash
autotrain-bench prepare --task examples/toy/task.json
```

Run a complete Track B episode with the example autonomous agent:

```bash
autotrain-bench start --task examples/toy/task.json --track B
```

For a still inexpensive but substantially more realistic rehearsal, prepare the
fixed TinyStories mini benchmark and run its five-minute Track A:

```bash
python3 examples/mini_lm/prepare_data.py
autotrain-bench prepare --task examples/mini_lm/task.json
autotrain-bench start --task examples/mini_lm/task.json --track A
```

This second example downloads about 0.98 MB of real text, uses a fixed byte-level
tokenizer, hidden 8K-token VAL/TEST splits, pinned container images, and three
validation-guided training leases. See `examples/mini_lm/README.md` for the full
official-to-mini mapping and reproducibility details.

A participant Agent Container can be selected without changing the benchmark task:

```bash
autotrain-bench start \
  --task path/to/task.json \
  --track B \
  --agent-image participant-agent:version \
  --agent-command "python /workspace/agent.py"
```

## Bring your own Agent: controlled inference

AutoTrain-Bench is intended to benchmark participant-defined autonomous Agent
systems, including systems whose researcher model is served remotely. The Agent
execution loop still runs inside the Agent Container so the benchmark can enforce
its policy boundary; remote model inference is exposed through a narrow
official Gateway sidecar rather than arbitrary public Internet access.

In the normal `sandboxed` mode an Agent without remote inference uses
`--network none`. When a registered inference channel is configured,
AutoTrain-Bench creates an episode-private Docker `--internal` network shared
only by the Agent and a trusted OpenAI-compatible Gateway at:

```text
http://inference-gateway:8765/v1
```

The Agent has no route from that internal network to the public Internet. The
official Gateway alone also joins a separate egress network and owns the real
provider credential. It is not a generic HTTP proxy: the task configuration fixes
the upstream origin, allowed API paths, and model names before task reveal;
redirects and provider-hosted external tools are rejected. The Agent never
receives the provider credential or chooses arbitrary destinations.

A minimal configuration looks like:

```json
{
  "images": {
    "inference_gateway": "python:3.11-slim"
  },
  "agent": {
    "command": ["python", "/workspace/agent.py"],
    "mode": "sandboxed",
    "environment": [],
    "inference": {
      "protocol": "openai_compatible",
      "base_url": "https://api.openai.com",
      "api_key_env": "OPENAI_API_KEY",
      "allowed_models": ["registered-researcher-model"],
      "allowed_paths": ["/v1/responses", "/v1/chat/completions"]
    }
  }
}
```

The user exports the credential only in the host environment before starting the
episode:

```bash
export OPENAI_API_KEY="..."
autotrain-bench start --task path/to/task.json --track B
```

For compatibility with common OpenAI-style clients, the Agent Container receives
`OPENAI_BASE_URL=http://inference-gateway:8765/v1` and a dummy local API key. The
real key is copied only into the trusted Gateway. V0 permits ordinary
function-tool schemas because the tool is executed by the sandboxed Agent, but
rejects provider-hosted web/retrieval tools that would create a second
external-information channel. JSON and SSE responses are forwarded incrementally;
the Gateway writes model calls and provider-reported usage to the append-only
`inference/inference.jsonl` trace. Missing usage is explicitly marked incomplete.
Other provider protocols can be added as small adapters later.

The Gateway is a separate official container, so enabling inference does not
replace or wrap the participant command. Node, Go, Rust, Python, and other Agent
images continue to run their registered entrypoint unchanged.

The Agent can use the benchmark package only as the control plane for training:

```bash
python -m autotrain_bench.cli agent status
python -m autotrain_bench.cli agent lease-start
python -m autotrain_bench.cli agent lease-exec -- python /workspace/train.py
python -m autotrain_bench.cli agent lease-commit candidate
python -m autotrain_bench.cli agent finish
```

The submitted Agent command must be autonomous and non-interactive. Docker does
not attach a user input stream after task reveal, so the user cannot continue
steering the Agent during the episode. Each episode receives a private copy of
the prepared Agent workspace.

For debugging only, `agent.mode` may be set to `unrestricted-dev`. That gives the
Agent Container ordinary Docker bridge egress and marks the run
`policy_compliant: false`. It is not a closed AutoTrain-Bench result and exists
only to make local integration easier.

The Supervisor prints the episode directory. Its final state can be inspected later with:

```bash
autotrain-bench report --episode .autotrain/episodes/<episode-id>
```

The example proves the complete protocol:

```text
task reveal
→ interactive lease
→ TRAIN-only experimentation
→ candidate commit
→ hidden VAL scalar
→ autonomous revision
→ finish
→ episode closed
→ hidden TEST
```

## v0 scope and security

The reference implementation uses an official host-side Supervisor, an Agent Container, a thin trusted Inference Gateway when registered, disposable Training Containers, and evaluator-only Eval Containers. No participant container receives the Docker socket or VAL/TEST content. Agent public networking is disabled in policy-compliant runs. The Gateway handles only frozen researcher-inference endpoints while the Supervisor independently enforces episode and GPU-lease deadlines. Training and Eval networking remains disabled. `unrestricted-dev` exists only as a non-compliant debugging mode.

The v0 threat model protects benchmark policy from the Agent, not from the user or machine administrator. Docker isolation is a development reference boundary and is not presented as protection against kernel, container-runtime, or GPU-driver exploits. Official adversarial evaluation may require stronger worker or VM isolation.

The bundled evaluator uses a provisional dependency-free Candidate ABI. It is intentionally not the final high-performance serving interface.

---

# Motivation

Most training benchmarks deliberately restrict the search space.

This is useful when the goal is to isolate a particular algorithmic or systems contribution, but it is less suitable for evaluating autonomous ML research systems.

A capable autonomous researcher should be able to decide not only:

> What learning rate should I use?

but also:

> What should I train?

> How large should the model be?

> Should this even be a Transformer?

> Which parts of the dataset are worth spending compute on?

> Is my current bottleneck algorithmic or infrastructural?

> Should I improve the optimizer, the architecture, the kernels, or the research process itself?

> What experiment should I run next?

AutoTrain-Bench therefore treats the **entire training system as the search space**.

The benchmark is intended to measure the capability of a system to perform autonomous ML systems research, rather than its ability to optimize a predefined training script.

---

# Core Benchmark Question

Let:

* $S_0$ be the AutoTrain system at the moment the benchmark task is revealed;
* $S_t$ be the system after autonomous improvement during the episode;
* $M_t$ be a target model or training system produced by $S_t$.

AutoTrain-Bench studies trajectories of the form:

$$
S_0
\rightarrow
S_1
\rightarrow
S_2
\rightarrow
\cdots
$$

while simultaneously producing increasingly capable training systems:

$$
S_t
\rightarrow
M_t.
$$

The benchmark therefore does **not** require the AutoTrain system itself to remain static.

It only requires its **initial state and external boundary** to be fixed.

The central principle is:

> **External improvement is frozen. Autonomous improvement is allowed.**

After task reveal, no human developer may patch, steer, upgrade, or otherwise modify the submitted AutoTrain system.

However, the AutoTrain system may autonomously modify its own task-local behavior, software, tools, memory, workflow, experimental strategy, and other artifacts within the resources and permissions of the benchmark.

This distinction is fundamental to the benchmark.

---

# System Boundaries

AutoTrain-Bench v0 separates four components.

```text
User-owned host
└── trusted AutoTrain-Bench Supervisor
    ├── untrusted Agent Container
    ├── disposable Training Container
    └── evaluator-only Eval Container
```

The official Supervisor records task reveal time, owns Track A deadlines and Track B lease counters, controls container lifecycle, exposes already-allocated local GPUs, and invokes evaluation. The Agent cannot access the Supervisor's state or Docker socket.

The Agent receives interactive access to a Training Container only through the Supervisor's narrow command channel during an active **Training Lease**. It may inspect logs, edit the shared workspace, compile code, profile, and run any number of TRAIN-only experiments inside that lease.

VAL and TEST content exists only in Eval Containers. A committed candidate receives at most one official VAL evaluation per lease, and the Agent receives only the permitted aggregate metric. TEST cannot run until the episode has been permanently closed.

---

# The Benchmark Task

Each benchmark release defines a task containing:

### TRAIN

A large fixed training corpus.

The TRAIN pool should be sufficiently large that consuming all available data is not automatically the optimal strategy under the benchmark compute budget.

The AutoTrain system may choose which portion of TRAIN to use.

For example, it may autonomously explore:

* random subsets;
* quality filtering;
* domain mixtures;
* deduplication;
* difficulty filtering;
* curriculum learning;
* dynamic sampling;
* sequence packing;
* sample weighting;
* repeated exposure;
* model-dependent sampling.

Importantly, **data policy is part of the research problem**.

The system is explicitly allowed to modify its data policy in response to validation results.

For example:

```text
Run 1
random subset
↓
VAL = 2.95

Run 2
quality-filtered subset
↓
VAL = 2.91

Run 3
new domain mixture
↓
VAL = 2.87

Run 4
dynamic curriculum
↓
VAL = 2.84
```

This is intended behavior.

AutoTrain-Bench does not attempt to prevent validation-guided dataset research.

Generalization is enforced by TEST.

---

## VAL

VAL is the benchmark's **research feedback channel**.

After a Training Lease commits a candidate and closes, the official evaluator may return the fixed validation metric to the AutoTrain system.

The AutoTrain system is free to use this result to change anything it controls, including:

* architecture;
* model size;
* optimizer;
* data selection;
* data mixture;
* training objective;
* infrastructure;
* experiment strategy;
* or its own autonomous research process.

By default, the benchmark returns an aggregate validation metric rather than arbitrary per-example queries.

The core metric is autoregressive validation negative log-likelihood under the benchmark tokenizer.

Exact auxiliary metrics, if any, are defined by the benchmark release.

Overfitting VAL is not considered a rules violation.

It is considered a research failure if it harms TEST performance.

---

## TEST

TEST is the benchmark's final generalization metric.

TEST is:

* hidden from the AutoTrain system;
* unavailable through the training environment;
* unavailable through the agent environment;
* unavailable during the active research episode.

TEST results are **not** computed and returned after each experiment.

Instead, the benchmark first permanently terminates the research episode and freezes the relevant artifacts.

Only then does the official evaluator run TEST.

Therefore:

$$
\boxed{\text{VAL is part of the research loop}}
$$

while:

$$
\boxed{\text{TEST is post-mortem only}}
$$

The system cannot resume research after receiving TEST results.

---

# Fixed Evaluation Semantics

The benchmark fixes the tokenizer and evaluation semantics.

A benchmark release specifies:

* tokenizer;
* vocabulary;
* tokenization procedure;
* context/evaluation protocol;
* evaluation dataset;
* loss implementation.

The default primary metric is autoregressive token negative log-likelihood:

$$
L
=
-\frac{1}{N}
\sum_t
\log p(x_t \mid x_{<t}).
$$

The internal architecture is unrestricted.

A submission may use:

* Transformers;
* recurrent networks;
* state-space models;
* mixtures of experts;
* hybrids;
* novel architectures;
* or future architectures not anticipated by this specification.

The only requirement is that the final artifact implements the benchmark evaluation interface.

Fixing the tokenizer ensures that loss remains comparable across models.

---

# Open Training Stack

Unless explicitly restricted by a benchmark release, the AutoTrain system may modify any participant-controlled component of training.

Examples include:

### Model

* architecture;
* depth;
* width;
* parameter count;
* attention structure;
* recurrence;
* sparsity;
* MoE routing;
* normalization;
* activation functions;
* positional representation.

### Optimization

* optimizer;
* learning-rate schedule;
* initialization;
* gradient clipping;
* regularization;
* loss functions;
* auxiliary objectives;
* batch schedule.

### Data

* subset selection;
* filtering;
* ordering;
* weighting;
* curriculum;
* packing;
* masking;
* transformations;
* dynamic sampling.

### Systems

* CUDA kernels;
* Triton kernels;
* operator fusion;
* precision;
* compilation;
* graph capture;
* memory layouts;
* activation checkpointing;
* distributed parallelism;
* communication overlap;
* dataloading;
* checkpointing;
* runtime scheduling.

### Research Process

* experiment design;
* experiment allocation;
* automated profiling;
* code generation;
* failure analysis;
* hypothesis generation;
* research memory;
* tool creation;
* subagents;
* task-local self-improvement.

There is intentionally no fixed model size.

Choosing the appropriate model scale is itself part of the benchmark.

---

# Training Leases

A **Training Lease** is a fixed-duration interactive window in which the Agent may use the benchmark's already-allocated training hardware.

```text
Agent
  │
  │ lease start
  ▼
Supervisor creates Training Container
  │
  ├── edit shared workspace
  ├── execute commands
  ├── inspect logs
  ├── compile and profile
  └── run unlimited TRAIN-only experiments
  │
  │ commit one candidate, stop, or time out
  ▼
lease permanently ends
  │
  └── committed candidate → one hidden VAL query → aggregate metric
```

The Agent interacts with the Training Container only through Supervisor-controlled operations. It never receives Docker or host control.

Within a lease, the search process is unrestricted by experiment count. The Agent may repeatedly change architecture, optimization, data policy, kernels, distributed code, or its own research tools as long as it remains within the lease's hardware and time limits.

Each lease may commit at most one candidate. Commit immediately terminates the lease before VAL is evaluated, so VAL feedback can influence only a later lease. A lease may also end without a candidate because of failure, timeout, or an explicit stop.

The lease is consumed when the Training Container starts successfully. Participant-controlled crashes, OOMs, compile failures, early termination, or failure to commit do not restore it. A failure before the Training Container starts is an infrastructure failure and does not consume the lease.

---

# Track A — Time-Bounded Autonomous Research

Track A constrains total elapsed research time.

The benchmark defines:

$$
T_{\text{total}}.
$$

The clock begins when the task is revealed to the submitted AutoTrain system.

The same clock covers the complete autonomous trajectory:

```text
TASK REVEAL
│
├── reasoning
├── data analysis
├── coding
├── self-improvement
├── Training Lease 1
├── VAL feedback
├── analysis
├── Training Lease 2
├── infrastructure changes
├── architecture changes
├── more research
│
└── DEADLINE
```

The system may open as many Training Leases as fit inside the total budget. A lease can never extend beyond the Track A deadline.

There is no requirement to follow a particular experiment strategy.

A system may choose to:

* perform many small experiments;
* perform a few large experiments;
* spend substantial time reasoning before training;
* build additional research tooling;
* rewrite its training stack;
* improve its own autonomous workflow.

All of these decisions compete for the same wall-clock budget.

The primary Track A score is:

$$
\boxed{
L_{\mathrm{TEST}}(T_{\mathrm{total}})
}
$$

for the system's final declared incumbent at the deadline.

### What Track A measures

Track A asks:

> **How much useful training capability can an autonomous system create within a fixed amount of real time?**

It rewards both research quality and research speed.

---

# Track B — Lease-Bounded Autonomous Research

Track B constrains the system's access to the expensive training environment rather than its total research wall-clock.

A benchmark release defines:

$$
N_{\mathrm{leases}}
$$

and:

$$
T_{\mathrm{lease}}.
$$

For example:

```text
Maximum Training Leases:   TBD
Maximum Time per Lease:    TBD
Training Hardware:         TBD
VAL Queries per Lease:     1
Agent Research Time:       Unbounded by benchmark score
```

The AutoTrain system may reason, write code, analyze previous results, and autonomously improve for as long as it chooses between Training Leases.

However, it may obtain interactive access to the training hardware only a fixed number of times.

Conceptually:

```text
TASK
 │
 ▼
Agent research
 │
 ▼
LEASE 1
 │
 ▼
VAL 1
 │
 ▼
Agent research / self-improvement
 │
 ▼
LEASE 2
 │
 ▼
VAL 2
 │
 ...
 ▼
LEASE N
 │
 ▼
FINAL SUBMISSION
```

A lease that starts consumes one lease budget even if participant code crashes, OOMs, diverges, stops early, or otherwise fails.

Failures that occur before the Training Container starts do not consume the participant's lease budget.

The primary Track B score is:

$$
\boxed{
L_{\mathrm{TEST}}(N_{\mathrm{leases}})
}
$$

for the final incumbent selected by the AutoTrain system.

### What Track B measures

Track B asks:

> **How effectively can an autonomous researcher use a limited number of training-hardware windows and official validation-feedback rounds?**

It emphasizes:

* experiment selection;
* strength of prior reasoning;
* information gained per lease and VAL query;
* interpretation of failures;
* hypothesis quality;
* efficiency of autonomous research under bounded accelerator access.

Track B intentionally does **not** optimize total research wall-clock.

Agent-side reasoning time may be unlimited.

Agent inference usage is still recorded.

---

# Track A vs. Track B

The two tracks constrain different scarce resources.

|                     | Track A                          | Track B                                        |
| ------------------- | -------------------------------- | ---------------------------------------------- |
| Primary constraint  | Total wall-clock                 | Training Lease count and duration              |
| Per-lease time      | Bounded by remaining total time  | Fixed maximum                                  |
| Agent research time | Counts against deadline          | Not score-limited                              |
| Number of leases    | Unlimited within deadline        | Fixed                                          |
| Experiments in lease| Unlimited within time/hardware   | Unlimited within time/hardware                 |
| Official VAL        | At most once per committed lease | At most once per committed lease               |
| Self-improvement    | Allowed                          | Allowed                                        |
| Human intervention  | Forbidden                        | Forbidden                                      |
| Primary question    | How fast can the system improve? | How well can it use bounded GPU access?         |

Together they expose two different scaling behaviors.

Track A measures:

$$
L_{\mathrm{TEST}}(t).
$$

Track B measures:

$$
L_{\mathrm{TEST}}(n).
$$

where $t$ is autonomous research time and $n$ is the number of Training Leases and official feedback rounds.

---

# Autonomous Self-Improvement

The submitted system is frozen only at the **start boundary**.

Before task reveal, the participant registers the initial AutoTrain system $S_0$.

After task reveal:

> Anything generated autonomously and causally by $S_0$, using permitted resources, may become part of the system's subsequent trajectory.

Therefore, a system may autonomously:

* rewrite its own task code;
* build new research tools;
* modify prompts;
* restructure subagents;
* create new memory mechanisms;
* change experiment-management logic;
* improve its profiling system;
* generate new search procedures;
* alter its task-local research workflow.

If future versions allow the system to use benchmark training compute to improve components of the researcher itself, that compute must consume the same benchmark resources that would otherwise be available for training.

What is forbidden is **external intervention**.

After task reveal, a human developer may not:

* inspect results and push a patch;
* tell the agent what experiment to run;
* modify prompts manually;
* repair participant code;
* change the agent model;
* provide new task-specific information.

The benchmark freezes the outside world, not the system's internal state.

---

# Agent System Definition

An AutoTrain submission includes its initial autonomous research system.

This may contain:

* one or more foundation models;
* prompts;
* orchestrators;
* planners;
* coding agents;
* subagents;
* memory systems;
* context-management systems;
* static tools;
* experiment managers;
* model-routing logic.

The complete initial system configuration must be registered before task reveal.

Multi-agent systems are allowed.

The benchmark evaluates the **whole submitted AutoTrain system**, not an isolated foundation model.

---

# Agent Tokens

Agent inference is a first-class measured resource.

The benchmark records, at minimum:

* input tokens;
* output tokens;
* total model calls;
* token usage over time;
* token usage between Training Leases.

For multi-agent systems, usage is aggregated across the complete AutoTrain system.

This makes it possible to study:

$$
L_{\mathrm{TEST}}
= f(
\text{agent tokens},
\text{research time},
\text{lease and validation interactions}
).
$$

In Track A, agent reasoning naturally competes with training and engineering for the total wall-clock budget.

In Track B, agent reasoning time is not capped by the primary benchmark constraint, so token usage is especially important as a diagnostic measure of research efficiency.

Token count is not assumed to represent identical FLOPs across different underlying agent models.

The initial benchmark therefore treats tokens as a **first-class trace and scaling variable**, rather than using them as the sole cross-system compute normalization.

---

# Research History

The AutoTrain system is allowed to retain the complete history of its benchmark episode.

This may include:

* previous source code;
* experiment configurations;
* training logs;
* compiler errors;
* profiler results;
* throughput;
* model checkpoints;
* VAL scores;
* previous reasoning;
* research notes;
* experiment lineage.

There is no artificial memory reset between experiments.

This is intentional.

A capable autonomous researcher should be able to learn from its own previous work.

The benchmark controls information through evaluator and environment boundaries rather than by forcing the agent to forget legitimate history.

---

# Validation-Guided Data Research

AutoTrain-Bench explicitly permits using VAL results to change the training-data strategy.

For example:

$$
\text{VAL feedback}
\rightarrow
\text{new subset}
\rightarrow
\text{new mixture}
\rightarrow
\text{new curriculum}.
$$

This may eventually overfit VAL.

That is not prohibited.

The hidden TEST set exists specifically to distinguish genuine improvement from validation overfitting.

A capable AutoTrain system may therefore need to autonomously discover good research practices such as:

* internal holdouts;
* uncertainty estimation;
* replication;
* multiple seeds;
* robustness checks;
* avoiding overreaction to small validation changes.

The benchmark does not prescribe these practices.

Discovering when and how to use them is part of the task.

---

# Intermediate Incumbents and Research Curves

The benchmark should not record only a single final score.

Throughout an episode, the AutoTrain system may declare a **current incumbent**: the model it would submit if the benchmark ended immediately.

Each incumbent declaration records:

* artifact hash;
* timestamp;
* lease number;
* VAL score;
* model metadata;
* current research resource usage.

TEST remains unavailable.

After the episode is permanently closed, the evaluator may evaluate historical incumbents on TEST.

This produces a hidden generalization trajectory.

For Track A:

$$
\boxed{
L_{\mathrm{TEST}}(t)
}
$$

For Track B:

$$
\boxed{
L_{\mathrm{TEST}}(n_{\mathrm{leases}})
}
$$

These curves are first-class benchmark outputs.

They can reveal behavior that a final scalar cannot.

For example:

```text
System A:
strong initial solution
little subsequent improvement

System B:
weak initial solution
rapid autonomous improvement
eventually surpasses A
```

The second system may be more interesting from an RSI perspective even when the two systems have similar final scores.

---

# Primary and Secondary Metrics

## Primary metric

The primary leaderboard metric is:

$$
\boxed{\text{final hidden TEST NLL}}
$$

under the selected track's resource constraint.

Lower is better.

The benchmark should avoid combining many metrics into a complicated scoring formula.

---

## First-class diagnostic metrics

The benchmark also records:

* VAL NLL;
* TEST NLL trajectory;
* final VAL–TEST gap;
* agent input/output tokens;
* number of agent calls;
* number and duration of Training Leases;
* total training wall-clock;
* hardware-hours;
* tokens processed by the target model;
* model parameter count;
* active parameter count where relevant;
* throughput;
* compilation time;
* initialization time;
* run failures;
* incumbent changes.

These metrics are intended for scientific analysis rather than replacing the primary objective.

---

# Training-Time Accounting

Participant-controlled setup performed inside a Training Lease counts toward that lease's execution budget.

Examples include:

* CUDA compilation;
* Triton compilation;
* autotuning;
* `torch.compile`;
* graph capture;
* model initialization;
* participant-specific preprocessing;
* profiling;
* checkpoint loading;
* training itself.

Benchmark-owned preparation may occur outside participant timing.

Examples may include:

* provisioning the clean machine;
* mounting the official dataset;
* benchmark-standard tokenization or preprocessing;
* benchmark infrastructure startup.

The exact boundary is defined by each benchmark release and must be identical across participants.

---

# Isolation

The benchmark should enforce strong isolation between the autonomous researcher and the training environment.

The AutoTrain environment may:

* inspect permitted TRAIN data;
* reason;
* write code;
* modify its shared research workspace;
* analyze returned logs;
* store research state.

It may request and interact with a Training Container only through the Supervisor-controlled lease interface.

The Training Environment:

* executes on fixed, already-allocated local hardware;
* shares the Agent workspace during an active lease;
* permits repeated interactive commands inside that lease;
* has no Docker socket, public network, VAL, or TEST access;
* is forcibly terminated by the host Supervisor at lease expiry.

The Eval Environment receives only a read-only candidate and the selected hidden split. It is inaccessible to Agent and Training Containers and returns only benchmark-permitted aggregate metrics.

---

# External Resources

The core benchmark is intended to be reproducible and closed-task.

A benchmark release should define exactly which external capabilities the AutoTrain system may access.

The default direction for the core track is:

* no external datasets;
* no arbitrary Internet access;
* no external pretrained target-model checkpoints;
* no human assistance after task reveal.

Each release separates an online **Preparation Phase** from the benchmark episode. Preparation may download versioned official images, the tokenized TRAIN corpus, tokenizer assets, compilers, frameworks, and an offline package bundle. Dataset and image hashes are verified before task reveal.

The first official Training Environment will be fixed by the benchmark release rather than extended by each user. In policy-compliant runs, Agent and Training Containers have no arbitrary public-network access. The fixed training environment may still include CUDA, Triton, compilers, headers, source trees, and other tools that allow the Agent to build or replace participant-controlled infrastructure within its resource budget.

Remote researcher inference is allowed only through a registered, Supervisor-managed trusted Gateway. The upstream origin, permitted API paths, and permitted model identities are frozen before task reveal; provider credentials remain outside the Agent Container. The Gateway cannot redirect to another origin and records calls and provider-reported token usage as part of the episode trace.

The exact policy for synthetic training examples generated autonomously by the submitted system will be specified before the first official dataset release.

---

# Failures

Failures are part of autonomous research.

Examples include:

* syntax errors;
* compile failures;
* CUDA errors;
* OOMs;
* divergence;
* deadlocks;
* invalid checkpoints.

In Track B, a lease is consumed as soon as its Training Container starts successfully, even if participant-controlled code later fails.

Benchmark infrastructure failures are excluded and may be rerun.

The infrastructure-failure policy must be deterministic and auditable.

---

# Reproducibility and Logging

Every official episode should produce an append-only benchmark trace.

At minimum, the benchmark records:

```text
task manifest
initial AutoTrain system identity
agent token usage
agent/tool invocation metadata
lease start/end timestamps
lease command metadata
candidate artifact hashes
training logs
hardware telemetry
VAL outputs
incumbent declarations
final artifact
post-mortem TEST results
```

Where possible, official candidates and episode traces should be reproducible from immutable artifacts.

The benchmark should preserve enough information to distinguish:

* algorithmic improvements;
* infrastructure improvements;
* data-policy improvements;
* research-strategy improvements;
* stochastic variation.

---

# Leaderboards

AutoTrain-Bench should maintain separate leaderboards for the two tracks.

## Track A

Ranked primarily by:

```text
final TEST NLL at fixed total wall-clock
```

Reported alongside:

```text
research-time curve
agent tokens
Training Leases
training compute
model size
```

## Track B

Ranked primarily by:

```text
final TEST NLL after at most N Training Leases
```

with fixed maximum duration per lease and at most one official VAL query per committed lease.

Reported alongside:

```text
TEST loss by lease number
agent tokens
agent reasoning between leases
training compute
model size
```

Different benchmark releases may use different hardware or budget scales and should not be mixed into a single leaderboard.

---

# Example Benchmark Configuration

The executable v0 task manifest is JSON and keeps task and track budgets in one small file:

```json
{
  "task_id": "toy-bigram-v0",
  "paths": {
    "train": "data",
    "val": "private/val.json",
    "test": "private/test.json",
    "agent_workspace": "agent"
  },
  "images": {
    "agent": "python:3.11-slim",
    "training": "python:3.11-slim",
    "evaluator": "python:3.11-slim"
  },
  "agent": {
    "command": ["python", "/workspace/agent.py"],
    "mode": "sandboxed",
    "environment": []
  },
  "hardware": { "gpu_count": 0, "gpu_model": null },
  "track_a": { "total_seconds": 120 },
  "track_b": { "max_leases": 2, "lease_seconds": 30 },
  "evaluation": {
    "metric": "autoregressive_nll",
    "vocab_size": 3,
    "context_length": 16
  }
}
```

See `examples/toy/task.json` for the complete runnable manifest. Dataset and hardware selection for the first official release remain intentionally deferred.

---

# Design Principles

AutoTrain-Bench follows several principles.

### 1. Fix the task, not the solution

The benchmark fixes the environment and evaluation semantics while leaving the training system open.

### 2. TEST, not rules, should punish overfitting

If an autonomous system over-optimizes VAL, its TEST score should reveal the failure.

The benchmark should avoid unnecessary restrictions on legitimate research.

### 3. Treat infrastructure as part of ML research

Kernel engineering, compilation, parallelism, communication, memory management, and data pipelines are part of the search space.

### 4. Preserve autonomous history

Learning from previous experiments is a core capability, not leakage.

### 5. Permit autonomous self-improvement

The initial system is fixed. Its autonomous trajectory is not.

### 6. Prevent external intervention

The system should succeed or fail based on its own actions after task reveal.

### 7. Keep the primary objective simple

The final objective is hidden TEST loss.

### 8. Record trajectories, not only endpoints

The way a system improves can be as scientifically important as where it ends.

---

# What AutoTrain-Bench Is Intended to Measure

AutoTrain-Bench is designed to expose capabilities including:

* autonomous ML research;
* autonomous systems engineering;
* experimental design;
* scientific reasoning;
* data-policy research;
* infrastructure optimization;
* resource allocation;
* learning from failed experiments;
* autonomous code generation;
* autonomous research memory;
* task-local self-improvement;
* improvement scaling with additional time or empirical feedback.

It is particularly interested in the difference between:

> **already knowing a strong recipe**

and:

> **being able to discover a stronger recipe through autonomous research.**

---

# What AutoTrain-Bench Does Not Claim to Measure

AutoTrain-Bench is not a complete measure of general intelligence or general recursive self-improvement.

A high score demonstrates strong autonomous capability within a specific machine-learning research environment.

The benchmark should therefore be understood as an **RSI-oriented empirical testbed**, not as a universal measure of recursive self-improvement.

Its value comes from providing a concrete environment in which improvement itself can be measured repeatedly and quantitatively.

---

# Research Questions

Beyond leaderboard ranking, AutoTrain-Bench is intended to support questions such as:

### Research-time scaling

How does model quality scale with autonomous research time?

$$
L_{\mathrm{TEST}}(t)
$$

### Lease-efficiency scaling

How does model quality scale with the number of Training Leases and official validation-feedback rounds?

$$
L_{\mathrm{TEST}}(n)
$$

### Agent-compute scaling

How does performance scale with researcher inference?

$$
L_{\mathrm{TEST}}(N_{\mathrm{agent\ tokens}})
$$

### Research gain

How much better does a system become after receiving experimental feedback?

$$
\Delta L
=
L_{\mathrm{initial}}
-
L_{\mathrm{final}}.
$$

### Validation overfitting

When does continued autonomous optimization stop improving hidden generalization?

### Research strategy

Do strong systems win by:

* running more experiments;
* choosing better experiments;
* reasoning more between experiments;
* improving infrastructure;
* improving data;
* improving algorithms;
* or improving their own research process?

These questions are central to the project.

---

# Project Status

The repository contains a working v0 local reference prototype with:

* an official host-side Supervisor;
* Track A wall-clock enforcement;
* Track B interactive lease count and duration enforcement;
* a Supervisor-controlled Agent command mailbox;
* disposable Docker Training Containers;
* candidate hashing and immutable snapshots;
* sandboxed Agent execution with an optional trusted streaming inference sidecar;
* hidden VAL and post-closure TEST evaluation;
* append-only `events.jsonl` and `inference/inference.jsonl` traces;
* a deterministic CPU toy task plus a reproducible real-data TinyStories mini benchmark.

The first official benchmark instance has not been selected. Major remaining release decisions include the real TRAIN/VAL/TEST construction, tokenizer, hardware and budgets, fixed software image, production Candidate ABI, additional inference-provider adapters, synthetic-data policy, and stronger official isolation.

The v0 prototype is intentionally single-host and resource-allocation agnostic. Users obtain a machine matching the release specification before starting AutoTrain-Bench.

---

# Summary

AutoTrain-Bench evaluates an autonomous system starting from a fixed initial state:

$$
S_0.
$$

After task reveal, the system may autonomously research, experiment, write software, modify its training strategy, and improve its own task-local process.

The benchmark provides:

$$
\boxed{\text{Fixed TRAIN}}
$$

$$
\boxed{\text{Fixed tokenizer}}
$$

$$
\boxed{\text{Fixed training hardware}}
$$

$$
\boxed{\text{VAL feedback}}
$$

$$
\boxed{\text{Hidden TEST}}
$$

while leaving the training stack open.

Two complementary tracks constrain different resources:

$$
\boxed{
\text{Track A: fixed total autonomous research time}
}
$$

and:

$$
\boxed{
\text{Track B: fixed number and duration of Training Leases}
}
$$

with at most one official VAL query for each committed lease.

In both tracks:

> **The initial system is fixed. Autonomous improvement is allowed. External intervention is not.**

The final question is deliberately simple:

> **How good a training system can this autonomous researcher discover?**
