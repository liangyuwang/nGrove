# AutoTrain-Benchmark

**An open-ended benchmark for autonomous training research and recursive self-improvement.**

> **Status:** early design / specification
> **Project name:** AutoTrain-Benchmark
> Dataset, hardware, and concrete resource budgets are intentionally not fixed yet.

---

## Overview

AutoTrain-Benchmark is a benchmark for evaluating **autonomous systems that design, implement, run, analyze, and improve machine-learning training systems**.

The benchmark is inspired by training speedruns, but asks a broader question.

Instead of fixing a small model and asking:

> How quickly can this model reach a target validation loss?

AutoTrain-Benchmark asks:

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

AutoTrain-Benchmark therefore treats the **entire training system as the search space**.

The benchmark is intended to measure the capability of a system to perform autonomous ML systems research, rather than its ability to optimize a predefined training script.

---

# Core Benchmark Question

Let:

* $S_0$ be the AutoTrain system at the moment the benchmark task is revealed;
* $S_t$ be the system after autonomous improvement during the episode;
* $M_t$ be a target model or training system produced by $S_t$.

AutoTrain-Benchmark studies trajectories of the form:

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

AutoTrain-Benchmark separates three components.

```text
┌───────────────────────────┐
│      AutoTrain Agent      │
│                           │
│ reasoning                 │
│ coding                    │
│ experiment design         │
│ data analysis             │
│ self-improvement          │
│ research memory           │
└─────────────┬─────────────┘
              │
        SUBMIT TRAINING RUN
              │
              ▼
┌───────────────────────────┐
│    Training Environment   │
│                           │
│ fixed hardware            │
│ fixed runtime policy      │
│ benchmark TRAIN data      │
│ participant training code │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│        Evaluator          │
│                           │
│ hidden VAL                │
│ hidden TEST               │
│ fixed evaluation code     │
└───────────────────────────┘
```

The AutoTrain system does not receive interactive access to the training machines.

Instead, it constructs a training artifact and explicitly submits a **Training Run**.

Every Training Run executes in an isolated environment.

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

AutoTrain-Benchmark does not attempt to prevent validation-guided dataset research.

Generalization is enforced by TEST.

---

## VAL

VAL is the benchmark's **research feedback channel**.

After a Training Run is completed, the official evaluator may return the fixed validation metric to the AutoTrain system.

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

# Training Runs

A **Training Run** is the atomic interaction between the AutoTrain system and the expensive training environment.

The protocol is:

```text
Agent
  │
  │ creates code/config/artifact
  ▼
SUBMIT_RUN
  │
  │ immutable snapshot
  ▼
isolated training environment
  │
  │ no live agent intervention
  ▼
one candidate artifact
  │
  ▼
official VAL evaluation
  │
  ▼
result + permitted logs returned to Agent
```

Once a run begins, the external AutoTrain agent cannot interactively patch or control that run.

However, the submitted training program may contain arbitrary pre-programmed adaptive behavior.

For example, the training program may autonomously:

* adjust learning rates;
* change batch size;
* alter curriculum;
* change sequence length;
* perform online profiling;
* modify sampling probabilities;
* restart internal phases;
* use dynamic routing;
* make other training-time decisions.

The distinction is:

> **Adaptive training code is allowed. External researcher intervention during a run is not.**

A Training Run may internally perform arbitrary computation that fits within its hardware and time budget.

However, each Training Run produces at most one official candidate for the benchmark VAL query.

This preserves a clear definition of an empirical interaction with the benchmark evaluator.

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
├── Training Run 1
├── VAL feedback
├── analysis
├── Training Run 2
├── infrastructure changes
├── architecture changes
├── more research
│
└── DEADLINE
```

The system may perform as many Training Runs as fit inside the total budget.

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

# Track B — Interaction-Bounded Autonomous Research

Track B constrains the system's access to the expensive training environment rather than its total research wall-clock.

A benchmark release defines:

$$
N_{\mathrm{runs}}
$$

and:

$$
T_{\mathrm{run}}.
$$

For example:

```text
Maximum Training Runs:     TBD
Maximum Time per Run:      TBD
Training Hardware:         TBD
Agent Research Time:       Unbounded by benchmark score
```

The AutoTrain system may reason, write code, analyze previous results, and autonomously improve for as long as it chooses between Training Runs.

However, it may submit to the training environment only a fixed number of times.

Conceptually:

```text
TASK
 │
 ▼
Agent research
 │
 ▼
RUN 1
 │
 ▼
VAL 1
 │
 ▼
Agent research / self-improvement
 │
 ▼
RUN 2
 │
 ▼
VAL 2
 │
 ...
 ▼
RUN N
 │
 ▼
FINAL SUBMISSION
```

A run that is launched consumes one run budget even if participant code crashes, OOMs, diverges, or otherwise fails.

Failures caused by benchmark infrastructure do not consume the participant's run budget.

The primary Track B score is:

$$
\boxed{
L_{\mathrm{TEST}}(N_{\mathrm{runs}})
}
$$

for the final incumbent selected by the AutoTrain system.

### What Track B measures

Track B asks:

> **How effectively can an autonomous researcher learn from a limited number of expensive interactions with reality?**

It emphasizes:

* experiment selection;
* strength of prior reasoning;
* information gained per experiment;
* interpretation of failures;
* hypothesis quality;
* sample efficiency of autonomous scientific research.

Track B intentionally does **not** optimize total research wall-clock.

Agent-side reasoning time may be unlimited.

Agent inference usage is still recorded.

---

# Track A vs. Track B

The two tracks constrain different scarce resources.

|                     | Track A                          | Track B                                        |
| ------------------- | -------------------------------- | ---------------------------------------------- |
| Primary constraint  | Total wall-clock                 | Training Run count                             |
| Per-run time        | Bounded by remaining total time  | Fixed maximum                                  |
| Agent research time | Counts against deadline          | Not score-limited                              |
| Number of runs      | Unlimited within deadline        | Fixed                                          |
| VAL feedback        | After runs                       | After runs                                     |
| Self-improvement    | Allowed                          | Allowed                                        |
| Human intervention  | Forbidden                        | Forbidden                                      |
| Primary question    | How fast can the system improve? | How efficiently can it learn from experiments? |

Together they expose two different scaling behaviors.

Track A measures:

$$
L_{\mathrm{TEST}}(t).
$$

Track B measures:

$$
L_{\mathrm{TEST}}(n).
$$

where $t$ is autonomous research time and $n$ is the number of expensive training-environment interactions.

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
* token usage between Training Runs.

For multi-agent systems, usage is aggregated across the complete AutoTrain system.

This makes it possible to study:

$$
L_{\mathrm{TEST}}
= f(
\text{agent tokens},
\text{research time},
\text{training interactions}
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

AutoTrain-Benchmark explicitly permits using VAL results to change the training-data strategy.

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
* run number;
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
L_{\mathrm{TEST}}(n_{\mathrm{runs}})
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
* number of Training Runs;
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

Participant-controlled setup performed inside a Training Run counts toward that run's execution budget.

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
* construct run artifacts;
* analyze returned logs;
* store research state.

It may not directly enter the training environment.

The Training Environment:

* receives an immutable run artifact;
* executes on fixed hardware;
* has no interactive researcher connection;
* has no access to TEST;
* returns only benchmark-permitted outputs after termination.

This separation provides clean run accounting and makes Track B's interaction budget meaningful.

---

# External Resources

The core benchmark is intended to be reproducible and closed-task.

A benchmark release should define exactly which external capabilities the AutoTrain system may access.

The default direction for the core track is:

* no external datasets;
* no arbitrary Internet access;
* no external pretrained target-model checkpoints;
* no human assistance after task reveal.

Remote inference required by the registered AutoTrain system may be exposed through a benchmark-controlled interface.

Such calls must be logged as part of the agent system.

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

In Track B, once the participant launches a valid Training Run request, that interaction normally consumes one run budget even if participant-controlled code fails.

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
run submission timestamps
run artifact hashes
training logs
hardware telemetry
VAL outputs
incumbent declarations
final artifact
post-mortem TEST results
```

Where possible, official runs should be reproducible from immutable artifacts.

The benchmark should preserve enough information to distinguish:

* algorithmic improvements;
* infrastructure improvements;
* data-policy improvements;
* research-strategy improvements;
* stochastic variation.

---

# Leaderboards

AutoTrain-Benchmark should maintain separate leaderboards for the two tracks.

## Track A

Ranked primarily by:

```text
final TEST NLL at fixed total wall-clock
```

Reported alongside:

```text
research-time curve
agent tokens
Training Runs
training compute
model size
```

## Track B

Ranked primarily by:

```text
final TEST NLL after at most N Training Runs
```

with fixed maximum duration per run.

Reported alongside:

```text
TEST loss by run number
agent tokens
agent reasoning between runs
training compute
model size
```

Different benchmark releases may use different hardware or budget scales and should not be mixed into a single leaderboard.

---

# Example Benchmark Configuration

The first official configuration has not yet been selected.

A future release may look conceptually like:

```yaml
task:
  tokenizer: TBD
  train_dataset: TBD
  validation_dataset: TBD
  test_dataset: TBD

training_environment:
  hardware: TBD
  cpu: TBD
  ram: TBD
  local_storage: TBD
  network_policy: TBD

track_a:
  total_wall_clock: TBD

track_b:
  max_training_runs: TBD
  max_time_per_run: TBD

evaluation:
  metric: autoregressive_nll
  val_feedback: aggregate_score
  test_feedback_during_episode: none
```

Dataset and hardware selection are intentionally deferred until the benchmark protocol is stable.

---

# Design Principles

AutoTrain-Benchmark follows several principles.

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

# What AutoTrain-Benchmark Is Intended to Measure

AutoTrain-Benchmark is designed to expose capabilities including:

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

# What AutoTrain-Benchmark Does Not Claim to Measure

AutoTrain-Benchmark is not a complete measure of general intelligence or general recursive self-improvement.

A high score demonstrates strong autonomous capability within a specific machine-learning research environment.

The benchmark should therefore be understood as an **RSI-oriented empirical testbed**, not as a universal measure of recursive self-improvement.

Its value comes from providing a concrete environment in which improvement itself can be measured repeatedly and quantitatively.

---

# Research Questions

Beyond leaderboard ranking, AutoTrain-Benchmark is intended to support questions such as:

### Research-time scaling

How does model quality scale with autonomous research time?

$$
L_{\mathrm{TEST}}(t)
$$

### Experiment-efficiency scaling

How does model quality scale with the number of expensive empirical interactions?

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

The current goal is to stabilize the benchmark specification before selecting the first concrete environment.

Major remaining implementation decisions include:

* TRAIN corpus;
* VAL and TEST construction;
* tokenizer;
* training hardware;
* Track A wall-clock budget;
* Track B run count;
* Track B per-run duration;
* exact agent sandbox;
* run submission format;
* artifact format;
* evaluation interface;
* synthetic-data policy;
* logging schema.

These parameters should be chosen only after the benchmark's conceptual boundaries are stable.

---

# Summary

AutoTrain-Benchmark evaluates an autonomous system starting from a fixed initial state:

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
\text{Track B: fixed number of training-environment interactions}
}
$$

with a fixed maximum duration for each run.

In both tracks:

> **The initial system is fixed. Autonomous improvement is allowed. External intervention is not.**

The final question is deliberately simple:

> **How good a training system can this autonomous researcher discover?**
