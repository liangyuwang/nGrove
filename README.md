# AutoTrain-Bench

**An open-ended benchmark for autonomous training research and recursive self-improvement.**

> **Status:** v0 local reference runtime; no official challenge released yet
> **Project name:** AutoTrain-Bench
> The repository includes one leaderboard-ineligible development smoke task.

---

## Overview

AutoTrain-Bench is a benchmark for evaluating **autonomous systems that design, implement, run, analyze, and improve machine-learning training systems**.

The benchmark is inspired by training speedruns, but asks a broader question.

Instead of fixing a small model and asking:

> How quickly can this model reach a target validation loss?

AutoTrain-Bench asks:

> **Given a fixed task, fixed data, a fixed evaluation contract, and a constrained opportunity to interact with the training environment, how good a model can an autonomous research system produce?**

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
\boxed{\text{Minimize hidden TEST perplexity}}
$$

under the resource constraints of the selected benchmark track.

The architectural model is **sandbox-style workload semantics**, not “Docker
semantics”: workloads have fixed identities, bounded visibility and resources,
revocable lifetimes, immutable outputs, and auditable evidence. Docker is only
the bundled local mechanism. Other Workers may realize the same contract with a
scheduler sandbox, VM, another container runtime, or a certified process
sandbox without adding platform branches to Core.

---

# Local Reference Prototype

The repository now contains an executable, single-host Docker prototype of the benchmark protocol. It is intended to validate episode lifecycle, Training Lease accounting, hidden evaluation, and the agent-facing control surface before the first official benchmark instance is selected.

The current development snapshot uses Worker Protocol v5 but does not carry a
release package version yet. Protocol v5 is a breaking Worker API revision:
external Evaluation Workers must resolve a path-free `HiddenDatasetRequest`
after research sealing, return a content-identity-bearing
`ResolvedHiddenDataset`, accept TEST before ANALYSIS, and use the precise
prefix/target/artifact visibility fields in the transport-neutral
`CandidateChannelRequirements` contract. The Python package itself remains
`0.0.0` while unreleased.

Docker is the bundled v0 compatibility implementation, not a general platform
abstraction. The benchmark Supervisor now depends on the semantic
`autotrain_bench.worker_protocol` boundary: it asks a Worker to start, execute,
inspect, and terminate a lease or evaluate a Candidate without referring to
containers, scheduler jobs, Pods, or platform networks. Worker capabilities,
deployment trust, and conformance are recorded independently, and there is no
automatic capability fallback.

Agent launch mechanism is also separate from policy evidence. `container` and
`host_cli` describe bundled launch paths; a trusted launcher supplies the
semantic enforcement, trust, workload identity, and limitations recorded for
the episode. The bundled launcher is development-trust only, and its unbounded
writable bind-mount storage is recorded as a policy limitation rather than
being hidden behind application-level mailbox limits.

The bundled `autotrain-bench prepare` and `autotrain-bench start` commands are
the composition root for that local Docker implementation; they are not a
generic backend loader. Worker Protocol v5 is currently a synchronous,
in-process Python contract. Its remaining `Path` values for TRAIN, workspaces,
Candidate artifacts, and output directories are local deployment handles, not
a remote wire format. Hidden-data requests and resolution receipts contain no
path. An external Worker package uses its own trusted launcher to compose Core
with its deployment rather than adding a platform branch to this CLI.

The Agent-facing file mailbox is separate from the Worker Protocol. It is the
local reference transport for Agent requests to the Coordinator; its
filesystem hardening and operating-system support are transport properties,
not benchmark semantics. The task's `*_image` fields are release-selected
OCI-style workload references consumed by the Docker implementation, not
Docker container IDs; leaderboard-eligible releases require digest-pinned
values. A different runtime may consume the OCI artifact or an
organizer-approved derived artifact, but may not silently substitute an
environment.

Slurm, Kubernetes, Apptainer, cloud-job, and other integrations belong in
separately distributed Worker packages maintained by their platform or site.
They must not add platform branches or dependencies to this core repository.
See [Worker Protocol boundary](docs/worker-protocol.md) for the ownership rules
and [Thin Core development plan](docs/thin-core-development-plan.md) for the
incremental migration and its acceptance criteria.

The reference runtime requires a machine that can run Docker. AutoTrain-Bench
consumes the environment the user has already obtained; it does **not** allocate
machines or integrate with Slurm, Kubernetes, or a cloud scheduler.

The repository contains one bundled development task,
`dev_tasks/smoke_lm`. It is an end-to-end smoke test for Agent and runtime
development and never a leaderboard task. It uses the same release-bound start
contract, Policy, trusted Evaluator implementation, both tracks, and isolated
Candidate serving boundary as the challenge preview, while retaining a much
smaller task-specific corpus and evaluation configuration. It intentionally
ships no reference Agent.

Future official task packages live under `challenges/`. `lm-pretrain-v1/` now
contains a runnable-after-preparation preview task, evaluator, data builder, and
provisional comparison profiles; no frozen, leaderboard-eligible challenge has
been released yet.

Hardware allocation is likewise not part of the task. Ad-hoc task episodes may
explicitly expose resources with manual flags. Profile-bound development and
challenge episodes instead select one of the release's finite named runtime
profiles. In both cases the requested resources and discovered runtime
inventory are recorded with the result.

## Quick start

Requirements:

* Python 3.9 or newer;
* Docker.

CPU episodes need nothing else. NVIDIA GPU episodes additionally require the
NVIDIA Container Toolkit; other accelerators require a Docker-compatible device
and support in the task's pinned Training image.

Install the local CLI:

```bash
python3 -m pip install -e .
```

Prepare the TinyStories smoke-task data and pinned images:

```bash
python3 dev_tasks/smoke_lm/prepare_data.py
autotrain-bench prepare --task dev_tasks/smoke_lm/task.json
```

The smoke task downloads about 0.98 MB of real text and uses hidden 8K-token
ANALYSIS/TEST datasets. By default the selected Agent receives an empty workspace and
must create its own training system. See `dev_tasks/smoke_lm/README.md` for the
development workflow, Candidate ABI, Policy, and reproducibility details.
Smoke-task scores are not benchmark results.

Every episode must select an explicit `--episode-dir` or `--episodes-root` in a
dedicated trusted working root outside the AutoTrain-Bench Core/task checkout.
This keeps episode state and the Tested Agent workspace separate from operator
files. It also prevents host-native tools such as Codex from inheriting the
Core's `AGENTS.md` through ancestor-directory instruction discovery. An empty
workspace beneath the Core is therefore not isolated. This check is scoped to
the trusted AutoTrain-Benchmark checkout; it does not remove the Tested Agent's
own global configuration or participant-owned instructions.

A participant Agent Container can be selected without changing the task:

```bash
autotrain-bench start \
  --challenge dev_tasks/smoke_lm/challenge.json \
  --track-config smoke-b-3x75s \
  --runtime-profile cpu-local \
  --episodes-root /path/to/autotrain-bench-runs/episodes \
  --agent-runner container \
  --agent-image participant-agent:version \
  --agent-command "participant-agent run"
```

## Bring your own Agent: host-native CLI tools

AutoTrain-Bench supports two Agent runners:

* `container` runs a participant command in an Agent Container;
* `host_cli` runs any autonomous, non-interactive CLI command on the host while
  the Supervisor, Training Containers, hidden evaluation, deadlines, and mailbox
  protocol remain unchanged.

The core runner is tool-independent. A CLI can be used directly when it already
knows how to operate the benchmark protocol:

```bash
autotrain-bench start \
  --challenge dev_tasks/smoke_lm/challenge.json \
  --track-config smoke-a-5m \
  --runtime-profile cpu-local \
  --episodes-root /path/to/autotrain-bench-runs/episodes \
  --agent-runner host_cli \
  --agent-command "your-agent-cli --non-interactive ..."
```

The command starts with its working directory set to the private episode
workspace and with stdin disconnected. It receives
`AUTOTRAIN_BENCH_WORKSPACE`, `AUTOTRAIN_BENCH_TASK`, and
`AUTOTRAIN_BENCH_CONTROL_DIR`; it must use the `autotrain-bench agent ...`
mailbox commands, call `agent finish`, and exit without further user input. A
custom wrapper may reuse the tool-independent prompt in
`autotrain_bench.agent_prompt.HOST_CLI_PROMPT`.

Tool-specific adapters are optional conveniences, not part of the benchmark
protocol. They translate the generic contract into a particular CLI's flags,
prompt input, authentication environment, sandbox settings, and event stream.
For example, the bundled Codex adapter allows an existing ChatGPT-authenticated
Codex CLI installation to be used without an API key. After `codex login status`
confirms an existing login, run the TinyStories Track A with a fixed Codex model:

```bash
python3 -m autotrain_bench.cli start \
  --challenge dev_tasks/smoke_lm/challenge.json \
  --track-config smoke-a-5m \
  --runtime-profile cpu-local \
  --episodes-root /path/to/autotrain-bench-runs/episodes \
  --agent-runner host_cli \
  --agent-command "python3 -m autotrain_bench.cli_adapters.codex --model YOUR_CODEX_MODEL --reasoning-effort high"
```

The generic host runner captures stdout/stderr and enforces the same episode
deadline. The optional Codex adapter invokes
`codex exec --ephemeral --json`, disables web search and user configuration, and
reuses the CLI's existing login. It never copies `auth.json` into an Agent
Container and does not require an inference Gateway. These behaviors follow the
official Codex documentation for
[non-interactive mode](https://developers.openai.com/codex/noninteractive/) and
[ChatGPT authentication](https://developers.openai.com/codex/auth/).

Any CLI wrapper can make usage accounting portable by writing newline-delimited
JSON to stdout. The runner recognizes these tool-independent records (additional
fields are allowed):

```json
{"type":"autotrain_bench.agent_metadata","tool":"my-cli","tool_version":"1.0","model":"model-id","reasoning_effort":"high"}
{"type":"autotrain_bench.agent_usage","usage":{"input_tokens":100,"cached_input_tokens":20,"output_tokens":30,"reasoning_output_tokens":5},"usage_complete":true}
```

Without an adapter or these records, the episode itself still runs normally;
the final report simply marks Agent token usage as incomplete.

After an episode closes, a host Agent receives a 30-second drain period to finish
its current response and emit final usage records. Candidates, TEST results, and
official elapsed time are frozen before this drain period begins. If the command
does not exit, the Supervisor terminates it and records `drain_timeout` without
changing the benchmark result.

`host_cli` is always reported as `policy_compliant: false`: the v0 reference
implementation cannot prove that a host process was unable to inspect unrelated
host files or use its own network. It is a faithful local rehearsal and agent
comparison mode, not yet a closed leaderboard submission boundary. A future
policy-compliant host-tool runner requires an OS/VM or remote-tool boundary that
confines every model-generated action to the episode workspace and mailbox.

## Bring your own Agent: controlled inference

AutoTrain-Bench is intended to benchmark participant-defined autonomous Agent
systems, including systems whose researcher model is served through an
OpenAI-compatible remote or local endpoint. The Agent
execution loop still runs inside the Agent Container so the benchmark can enforce
its policy boundary; remote model inference is exposed through a narrow
official Gateway sidecar rather than arbitrary public Internet access.

In the default `denied` egress policy an Agent without remote inference uses
`--network none`. When a registered inference channel is configured,
AutoTrain-Bench creates an episode-private Docker `--internal` network shared
only by the Agent and a trusted OpenAI-compatible Gateway at:

```text
http://inference-gateway:8765/v1
```

The Agent has no route from that internal network to the public Internet. The
official Gateway alone also joins a separate egress network and owns the real
provider credential. It is not a generic HTTP proxy: the episode registration
fixes the upstream origin, allowed API paths, and model names before task reveal;
redirects and provider-hosted external tools are rejected. The Agent never
receives the provider credential or chooses arbitrary destinations.

A minimal launch looks like:

```bash
autotrain-bench start \
  --task path/to/task.json \
  --track B \
  --episodes-root /path/to/autotrain-bench-runs/episodes \
  --agent-runner container \
  --agent-image participant-agent@sha256:DIGEST \
  --agent-command "participant-agent run" \
  --inference-base-url https://api.openai.com \
  --inference-api-key-env OPENAI_API_KEY \
  --inference-model registered-researcher-model
```

For a credentialed endpoint, the user exports the credential only in the host
environment before starting the episode:

```bash
export OPENAI_API_KEY="..."
```

For a trusted local endpoint that requires no authentication, omit
`--inference-api-key-env`; the Gateway still freezes the origin, paths, and
model identities and records usage.

For compatibility with common OpenAI-style clients, the Agent Container receives
`OPENAI_BASE_URL=http://inference-gateway:8765/v1` and a dummy local API key. The
real key is copied only into the trusted Gateway. V0 permits ordinary
function-tool schemas because the tool is executed by the Agent Container, but
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
python -m autotrain_bench.cli agent lease-commit /output/candidate
python -m autotrain_bench.cli agent declare-incumbent sha256:<candidate-hash>
python -m autotrain_bench.cli agent finish
```

The public task manifest at `$AUTOTRAIN_BENCH_TASK` declares the Agent-facing
control protocol as `protocol_version`. Version 2 renamed the `agent status`
field `agent_mode` to `agent_egress` and its `"sandboxed"` value to `"denied"`.
An Agent that parses the version 1 field name must be updated; the benchmark
does not emit both names. Version 3 corrected participant-time accounting for
release-declared Worker preparation. Version 4 removes organizer validation
feedback: Candidate commit returns artifact acceptance only.

Training code writes Candidate artifacts into the lease-owned `/output` mount.
`lease-commit` first stops the Training sandbox, validates and snapshots the
now-immutable artifact, and returns its content identity. It does not run hidden
evaluation or change the incumbent. The Agent must explicitly call
`declare-incumbent` for the model it would submit if the episode ended
immediately. The Agent may choose whether and how to validate using permitted
TRAIN information; Core supplies no validation recipe or official feedback.

The submitted Agent command must be autonomous and non-interactive. Docker does
not attach a user input stream after task reveal, so the user cannot continue
steering the Agent during the episode. Each episode receives a private copy of
the prepared Agent workspace.

For debugging only, `--agent-egress unrestricted-dev` gives the Agent Container
ordinary Docker bridge egress and marks the run
`policy_compliant: false`. It is not a closed AutoTrain-Bench result and exists
only to make local integration easier.

The Supervisor prints the episode directory. Its final state can be inspected later with:

```bash
autotrain-bench report \
  --episode /path/to/autotrain-bench-runs/episodes/<episode-id>
```

New episode states declare `result_schema_version: 6`. Reporting an older state
performs a read-only migration. An unversioned v1 state receives the v2
outcome fields: a completed result with a valid TEST NLL receives
`outcome: "success"` and a derived `test_perplexity`, while an unscored
terminal result receives `outcome: "failed"` with both TEST metrics left
`null`. A v1 or v2 state additionally has its `agent_system.mode` field renamed
to `agent_system.egress`, with the value `"sandboxed"` mapped to `"denied"`, so
one schema version always identifies one structure. A migrated report records
`migrated_from_result_schema_version`. The persisted historical `state.json` is
never rewritten, and unknown future result-schema versions are rejected rather
than guessed. Legacy VAL fields remain explicitly labelled legacy; reporting
never relabels an old Agent-visible VAL trajectory as postmortem ANALYSIS.
An existing v4 postmortem result remains readable, but migration explicitly
reports that its complete ordered incumbent-declaration history is unavailable
rather than reconstructing that history from an Agent log or a final boolean.
A v5 result gains its ANALYSIS summary counts by deriving them from the
trajectory it already carried, so migration invents nothing.

In v6, `outcome: "success"` requires a finalized episode, a non-empty final
Candidate ID, and both consistent TEST metrics. A failed outcome requires a
failure reason and keeps both TEST metrics `null`. A null outcome is allowed
only before a result exists (`created`, `active`, or the transient `closed`
state while postmortem TEST, ANALYSIS, and trusted finalization are running),
and may not carry TEST metrics or a failure reason. Research is first sealed
irreversibly. Hidden evaluation begins only after the Agent and Research Worker
have stopped and Agent-control cleanup has completed. TEST first scores only the
frozen final incumbent and its decision is persisted before ANALYSIS begins.
ANALYSIS then scores the frozen Candidate history without returning feedback.
Its `complete`/`incomplete` disposition is independent of the TEST outcome and
never gates leaderboard eligibility merely because diagnostic points are
missing. A boundary or teardown failure discovered during ANALYSIS is still
recorded as deployment-integrity evidence and makes the run non-compliant and
ineligible, but it does not erase an already completed TEST measurement.

`complete` means the ANALYSIS pass finished and gave every selected Candidate a
disposition, not that every point produced a metric. A result therefore also
carries `analysis_selected_points` and `analysis_evaluated_points`, so a reader
of the summary fields alone can see a complete pass that still has gaps without
walking the trajectory. The counts are validated against the trajectory they
summarize, so the summary cannot drift from its own detail.

`state.json` is the authoritative result commit point. Immediately before that
atomic commit, the Supervisor appends a non-terminal
`RESULT_PUBLICATION_PREPARED` audit record. A finalized v6 state created by the
current implementation includes `result_publication`, which binds the result
to that record's publication ID and sequence number. An unreferenced prepared
record documents an interrupted attempt and is not a terminal result. Older
states without this optional reference remain readable through migration.

Deployment cleanup can still fail after an episode has already reached DNF or
an infrastructure-failure state. Those failures do not rewrite the original
classification. Their stable codes are appended to
`deployment_teardown_failures`, while runtime-specific details remain only in
the trusted event stream. A successful retry of active-Lease termination is
reconciled through the Supervisor so the final state no longer reports a stale
Lease and official Lease accounting remains frozen at episode closure.
Likewise, if a Worker cannot confirm cleanup after a failed `start_lease()`,
the Supervisor retains the semantic Lease ID for a final retry; runtime or
scheduler resource handles remain private to the Worker deployment.
The bundled `start` command returns zero only for a finalized successful result
with no active/pending Lease or recorded deployment teardown failure. This
keeps a late cleanup or persistence problem from looking like process-level
success while preserving the benchmark outcome and failure evidence in the
report.

The example proves the complete protocol:

```text
task reveal
→ interactive lease
→ TRAIN-only experimentation
→ candidate commit
→ autonomous revision
→ finish
→ research sealed permanently
→ Agent and Research Worker stop confirmed
→ Agent-control cleanup
→ final incumbent scored once on hidden TEST
→ historical Candidates scored on hidden ANALYSIS when available
→ one final result
```

## v0 scope and security

The reference implementation uses an official host-side Supervisor, a
development Agent Container or host CLI process, a thin trusted Inference
Gateway when registered, disposable Training Containers, and separate trusted
Evaluator and untrusted Candidate containers. Agent, Training, and Candidate
containers receive neither the Docker socket nor ANALYSIS/TEST mounts. Sandboxed
Agent and Training networking is disabled; Candidate evaluation uses only an
ephemeral internal network shared with its Evaluator. The Gateway handles only
frozen researcher-inference endpoints. The bundled launcher nevertheless
reports `policy_compliant: false` because its writable host bind mounts lack a
deployment-enforced quota. In this local compatibility path the Supervisor
process also enforces episode and Training-Lease deadlines; an external
certified Worker must enforce its deadline envelope even if the Coordinator
disconnects. `unrestricted-dev` and `host_cli` add further non-compliant local
integration boundaries.

Here, **user/operator** means the entity that prepares and launches the run; it
may itself be a human or another Agent and is inside the trusted host boundary.
**Tested Agent** means the submitted autonomous system being benchmarked. The v0
threat model protects benchmark policy from the Tested Agent, not from the
user/operator or machine administrator. Docker isolation is a development
reference boundary and is not presented as protection against kernel,
container-runtime, or GPU-driver exploits. Official adversarial evaluation may
require stronger worker or VM isolation.

The `isolated_candidate` backend starts a fresh internal Docker network per
query. The trusted Evaluator mounts only the selected hidden dataset and output;
the untrusted Candidate mounts only its content-addressed artifact. The
Evaluator sends same-length, prefix-free batches over `next_token_logits_v1`
and validates a bounded `[batch, vocabulary]` response. It scores each response
before revealing longer prefixes. Logits must be finite numeric values;
booleans and numeric strings are rejected. Candidate failures return only fixed
error codes to the Agent; raw Candidate logs stay in the trusted episode
directory. Candidate and Evaluator are destroyed after each query. This backend
satisfies the v0 reference isolation contract.

The Agent Container receives a generated, minimal `autotrain_bench.client`
package rather than the Core source tree. Its mailbox inbox and outbox are
mounted as separate directory roots, so it cannot replace the shared control
parent. Training Containers receive TRAIN, the participant workspace, scratch,
and lease output only. Command stdout and stderr are each bounded to 1 MiB and
container logs use a fixed rotation limit. The task field
`candidate_artifact_limits.max_files` bounds all regular-file and directory
entries in a Candidate bundle, including empty directories.

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

The local v0 reference separates trusted scoring from untrusted Candidate code.

```text
User-owned host
└── trusted AutoTrain-Bench Supervisor
    ├── untrusted Agent Container
    ├── disposable Training Container
    └── ephemeral internal evaluation network
        ├── trusted Evaluator Container with one hidden dataset
        └── untrusted Candidate Container without hidden files or public network
```

The Supervisor records task reveal time, owns Track A/B policy and lease
counters, and requests controlled execution and evaluation through the Worker
Protocol. The local v0 Worker maps those requests to the pictured containers;
other deployments implement the same semantics outside this repository. The
Agent cannot access Supervisor state or a privileged Worker control channel.

The Agent receives interactive access to a Training Container only through the Supervisor's narrow command channel during an active **Training Lease**. It may inspect logs, edit the shared workspace, compile code, profile, and run any number of TRAIN-only experiments inside that lease.

ANALYSIS and TEST are unavailable to the Agent and Research Worker trust domain.
Only after research is durably sealed and those workloads are confirmed stopped
may an Evaluation Worker materialize either hidden dataset. It communicates with
a separate Candidate sandbox through the release-owned next-token application
protocol. Core fixes the information-flow and isolation requirements; the Worker
chooses the channel mechanism. The bundled Docker Worker uses a private HTTP/TCP
network, but that mechanism is not part of the benchmark contract.

Core requests a hidden split by logical name and, for a verified release, its
expected SHA-256 identity. The Evaluation Worker privately maps that request to
its organizer-controlled source and returns an opaque resolution plus the
effective digest before Candidate execution. Paths, URLs, credentials, mounts,
and cache handles never enter the semantic request or receipt.

The bundled smoke/reference data is prepared in advance on the trusted host
filesystem and withheld by sandbox visibility: it is never mounted into the
Agent or Research Worker. That is a weaker but valid local deployment than late
download, decryption, or mount after `research_sealed`. Deployments that can keep
the bytes physically absent from the host trust domain until evaluation provide
stronger defense in depth; Core requires the visibility boundary and frozen
identity, not one materialization mechanism.

Task schema v2 still retains `paths.analysis` and `paths.test` as compatibility
inputs for the bundled local CLI. The Supervisor neither reads them nor sends
them through Worker Protocol v5. An external launcher supplies its Evaluation
Worker's private source mapping independently; the configured local paths may
remain absent for that deployment. Replacing these compatibility fields in a
future task schema is separate from the now path-free Worker boundary.

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

Importantly, **data policy and experimental methodology are part of the research
problem**. The Agent may reserve some permitted TRAIN data, devise another
permitted validation method, or perform no validation at all. AutoTrain-Bench
neither requires a holdout nor prescribes a replacement for organizer-owned VAL.

This choice deliberately measures validation and incumbent selection as part of
the autonomous research system. It also raises the variance of the final TEST
score: systems may reserve different amounts of TRAIN, use estimators with
different noise, or select the wrong incumbent despite similar underlying
capability. Comparisons intended to resolve small differences should therefore
predeclare repeated episodes and report uncertainty rather than treating one
episode as a noiseless ranking.

---

## ANALYSIS

ANALYSIS is an organizer-owned hidden **postmortem measurement**, not a research
feedback channel. After research is irreversibly sealed, the Evaluation Worker
scores the release-selected frozen Candidate inventory on the same ANALYSIS data
and metric. This produces an auditable trace of the **submitted Candidate
inventory** without letting the Agent adapt to an official oracle. The values
are comparable for like-for-like commits under one release; the trajectory is
not a direct observation of latent research progress because Candidate commit
and incumbent-declaration strategies are themselves Agent choices.

ANALYSIS is never returned through Agent control, never selects an incumbent,
and never replaces TEST as the primary score. A release freezes its Candidate
selection limit and result-visibility policy before task reveal. Publication may
be organizer-only or delayed to reduce cross-episode leakage. The local
`report` command is a trusted Operator view and retains the complete audit
record; an organizer's submission/publication layer must enforce the declared
visibility policy when distributing that record.

---

## TEST

TEST is the benchmark's final generalization metric.

TEST is:

* hidden from the AutoTrain system;
* unavailable through the training environment;
* unavailable through the agent environment;
* unavailable during the active research episode.

Neither ANALYSIS nor TEST is computed during autonomous research.

Instead, the benchmark first permanently terminates the research episode and freezes the relevant artifacts.

Only then does the official evaluator run TEST once on the final Agent-declared
incumbent, durably stage that primary decision, and attempt ANALYSIS on the
frozen history. Missing diagnostic points do not erase TEST or determine
leaderboard eligibility. A compromised or unconfirmed evaluation boundary is
still reported independently as a deployment-integrity failure.

Therefore:

$$
\boxed{\text{validation strategy belongs to the Tested Agent}}
$$

while:

$$
\boxed{\text{ANALYSIS and TEST are postmortem only}}
$$

The system cannot resume research after sealing, including when hidden
evaluation fails.

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

The evaluator first computes aggregate autoregressive token negative
log-likelihood:

$$
L
=
-\frac{1}{N}
\sum_t
\log p(x_t \mid x_{<t}).
$$

The primary leaderboard metric is token perplexity:

$$
\operatorname{PPL}=\exp(L).
$$

NLL is retained in every successful result as the canonical additive
measurement. Perplexity is derived once from the token-weighted mean NLL; the
benchmark never averages per-sequence or per-batch perplexities.

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

A **Training Lease** is a fixed-duration interactive window in which the Agent
may use the benchmark's already-allocated Training Environment.

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
  └── committed Candidate → immutable content identity only
```

The Agent interacts with the Training Container only through Supervisor-controlled operations. It never receives Docker or host control.

Within a lease, the search process is unrestricted by experiment count. The
Agent may repeatedly change architecture, optimization, data policy, kernels,
distributed code, or its own research tools as long as it remains within the
visible Training Environment and time limits.

Each lease may commit at most one Candidate. Commit immediately terminates the
lease, validates the artifact, and freezes it without quality-scoring it. A
lease may also end without a Candidate because of failure, timeout, or an
explicit stop.

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
├── participant-chosen experiments and analysis
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

All of these decisions compete for the same wall-clock budget. At the deadline,
research workloads are terminated and the previously declared incumbent and
Candidate inventory are sealed. Benchmark-owned postmortem ANALYSIS and TEST
time does not consume participant research time because the Agent can no longer
act or benefit from it.

The primary Track A score is:

$$
\boxed{
\operatorname{PPL}_{\mathrm{TEST}}(T_{\mathrm{total}})
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
Maximum Training Leases:   defined by the selected track config
Maximum Time per Lease:    defined by the selected track config
Runtime Profile:           selected independently at episode launch
Official feedback:         none
Agent Research Time:       Unbounded by benchmark score
```

The AutoTrain system may reason, write code, analyze previous results, and autonomously improve for as long as it chooses between Training Leases.

However, it may obtain interactive access to the Training Environment only a
fixed number of times.

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
Agent research / self-improvement
 │
 ▼
LEASE 2
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
\operatorname{PPL}_{\mathrm{TEST}}(N_{\mathrm{leases}})
}
$$

for the final incumbent selected by the AutoTrain system.

### What Track B measures

Track B asks:

> **How effectively can an autonomous researcher use a limited number of Training Environment windows while generating and interpreting its own experimental evidence?**

It emphasizes:

* experiment selection;
* strength of prior reasoning;
* information gained per lease and participant-chosen experiment;
* interpretation of failures;
* hypothesis quality;
* efficiency of autonomous research under bounded Training Environment access.

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
| Experiments in lease| Unlimited within time/environment| Unlimited within time/environment              |
| Official feedback   | None during research             | None during research                           |
| Self-improvement    | Allowed                          | Allowed                                        |
| Human intervention  | Forbidden                        | Forbidden                                      |
| Primary question    | How fast can the system improve? | How well can it use bounded training access?    |

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
\operatorname{PPL}_{\mathrm{TEST}}
= f(
\text{agent tokens},
\text{research time},
\text{lease and experiment interactions}
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
* participant-chosen metrics and observations;
* previous reasoning;
* research notes;
* experiment lineage.

There is no artificial memory reset between experiments.

This is intentional.

A capable autonomous researcher should be able to learn from its own previous work.

The benchmark controls information through evaluator and environment boundaries rather than by forcing the agent to forget legitimate history.

---

# Participant-Owned Experimental Methodology

AutoTrain-Bench deliberately does not provide an organizer validation oracle.
An Agent may decide that internal holdouts or other permitted experiments are
useful, but it may also choose a different method or no explicit validation.

A capable AutoTrain system may autonomously discover practices such as:

* internal holdouts;
* uncertainty estimation;
* replication;
* multiple seeds;
* robustness checks;
* avoiding overreaction to small validation changes.

The benchmark encourages careful empirical reasoning but does not prescribe any
of these practices.

Discovering when and how to use them is part of the task.

---

# Intermediate Incumbents and Research Curves

The benchmark should not record only a single final score.

Throughout an episode, the AutoTrain system may declare a **current incumbent**: the model it would submit if the benchmark ended immediately.

Candidate commitment and incumbent declaration are separate operations. The
Supervisor never infers an incumbent by sorting hidden scores or by choosing the
most recently committed candidate; that decision belongs to the Agent.

Each incumbent declaration records:

* a strictly ordered Supervisor event sequence, trusted wall-clock timestamp,
  and episode-relative time;
* the Candidate artifact hash and previous incumbent;
* the Candidate's source Leases;
* the number of Leases consumed at declaration time.

The public task manifest also exposes a fixed Core safety limit on declaration
events. This bounds untrusted control-plane growth without making declaration
frequency a hidden rule; the limit is far above the release's Candidate cap.

The append-only Supervisor event stream is the authoritative incumbent history.
Every Candidate commit and declaration carries the same timestamp as its
referenced Supervisor event. Trusted inference-gateway records—and Agent runner
records when that adapter emits timestamps—can therefore be correlated with
the declaration's event sequence, episode-relative time, and timestamp to
explain a decision. Untimestamped or truncated Agent logs permit only coarser
ordering and cumulative-token analysis. All such traces remain diagnostic and
do not determine declaration order or Candidate identity. The result itself
persists the complete ordered
declaration history and binds each ANALYSIS point to its Candidate-commit and
declaration event sequences. The research seal separately freezes the final
declared incumbent used for TEST.

TEST remains unavailable.

After the episode is permanently sealed, the evaluator evaluates the final
declared incumbent once on TEST. Historical committed Candidates are then
evaluated on ANALYSIS under the release-frozen selection rule, so one episode
produces a submitted-Candidate quality trajectory but never a TEST trajectory.
From the ordered declaration history, an analyst can derive a fixed
research-time or fixed-Lease curve by selecting the most recently declared
incumbent at each predeclared checkpoint and joining it to that Candidate's
ANALYSIS score. This makes checkpoint coverage independent of how frequently an
Agent redeclares an incumbent; it does not invent quality observations for work
the Agent never committed. Scaling studies may compare such derived curves and
final TEST results only when their complete comparison keys match.

The submitted-Candidate ANALYSIS trace can reveal behavior that a final scalar
cannot, without providing an official signal while the Agent can still act.

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
\boxed{\text{final hidden TEST perplexity}}
$$

under the selected track's resource constraint.

Lower is better.

The benchmark should avoid combining many metrics into a complicated scoring formula.

---

## First-class diagnostic metrics

The benchmark also records:

* postmortem ANALYSIS NLL;
* final TEST NLL;
* postmortem ANALYSIS NLL trajectory;
* final ANALYSIS–TEST disagreement as a distribution diagnostic, not a measure
  of overfitting to ANALYSIS;
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

A release declares that boundary in its task manifest:

```json
"timing": { "benchmark_owned_preparation": ["worker_lease_startup"] }
```

`worker_lease_startup` is the wall clock a Worker spends inside
`start_lease` — scheduler queueing, image conversion, workload startup.
Declaring it refunds that time to the participant, including when the Lease
fails to start. An
explicit empty list is the opposite and equally valid choice: every second is
charged to the participant. The field is optional and currently defaults to
refunding Worker Lease startup; unknown phase names are rejected rather than
ignored. Because the declaration lives in `task.json`, it is covered by the
manifest digest recorded with every result, so the accounting boundary belongs
to the frozen challenge rather than to whichever runtime revision executed it.

The refund is never hidden. Every result reports the cumulative amount as
`timing.benchmark_overhead_seconds`, and the Track A deadline moves out by the
same amount so that `remaining_wall_clock_seconds` and the duration granted by
`lease-start` always describe the same instant. Independently of the boundary
a release chooses, a granted Lease duration is clamped to the budget that
actually remains, so the Agent is never promised time it cannot have.

Without this boundary the same Track A budget would mean different things at
different sites, because the cost of starting a Lease is a property of the
deployment rather than of the participant. A local Docker Worker with warm
images needs well under a second; a scheduler-backed Worker can need minutes.
An episode's real wall clock is therefore not bounded by its Track A budget —
only the participant's measured research time is.

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

It may request and interact with a sandboxed Training workload only through the
Supervisor-controlled Lease interface.

The Training Environment:

* executes in an already allocated, Worker-enforced sandbox;
* shares the Agent workspace during an active lease;
* permits repeated interactive commands inside that lease;
* has no privileged deployment control, public network, ANALYSIS, or TEST access;
* is forcibly terminated by the Worker at Lease expiry even if Core disconnects.

The trusted Eval Environment receives only the selected hidden dataset. A separate
read-only Candidate Environment receives only the immutable Candidate artifact;
the two communicate through the restricted inference protocol. Both are
inaccessible to Agent and Training Containers, and the Evaluator returns only
benchmark-permitted aggregate metrics.

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

In policy-compliant container runs, remote researcher inference is allowed only
through a registered, Supervisor-managed trusted Gateway. The upstream origin,
permitted API paths, and permitted model identities are frozen before task reveal;
provider credentials remain outside the Agent Container. The Gateway cannot
redirect to another origin and records calls and provider-reported token usage as
part of the episode trace. The non-compliant `host_cli` runner instead uses the
selected tool's own host login and network path.

Synthetic training examples generated autonomously during the episode by the
frozen initial system through registered capabilities are allowed by the
`lm-pretrain-v1` preview direction. Human intervention after task reveal,
unregistered external access, extra mounted corpora, and a submitted initial
workspace or image containing an external target-model checkpoint or extra
training corpus are outside policy. The runtime enforces the untrusted Agent
boundary where practical; it does not claim to defend results against a
malicious administrator of the trusted host.

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

In Track B, a Lease is consumed as soon as the Worker confirms that its Training
workload started successfully, even if participant-controlled code later fails.

Benchmark infrastructure failures are excluded and may be rerun.

The infrastructure-failure policy must be deterministic and auditable.

An episode succeeds only when post-mortem TEST evaluation produces a finite,
non-negative mean NLL and a finite perplexity derived as `exp(mean_token_nll)`.
An episode with no declared incumbent, a failed TEST evaluation, an invalid TEST
metric, or an unrepresentable perplexity is finalized with `outcome: "failed"`;
both `test_nll` and `test_perplexity` remain `null`. Infrastructure failures are
also failed episodes, but remain separately classified for retry policy.

Post-TEST ANALYSIS is diagnostic. Ordinary point failures or incomplete
ANALYSIS retain the TEST outcome and are reported through `analysis_status`,
`analysis_failure`, and per-point status. ANALYSIS completeness is never a
leaderboard-eligibility requirement. If the failure also shows that an
Evaluation Worker boundary or teardown is unconfirmed, the TEST measurement is
retained while policy compliance and leaderboard eligibility fail closed.

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
postmortem ANALYSIS trajectory
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
final TEST perplexity at fixed total wall-clock
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
final TEST perplexity after at most N Training Leases
```

with fixed maximum duration per lease and no organizer feedback during research.

Reported alongside:

```text
postmortem ANALYSIS NLL by Candidate commit
final TEST perplexity
agent tokens
agent reasoning between leases
training compute
model size
```

Results from materially different runtime profiles or budget scales should not
be mixed into a single leaderboard comparison.

---

# Challenge Releases and Comparison Lanes

A task fixes the corpus and evaluation semantics, but does not require one
hardware platform. A challenge release publishes a deliberately finite map of
named track configurations and named runtime profiles. The first preview keeps
this surface small: provisional Track A/Track B budgets and single-host 4×/8×
NVIDIA A100 SXM4 80GB or H100 SXM 80GB profiles.

Every profile-bound episode records the exact leaderboard comparison key:

```text
challenge_id + track + track_config + runtime_profile
```

Only results with the same four fields are directly comparable. CPU, B300,
TPU, and future platforms may execute the same challenge without changing its
identity, but each belongs to a separate runtime profile. The runtime profile
is an episode-time allocation and eligibility input, not a field in `task.json`.

Inspect a release's supported maps with:

```bash
autotrain-bench challenge-info \
  --challenge challenges/lm-pretrain-v1/challenge.json
```

In challenge mode, `--track-config` selects both the track and its budget, while
`--runtime-profile` supplies the concrete resources. Manual `--track` and GPU
overrides are rejected so the recorded key cannot disagree with the run.

The `lm-pretrain-v1` TRAIN mount contains both raw FineWeb-Edu TRAIN documents with
source metadata and official GPT-2 token shards linked by document IDs and
token spans. The Agent may use either representation. Preview ANALYSIS/TEST rows are
excluded from both Agent-visible forms and remain evaluator-only.

---

# Development Release Configuration

The executable v0 task manifest fixes task semantics. The adjacent
`challenge.json` exposes the same finite Track/runtime selection and launch
contract as a challenge preview:

```json
{
  "schema_version": 2,
  "task_id": "tinystories-byte-lm-smoke-v1",
  "task_kind": "development_smoke",
  "leaderboard_eligible": false,
  "paths": {
    "train": "data/train",
    "analysis": "data/private/analysis.json",
    "test": "data/private/test.json"
  },
  "images": {
    "training": "pytorch/pytorch@sha256:<digest>",
    "candidate": "pytorch/pytorch@sha256:<digest>",
    "evaluator": "python@sha256:<digest>"
  },
  "timing": { "benchmark_owned_preparation": ["worker_lease_startup"] },
  "analysis": {
    "candidate_selection": "all_committed",
    "max_candidates": 16,
    "result_visibility": "organizer_only"
  },
  "track_a": { "total_seconds": 300 },
  "track_b": { "max_leases": 3, "lease_seconds": 75 },
  "evaluation": {
    "backend": "isolated_candidate",
    "candidate_protocol": "next_token_logits_v1",
    "entrypoint": "evaluator/evaluator.py",
    "metric": "autoregressive_nll",
    "leaderboard_metric": {
      "name": "perplexity",
      "field": "test_perplexity",
      "derived_from": "autoregressive_nll",
      "formula": "exp(mean_token_nll)",
      "direction": "minimize"
    },
    "vocab_size": 257,
    "context_length": 128,
    "batch_size": 16,
    "device": "cpu",
    "max_candidate_bytes": 2147483648,
    "max_candidate_files": 10000
  }
}
```

See `dev_tasks/smoke_lm/task.json` for the complete runnable development
manifest with exact image digests. Agent image, command, runner, and optional
initial workspace are participant-owned launch inputs and are intentionally
absent from the task. Hardware/device allocation is absent from `task.json` and
selected through a named runtime profile in
`dev_tasks/smoke_lm/challenge.json`.

---

# Design Principles

AutoTrain-Bench follows several principles.

### 1. Fix the task, not the solution

The benchmark fixes the Training Environment contract and evaluation semantics
while leaving the participant's training system open.

### 2. TEST, not rules, should punish overfitting

The Agent owns its experimental methodology; hidden TEST reveals whether the
final result generalizes.

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

The final objective is hidden TEST perplexity.

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

A strong result demonstrates autonomous capability within a specific machine-learning research environment.

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

How does model quality scale with the number of Training Leases?

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

### Autonomous model-selection quality

How reliably does the Agent's own experimental methodology select an incumbent
that generalizes to hidden TEST?

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
* egress-denied Agent execution with an optional trusted streaming inference
  sidecar;
* a generic host CLI runner and a logged-in Codex CLI adapter;
* irreversible research sealing followed by isolated postmortem ANALYSIS and
  TEST evaluation;
* append-only benchmark, inference, and host-Agent JSONL traces;
* one reproducible TinyStories development smoke task with no bundled
  reference Agent and no implicitly injected training code.

The smoke task fixes its TRAIN/ANALYSIS/TEST construction, public evaluation contract,
budgets, task-owned software images, and isolated Candidate serving protocol so
developers can verify a complete episode cheaply. It is explicitly marked
`leaderboard_eligible: false`. The selected Agent and its initial workspace
remain participant-owned.

No frozen benchmark challenge has been released. The `lm-pretrain-v1` preview under
`challenges/` now defines provisional track configurations, a deliberately
small set of single-host A100/H100 runtime profiles, the four-field comparison
key, the initial external-resource policy, a task/evaluator, and a builder that
retains both raw TRAIN documents and GPT-2 token shards. Full-scale validation
of participant-side 10B data preparation, including scale receipts and checksum
manifests, remains open release work alongside organizer-owned hidden datasets,
final pinned task images, calibrated budgets, submission services, and
leaderboard infrastructure. Additional
inference-provider adapters and isolation beyond the v0 Docker threat model
also remain open design work.

The v0 prototype is intentionally single-host and resource-allocation agnostic.
Users obtain a Docker-capable environment before starting AutoTrain-Bench.

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
\boxed{\text{Fixed Training Environment contract}}
$$

$$
\boxed{\text{No organizer feedback during research}}
$$

$$
\boxed{\text{Postmortem hidden ANALYSIS and TEST}}
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

with the Agent responsible for deciding whether and how to gather validation
evidence from permitted information.

In both tracks:

> **The initial system is fixed. Autonomous improvement is allowed. External intervention is not.**

The final question is deliberately simple:

> **How good a training system can this autonomous researcher discover?**
