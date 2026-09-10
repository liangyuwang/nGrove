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

The evaluation objective is two-dimensional:

$$
\boxed{\text{maximize hidden TEST performance score}}
\qquad
\boxed{\text{minimize hidden TEST time per task}}
$$

under the resource constraints of the selected benchmark track. Results are
compared on a Performance–Cost Pareto frontier rather than collapsed into an
opaque scalar.

The architectural model is **sandbox-style workload semantics**, not “Docker
semantics”: workloads have fixed identities, bounded visibility and resources,
revocable lifetimes, immutable outputs, and auditable evidence. Docker is only
the bundled local mechanism. Other Workers may realize the same contract with a
scheduler sandbox, VM, another container runtime, or a certified process
sandbox without adding platform branches to Core.

---

# Local Reference Prototype

Container runs now record a private Worker cleanup journal outside episode
storage. The launcher prints its location; use `--worker-state-dir PATH` to
select it. The container CLI additionally runs an independent parent watchdog
that terminates an unresponsive Worker and invokes journaled cleanup. To retry
unconfirmed cleanup, invoke
`autotrain-bench worker-recover --state-dir PATH` in the same Worker environment
to verify and reclaim its resources. This preserves existing results and does
not resume the episode. See [Worker recovery](docs/worker-recovery.md) for the
ownership checks, supported scope and unresolved-outcome handling.

The repository now contains an executable, single-host Docker prototype of the benchmark protocol. It is intended to validate episode lifecycle, Training Lease accounting, hidden evaluation, and the agent-facing control surface before the first official benchmark instance is selected.

The current development snapshot uses Worker Protocol v16 but does not carry a
release package version yet. Protocol v7 introduced a breaking Worker API
revision: every successful Training Lease startup returns operation-bound
resource allocation evidence. The Release selects an extensible semantic resource
envelope; the Worker separately reports allocator-visible, workload-visible,
and enforcement evidence, and Core rejects a Lease whose evidence is missing,
misbound, inconsistent, or weaker than the release requirement. Version 7
retains the version 6 workload-resolution contract: Workers first resolve each
Worker-owned release workload into immutable effective artifact evidence, and
the resulting opaque authorization is bound to the
Lease or evaluation operation that launches it. An unchanged digest-pinned OCI
artifact is accepted by identity; a derived SIF, VM image, or other artifact
must match an organizer-controlled acceptance in the trusted release receipt.
Version 8 retains those contracts and adds required, bounded resolution
deadlines to the workload and hidden-data resolution calls. It also retains the
version 5 hidden-data contract: external Evaluation Workers must resolve a
path-free `HiddenDatasetRequest` after research sealing, return a
content-identity-bearing `ResolvedHiddenDataset`, accept TEST before ANALYSIS,
and use the precise
context/target/artifact visibility fields in the transport-neutral
`CandidateChannelRequirements` contract. Version 9 removes conformance evidence
from the Worker's self-description: a trusted Operator composition supplies a
versioned receipt independently, bound to the assessed Worker ID, issuing
authority, and organizer-owned report digest. Version 10 replaced progressive
all-prefix scoring with a single-target context contract. Version 11 replaces
the request-at-a-time channel with `offline_next_token_logits_v3`: after
research sealing, the Worker materializes target-free `contexts.json` for the
Candidate and retains a separately bound `targets.json` for trusted scoring.
The Candidate serving system reads the complete context artifact, chooses its
own batching and scheduling, and writes one ordered logit vector per context.
Trusted timing records that complete serving invocation independently from
NLL. Version 12 standardizes the two-file hidden-bundle digest, requires a hard
byte-and-entry quota for untrusted prediction storage, and makes prediction
artifacts ephemeral once evaluation ends and workload teardown is confirmed.
Version 13 replaces the LM-specific Worker result with a generic `[0, 100]`,
maximize-oriented performance score, exact release-declared raw metrics, and a
score definition that is comparable only within one benchmark release.
Version 14 promotes end-to-end Candidate cost to a release-scoped second axis.
Its primary metric is `time_per_task`: Candidate wall time divided by the
frozen task count. Total wall time and throughput remain auditable cost
submetrics, and comparisons additionally require the same hidden dataset
identity and runtime profile.
The Python package itself remains `0.0.0` while unreleased.

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
the episode. The bundled launcher is development-trust only. Without successful
operator storage admission, its unbounded writable bind-mount storage is recorded
as a policy limitation rather than being hidden behind application-level mailbox
limits. Prediction output uses its own bounded allocation and strict final
validation, independently of this private operator setting.

The bundled `autotrain-bench prepare` and `autotrain-bench start` commands are
the composition root for that local Docker implementation; they are not a
generic backend loader. Worker Protocol v16 is currently a synchronous,
in-process Python contract. Resolution calls receive explicit deadlines, and
TEST workload/data resolution consumes the same operation budget as evaluation;
those deadlines are cooperative, so a future asynchronous operation/cancellation
protocol is still needed to preempt a nonconforming or wedged in-process Worker.
Its remaining `Path` values for TRAIN, workspaces,
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
external Worker deployments. These may be maintained platform packages or
temporary implementations created by the trusted operator during authorized
preparation, with source and identities preserved for the run. They must not
add platform branches or dependencies to this core repository.
See [Worker Protocol boundary](docs/worker-protocol.md) for the ownership rules
and [Worker integration](docs/worker-integration.md) for composition order,
Agent tool admission, monitoring, cancellation/recovery responsibilities, and
staged verification. The [Thin Core development plan](docs/thin-core-development-plan.md)
describes the incremental migration and its acceptance criteria.

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
profiles. In both cases the semantic resource envelope is recorded. Every
successful Training Lease also records a Worker receipt separating allocator
observations, workload-visible values, and enforcement evidence. Inventory
alone remains diagnostic metadata; it is not treated as proof of allocation.

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

Preparation writes a complete public TRAIN file inventory. The local `prepare`
and `start` launchers verify file names, byte lengths, and SHA-256 before task reveal;
old data without this inventory must be prepared again. These checks do not
read hidden evaluation data. Core receives the trusted verification result and
checks its binding to the root manifest without scanning the dataset. See
[TRAIN integrity](docs/training-data.md) for
the manifest format and the Worker's responsibility to keep TRAIN immutable.

The smoke task downloads about 0.98 MB of real text and uses 512 hidden
single-target contexts per ANALYSIS/TEST split. By default the selected Agent receives an empty workspace and
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
mailbox commands for benchmark control and run without further user input. A
custom wrapper may reuse the tool-independent prompt in
`autotrain_bench.agent_prompt.HOST_CLI_PROMPT`.

The supplied prompt describes the evaluation objective, allowed interfaces, and
episode lifecycle. Research methods, experiment scheduling, status polling, and
budget allocation belong to the participant. Track A seals automatically at its
research deadline. To end an active episode voluntarily, the Agent commits or
stops its active lease, calls `agent finish`, and exits; exiting while research
is still active records an unfinished episode (DNF).

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
deadline. Short output is flushed to its log files while the process is still
running. Each stream retains at most 10 MiB including its truncation marker;
excess output is still drained so the log limit does not block the Agent.
Truncated or malformed usage records remain explicitly incomplete.
If a log write or flush fails (for example, disk space or quota exhaustion),
the runner remembers the failure and continues draining that pipe. Final
`agent_usage.log_capture` records each stream's completion, truncation, and
first I/O error. Incomplete stdout capture makes `usage_complete: false` even
if its retained prefix contains valid usage records; stderr diagnostic loss is
recorded separately. Log-capture failure does not imply process-termination
failure.
The optional Codex adapter invokes
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
feedback: Candidate commit returns artifact acceptance only. Version 5 adds the
exact `predictions.json` contract and optional generated helper signatures under
`evaluation.candidate_boundary` so a Candidate can be implemented without
reading benchmark source code. Version 6 promotes Candidate cost to an
independent evaluation axis. Version 7 adds the existing Candidate invocation
contract at `evaluation.candidate_boundary.entrypoint`: `candidate.py` in the
artifact root is imported as a Python module, then
`run_offline(candidate_dir, contexts_path, output_dir, device)` is called
synchronously with positional arguments. The first three arguments are absolute
`pathlib.Path` objects for the immutable artifact directory, target-free context
JSON, and writable prediction output directory; `device` is the string from
`evaluation.device`. The return value is ignored; output files carry predictions.
`sys.argv` belongs to the Worker launcher and is not a Candidate interface.
Version 8 publishes `evaluation.candidate_boundary.context_manifest`, the exact
`contexts.json` schema, including `examples[].example_id` and
`examples[].input_ids`. It also declares that `load_context_request()` returns
the parsed dictionary with unchanged keys and values, and publishes the binary
logit encoding, shape, and row/column order under
`evaluation.candidate_boundary.prediction_manifest.logits_artifact`.
Version 9 removes `evaluation.context_lengths` from the research-visible
manifest and renames its `evaluation.context_length` limit to
`evaluation.max_context_tokens`. `evaluation.context_policy` declares:

> Evaluation inputs are variable-length token sequences. The exact length
> distribution is not disclosed during research. Participants choose how to use
> context to optimize next-token prediction within the resource budget.

The maximum is an input bound, not a guarantee that any evaluation sample
reaches it. Exact sampling lengths, proportions, and construction parameters
stay in trusted evaluation configuration. The `contexts.json.context_lengths`
values describe only the actual request received by the frozen Candidate after
research sealing; this field is not a research-stage sampling plan. The
Candidate ABI and hidden dataset format are unchanged.
Version 10 restricts the Agent-visible `challenge_release` to `challenge_id`,
`release_stage`, `leaderboard_eligible`, `track_config_id`, and
`runtime_profile_id`. Full task/challenge manifest hashes and other trusted
release metadata are not copied into the Agent manifest. The public budget and
Training resource contract remain in `budget` and `training_environment`.
`challenge-info` likewise omits full task/challenge file hashes. Release IDs are
public labels; complete file identities remain in trusted verification and
episode audit records. An unsalted hash of an enumerable private configuration
can reveal it through offline guessing even when the configuration itself is
absent from the public manifest.

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

The Supervisor prints the episode directory. The trusted Operator can inspect
the latest Core snapshot and, when supplied by the deployment, current execution
evidence with:

```bash
autotrain-bench status \
  --episode /path/to/autotrain-bench-runs/episodes/<episode-id>
```

Without fresh, episode-bound deployment evidence, this view reports current
execution as unknown even if the Core snapshot says `active`. Finalized results
remain authoritative. Add `--observation PATH` for a deployment observation or
`--json` for the structured view; see [Operator status](docs/operator-status.md)
for the schema, freshness rules, and separate cleanup evidence. This view does
not query an allocator or change the Agent-visible status API.

The full Core report remains available with:

```bash
autotrain-bench report \
  --episode /path/to/autotrain-bench-runs/episodes/<episode-id>
```

Episode states declare `result_schema_version: 14`. `report` also accepts v12/v13
reports under their original semantics; v12 cannot contain operator-protection
fields or aborted outcomes. Neither old version carries verified evaluation
resource evidence. Unversioned, earlier, and unknown future versions
are rejected rather than guessed. This repository is still pre-preview. The first
published preview establishes the result-compatibility baseline; later schema
changes must then define an explicit compatibility policy.

Worker v16 and Challenge schema v3 add an independent
`evaluation_resource_contract` to a runtime profile. Core passes the selected
envelope and a fresh evaluation ID to each TEST/ANALYSIS invocation, checks the
Worker's allocation receipt before accepting metrics, and records the contract,
receipt, and assessment in the v14 report. Missing contracts remain explicit
(`evaluation_resource_envelope: null`) in previews; they do not establish cost
comparability. Existing Challenge v2 profiles remain readable without this field.
Formal releases require an enforced evaluation contract and accepted TEST
resource evidence. See [evaluation resource contracts](docs/evaluation-resources.md)
for the Worker interface, local limits, and remaining comparison conditions.

In v13, `outcome: "success"` requires a finalized episode, a non-empty final
Candidate ID, a valid TEST performance score with the exact release-declared raw
metrics, and a complete trusted TEST cost measurement. Current reports identify hidden split digests with
`artifact_identity_algorithms.hidden_evaluation` and a formal organizer receipt
must declare the same algorithm. A failed outcome requires a failure reason and
keeps performance and cost measurements `null`. A null outcome is allowed
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

Version 13 additionally records operator protection in the trusted-only
`operator_limits` audit field. When that protection stops research, the run has
`status: "operator_stopped"`, `outcome: "aborted"`, and
`failure: "operator_time_limit_reached"`. It has no TEST measurement and is
not a participant failure or a normal Track B result. Agent Protocol v11
recognizes the additional terminal lifecycle; it exposes no protection setting
or remaining operator allowance. Task and Worker protocols are unchanged.

`complete` means the ANALYSIS pass finished and gave every selected Candidate a
disposition, not that every point produced a metric. A result therefore also
carries `analysis_selected_points` and `analysis_evaluated_points`, so a reader
of the summary fields alone can see a complete pass that still has gaps without
walking the trajectory. The counts are validated against the trajectory they
summarize, so the summary cannot drift from its own detail.

`state.json` is the authoritative result commit point. Immediately before that
atomic commit, the Supervisor appends a non-terminal
`RESULT_PUBLICATION_PREPARED` audit record. A finalized v11 state created by the
current implementation includes `result_publication`, which binds the result
to that record's publication ID and sequence number. An unreferenced prepared
record documents an interrupted attempt and is not a terminal result. The
reference remains optional within the current development schema so an
interrupted publication rollout does not manufacture a terminal result.

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
Evaluator and untrusted Candidate containers. Agent and Training containers
receive neither the Docker socket nor ANALYSIS/TEST mounts. The Candidate
receives only the target-free context file, never the hidden split directory or
target file. Sandboxed Agent, Training, Candidate, and scoring networking is
disabled. The Gateway handles only frozen researcher-inference endpoints. The
bundled launcher nevertheless
reports `policy_compliant: false` because its writable host bind mounts lack a
deployment-enforced quota. In this local compatibility path the Supervisor
process also enforces episode and Training-Lease deadlines; an external
certified Worker must enforce its deadline envelope even if the Coordinator
disconnects. `unrestricted-dev` and `host_cli` add further non-compliant local
integration boundaries.

The Docker Worker verifies a numeric execution UID/GID before launch using
disposable `0400` input and `0700` output storage. It discovers the owner as
seen by the Docker daemon and checks that container-created private files are
readable, owned, and removable by the Worker. Failed verification stops setup;
there is no fallback to broader permissions or additional capabilities.

Agent, Training, Candidate, Scorer, and Gateway containers explicitly run as
that verified owner, overriding the image's `USER`. Images must support this
arbitrary numeric UID/GID and keep required executables and dependencies
accessible to it. Home/scratch directories are writable by that identity;
hidden snapshots remain `0400`, and writable evaluation directories are `0700`.
The trusted runtime record includes `container_execution_identity`. Role
separation still comes from the distinct mounts and network boundaries, not
from assigning different host UIDs to the roles. After confirmed termination,
prediction cleanup can restore owner access on locked directories without
following symlinks or changing hidden inputs. Existing artifacts owned by an
unmapped or different operator account require deployment-side migration.

Here, **user/operator** means the entity that prepares and launches the run; it
may itself be a human or another Agent and is inside the trusted host boundary.
**Tested Agent** means the submitted autonomous system being benchmarked. The v0
threat model protects benchmark policy from the Tested Agent, not from the
user/operator or machine administrator. Docker isolation is a development
reference boundary and is not presented as protection against kernel,
container-runtime, or GPU-driver exploits. Official adversarial evaluation may
require stronger worker or VM isolation.

The `isolated_candidate` backend first validates both halves of one selected
hidden split in a trusted, network-disabled scorer. It then launches the
untrusted Candidate with only its content-addressed artifact, a read-only
`contexts.json`, and isolated writable prediction storage. The Candidate reads
and schedules the full request itself and writes `predictions.json` plus a
row-major little-endian float32 `logits.f32`. A second trusted scorer joins
those outputs to the unmounted `targets.json`. Exactly one vector is required
per context in request order; targets are never returned, and the hidden schema
rejects overlapping protected source spans. Logits must be finite numeric
values. The Agent-visible task manifest publishes every exact manifest field,
its constant or request-derived source, and the rule that additional fields are
forbidden; it also names the optional public reader and writer helpers. Hidden
Candidate failures still return only fixed error codes to the Agent; private
diagnostics stay in the trusted episode directory. Candidate and scorer
containers are destroyed after each evaluation. A conforming Evaluation Worker
also supplies an independent output allocation before Candidate startup:
`round_up(max(16 MiB, 2 * (examples * vocab_size * 4 + 65536)), 1 MiB)` of
allocated storage and at most 128 filesystem inodes, including the root.
Temporary files are allowed during execution; the final artifact must still
contain exactly the two specified regular files with exact logits length and
valid format. The bundled Docker Worker enforces this boundary with a bounded
Linux Docker tmpfs volume, independently of episode logs/results. It verifies
the allocation before launching Candidate and refuses unbounded fallbacks.
A read-only trusted holder preserves outputs until scoring; all consumers must
be confirmed stopped before holder/volume removal. Failed cleanup remains
registered for retry. This supports `candidate_prediction_storage_quota` under
Worker Protocol v15, but does not change development trust or formal eligibility.
See [Worker prediction storage](docs/worker-protocol.md) for memory/swap needs,
deployment support, cleanup limits and the v14 compatibility break.

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
    └── offline evaluation sequence
        ├── trusted validation with contexts + targets
        ├── untrusted Candidate with contexts only, no network
        └── trusted scoring with targets + Candidate predictions
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
also fixes the offline file ABI, while the Worker chooses how to materialize
and isolate those files. The bundled Docker Worker uses bind-mounted artifacts
and no evaluation network.

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

Task schema v7 still retains `paths.analysis` and `paths.test` as compatibility
inputs for the bundled local CLI. The Supervisor neither reads them nor sends
them through Worker Protocol v16. An external launcher supplies its Evaluation
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
and never replaces the final TEST performance and cost measurements. A release
freezes its Candidate selection limit and result-visibility policy before task
reveal. Publication may be organizer-only or delayed to reduce cross-episode
leakage. The local
`report` command is a trusted Operator view and retains the complete audit
record; an organizer's submission/publication layer must enforce the declared
visibility policy when distributing that record.

---

## TEST

TEST provides the final performance and cost measurements for the Agent-declared
incumbent. Performance measures generalization; cost measures the complete
Candidate invocation per prediction task.

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

The evaluator contains a release-frozen set of separately scored examples
$(c_i, y_i)$ with variable context lengths within the public input bound.
The research-visible contract publishes that bound and the scoring rules;
the actual length distribution and sampling parameters belong to trusted
evaluation configuration frozen before research starts. Participants choose
how much context to use. After research sealing, the Worker exposes the complete,
target-free context collection to the Candidate and requires exactly one output
vector for each $c_i$, while keeping the corresponding next-token target $y_i$
permanently private. The Candidate controls ordering, batching, parallelism,
and model lifetime internally, but restores request order in its output.
Protected source spans may not overlap, so a target
cannot be recovered later from a longer or neighboring context. Each span
covers the source tokens represented in a context plus that example's target
occurrence. If $q_i=\operatorname{softmax}(z_i)$ is the distribution formed from
the logit vector returned for context $c_i$, the aggregate target negative
log-likelihood is:

$$
L
=
-\frac{1}{N}
\sum_i
\log q_i[y_i].
$$

The current LM release retains mean NLL as its canonical raw metric. Token
perplexity can be derived for display when representable:

$$
\operatorname{PPL}=\exp(L).
$$

Perplexity is derived once from the mean of the single-target losses; the
evaluator never averages per-sequence or per-batch perplexities. This release
maps the raw measurement into the generic performance interface:

$$
\operatorname{performance\_score}
=100\max\left(0,\min\left(1,1-\frac{L}{\log V}\right)\right).
$$

The score is finite, bounded to `[0, 100]`, and higher is better. A score of 100
represents perfect prediction; uniform-over-vocabulary prediction and worse map
to 0. The canonical `mean_nll` raw metric retains distinctions among results at
that floor. The definition ID, range, direction, raw metric names, and
`benchmark_release` comparison scope are part of the task manifest. Core
validates this generic contract and its result shape; the frozen trusted task
evaluator owns the formula. Another benchmark release may use a different
dataset, raw metric, or formula without changing the result interface, and its
score is not comparable with this one. Perplexity is not a required raw field,
so overflow in that derived display value cannot invalidate an otherwise finite
measurement.

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

The Track A result is the final TEST performance–cost pair:

$$
R_{\mathrm{TEST}}(T_{\mathrm{total}})
=\left(
\operatorname{performance\_score}_{\mathrm{TEST}},
\operatorname{time\_per\_task}_{\mathrm{TEST}}
\right).
$$

Both measurements use the final Agent-declared incumbent when research seals,
under the total research budget $T_{\mathrm{total}}$. Higher performance and lower
time per task are better; the two axes remain separate.

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

The bundled launcher enables private research-time protection by default for
both bundled Track B tasks:

| Task | Default operator protection |
| --- | --- |
| `lm-pretrain-v1` | 24 hours (86,400 seconds) |
| `smoke_lm` | 30 minutes (1,800 seconds) |

Each task keeps this trusted setting in a separate `operator-defaults.json`
beside `task.json`, with the key `track_b_max_research_seconds`. The optional
`max_episode_storage_bytes` key configures separate storage protection described
below. These settings are not benchmark budgets and do not enter the task/release
manifest digests or the comparison key. A custom task without that file has no default protection;
the launcher warns explicitly when protection is disabled.

Before preparation and Agent launch, `start` tells the operator the effective
limit, how to change or disable it, and that disabling protection risks extremely
high token consumption. To override the default for one run, add:

```text
--operator-max-research-seconds 7200
```

To remove this protection for one run, add:

```text
--no-operator-research-limit
```

These options are mutually exclusive and rejected for Track A. A custom limit
must be a finite positive number of seconds. Omitting both options uses the
task's default for Track B and has no effect on Track A. The default file must
contain a finite positive value; invalid defaults fail before runtime setup
unless an explicit run option overrides them.

The protection clock starts at task reveal (`Supervisor.begin`),
includes Agent startup, research, waiting between Leases, and Worker preparation,
and receives no accounting refunds. Pre-reveal preparation is outside this clock.
The setting is fixed before reveal and recorded only in trusted episode state
and `report`; the default file, launch notices, and effective value are absent
from the public task, Agent instructions/status, and
the advertised Lease budgets. It does not alter the comparison key.

On expiry, research is irreversibly stopped. The local launcher revokes the
inference Gateway, Agent, and active or pending Training workloads, then performs
normal cleanup without a finish-drain allowance. It records an operator-aborted
run, returns a nonzero exit code, and performs neither TEST nor ANALYSIS, even
if an incumbent exists. Successful normal finish before the deadline disarms
this research protection; postmortem evaluation keeps its own existing timeouts.

### Private operator storage protection

On both Tracks, `lm-pretrain-v1` defaults to **100 GiB** and `smoke_lm` to
**2 GiB** of episode filesystem capacity. These are private operator settings,
not Track budgets, Candidate artifact limits, or evaluation targets. They are
not included in the public task or Agent instructions. Filesystem capacity may
still be observable by code running inside the workload.

The Worker deployment must provision a dedicated, empty, bounded filesystem
for each episode. The parent of `--episode-dir`, or `--episodes-root` itself,
must be the root of that filesystem. Its reported capacity must be positive
and no greater than the requested limit, with bounded metadata/inode storage.
An ordinary directory, an ordinary Docker named volume, or a free-space check
does not implement this protection. Filesystem/volume creation, quota setup,
ownership, capacity reservation and eventual disposal belong to the deployment;
the benchmark does not format disks, mount filesystems, or resize existing data.

Before copying the Agent workspace or revealing the task, the bundled Worker
verifies the filesystem boundary, capacity and inode observation. A trusted,
network-disabled probe must observe the same capacity and a disposable shared
identity from Docker. Each workload's writable paths are checked against the
admitted filesystem, rejecting escaped paths and nested/replaced filesystems.
With protection enabled, images declaring Docker `VOLUME` mounts are rejected
before container creation, including declarations under an explicit mount.
This covers Agent, Training, Candidate, scorer, Gateway and trusted probes.
Launches use the inspected image ID so changing a tag cannot introduce an
unchecked anonymous volume. Images must omit `VOLUME` declarations for this mode.
Storage admission does not constitute independent certification of an arbitrary
filesystem provider; the deployment must actually enforce its reported bounds,
including separately bounded filesystem metadata overhead.

For example, with an already provisioned empty filesystem at `/worker/episode-volume`,
use `--episode-dir /worker/episode-volume/run`. To override or disable the
protection for one run, respectively:

```text
--operator-max-storage-bytes 107374182400
--no-operator-storage-limit
```

These options are mutually exclusive. An override must be a positive integer
of bytes. The launcher announces the effective choice and the disk-exhaustion
risk of disabling it. The `host_cli` preview requires explicit disabling because
a host-native Agent can write outside the volume. A custom task without storage
defaults has protection disabled unless an override is supplied. Missing or
incompatible bounded storage fails before episode creation; there is no silent
fallback to ordinary directories.

Each protection validates only the default it actually uses. An invalid storage
default cannot block an explicit storage override/disable, and an invalid Track B
time default cannot block Track A or an explicit time override/disable. When a
default is needed, the shared file must still be valid JSON with known keys.

The protected scope is the episode directory: workspace, mailbox, Lease scratch
and output, captured Candidates and episode logs/audits.
Prediction output has its own bounded allocation, separate from this volume.
Prepared TRAIN/hidden data and images are separate, as are the runtime's already
bounded temporary memory filesystems and rotating container logs. Research and
evaluation retain their separate mount views; sharing a capacity boundary does
not grant access to each other's files. Hidden materialization remains under the
Evaluation Worker's existing private lifecycle.

The filesystem refuses allocations when full, including storage held by an
open-but-unlinked file. It may allow progress again after space is released;
this is not the research timer and does not itself issue an episode stop.
An unrecovered I/O failure follows existing error/teardown handling and may
prevent writing complete episode records to the full volume. Operators must
retain Worker-side failure logs and clean up only after workload termination.
No new participant penalty or evaluation scoring rule is introduced.

Storage evidence is retained in the private Worker environment record as
`operator_storage_protection`; Core does not receive filesystem provisioning
commands. For remote execution, the remote Worker/launcher must enforce and
record the protection where the bytes are written. Checking the operator's
laptop disk is not sufficient. The bundled local composition still requires
shared paths with Docker: selecting a remote Docker context alone does not
provide a remote Worker. Separate Worker storage domains need their own
deployment-controlled allocations; a receipt from one domain does not bound
another. Neither the Worker protocol nor the comparison key changes.

The separate `candidate_prediction_storage_quota` guarantee uses the revised
Worker v15 allocation rule above. Disabling private episode storage protection
does not disable prediction output protection or exact final artifact validation.

The bundled launcher uses a separate guard thread and local runtime client so
ordinary revocation does not wait for the Coordinator's synchronous Worker
lock. The container CLI also runs an independent parent watchdog: missed
heartbeats or operation deadlines beyond a short grace trigger process-group
termination and journaled cleanup. Its default heartbeat tolerance, deadline
grace and control-call fallback are 10, 5 and 120 seconds; the launcher prints
their `--worker-watchdog-*-seconds` overrides before workloads start. See
[Worker recovery](docs/worker-recovery.md) for the precise scope and failure
report. `host_cli` and direct Python integrations require their own external
supervision. Runtime/host outages may still prevent termination; cleanup can
outlast the research limit, which is not a token or monetary cap. This feature
does not upgrade the local deployment's policy-compliance claims.

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

The Track B result uses the same TEST performance–cost pair:

$$
R_{\mathrm{TEST}}(N_{\mathrm{leases}}, T_{\mathrm{lease}})
=\left(
\operatorname{performance\_score}_{\mathrm{TEST}},
\operatorname{time\_per\_task}_{\mathrm{TEST}}
\right).
$$

Both measurements use the final Agent-declared incumbent under the selected
lease-count and per-lease time limits. Higher performance and lower time per task
are better; the two axes remain separate.

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
| Evaluation axes     | TEST performance and cost        | TEST performance and cost                      |
| Self-improvement    | Allowed                          | Allowed                                        |
| Human intervention  | Forbidden                        | Forbidden                                      |
| Primary question    | How fast can the system improve? | How well can it use bounded training access?    |

Together they expose two different scaling behaviors.

Across completed episodes at different budgets, Track A measures:

$$
R_{\mathrm{TEST}}(t).
$$

Track B measures:

$$
R_{\mathrm{TEST}}(n, T_{\mathrm{lease}}).
$$

Here $R_{\mathrm{TEST}}$ is the performance–cost pair, $t$ is the research time
budget, and $n$ is the Training Lease budget with per-lease duration
$T_{\mathrm{lease}}$ held fixed. Neither track supplies organizer feedback during
research. Each episode contributes one final TEST result; within-episode
Candidate trajectories are measured afterward on ANALYSIS.

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
R_{\mathrm{TEST}}
= f(
\text{agent tokens},
\text{research time},
\text{lease and experiment interactions}
).
$$

Here $R_{\mathrm{TEST}}$ contains both final performance and time per task. Agent
token usage is a separate research diagnostic; it is not Candidate serving cost.

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

The benchmark records the final performance–cost pair together with the submitted
Candidate and incumbent history.

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

The submitted-Candidate ANALYSIS trace shows how the committed artifacts changed
before the final result, without providing an official signal while the Agent can
still act.

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

These histories distinguish research trajectories even when the two systems have
similar final performance and cost.

---

# Evaluation Axes

## Performance

The performance axis is:

$$
\boxed{\text{final hidden TEST performance score}}
$$

under the selected track's resource constraint.

Higher is better. The score is always finite and lies in `[0, 100]`; its
release-owned definition converts the task's raw metrics into that interface.
Scores from different benchmark releases are not comparable.

The performance-score mapping must not combine unrelated axes into an opaque
formula.

## Cost

The independent cost axis is:

$$
\boxed{\operatorname{time\_per\_task}
=\frac{\operatorname{candidate\_wall\_seconds}}
{\operatorname{task\_count}}}
$$

Lower is better. For `lm-pretrain-v1`, one task is one next-token prediction
from one hidden prefix. This is an amortized workload cost, not the latency of
an individual request: the Candidate controls batching and scheduling across
the complete hidden context file. The denominator is the release-required task
count, never a participant-selected count of successful outputs. Missing or
malformed predictions fail the evaluation instead of reducing reported cost.

Candidate wall time starts immediately before the isolated Candidate
invocation and stops after process exit and prediction-file flush. It includes
sandbox startup, Candidate import, model loading, input-file reading,
participant-controlled scheduling and inference, and output-file writing.
Hidden-data materialization, trusted input validation, target joining, and
metric computation are excluded.

Performance–Cost comparison uses Pareto dominance: result A dominates result B
when A has no lower performance and no higher time per task, with at least one
strict improvement. The leaderboard frontier contains the non-dominated
results. The two axes are never collapsed into a scalar. Comparisons require
the same benchmark release, hidden input identity, and runtime profile.

---

## First-class diagnostic metrics

The benchmark also records:

* postmortem ANALYSIS performance score and release-declared raw metrics;
* final TEST performance score and release-declared raw metrics;
* Candidate wall time, task count, tasks/second, input tokens/second, and
  prediction bytes as cost submetrics for TEST and each measured ANALYSIS point;
* postmortem ANALYSIS performance-score trajectory;
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

These metrics support scientific analysis without replacing either evaluation
axis.

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

The trusted scoring environment receives only the selected hidden split and
isolated output location. A separate Candidate environment receives the
immutable Candidate artifact, the target-free context file read-only, and a
private writable prediction directory with a release-derived hard byte and
entry budget. The Worker runs these environments sequentially and transfers
only the prescribed files; neither has a network. Both are inaccessible to
Agent and Training Containers, and the scorer returns only
benchmark-permitted aggregate performance and cost evidence. Prediction files
are not episode audit artifacts: they are removed after scoring or failure and
confirmed evaluation teardown.

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

An episode succeeds only when post-mortem TEST evaluation produces a finite
performance score in `[0, 100]`, the exact finite raw metrics declared by the
release, and a complete trusted Candidate cost measurement.
An episode with no declared incumbent, a failed TEST evaluation, an invalid TEST
performance measurement, or an invalid raw metric is finalized with
`outcome: "failed"`; `test_performance_score`, `test_raw_metrics`, and
`test_cost` remain `null`.
Infrastructure failures are
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
per-evaluation Candidate time per task and supporting cost measurements
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
  "schema_version": 7,
  "task_id": "tinystories-byte-lm-smoke-v1",
  "task_kind": "development_smoke",
  "leaderboard_eligible": false,
  "paths": {
    "train": "data/train",
    "analysis": "data/private/analysis",
    "test": "data/private/test"
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
    "candidate_protocol": "offline_next_token_logits_v3",
    "entrypoint": "evaluator/evaluator.py",
    "performance_score": {
      "definition_id": "uniform_normalized_log_score_v1",
      "range": {"minimum": 0.0, "maximum": 100.0},
      "direction": "maximize",
      "comparison_scope": "benchmark_release",
      "raw_metrics": ["mean_nll"]
    },
    "cost": {
      "definition_id": "candidate_end_to_end_time_per_task_v1",
      "task_unit": "next_token_prediction",
      "primary_metric": "time_per_task",
      "unit": "seconds_per_task",
      "direction": "minimize",
      "aggregation": "candidate_wall_seconds_divided_by_task_count",
      "comparison_scope": "benchmark_release",
      "comparison_requires": ["hidden_dataset_identity", "runtime_profile"],
      "submetrics": [
        "candidate_wall_seconds",
        "task_count",
        "input_tokens",
        "prediction_bytes",
        "tasks_per_second",
        "input_tokens_per_second"
      ]
    },
    "vocab_size": 257,
    "context_length": 128,
    "context_lengths": [8, 16, 32, 64, 128],
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
This example is trusted task configuration. Its `context_length` is the input
bound and `context_lengths` selects the local evaluation lengths, all of which
must fit that bound. The Agent receives a filtered manifest with
`max_context_tokens` and the variable-length declaration instead of that list.
The reproducible smoke/preview builders expose their local recipes in source;
filtering the episode manifest does not make those public recipes secret. A
formal release must keep its actual evaluation construction parameters and
datasets organizer-owned and outside participant-visible artifacts.

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

### 7. Keep evaluation axes explicit

The final objective is the release-defined hidden TEST Performance–Cost Pareto
frontier. Performance and time per task remain independently interpretable.

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

These studies may use the release's raw mean NLL $L$ as a diagnostic alongside
the final performance–cost pair. TEST scaling curves compare completed episodes
at different budgets; within-episode Candidate trajectories use postmortem
ANALYSIS.

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

How much better does a system become through evidence from its own experiments?

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
retains both raw TRAIN documents and GPT-2 token shards. Source bytes are checked
against a pinned upstream inventory, and TRAIN files are fully verified before
task reveal. Full-scale validation of the 10B build and its actual scale remains
open release work alongside organizer-owned hidden datasets,
final pinned task images, calibrated budgets, submission services, and
leaderboard infrastructure. Additional
inference-provider adapters and isolation beyond the v0 Docker threat model
also remain open design work.

The v0 prototype is intentionally single-host and never acquires machines or
scheduler allocations. Users obtain a Docker-capable environment before
starting AutoTrain-Bench; the Worker then reports what the Training workload
actually received through the platform-neutral allocation receipt.

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
