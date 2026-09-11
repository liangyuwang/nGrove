# Smoke LM pretrain development task

`smoke_lm-pretrain` is the single bundled end-to-end development smoke task for
AutoTrain-Bench. It exercises the same release-bound launch contract,
task/participant boundary, Policy, trusted Evaluator implementation, Track
A/Track B state machine, Candidate isolation, and incumbent semantics as
`lm-pretrain`. Its corpus, tokenization, context, data layout, split recipe,
evaluation resources, and budgets are deliberately smaller development
choices. Its scores are not benchmark results and it is never leaderboard
eligible.

## Ownership boundary

The smoke task owns only:

- the public TRAIN split and its preparation recipe;
- the hidden ANALYSIS/TEST datasets and trusted Evaluator;
- the public metric and Candidate ABI;
- the Training, Candidate, Evaluator, and Gateway images;
- the finite Track/runtime profile maps, budgets, and artifact limits;
- the release Policy.

The participant owns the Agent image or CLI, its command, and every initial
workspace file. If `--agent-workspace` is omitted, AutoTrain-Bench creates an
empty episode workspace. The Agent must create its own training program, model,
experiments, Candidate implementation, and incumbent decisions after task
reveal. This task does not ship a reference or starter Agent. An egress-denied
Agent sees only its episode workspace, TRAIN, the public task manifest, and the
control mailbox; it cannot see the trusted Core checkout.

The task provides no organizer validation feedback during research. The Agent
may decide whether and how to validate using permitted TRAIN information, or may
decide not to create a holdout. After research is permanently sealed, ANALYSIS
scores the frozen Candidate history and TEST scores only the declared final
incumbent.

## Fixed smoke task

| Smoke-task property | Value |
| --- | --- |
| Corpus | 1,200 pinned TinyStories documents |
| TRAIN | 1,000 documents, about 818K target bytes |
| ANALYSIS / TEST | 512 hidden next-token targets each |
| Token representation | UTF-8 bytes plus BOS, vocabulary 257 |
| Evaluation input bound | 128 tokens; variable lengths, no guarantee of a sample at the bound |
| Track A | 300 seconds total wall clock |
| Track B | Three leases, at most 75 seconds each |
| Evaluation axes | Performance score and end-to-end time per task |
| Performance score | `100 * max(0, min(1, 1 - mean_token_nll / log(vocab_size)))`, higher is better |
| Cost | Candidate wall seconds / 512 next-token tasks, lower is better |
| Candidate protocol | `offline_next_token_logits_v3` |

The trusted evaluation path runs on CPU. A participant may train with any
approach supported by the selected Training Environment and must export a
Candidate that satisfies the public ABI. Concrete runtime provenance and Worker
capability evidence belong to the episode result. A development deployment
cannot become leaderboard eligible merely by reporting the expected hardware.

The Agent receives the same variable-length declaration as the challenge:
evaluation inputs are variable-length token sequences, the exact length
distribution is not disclosed during research, and participants choose how to
use context to optimize next-token prediction within the resource budget.
`evaluation.max_context_tokens` publishes only the input bound. The local smoke
recipe remains reproducible from its public source; this visibility convention
does not make that recipe suitable as a secret formal evaluation.

Training resources are selected from the finite map in `challenge.json`, never
by `task.json` or ad-hoc launch flags. `cpu-local` keeps the smoke path cheap;
the GPU profiles mirror the [lm-pretrain hardware map](../../challenges/lm-pretrain/README.md#supported-preview-lanes):
A100 SXM4 80GB, H100 SXM 80GB, H200 SXM 141GB, and H200 NVL 141GB, each with
1, 2, 4, or 8 full GPUs on one host. These select Training resources; this smoke
task's trusted evaluation remains on CPU. The budgets remain five minutes for
Track A and three 75-second Leases for Track B at every GPU count. Inspect the
supported Track and runtime choices before launch:

```bash
autotrain-bench challenge-info \
  --challenge dev_tasks/smoke_lm-pretrain/challenge.json
```

These profiles are integration configurations, not official comparison
environments. For custom hardware and independent Agent comparisons, follow the
[reference environment and comparison policy](../../challenges/README.md#reference-environments-and-custom-comparisons).

## Prepare

Install Core first with `python3 -m pip install -e .` from the repository root.
The data builder and integrity verifier otherwise require only the standard
library.

```bash
python3 dev_tasks/smoke_lm-pretrain/prepare_data.py
autotrain-bench prepare --task dev_tasks/smoke_lm-pretrain/task.json
```

Preparation downloads a deterministic slice of
`karpathy/tinystories-gpt4-clean` at revision
`0397e27157956705a0260709da3095bb9c43d6a7`, verifies the ordered source digest,
and writes ignored generated artifacts under `dev_tasks/smoke_lm-pretrain/data/`.

`data/train/manifest.json` inventories the public TRAIN files without including
hidden evaluation metadata. Cached preparation verifies this inventory;
legacy or damaged data is rebuilt. `autotrain-bench prepare` and every new
episode independently check the full TRAIN view before the Agent starts.
See [TRAIN integrity](../../docs/training-data.md).

## Run an Agent integration

Track B (`smoke-b-3x75s`) defaults to a **30-minute private operator research-time
protection**, configured in `operator-defaults.json`. The launcher announces it
before run preparation. Add `--operator-max-research-seconds SECONDS` to change
it or `--no-operator-research-limit` to disable it for one run. Disabling protection
risks extremely high token consumption. The setting and notices are never sent
to the Tested Agent and do not change the three 75-second Leases or comparison key.
See [operator protection semantics](../../docs/benchmark-reference.md#track-b--lease-bounded-autonomous-research).

Both Tracks also default to **2 GiB of private operator storage protection**.
The Worker deployment must provide an empty, bounded filesystem at the selected
episode parent directory; an ordinary directory will fail admission. Use
`--operator-max-storage-bytes BYTES` to override the limit or
`--no-operator-storage-limit` to explicitly accept unprotected storage. The
`host_cli` preview requires explicit disabling. This does not change Track
budgets or the prediction-output protocol. Remote Workers must enforce this in
their execution environment. See
[storage protection requirements](../../docs/benchmark-reference.md#private-operator-storage-protection).

An egress-denied participant is supplied explicitly as an Agent Container.
The image contains the Agent executable, while the writable episode workspace
starts empty:

```bash
autotrain-bench start \
  --challenge dev_tasks/smoke_lm-pretrain/challenge.json \
  --track-config smoke-a-5m \
  --runtime-profile cpu-local \
  --episodes-root /path/to/autotrain-bench-runs/episodes \
  --agent-runner container \
  --agent-image participant-agent@sha256:DIGEST \
  --agent-command "participant-agent run"
```

The same smoke task can exercise a published accelerator profile without
changing the task or using manual resource overrides:

```bash
autotrain-bench start \
  --challenge dev_tasks/smoke_lm-pretrain/challenge.json \
  --track-config smoke-a-5m \
  --runtime-profile h100-sxm-80gb-x4 \
  --episodes-root /path/to/autotrain-bench-runs/episodes \
  --agent-runner container \
  --agent-image participant-agent@sha256:DIGEST \
  --agent-command "participant-agent run"
```

A logged-in CLI such as Codex can be rehearsed from the same default empty
workspace by omitting `--agent-workspace`. The episode root remains explicit
and outside the Core checkout. This matters for host-native tools that discover
instructions from parent directories: an empty workspace nested beneath the
Core would still inherit the operator's `AGENTS.md` and is rejected before task
reveal. The Tested Agent's own global configuration and participant-owned
instructions remain in scope.

```bash
autotrain-bench start \
  --challenge dev_tasks/smoke_lm-pretrain/challenge.json \
  --track-config smoke-a-5m \
  --runtime-profile cpu-local \
  --episodes-root /path/to/autotrain-bench-runs/episodes \
  --agent-runner host_cli \
  --agent-command "python3 -m benchcore.cli_adapters.codex --model MODEL --reasoning-effort high"
```

The `host_cli` run reports `policy_compliant: false` because a host process can
access files and networking outside the enforced episode boundary. The
Candidate/Evaluator boundary remains isolated. An egress-denied participant
container exercises the strongest local v0 path, but the bundled Worker is a
Development deployment and its deadline guard shares the Coordinator process.
It therefore cannot produce a formal leaderboard result.

The Tested Agent is governed by the same [preview Policy](POLICY.md) as
`lm-pretrain`. In particular, the initial Agent image/workspace may not
contain extra training corpora or a pretrained target-model checkpoint.

## Candidate artifact contract

Training commands write Candidate artifacts beneath the lease-owned `/output`
mount. A participant commits an artifact through the mailbox:

```bash
python -m benchcore.cli agent lease-commit /output/candidate
```

Commit stops the Training sandbox before validating and snapshotting the
artifact. It returns a content identity but no organizer score. Candidate
bundles may contain only regular files and directories,
may not contain symbolic or hard links, and must fit the task-defined file and
byte limits.

Hidden evaluation uses non-overlapping context windows within the input bound.
The actual sampling lengths and proportions stay in trusted task configuration.
Each window contributes exactly one prediction:
the Candidate receives the context and returns next-token logits, while the
target remains permanently inside the trusted Evaluator. A target is never
revealed later through a longer prefix or reused inside another context from
the same source span. The evaluator retains mean NLL as its raw metric, then
maps it to the release's generic `[0, 100]` performance score. Perplexity can be
derived for display when representable. This development result is not
comparable with a different benchmark release.

The public task manifest declares the invocation contract under
`evaluation.candidate_boundary.entrypoint`. The Worker imports `candidate.py`
from the artifact root and calls its synchronous function with four positional
arguments:

```python
def run_offline(candidate_dir, contexts_path, output_dir, device): ...
```

`candidate_dir`, `contexts_path`, and `output_dir` are absolute `pathlib.Path`
objects naming the immutable artifact directory, the context JSON file, and
isolated writable prediction storage. `device` is the string in
`evaluation.device` (`cpu` or `cuda`). Module-level code executes during import;
`__name__` is not `__main__`, and `sys.argv` is not a Candidate interface. The
return value is ignored; the output files must be written before returning.

The exact input schema is declared at
`evaluation.candidate_boundary.context_manifest`. `contexts_path` contains a
read-only JSON object with exactly `schema_version` (integer 2),
`candidate_protocol` (string `offline_next_token_logits_v3`), `dataset_id`
(opaque string), `vocab_size` (integer), `context_lengths` (increasing positive
integer array), and `examples` (nonempty array). Each example contains exactly
`example_id` (unique opaque string) and `input_ids` (integer token array in
oldest-to-newest order). Context lengths belong to `context_lengths`; token IDs
are in `[0, vocab_size)`. Input lengths must not exceed
`evaluation.max_context_tokens`. The `context_lengths` values describe the
actual request, visible to the frozen Candidate only after research sealing.
No targets or split label are included.
The serving system owns file reading, model loading,
batching, ordering, and parallelism. It writes exactly `predictions.json` and
`logits.f32` under `output_dir`; logits are finite row-major little-endian
float32 vectors in request order, one `[vocabulary]` row per example. The
output manifest is a JSON object built from the request with exactly these
fields; member order and whitespace do not matter, but additional fields are
rejected:

```python
manifest = {
  "schema_version": 1,
  "candidate_protocol": "offline_next_token_logits_v3",
  "dataset_id": request["dataset_id"],
  "examples": len(request["examples"]),
  "vocab_size": request["vocab_size"],
  "dtype": "float32-le",
  "logits_file": "logits.f32"
}
```

The generated public task at `$AUTOTRAIN_BENCH_TASK` carries the same
machine-readable contract under
`evaluation.candidate_boundary.prediction_manifest`. The generated public
`benchcore.candidate_protocol` API is explicitly allowed in Agent and
Training environments: `load_context_request(path)` reads the request and
`write_prediction_artifact(output_dir, *, dataset_id, logits, examples,
vocab_size)` writes both output files. The reader returns the parsed dictionary
without renaming keys or wrapping values; its return schema is the same
`context_manifest`. These helpers are optional. The prediction declaration's
`logits_artifact` specifies finite IEEE-754 float32 values in little-endian,
headerless row-major order: row `i` corresponds to `examples[i]`, and column `j`
corresponds to token ID `j`.

A conforming Worker provides a separate output allocation of
`round_up(max(16 MiB, 2 * (examples * vocab_size * 4 + 65536)), 1 MiB)` and at most
128 filesystem inodes including the root. Temporary files are allowed while
Candidate runs. Final validation still requires exactly the two specified
files, exact logits length, and a manifest no larger than 65536 bytes. The
Docker Worker verifies and enforces an independent tmpfs allocation; filling it
does not consume episode log/result filesystem capacity. See
[Worker Protocol v15](../../docs/worker-protocol.md) for memory/swap needs,
cleanup behavior and compatibility. This does not grant formal compliance.

After research sealing and confirmed Agent/Research teardown, the bundled
Docker Worker runs three network-disabled steps for each ANALYSIS/TEST evaluation:

```text
trusted validation: contexts + targets
untrusted Candidate: Candidate files + contexts, never targets
trusted scoring: contexts + targets + read-only Candidate predictions
```

No target feedback is ever sent, and the hidden dataset schema rejects
overlapping source spans. The Candidate sees all unlabeled contexts for the
selected split at once and may schedule them jointly; the v3 contract does not
prescribe batches or reset its process or model between examples.
Candidate failures expose only a fixed error code to the Agent; bounded runtime
diagnostics remain inside the trusted boundary. Each container is removed after
the evaluation step. Prediction files are ephemeral and are deleted after
scoring or failure and confirmed teardown; an unconfirmed teardown retains
them only for a cleanup retry.

On an execution failure, the Worker records the stage in `<split>.failure.json`
and attempts to retain `<split>.candidate.stdout.log` and
`<split>.candidate.stderr.log` before removing the container. It stops Candidate
execution before collecting diagnostics, limits each stream to 1 MiB, and bounds
log collection time. Files are private (`0600`); raw output may contain hidden
contexts and must not be returned to the Agent or treated as a public report.
OOM/exit state and log truncation or collection failures are recorded separately.
Failure to collect diagnostics never skips workload cleanup or changes the fixed
public failure code.

Candidate wall time covers the entire Candidate invocation: sandbox startup,
Candidate import, model loading, context-file reading, participant-controlled
scheduling/inference, prediction writing, flushing, and process exit. Trusted
hidden-data materialization, validation, target joining, and scoring are
excluded. Results record the performance score, its mean-NLL raw metric, and
the independent `time_per_task` cost axis. `time_per_task` divides that wall
time by the frozen number of next-token tasks, while total time and throughput
remain cost submetrics. The two axes form a Pareto frontier and are not folded
into one score.
