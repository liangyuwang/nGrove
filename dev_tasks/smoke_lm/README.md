# Smoke LM development task

`smoke_lm` is the single bundled end-to-end development smoke task for
AutoTrain-Bench. It exercises the same release-bound launch contract,
task/participant boundary, Policy, trusted Evaluator implementation, Track
A/Track B state machine, Candidate isolation, and incumbent semantics as
`lm-pretrain-v1`. Its corpus, tokenization, context, data layout, split recipe,
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
| Evaluation context lengths | 8, 16, 32, 64, and 128 tokens |
| Track A | 300 seconds total wall clock |
| Track B | Three leases, at most 75 seconds each |
| Evaluator measurement | Mean single-target next-token NLL |
| Leaderboard-style score | Perplexity, `exp(mean_token_nll)`, lower is better |
| Candidate protocol | `offline_next_token_logits_v3` |

The trusted evaluation path runs on CPU. A participant may train with any
approach supported by the selected Training Environment and must export a
Candidate that satisfies the public ABI. Concrete runtime provenance and Worker
capability evidence belong to the episode result. A development deployment
cannot become leaderboard eligible merely by reporting the expected hardware.

Training resources are selected from the finite map in `challenge.json`, never
by `task.json` or ad-hoc launch flags. `cpu-local` keeps the smoke path cheap;
the A100/H100 profiles exercise the same accelerator-profile contract as
`lm-pretrain-v1`. Inspect the supported Track and runtime choices before launch:

```bash
autotrain-bench challenge-info \
  --challenge dev_tasks/smoke_lm/challenge.json
```

## Prepare

```bash
python3 dev_tasks/smoke_lm/prepare_data.py
autotrain-bench prepare --task dev_tasks/smoke_lm/task.json
```

Preparation downloads a deterministic slice of
`karpathy/tinystories-gpt4-clean` at revision
`0397e27157956705a0260709da3095bb9c43d6a7`, verifies the ordered source digest,
and writes ignored generated artifacts under `dev_tasks/smoke_lm/data/`.

## Run an Agent integration

An egress-denied participant is supplied explicitly as an Agent Container.
The image contains the Agent executable, while the writable episode workspace
starts empty:

```bash
autotrain-bench start \
  --challenge dev_tasks/smoke_lm/challenge.json \
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
  --challenge dev_tasks/smoke_lm/challenge.json \
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
  --challenge dev_tasks/smoke_lm/challenge.json \
  --track-config smoke-a-5m \
  --runtime-profile cpu-local \
  --episodes-root /path/to/autotrain-bench-runs/episodes \
  --agent-runner host_cli \
  --agent-command "python3 -m autotrain_bench.cli_adapters.codex --model MODEL --reasoning-effort high"
```

The `host_cli` run reports `policy_compliant: false` because a host process can
access files and networking outside the enforced episode boundary. The
Candidate/Evaluator boundary remains isolated. An egress-denied participant
container exercises the strongest local v0 path, but the bundled Worker is a
Development deployment and its deadline guard shares the Coordinator process.
It therefore cannot produce a formal leaderboard result.

The Tested Agent is governed by the same [preview Policy](POLICY.md) as
`lm-pretrain-v1`. In particular, the initial Agent image/workspace may not
contain extra training corpora or a pretrained target-model checkpoint.

## Candidate artifact contract

Training commands write Candidate artifacts beneath the lease-owned `/output`
mount. A participant can validate the public ABI before committing the artifact:

```bash
python -m autotrain_bench.cli agent lease-exec -- \
  python -m autotrain_bench.candidate_preflight \
  --candidate /output/candidate \
  --device cpu --vocab-size 257 --context-lengths 8 16 32 64 128

python -m autotrain_bench.cli agent lease-commit /output/candidate
```

Commit stops the Training sandbox before validating and snapshotting the
artifact. It returns a content identity but no organizer score. Candidate
bundles may contain only regular files and directories,
may not contain symbolic or hard links, and must fit the task-defined file and
byte limits.

Hidden evaluation uses balanced, non-overlapping context windows at the five
release-declared lengths above. Each window contributes exactly one prediction:
the Candidate receives the context and returns next-token logits, while the
target remains permanently inside the trusted Evaluator. A target is never
revealed later through a longer prefix or reused inside another context from
the same source span. TEST perplexity is `exp(mean_token_nll)` over those
single-target examples; it is not the legacy all-prefix corpus perplexity.

`candidate.py` defines one offline entry point:

```python
def run_offline(candidate_dir, contexts_path, output_dir, device): ...
```

`contexts_path` is a read-only JSON artifact containing example IDs and token
contexts but no targets or split label. The serving system owns file reading, model loading,
batching, ordering, and parallelism. It writes exactly `predictions.json` and
`logits.f32` under `output_dir`; logits are finite row-major little-endian
float32 vectors in request order, one `[vocabulary]` row per example. The
public `autotrain_bench.candidate_protocol` helpers can read the request and
write the output, but their use is optional.

A conforming Worker hard-limits this untrusted directory to two entries and
`examples * vocab_size * 4 + 65536` bytes; the manifest itself is limited to
65536 bytes. The bundled local Docker preview cannot enforce a hard quota on
its host bind mount and reports that capability as unsupported rather than
claiming formal compliance.

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

Serving wall time covers the entire Candidate invocation: sandbox startup,
Candidate import, model loading, context-file reading, participant-controlled
scheduling/inference, prediction writing, flushing, and process exit. Trusted
hidden-data materialization, validation, target joining, and scoring are
excluded. Results record NLL/perplexity and raw serving efficiency separately;
this development task does not yet combine them into one ranking score.
