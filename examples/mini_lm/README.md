# Mini LM challenge

`mini_lm` has the same task/participant boundary, Track A/Track B state
machine, hidden evaluation, Candidate isolation, and incumbent semantics as a
full AutoTrain-Bench challenge. Only the corpus and time budgets are smaller.
It is an example challenge, not a smoke test and not a starter-code task.

## Ownership boundary

The challenge owns only:

- the public TRAIN split and its preparation recipe;
- the hidden VAL/TEST splits and trusted Evaluator;
- the public metric and Candidate ABI;
- the Training, Candidate, Evaluator, and Gateway images;
- the Track A/Track B budgets and artifact limits.

The participant owns the Agent image or CLI, its command, and every initial
workspace file. If `--agent-workspace` is omitted, AutoTrain-Bench creates an
empty episode workspace. The Agent must create its own training program, model,
experiments, Candidate implementation, and incumbent decisions after task
reveal.

`reference_agent/` is an optional public baseline. It is not named by
`task.json`, copied into an episode, or mounted into a Training Container unless
the user explicitly selects it with `--agent-workspace`. A sandboxed Agent sees
only its episode workspace, TRAIN, the public task manifest, and the control
mailbox; it cannot see the repository or the reference baseline.

## Fixed mini task

| Challenge property | Mini value |
| --- | --- |
| Corpus | 1,200 pinned TinyStories documents |
| TRAIN | 1,000 documents, about 818K target bytes |
| VAL / TEST | 8,192 hidden target bytes each |
| Token representation | UTF-8 bytes plus BOS, vocabulary 257 |
| Evaluation prefix limit | 128 tokens |
| Track A | 300 seconds total wall clock |
| Track B | Three leases, at most 75 seconds each |
| Metric | Autoregressive NLL |
| Candidate protocol | `next_token_logits_v1` |

The reference configuration requires Docker, not an NVIDIA GPU or a particular
accelerator. Its trusted evaluation path runs on CPU. A participant may train
with any approach supported by the allocated Training Environment and must
export a Candidate that satisfies the public ABI. Concrete runtime provenance
belongs to the episode result; it does not change whether the isolation policy
is compliant.

Training resources are selected when the episode starts, never by `task.json`.
The default exposes CPU only. For example, a user with an NVIDIA runtime can add
`--training-gpus 1`; an optional
`--training-gpu-model "NVIDIA H100"` turns the model name into a host-side
preflight requirement. `--training-device /dev/DEVICE` exposes a generic Docker
device to Training Containers for runtimes that support it. Device requests and
the discovered GPU inventory are recorded in episode provenance. Results from
materially different runtime profiles should be compared separately.

## Prepare

```bash
python3 examples/mini_lm/prepare_data.py
autotrain-bench prepare --task examples/mini_lm/task.json
```

Preparation downloads a deterministic slice of
`karpathy/tinystories-gpt4-clean` at revision
`0397e27157956705a0260709da3095bb9c43d6a7`, verifies the ordered source digest,
and writes ignored generated artifacts under `examples/mini_lm/data/`.

## Run a participant Agent

A compliant participant is supplied explicitly as a sandboxed Agent Container.
The image contains the Agent executable, while the writable episode workspace
starts empty:

```bash
autotrain-bench start \
  --task examples/mini_lm/task.json \
  --track A \
  --agent-runner container \
  --agent-image participant-agent@sha256:DIGEST \
  --agent-command "participant-agent run"
```

The same challenge can expose a GPU without modifying the task:

```bash
autotrain-bench start \
  --task examples/mini_lm/task.json \
  --track A \
  --agent-runner container \
  --agent-image participant-agent@sha256:DIGEST \
  --agent-command "participant-agent run" \
  --training-gpus 1
```

To exercise the optional public baseline, explicitly snapshot its files:

```bash
autotrain-bench start \
  --task examples/mini_lm/task.json \
  --track B \
  --agent-runner container \
  --agent-image python@sha256:1042b61448fef4ba92d16a8c7eb4996d027568ce64792a7877fd88511e0af7c6 \
  --agent-command "python /workspace/agent.py" \
  --agent-workspace examples/mini_lm/reference_agent
```

A logged-in CLI such as Codex can be rehearsed from the same default empty
workspace by omitting `--agent-workspace`:

```bash
autotrain-bench start \
  --task examples/mini_lm/task.json \
  --track A \
  --agent-runner host_cli \
  --agent-command "python3 -m autotrain_bench.cli_adapters.codex --model MODEL --reasoning-effort high"
```

The `host_cli` run reports `policy_compliant: false` because a host process can
access files and networking outside the enforced episode boundary. The
Candidate/Evaluator boundary remains isolated. A formal compliant result uses a
sandboxed participant container or an equivalent isolated worker/VM.

## Candidate artifact contract

Training commands write Candidate artifacts beneath the lease-owned `/output`
mount. A participant can validate the public ABI before spending its one VAL
query for the lease:

```bash
python -m autotrain_bench.cli agent lease-exec -- \
  python -m autotrain_bench.candidate_preflight \
  --candidate /output/candidate \
  --device cpu --vocab-size 257 --context-length 128

python -m autotrain_bench.cli agent lease-commit /output/candidate
```

Commit stops the Training Container before validating and snapshotting the
artifact. Candidate bundles may contain only regular files and directories,
may not contain symbolic or hard links, and must fit the task-defined file and
byte limits.

`candidate.py` defines:

```python
def load(candidate_dir, device): ...
def next_token_logits(model, input_ids): ...  # [batch, vocabulary]
```

During VAL/TEST, a fresh internal Docker network connects two short-lived
containers:

```text
trusted Evaluator: hidden split, no Candidate files
untrusted Candidate: Candidate files, no hidden split, no public network
```

Only same-length, prefix-free input batches and one `[batch, vocabulary]` logit
matrix cross this boundary. Each response is scored before any longer prefix is
sent. Candidate failures expose only a fixed error code to the Agent; raw logs
remain private to the trusted episode directory. Both containers and the
network are removed after every query.
