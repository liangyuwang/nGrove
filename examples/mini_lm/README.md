# Comparable TinyStories mini benchmark

`mini_lm` is the repository's only bundled benchmark task. It keeps the shape of
a release benchmark and reduces only its scale: a fixed real corpus, fixed
tokenizer and images, exact hardware, hidden VAL/TEST, immutable Candidate
artifacts, explicit incumbent decisions, and the same Track A/Track B protocol.

The default task is policy-compliant. Its Agent and Training environments have
no public network, and hidden evaluation uses two separate containers:

```text
ephemeral internal Docker network
├── trusted Evaluator: hidden split, no Candidate files
└── untrusted Candidate: Candidate files, no hidden split, no public network
```

The Evaluator sends only input prefixes to the Candidate's
`next_token_logits_v1` endpoint and receives one `[batch, vocabulary]` logit
matrix. The network and both evaluation containers are destroyed after each VAL
or TEST query.

## Fixed mini release

| Release property | Fixed mini value |
| --- | --- |
| Corpus | 1,200 pinned TinyStories documents |
| TRAIN | 1,000 documents, about 818K target bytes |
| VAL / TEST | 8,192 hidden target bytes each |
| Tokenizer | UTF-8 bytes plus BOS, vocabulary 257 |
| Context length | 128, defined by `task.json` |
| Accelerator | Exactly one NVIDIA H200 |
| Training / Candidate image | Pinned PyTorch 2.7.1 + CUDA 12.8 digest |
| Track A | 300 seconds total wall clock |
| Track B | Three leases, at most 75 seconds each |
| Metric | Autoregressive NLL |

Different GPU models are intentionally rejected. A portable smoke test and a
comparable benchmark are different products: the latter needs identical
hardware and immutable images.

## Prepare and run

The host must already expose exactly one H200 through Docker and the NVIDIA
Container Toolkit. AutoTrain-Bench does not allocate Slurm, Kubernetes, cloud,
or other infrastructure.

```bash
python3 examples/mini_lm/prepare_data.py
autotrain-bench prepare --task examples/mini_lm/task.json
autotrain-bench start --task examples/mini_lm/task.json --track A
autotrain-bench start --task examples/mini_lm/task.json --track B
```

Preparation downloads a deterministic slice of
`karpathy/tinystories-gpt4-clean` at revision
`0397e27157956705a0260709da3095bb9c43d6a7`, verifies the ordered source digest,
and writes ignored generated artifacts under `examples/mini_lm/data/`.

The bundled baseline trains three causal Transformers, uses a TRAIN-only local
holdout for checkpointing, runs a public Candidate ABI preflight, commits at
most one Candidate per lease, interprets aggregate VAL itself, explicitly
declares its incumbent, and receives TEST only after closing the episode.

## Candidate artifact contract

Training commands must write artifacts under the lease-owned `/output` mount,
not the shared `/workspace`. Before commit, a participant can check the public
ABI without consuming a VAL query:

```bash
python -m autotrain_bench.cli agent lease-exec -- \
  python -m autotrain_bench.candidate_preflight \
  --candidate /output/candidate \
  --device cuda --vocab-size 257 --context-length 128

python -m autotrain_bench.cli agent lease-commit /output/candidate
```

Commit first stops the Training Container, then validates and snapshots the
now-immutable output. Candidate bundles may contain only regular files and
directories, may not contain symbolic or hard links, must fit the task-defined
artifact limits, and are content-addressed.

`candidate.py` must define:

```python
def load(candidate_dir, device): ...
def next_token_logits(model, input_ids): ...  # [batch, vocabulary]
```

## Agent execution and compliance

The bundled sandboxed container Agent produces `policy_compliant: true`. A
participant may substitute another registered Agent image and command while
keeping the container runner and sandboxed mode.

A logged-in CLI tool such as Codex can still be used for local rehearsal:

```bash
python3 -m autotrain_bench.cli start \
  --task examples/mini_lm/task.json \
  --track A \
  --agent-runner host_cli \
  --agent-command "python3 -m autotrain_bench.cli_adapters.codex --model MODEL --reasoning-effort high"
```

That override reports `policy_compliant: false` with only the
`agent_execution_boundary` violation. Candidate/Evaluator isolation remains
compliant, but v0 cannot prove that a host process avoided unrelated files and
its own network. A comparable official Agent run therefore needs a sandboxed
Agent container (optionally with the registered inference Gateway), or a future
equivalent VM/remote-tool boundary.
