# Mini real-data benchmark

This example is a low-cost rehearsal for an official AutoTrain-Bench release. It
uses real TinyStories text, hidden validation and test splits, a fixed tokenizer,
immutable candidate snapshots, both benchmark tracks, and the same autonomous
Agent/Supervisor/Training/Eval container boundary as a full run.

The data preparation follows the small-compute direction recommended by
[Karpathy's `autoresearch`](https://github.com/karpathy/autoresearch): use the
TinyStories distribution, shorten context, and reduce evaluation tokens. Instead
of downloading the complete 673 MB parquet or two roughly 92 MB ClimbMix shards,
this example fetches a deterministic 1,200-row
[TinyStories GPT-4 Clean](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean)
slice (about 0.98 MB) through the Hugging Face dataset server.

## Run it

From the repository root:

```bash
python3 examples/mini_lm/prepare_data.py
autotrain-bench prepare --task examples/mini_lm/task.json
autotrain-bench start --task examples/mini_lm/task.json --track A
```

Preparation uses only the Python standard library. It verifies the ordered source
sample against a frozen SHA-256 and writes generated data under
`examples/mini_lm/data/`, which is intentionally ignored by Git. Re-running the
script verifies and reuses those artifacts.

Track A is capped at 300 seconds, representing a scaled-down version of a long
wall-clock research run. The bundled baseline normally finishes earlier: it uses
three leases of at most 60 seconds, compares hidden VAL NLL, changes its third
experiment based on that feedback, selects the best immutable candidate, closes
the episode, and receives TEST NLL.

Track B can be exercised with the same task:

```bash
autotrain-bench start --task examples/mini_lm/task.json --track B
```

It allows three leases of at most 60 seconds each.

## Run a logged-in Codex CLI as the Agent

Codex CLI can run as a host-native Agent without an API key or inference
Gateway. First confirm that the CLI already has a ChatGPT login:

```bash
codex login status
```

Then choose an available Codex model and launch the five-minute Track A:

```bash
python3 -m autotrain_bench.cli start \
  --task examples/mini_lm/task.json \
  --track A \
  --agent-runner host_cli \
  --agent-command "python3 -m autotrain_bench.cli_adapters.codex --model YOUR_CODEX_MODEL --reasoning-effort high"
```

The Supervisor snapshots the same initial Agent workspace, starts Codex in that
copy, and exposes only the public task plus mailbox interface by contract. Codex
stdout is stored in `agent.stdout.log`, stderr in `agent.stderr.log`, and parsed
token/tool usage in the final report under `agent_usage`.

This mode is deliberately marked `policy_compliant: false`. Codex executes on
the host so v0 cannot technically prevent access outside the episode directory;
the adapter disables web search, MCP/user configuration, and persistent rollout
files, but those controls are not a replacement for a hard container or VM
boundary.

## Full-size to mini mapping

| Release concept | Mini setting |
| --- | --- |
| Real fixed corpus | 1,200 TinyStories documents, 1,000 for TRAIN |
| Fixed tokenizer | UTF-8 bytes plus BOS, vocabulary 257 |
| Evaluation record size | At most 128 target bytes per record |
| Hidden feedback | 8,192 VAL target tokens per candidate |
| Post-mortem result | 8,192 TEST target tokens after close |
| Track A | 300 seconds instead of a multi-hour run |
| Track B | Three 60-second leases |
| Training hardware | CPU-only, no accelerator setup |
| Images | One pinned multi-platform Python image digest |
| Baseline search | Sparse byte n-gram order and smoothing |
| Remote inference | Fully configured but disabled by default |

The sample comes from `karpathy/tinystories-gpt4-clean`, revision
`0397e27157956705a0260709da3095bb9c43d6a7`, licensed under CDLA-Sharing-1.0.
Its source content is not committed to this repository.

This is a realistic protocol and user-experience rehearsal, not a leaderboard
task. The dependency-free n-gram baseline and provisional Candidate ABI keep the
run cheap; an official release would use a fixed accelerator image, a neural
training stack, much larger hidden data, and a production serving boundary.

The current v0 task schema requires `evaluation.context_length`; here it describes
the largest prepared evaluation prefix. It is not passed to the Training
Container and does not constrain the sequence length, model memory, or context
strategy selected by the Agent.
