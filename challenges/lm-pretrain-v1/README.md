# LM pretrain v1 preview

`lm-pretrain-v1` is the first formal language-model pretraining challenge preview. Its identity
is independent of its initial corpus: this version builds an approximately
10-billion-token TRAIN pool from a pinned FineWeb-Edu revision and fixes GPT-2
token IDs only at the evaluation boundary.

The package now contains `task.json`, a trusted evaluator, and a reproducible
data builder. It remains `leaderboard_eligible: false`: the current budgets,
task images, local preview ANALYSIS/TEST size, and runtime profiles are provisional,
and a verified release will replace the reproducible local holdouts with
organizer-owned hidden datasets.

## Public TRAIN layout

Data preparation deliberately retains both representations:

```text
data/generated/
├── public/                    mounted read-only at /benchmark/train
│   ├── raw/                   TRAIN text + original source metadata, Parquet
│   ├── tokenized/             official GPT-2 little-endian uint16 shards
│   │   ├── documents.jsonl    raw/tokenized document mapping and token spans
│   │   └── manifest.json
│   └── manifest.json
├── private/                   mounted only in trusted Evaluator containers
│   ├── analysis/
│   │   ├── contexts.json      Candidate-visible only during ANALYSIS serving
│   │   └── targets.json       trusted scorer only
│   └── test/
│       ├── contexts.json      Candidate-visible only during TEST serving
│       └── targets.json       trusted scorer only
└── manifest.json
```

The Agent may use the ready-made token shards or ignore them and work from raw
TRAIN documents. It can filter, deduplicate, rescore, reorder, retokenize, or
construct a curriculum. Raw public Parquet contains only TRAIN documents; rows
selected for preview ANALYSIS/TEST are not copied into the Agent-visible directory.
Evaluation always sends and scores GPT-2 vocabulary IDs.

The local preview builder pre-stages ANALYSIS/TEST under the trusted task root,
but neither path is mounted into the Agent or Research Worker. After research
sealing, the Evaluation Worker may materialize the frozen pair. It validates
both files first, mounts only `contexts.json` read-only into the Candidate, and
keeps `targets.json` inside trusted scoring. A formal Worker may instead fetch,
mount, or decrypt the same release-frozen identity at that point; Core does not
prescribe the storage mechanism.

Evaluation inputs are variable-length token sequences. The exact length
distribution is not disclosed in the research-visible manifest. Participants
choose how to use context to optimize next-token prediction within the resource
budget. `evaluation.max_context_tokens` is 2,048 for this preview; it is an
input bound, not a guarantee that any evaluation example reaches it. Sampling
lengths and proportions remain in trusted evaluation configuration. The local
builder is publicly reproducible, so its recipe is not secret; a formal release
must use organizer-owned evaluation construction parameters and data.

The Evaluator uses non-overlapping contexts. Each contributes one hidden next-token
target. The Candidate receives the complete unlabeled context artifact, owns
its serving schedule and batching, and writes one ordered logit vector per
example. It never receives the target later through a longer prefix or another
overlapping source window. The
Evaluator preserves mean target NLL as the canonical raw metric. ANALYSIS
produces a postmortem trajectory over frozen Candidates and is never Agent
feedback. The performance axis is a `[0, 100]` TEST performance score, where
higher is better. This release defines it as
`100 * max(0, min(1, 1 - mean_token_nll / log(vocab_size)))`: 100 is perfect
prediction, while uniform-over-vocabulary prediction and worse map to 0. The
canonical mean NLL raw metric retains distinctions among results at that floor.
The frozen trusted Evaluator owns this formula; Core validates only the generic
score contract and result shape. Perplexity may be derived for display but is
not a required finite result field. The score definition and raw metric names
are frozen by the release and are not comparable with another benchmark
release. A run without a successful finite TEST measurement has no performance
score or raw metrics and is reported as a failed experiment.

The independent cost axis is Candidate `time_per_task`, defined as end-to-end
Candidate wall time divided by the frozen number of next-token prediction
tasks. Lower is better. The result retains total Candidate wall time, task
count, input tokens, prediction bytes, and derived throughput as cost
submetrics. The clock includes Candidate sandbox startup, model loading, file
I/O, scheduling, and inference, but excludes hidden-data materialization and
trusted validation/scoring. Results form a Performance–Cost Pareto frontier;
the axes are not collapsed into a scalar. Cost comparisons require the same
hidden input identity and runtime profile. A conforming Worker hard-limits
prediction storage to two entries and
`examples * vocab_size * 4 + 65536` bytes, then removes it after scoring or
failure and confirmed teardown. The bundled local Docker preview reports its
host bind mount quota as unsupported.

The Agent-visible task manifest declares the invocation contract at
`evaluation.candidate_boundary.entrypoint`: the Worker imports `candidate.py`
from the artifact root and synchronously calls
`run_offline(candidate_dir, contexts_path, output_dir, device)` with positional
arguments. The first three are absolute `pathlib.Path` objects for the immutable
Candidate directory, read-only target-free context JSON, and writable prediction
storage. The fourth is the `evaluation.device` string (`cpu` or `cuda`).
`sys.argv` is not a Candidate interface; the module is imported with
`__name__ != '__main__'`. The return value is ignored; predictions are written
to the output directory before the function returns.

The input object is specified at
`evaluation.candidate_boundary.context_manifest`: its exact fields are
`schema_version`, `candidate_protocol`, `dataset_id`, `vocab_size`,
`context_lengths`, and `examples`. Every example has exactly `example_id`
(a unique opaque string) and `input_ids` (integer tokens in oldest-to-newest
order). Tokens are in `[0, vocab_size)` and each array length belongs to
`context_lengths` and is at most `evaluation.max_context_tokens`.
`contexts.json.context_lengths` describes the actual request and becomes visible
to the frozen Candidate only after research sealing; it is not published as a
research-stage sampling plan. The declaration specifies every field's type,
constants, and constraints. No targets or split labels are included. The optional
`load_context_request()` helper returns this parsed dictionary unchanged.

The same manifest defines the exact output object at
`evaluation.candidate_boundary.prediction_manifest`: `predictions.json` must
contain only `schema_version`, `candidate_protocol`, `dataset_id`, `examples`,
`vocab_size`, `dtype`, and `logits_file`, with constants and request-derived
values specified there. JSON member order and whitespace are irrelevant;
additional fields are rejected. The same section identifies the optional
generated `autotrain_bench.candidate_protocol` reader and writer helpers.
`prediction_manifest.logits_artifact` declares the binary file as headerless,
row-major, little-endian IEEE-754 float32. Row `i` corresponds to `examples[i]`,
column `j` to token ID `j`, and every value must be finite.

## Prepare data

Install the isolated preparation dependencies:

```bash
python3 -m pip install -r challenges/lm-pretrain-v1/data/requirements.txt
```

The default source download is roughly 28.5 GB and the tokenized output is
about 20 GB, in addition to the public raw TRAIN Parquet:

```bash
python3 challenges/lm-pretrain-v1/prepare_data.py --accept-large-download
autotrain-bench prepare --task challenges/lm-pretrain-v1/task.json
```

An existing pinned FineWeb-Edu `sample/10BT` download can be reused:

```bash
python3 challenges/lm-pretrain-v1/prepare_data.py \
  --source-dir /path/to/fineweb-edu-sample-10BT
```

The builder records the exact resulting token count, GPT-2 encoding
fingerprint, raw and tokenized shard checksums, document mapping, dependency
versions, balanced hidden context-length histograms, and preview hidden-dataset
checksums. It does not force the canonical stream to contain exactly
10,000,000,000 tokens.

`--force` replaces only a recognized build of this exact task. The builder
inventories every existing entry before and immediately around an atomic
directory swap; if validation or installation fails, it restores the old
build or preserves it under a named backup path instead of deleting it first.
An existing build using the former progressive-prefix dataset must be rebuilt
with `--force`; its scores are not comparable with this single-target metric.

## Supported preview lanes

The initial map is intentionally small:

- Track A: `preview-a-8h`;
- Track B: `preview-b-6x1h`;
- runtime: single-host 4×/8× NVIDIA A100 SXM4 80GB or H100 SXM 80GB.

CPU, B300, TPU, and other platforms may run the same task when runtime support
exists, but each belongs to a separate runtime profile. Results are directly
comparable only when this complete key matches:

```text
challenge_id + track + track_config + runtime_profile
```

Inspect the finite maps:

```bash
autotrain-bench challenge-info \
  --challenge challenges/lm-pretrain-v1/challenge.json
```

After data preparation, start a profile-bound episode. `task.json` is inferred
from the challenge release:

```bash
autotrain-bench start \
  --challenge challenges/lm-pretrain-v1/challenge.json \
  --track-config preview-b-6x1h \
  --runtime-profile h100-sxm-80gb-x4 \
  --episodes-root /path/to/autotrain-bench-runs/episodes \
  --agent-runner container \
  --agent-image participant-agent@sha256:DIGEST \
  --agent-command "participant-agent run"
```

Challenge mode rejects `--track`, `--training-gpus`, and
`--training-gpu-model`; those values come from the selected profiles so the
recorded comparison key cannot disagree with the actual episode.
