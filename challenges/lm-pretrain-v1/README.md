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
hidden input identity and runtime profile. A conforming Worker provides a
separate prediction allocation of twice the maximum final artifact size, at
least 16 MiB and rounded up to MiB, with at most 128 filesystem inodes including
the root. Temporary entries are allowed during execution; the final artifact
still consists of exactly `logits.f32` and `predictions.json` with strict sizes
and format. The Docker Worker enforces this allocation using bounded tmpfs and
removes it after scoring or failure and confirmed teardown. See
[Worker Protocol v15](../../docs/worker-protocol.md) for the allocation formula,
memory provisioning and compatibility changes.

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
generated `benchcore.candidate_protocol` reader and writer helpers.
`prediction_manifest.logits_artifact` declares the binary file as headerless,
row-major, little-endian IEEE-754 float32. Row `i` corresponds to `examples[i]`,
column `j` to token ID `j`, and every value must be finite.

## Prepare data

Install Core and the isolated preparation dependencies:

```bash
python3 -m pip install -e .
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

Both downloaded and local sources must match all 14 Parquet identities in
[`data/source-files.json`](data/source-files.json), frozen from the pinned
revision's upstream LFS metadata. Verification checks file names, byte lengths,
and SHA-256 before tokenization. `--source-dir` accepts either the `sample/10BT/`
layout or a flat directory containing those files. Missing, extra, duplicated,
or changed Parquet files are rejected; materialize ordinary files instead of
passing a cache's symbolic or hard links. Arbitrary local corpora cannot be
labeled as this release's source.

The preview hashes each source row identity into 2,048 buckets, reserving bucket
0 for ANALYSIS and bucket 1 for TEST while their quotas remain unfilled. This
enlarges the candidate pool from the earlier 4,096-bucket recipe, which could
exhaust the source before filling the longest-context quotas. The non-overlapping
round-robin window algorithm and per-length quotas are unchanged. A build still
fails if any quota is short; the error reports the actual and required counts
for each affected split and context length.

The trusted root build manifest records `preview_hidden_evaluation.split_buckets`
alongside the actual histograms and hidden artifact checksums. Older manifests
without this field do not establish which split recipe was used. Rebuild an old
preview with `--force` when adopting this recipe, and compare results only when
their release and data identities match. Doubling the nominal holdout sampling
rate does not imply a fixed percentage change in TRAIN tokens: document lengths,
quota completion, and the TRAIN target also affect the final size.

The builder records the exact resulting token count, GPT-2 encoding
fingerprint, raw and tokenized shard checksums, document mapping, dependency
versions, balanced hidden context-length histograms, and preview hidden-dataset
checksums. It does not force the canonical stream to contain exactly
10,000,000,000 tokens.

The upstream `sample-10BT` name denotes [around 10B GPT-2 tokens](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu#smaller-sample-versions),
not a minimum TRAIN size. The default build processes the complete pinned sample
under the preview split recipe. Tokenization, document separators and the local
holdout affect the exact TRAIN count. A complete build below 10B is not a scale
failure, and the builder never pads or repeats documents to reach that number.

The generated `tokenized_train` metadata separates scale from completeness:

| Field | Meaning |
| --- | --- |
| `nominal_scale_tokens` | The approximate 10B source-sample scale, not an acceptance threshold |
| `exact_tokens` | Actual GPT-2 TRAIN tokens, including document separators |
| `requested_token_limit` | Explicit `--target-train-tokens` stopping threshold, or `null` for the default full-source build |
| `source_exhausted` | Whether the builder reached the end of the source iterator |
| `train_truncated_by_token_limit` | Whether any TRAIN rows were skipped because the explicit token threshold had been reached |

A full-source build has `source_exhausted: true` and
`train_truncated_by_token_limit: false`. Successful publication still requires
verified source identities, complete ANALYSIS/TEST quotas and the TRAIN integrity
inventory. Report the exact token count alongside that completion evidence;
reserve "below the required scale" for a release with an explicit exact or
minimum-size requirement.

For a smaller development build, pass `--target-train-tokens N`. The final TRAIN
document remains whole, so its tokens can take the count above N. The builder
may continue scanning to fill hidden quotas after it stops adding TRAIN rows;
reaching the end of that scan does not make truncated TRAIN a full-source build.
If the source ends below N, the actual count is retained without padding.

Historical manifests remain readable and must not be rewritten. Their
`target_scale_tokens` and `target_scale_reached` fields record the old numeric
threshold comparison, not source completeness. When the new fields are absent,
use the preserved build evidence to establish completeness; do not infer it from
token count alone. New builds have new manifest identities, even when their
corpus bytes are unchanged. This metadata change does not require rebuilding
existing verified data.

The root `public/manifest.json` also inventories every public TRAIN file,
including the document index and nested tokenized manifest. Paths in both
manifests are relative to the public TRAIN root; the nested manifest carries
task metadata, while the root carries the authoritative integrity inventory.
`prepare` and each episode startup read and verify the complete TRAIN view
before the Agent starts. Source verification reads roughly 28.5 GB once; TRAIN
verification adds a full read of the generated public data at each check.
See [TRAIN integrity](../../docs/training-data.md) for the format and Worker
immutability responsibilities. Older builds lacking the inventory must be
rebuilt with `--force`.

`--force` replaces only a recognized build of this exact task. The builder
inventories every existing entry before and immediately around an atomic
directory swap; if validation or installation fails, it restores the old
build or preserves it under a named backup path instead of deleting it first.
An existing build using the former progressive-prefix dataset must be rebuilt
with `--force`; its scores are not comparable with this single-target metric.

Challenge-local regression checks use the real hash split and default context
quotas on synthetic documents, without downloading or rebuilding the corpus:

```bash
python3 -m unittest discover -s challenges/lm-pretrain-v1 -p 'test_prepare_data.py' -v
```

These checks exercise the pool-sizing failure and shortage diagnostics; a full
pinned-source build is still required to verify real corpus capacity and scale.

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

Track B defaults to a **24-hour private operator research-time protection**,
configured in `operator-defaults.json`. The launcher announces it before run
preparation. Add `--operator-max-research-seconds SECONDS` to change it or
`--no-operator-research-limit` to disable it for one run. Disabling protection
risks extremely high token consumption. The setting and notices are never sent
to the Tested Agent and do not change the six one-hour Leases or comparison key.
See [operator protection semantics](../../docs/benchmark-reference.md#track-b--lease-bounded-autonomous-research).

Both Tracks also default to **100 GiB of private operator storage protection**.
The Worker deployment must provide an empty, bounded filesystem at the selected
episode parent directory; an ordinary directory will fail admission. Use
`--operator-max-storage-bytes BYTES` to override the limit or
`--no-operator-storage-limit` to explicitly accept unprotected storage. This
does not change Track budgets or the prediction-output protocol. Remote Workers
must enforce this in their execution environment. See
[storage protection requirements](../../docs/benchmark-reference.md#private-operator-storage-protection).

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
