# LM pretrain v1 preview

`lm-pretrain-v1` is the first formal language-model pretraining challenge preview. Its identity
is independent of its initial corpus: this version builds an approximately
10-billion-token TRAIN pool from a pinned FineWeb-Edu revision and fixes GPT-2
token IDs only at the evaluation boundary.

The package now contains `task.json`, a trusted evaluator, and a reproducible
data builder. It remains `leaderboard_eligible: false`: the current budgets,
task images, local preview VAL/TEST size, and runtime profiles are provisional,
and a verified release will replace the reproducible local holdouts with
organizer-owned hidden splits.

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
│   ├── val.json
│   └── test.json
└── manifest.json
```

The Agent may use the ready-made token shards or ignore them and work from raw
TRAIN documents. It can filter, deduplicate, rescore, reorder, retokenize, or
construct a curriculum. Raw public Parquet contains only TRAIN documents; rows
selected for preview VAL/TEST are not copied into the Agent-visible directory.
Evaluation always sends and scores GPT-2 vocabulary IDs.

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
versions, and preview split checksums. It does not force the canonical stream
to contain exactly 10,000,000,000 tokens.

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
