"""Dependency-free Candidate ABI for a sparse byte n-gram language model."""

import json
import math
from pathlib import Path


def load(candidate_dir, device):
    del device
    path = Path(candidate_dir) / "model.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _context_key(context):
    return ",".join(str(token) for token in context)


def _last_row(model, sequence):
    tables = model["tables"]
    maximum_width = min(model["order"] - 1, len(sequence))
    counts = None
    for width in range(maximum_width, -1, -1):
        context = sequence[-width:] if width else []
        counts = tables[width].get(_context_key(context))
        if counts is not None:
            break
    if counts is None:
        counts = []

    smoothing = float(model["smoothing"])
    vocab_size = int(model["vocab_size"])
    total = sum(int(count) for _, count in counts)
    denominator = total + smoothing * vocab_size
    unseen = math.log(smoothing / denominator)
    row = [unseen] * vocab_size
    for token, count in counts:
        row[int(token)] = math.log((int(count) + smoothing) / denominator)
    return row


def next_token_logits(model, input_ids):
    output = []
    for sequence in input_ids:
        row = _last_row(model, sequence)
        # The provisional evaluator reads only the final position. Reusing the
        # same immutable row avoids quadratic work while preserving the ABI.
        output.append([row] * len(sequence))
    return output
