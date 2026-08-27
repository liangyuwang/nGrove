"""Toy candidate adapter generated into each candidate bundle."""

import json
from pathlib import Path


def load(candidate_dir, device):
    del device
    path = Path(candidate_dir) / "model.json"
    return json.loads(path.read_text(encoding="utf-8"))


def next_token_logits(model, input_ids):
    transitions = model["transition_logits"]
    return [
        [[float(value) for value in transitions[token]] for token in sequence]
        for sequence in input_ids
    ]

