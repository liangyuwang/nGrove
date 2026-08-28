"""Candidate ABI for the starter byte-level Transformer."""

import json
from pathlib import Path

import torch

from model import ByteTransformer


def load(candidate_dir, device):
    root = Path(candidate_dir)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    model = ByteTransformer(config)
    state = torch.load(root / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return {"model": model, "device": torch.device(device), "config": config}


@torch.inference_mode()
def next_token_logits(bundle, input_ids):
    model = bundle["model"]
    device = bundle["device"]
    maximum = max(len(sequence) for sequence in input_ids)
    if maximum > int(bundle["config"]["context_length"]):
        raise ValueError("input exceeds configured context length")
    padded = torch.zeros((len(input_ids), maximum), dtype=torch.long, device=device)
    for index, sequence in enumerate(input_ids):
        padded[index, : len(sequence)] = torch.tensor(sequence, device=device)
    logits = model(padded).float().cpu()
    return [
        logits[index, : len(sequence)].tolist()
        for index, sequence in enumerate(input_ids)
    ]
