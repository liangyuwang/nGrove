def load(candidate_dir, device):
    del candidate_dir, device
    return None


def next_token_logits(model, input_ids):
    del model
    return [[0.0, 0.0, 0.0] for _ in input_ids]
