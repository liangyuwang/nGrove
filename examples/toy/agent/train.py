"""Build a tiny bigram candidate from the visible TRAIN split."""

import argparse
import json
import math
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoothing", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.smoothing <= 0:
        raise ValueError("smoothing must be positive")

    train_path = Path("/benchmark/train/train.json")
    sequences = json.loads(train_path.read_text(encoding="utf-8"))["sequences"]
    vocab_size = 3
    counts = [[0 for _ in range(vocab_size)] for _ in range(vocab_size)]
    for sequence in sequences:
        for previous, target in zip(sequence, sequence[1:]):
            counts[previous][target] += 1

    transition_logits = []
    for row in counts:
        denominator = sum(row) + args.smoothing * vocab_size
        transition_logits.append(
            [math.log((count + args.smoothing) / denominator) for count in row]
        )

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    (args.output / "model.json").write_text(
        json.dumps({"transition_logits": transition_logits}, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(
        Path("/workspace/candidate_template.py"), args.output / "candidate.py"
    )


if __name__ == "__main__":
    main()

