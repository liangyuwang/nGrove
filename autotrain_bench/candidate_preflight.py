"""Public Candidate ABI check intended to run inside a Training Lease."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .candidate_protocol import load_candidate, normalized_logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument("--context-lengths", required=True, type=int, nargs="+")
    args = parser.parse_args()

    context_lengths = tuple(args.context_lengths)
    if args.vocab_size <= 0:
        parser.error("--vocab-size must be positive")
    if tuple(sorted(set(context_lengths))) != context_lengths or any(
        length <= 0 for length in context_lengths
    ):
        parser.error("--context-lengths must be positive and strictly increasing")
    module, model = load_candidate(args.candidate, args.device)
    bos = args.vocab_size - 1
    alternative = (bos - 1) % args.vocab_size
    predictions = 0
    for length in context_lengths:
        contexts = [
            [bos, *([0] * (length - 1))],
            [bos, *([alternative] * (length - 1))],
        ]
        predictions += len(
            normalized_logits(
                module,
                model,
                contexts,
                vocab_size=args.vocab_size,
            )
        )
    print(
        json.dumps(
            {
                "status": "valid",
                "protocol": "next_token_logits_v2",
                "predictions": predictions,
                "vocab_size": args.vocab_size,
                "context_lengths": list(context_lengths),
                "device": args.device,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
