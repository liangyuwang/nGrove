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
    parser.add_argument("--context-length", required=True, type=int)
    args = parser.parse_args()

    module, model = load_candidate(args.candidate, args.device)
    bos = args.vocab_size - 1
    prefixes = [
        [bos],
        [bos, 0],
        [bos, *([0] * (args.context_length - 1))],
    ]
    logits = normalized_logits(
        module,
        model,
        prefixes,
        vocab_size=args.vocab_size,
    )
    print(
        json.dumps(
            {
                "status": "valid",
                "protocol": "next_token_logits_v1",
                "batch_size": len(logits),
                "vocab_size": args.vocab_size,
                "context_length": args.context_length,
                "device": args.device,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
