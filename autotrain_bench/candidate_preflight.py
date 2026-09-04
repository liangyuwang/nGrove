"""Validate the public offline Candidate ABI before committing an artifact."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .candidate_protocol import (
    CANDIDATE_PROTOCOL,
    DATASET_SCHEMA_VERSION,
    CandidateProtocolError,
    load_candidate,
    validate_prediction_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    try:
        module = load_candidate(args.candidate)
        boundary = args.vocab_size - 1
        alternative = (boundary - 1) % args.vocab_size
        examples = []
        for length in context_lengths:
            for fill in (0, alternative):
                examples.append(
                    {
                        "example_id": f"preflight-{length}-{len(examples)}",
                        "input_ids": [boundary, *([fill] * (length - 1))],
                    }
                )
        dataset_id = "autotrain-bench-candidate-preflight"
        request = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "candidate_protocol": CANDIDATE_PROTOCOL,
            "dataset_id": dataset_id,
            "vocab_size": args.vocab_size,
            "context_lengths": list(context_lengths),
            "examples": examples,
        }
        with tempfile.TemporaryDirectory(
            prefix="autotrain-candidate-preflight-"
        ) as root:
            temporary = Path(root)
            contexts_path = temporary / "contexts.json"
            output_dir = temporary / "predictions"
            output_dir.mkdir()
            contexts_path.write_text(
                json.dumps(request, separators=(",", ":"), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            module.run_offline(
                args.candidate.resolve(),
                contexts_path,
                output_dir,
                args.device,
            )
            prediction_bytes = validate_prediction_artifact(
                output_dir,
                dataset_id=dataset_id,
                examples=len(examples),
                vocab_size=args.vocab_size,
            )
    except CandidateProtocolError as exc:
        parser.exit(status=1, message=f"candidate preflight failed: {exc}\n")
    print(
        json.dumps(
            {
                "status": "valid",
                "protocol": CANDIDATE_PROTOCOL,
                "examples": len(examples),
                "prediction_bytes": prediction_bytes,
                "vocab_size": args.vocab_size,
                "context_lengths": list(context_lengths),
                "device": args.device,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
