"""Invoke a submitted offline Candidate inside its isolated container."""

from __future__ import annotations

import argparse
from pathlib import Path

from .candidate_protocol import load_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()

    module = load_candidate(args.candidate)
    module.run_offline(
        args.candidate.resolve(),
        args.contexts.resolve(),
        args.output.resolve(),
        args.device,
    )


if __name__ == "__main__":
    main()
