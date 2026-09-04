"""Exercise the public Candidate ABI on synthetic inputs inside a Training Lease.

This is a participant-side self-check, not an isolated evaluation or a promise
that hidden TEST will succeed. Its computation and artifact copy consume Lease
resources. No hidden input or target is used.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Sequence

from .candidate_protocol import (
    CANDIDATE_PROTOCOL,
    DATASET_SCHEMA_VERSION,
    CandidateProtocolError,
    validate_prediction_artifact,
)


def synthetic_request(
    *, vocab_size: int, context_lengths: Sequence[int], examples: int, seed: int
) -> Dict[str, Any]:
    """Mix ordered, repeated, boundary, and random tokens at every length."""

    random_source = random.Random(seed)
    lengths = [
        context_lengths[index % len(context_lengths)] for index in range(examples)
    ]
    random_source.shuffle(lengths)
    rows = []
    for index, length in enumerate(lengths):
        pattern = index % 4
        tokens = [vocab_size - 1]
        for position in range(length - 1):
            if pattern == 0:
                token = position % vocab_size
            elif pattern == 1:
                token = vocab_size - 1
            elif pattern == 2:
                token = 0
            else:
                token = random_source.randrange(vocab_size)
            tokens.append(token)
        rows.append({"example_id": f"preflight-{index:08d}", "input_ids": tokens})
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "candidate_protocol": CANDIDATE_PROTOCOL,
        "dataset_id": "autotrain-bench-candidate-preflight",
        "vocab_size": vocab_size,
        "context_lengths": list(context_lengths),
        "examples": rows,
    }


def _copy_read_only_candidate(source: Path, destination: Path) -> None:
    """Use a separate copy; never chmod the participant's original artifacts."""

    if source.is_symlink() or not source.is_dir():
        raise CandidateProtocolError("candidate must be a directory, not a symlink")
    if destination.resolve().is_relative_to(source.resolve()):
        raise CandidateProtocolError(
            "preflight scratch storage must be outside the Candidate"
        )
    for path in source.rglob("*"):
        metadata = path.lstat()
        if not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
            raise CandidateProtocolError(
                "candidate may contain only regular files and directories"
            )
        if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1:
            raise CandidateProtocolError("candidate files may not be hard-linked")
    shutil.copytree(source, destination)
    for path in destination.rglob("*"):
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    destination.chmod(stat.S_IMODE(destination.stat().st_mode) & ~0o222)


def _remove_temporary_tree(root: Path) -> None:
    # Restore only the disposable copy so read-only directories also clean up
    # on platforms whose rmtree requires writable files or directories.
    for directory, directories, files in os.walk(root, followlinks=False):
        Path(directory).chmod(0o700)
        for name in directories + files:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(0o700 if path.is_dir() else 0o600)
    shutil.rmtree(root)


def check_candidate(args: argparse.Namespace, request: Dict[str, Any]) -> int:
    # Candidate bundles can exceed the Training sandbox's small /tmp tmpfs.
    # Local Workers expose lease-owned scratch; other deployments can select it
    # explicitly without granting preflight any additional resources.
    scratch = args.scratch_dir or os.environ.get("AUTOTRAIN_SCRATCH_DIR")
    root = Path(
        tempfile.mkdtemp(prefix="autotrain-candidate-preflight-", dir=scratch)
    ).resolve()
    try:
        candidate = root / "candidate"
        _copy_read_only_candidate(args.candidate, candidate)
        contexts = root / "contexts.json"
        output = root / "predictions"
        output.mkdir()
        home = root / "candidate-home"
        home.mkdir()
        contexts.write_text(json.dumps(request) + "\n", encoding="utf-8")
        contexts.chmod(0o444)
        environment = dict(os.environ)
        environment.update(
            {
                "HOME": str(home),
                "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        # Use the same offline entrypoint and argv shape as Candidate execution,
        # in a fresh process with its artifact directory as cwd. The enclosing
        # Training sandbox and Python/image remain those of the active Lease.
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "autotrain_bench.candidate_offline_runner",
                "--candidate",
                str(candidate),
                "--contexts",
                str(contexts),
                "--output",
                str(output),
                "--device",
                args.device,
            ],
            cwd=candidate,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=sys.stderr,
            check=False,
        )
        if completed.returncode != 0:
            raise CandidateProtocolError(
                f"offline Candidate process exited with status {completed.returncode}; "
                "see its public-input traceback above"
            )
        return validate_prediction_artifact(
            output,
            dataset_id=request["dataset_id"],
            examples=len(request["examples"]),
            vocab_size=args.vocab_size,
        )
    finally:
        _remove_temporary_tree(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument("--context-lengths", required=True, type=int, nargs="+")
    parser.add_argument(
        "--examples",
        type=int,
        default=512,
        help="public synthetic workload size (default: 512)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="reproducible synthetic input seed (default: 0)",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        help="temporary copy/output storage; defaults to AUTOTRAIN_SCRATCH_DIR, then system temp",
    )
    args = parser.parse_args()

    context_lengths = tuple(args.context_lengths)
    if args.vocab_size <= 0:
        parser.error("--vocab-size must be positive")
    if tuple(sorted(set(context_lengths))) != context_lengths or any(
        length <= 0 for length in context_lengths
    ):
        parser.error("--context-lengths must be positive and strictly increasing")
    if not context_lengths or args.examples < len(context_lengths):
        parser.error("--examples must cover every declared context length")
    try:
        request = synthetic_request(
            vocab_size=args.vocab_size,
            context_lengths=context_lengths,
            examples=args.examples,
            seed=args.seed,
        )
        prediction_bytes = check_candidate(args, request)
    except (CandidateProtocolError, OSError) as exc:
        parser.exit(status=1, message=f"candidate preflight failed: {exc}\n")
    print(
        json.dumps(
            {
                "status": "public_abi_passed",
                "scope": "synthetic_inputs_in_training_environment",
                "checks": [
                    "offline_entrypoint_in_fresh_process",
                    "artifact_working_directory",
                    "read_only_copy_mode_bits",
                    "mixed_context_lengths_and_tokens",
                    "prediction_artifact",
                ],
                "not_checked": [
                    "candidate_image",
                    "candidate_sandbox_mounts",
                    "candidate_resource_limits",
                    "hidden_input_behavior",
                ],
                "hidden_evaluation_guaranteed": False,
                "protocol": CANDIDATE_PROTOCOL,
                "examples": args.examples,
                "seed": args.seed,
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
