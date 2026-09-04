"""Trusted Docker adapter for offline Candidate input validation and scoring."""

from __future__ import annotations

import argparse
import importlib.util
import json
import uuid
from pathlib import Path
from typing import Any, Callable


class AdapterError(RuntimeError):
    pass


def _load_callable(path: Path, name: str) -> Callable[..., Any]:
    module_name = f"autotrain_evaluator_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AdapterError("Evaluator module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = getattr(module, name, None)
    if not callable(value):
        raise AdapterError(f"Evaluator module must define {name}(...)")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate", "score"), required=True)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument("--context-lengths", required=True, type=int, nargs="+")
    args = parser.parse_args()

    if args.mode == "validate":
        if args.predictions is not None:
            parser.error("--predictions is not valid in validation mode")
        result = _load_callable(args.evaluator, "validate_inputs")(
            args.contexts,
            args.targets,
            args.vocab_size,
            context_lengths=args.context_lengths,
        )
    else:
        if args.predictions is None:
            parser.error("--predictions is required in score mode")
        result = _load_callable(args.evaluator, "evaluate")(
            args.contexts,
            args.targets,
            args.predictions,
            args.vocab_size,
            context_lengths=args.context_lengths,
        )
    if not isinstance(result, dict):
        raise AdapterError("Evaluator result must be an object")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
