"""Shared validation for the isolated Candidate serving protocol."""

from __future__ import annotations

import importlib.util
import math
import sys
import uuid
from numbers import Real
from pathlib import Path
from typing import Any, List, Sequence, Tuple


class CandidateProtocolError(RuntimeError):
    """The Candidate bundle does not implement the public serving contract."""


def load_candidate(candidate_dir: Path, device: str) -> Tuple[Any, Any]:
    """Import and initialize a Candidate inside an untrusted environment."""

    root = candidate_dir.resolve()
    module_path = root / "candidate.py"
    if not module_path.is_file():
        raise CandidateProtocolError("candidate.py is missing")
    sys.path.insert(0, str(root))
    module_name = f"autotrain_candidate_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise CandidateProtocolError("candidate.py could not be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "load", None)):
        raise CandidateProtocolError(
            "candidate.py must define load(candidate_dir, device)"
        )
    if not callable(getattr(module, "next_token_logits", None)):
        raise CandidateProtocolError(
            "candidate.py must define next_token_logits(model, input_ids)"
        )
    return module, module.load(root, device)


def validate_contexts(
    value: Any,
    *,
    vocab_size: int,
    context_length: int,
    maximum_batch_size: int,
) -> List[List[int]]:
    if not isinstance(value, list) or not value:
        raise CandidateProtocolError("input_ids must be a non-empty batch")
    if len(value) > maximum_batch_size:
        raise CandidateProtocolError("input_ids exceeds the maximum batch size")
    result: List[List[int]] = []
    for context in value:
        if not isinstance(context, list) or not context:
            raise CandidateProtocolError("every input context must be non-empty")
        if len(context) > context_length:
            raise CandidateProtocolError("input context exceeds the context length")
        if not all(
            isinstance(token, int)
            and not isinstance(token, bool)
            and 0 <= token < vocab_size
            for token in context
        ):
            raise CandidateProtocolError("input token is outside the vocabulary")
        result.append(context)
    return result


def normalized_logits(
    module: Any,
    model: Any,
    input_ids: Sequence[Sequence[int]],
    *,
    vocab_size: int,
) -> List[List[float]]:
    """Run the Candidate and normalize its ``[batch, vocabulary]`` response."""

    value = module.next_token_logits(model, input_ids)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != len(input_ids):
        raise CandidateProtocolError(
            "next_token_logits must return [batch, vocabulary]"
        )
    result: List[List[float]] = []
    for row in value:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, (list, tuple)) or len(row) != vocab_size:
            raise CandidateProtocolError(
                "candidate returned the wrong vocabulary dimension"
            )
        if any(isinstance(item, bool) or not isinstance(item, Real) for item in row):
            raise CandidateProtocolError("candidate logits must be numeric")
        try:
            normalized = [float(item) for item in row]
        except (OverflowError, TypeError, ValueError) as exc:
            raise CandidateProtocolError("candidate logits must be numeric") from exc
        if not all(math.isfinite(item) for item in normalized):
            raise CandidateProtocolError("candidate logits must be finite")
        result.append(normalized)
    return result
