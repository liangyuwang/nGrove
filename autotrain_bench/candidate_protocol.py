"""Helpers for the target-free offline Candidate file protocol."""

from __future__ import annotations

import importlib.util
import json
import math
import stat
import sys
import uuid
from array import array
from numbers import Real
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


DATASET_SCHEMA_VERSION = 2
PREDICTION_SCHEMA_VERSION = 1
CANDIDATE_PROTOCOL = "offline_next_token_logits_v3"
PREDICTIONS_FILENAME = "predictions.json"
LOGITS_FILENAME = "logits.f32"
PREDICTION_ARTIFACT_MAX_ENTRIES = 2
PREDICTION_MANIFEST_MAX_BYTES = 64 * 1024
PREDICTION_ARTIFACT_RETENTION = "ephemeral_until_evaluation_teardown"
_PREDICTION_MANIFEST_FIELD_SPECS = (
    ("schema_version", "integer", "constant", PREDICTION_SCHEMA_VERSION),
    ("candidate_protocol", "string", "constant", CANDIDATE_PROTOCOL),
    ("dataset_id", "string", "source", "contexts.json.dataset_id"),
    ("examples", "integer", "source", "len(contexts.json.examples)"),
    ("vocab_size", "integer", "source", "contexts.json.vocab_size"),
    ("dtype", "string", "constant", "float32-le"),
    ("logits_file", "string", "constant", LOGITS_FILENAME),
)
PREDICTION_MANIFEST_FIELDS = tuple(
    field for field, _json_type, _value_kind, _value in _PREDICTION_MANIFEST_FIELD_SPECS
)


def prediction_manifest_contract() -> Dict[str, Any]:
    """Return the exact Agent-facing ``predictions.json`` contract."""

    return {
        "filename": PREDICTIONS_FILENAME,
        "format": "json_object",
        "exact_fields": list(PREDICTION_MANIFEST_FIELDS),
        "additional_fields_allowed": False,
        "json_member_order_significant": False,
        "fields": {
            field: {"json_type": json_type, value_kind: value}
            for field, json_type, value_kind, value in (
                _PREDICTION_MANIFEST_FIELD_SPECS
            )
        },
    }


def candidate_helper_api_manifest() -> Dict[str, Any]:
    """Describe the optional generated helper API available to Candidates."""

    return {
        "module": "autotrain_bench.candidate_protocol",
        "request_reader": "load_context_request(path)",
        "prediction_writer": (
            "write_prediction_artifact(output_dir, *, dataset_id, logits, "
            "examples, vocab_size)"
        ),
        "use": "optional",
    }


def prediction_storage_limit(*, examples: int, vocab_size: int) -> Dict[str, int]:
    """Return the release-owned hard limits for one prediction workspace."""

    for value, field in ((examples, "examples"), (vocab_size, "vocab_size")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"prediction storage {field} must be a positive integer")
    logits_bytes = examples * vocab_size * 4
    return {
        "max_bytes": logits_bytes + PREDICTION_MANIFEST_MAX_BYTES,
        "max_entries": PREDICTION_ARTIFACT_MAX_ENTRIES,
        "logits_bytes": logits_bytes,
        "manifest_max_bytes": PREDICTION_MANIFEST_MAX_BYTES,
    }


def prediction_storage_manifest() -> Dict[str, Any]:
    """Return the machine-readable release rule used to derive hard limits."""

    return {
        "hard_limit": "required_before_candidate_start",
        "max_entries": PREDICTION_ARTIFACT_MAX_ENTRIES,
        "logit_bytes": 4,
        "logit_values": "examples*vocab_size",
        "manifest_max_bytes": PREDICTION_MANIFEST_MAX_BYTES,
        "max_bytes": "examples*vocab_size*4+manifest_max_bytes",
    }


def serving_timing_manifest() -> Dict[str, Any]:
    """Return the release-owned Candidate serving wall-clock boundary."""

    return {
        "clock": "trusted_monotonic_wall",
        "start": "immediately_before_candidate_process_invocation",
        "stop": "after_candidate_process_exit_and_prediction_flush",
        "includes": [
            "sandbox_startup",
            "candidate_import",
            "model_loading",
            "context_file_read",
            "participant_scheduling",
            "inference",
            "prediction_file_write",
        ],
        "excludes": [
            "hidden_data_materialization",
            "trusted_input_validation",
            "trusted_output_validation",
            "target_join",
            "metric_computation",
            "sandbox_teardown",
        ],
    }


class CandidateProtocolError(RuntimeError):
    """The Candidate bundle does not implement the public serving contract."""


def _expected_prediction_manifest(
    *, dataset_id: str, examples: int, vocab_size: int
) -> Dict[str, Any]:
    source_values = {
        "contexts.json.dataset_id": dataset_id,
        "len(contexts.json.examples)": examples,
        "contexts.json.vocab_size": vocab_size,
    }
    return {
        field: value if value_kind == "constant" else source_values[value]
        for field, _json_type, value_kind, value in (_PREDICTION_MANIFEST_FIELD_SPECS)
    }


def _validate_prediction_manifest(
    manifest: Any,
    *,
    dataset_id: str,
    examples: int,
    vocab_size: int,
) -> None:
    """Validate the public manifest with bounded, field-specific diagnostics."""

    if not isinstance(manifest, dict):
        raise CandidateProtocolError("prediction manifest must be a JSON object")

    expected = _expected_prediction_manifest(
        dataset_id=dataset_id,
        examples=examples,
        vocab_size=vocab_size,
    )
    expected_fields = set(PREDICTION_MANIFEST_FIELDS)
    actual_fields = set(manifest)
    missing = sorted(expected_fields - actual_fields)
    unexpected = sorted(actual_fields - expected_fields)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing fields {json.dumps(missing)}")
        if unexpected:
            details.append(f"unexpected fields {json.dumps(unexpected)}")
        raise CandidateProtocolError(
            "prediction manifest fields do not match: " + "; ".join(details)
        )

    field_contract = prediction_manifest_contract()["fields"]
    for field in PREDICTION_MANIFEST_FIELDS:
        value = manifest[field]
        expected_value = expected[field]
        required_type = int if field_contract[field]["json_type"] == "integer" else str
        if type(value) is required_type and value == expected_value:
            continue
        if "constant" in field_contract[field]:
            requirement = (
                f"the {field_contract[field]['json_type']} constant "
                f"{json.dumps(field_contract[field]['constant'])}"
            )
        else:
            requirement = (
                f"a {field_contract[field]['json_type']} equal to "
                f"{field_contract[field]['source']}"
            )
        raise CandidateProtocolError(
            f"prediction manifest field {json.dumps(field)} must be {requirement}"
        )


def load_candidate(candidate_dir: Path) -> Any:
    """Import an offline Candidate inside its untrusted evaluation process."""

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
    if not callable(getattr(module, "run_offline", None)):
        raise CandidateProtocolError(
            "candidate.py must define run_offline(candidate_dir, contexts_path, "
            "output_dir, device)"
        )
    return module


def load_context_request(path: Path) -> Dict[str, Any]:
    """Read and validate the target-free request visible to a Candidate."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise CandidateProtocolError("context request is invalid") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "candidate_protocol",
        "dataset_id",
        "vocab_size",
        "context_lengths",
        "examples",
    }:
        raise CandidateProtocolError("context request has an invalid schema")
    if payload["schema_version"] != DATASET_SCHEMA_VERSION:
        raise CandidateProtocolError("context request has an unsupported schema")
    if payload["candidate_protocol"] != CANDIDATE_PROTOCOL:
        raise CandidateProtocolError("context request uses a different protocol")
    dataset_id = payload["dataset_id"]
    vocab_size = payload["vocab_size"]
    lengths = payload["context_lengths"]
    examples = payload["examples"]
    if (
        not isinstance(dataset_id, str)
        or not dataset_id
        or len(dataset_id) > 512
        or "\0" in dataset_id
    ):
        raise CandidateProtocolError("context request dataset_id is invalid")
    if (
        isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size <= 0
    ):
        raise CandidateProtocolError("context request vocabulary is invalid")
    if (
        not isinstance(lengths, list)
        or not lengths
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in lengths
        )
        or sorted(set(lengths)) != lengths
    ):
        raise CandidateProtocolError("context request lengths are invalid")
    if not isinstance(examples, list) or not examples:
        raise CandidateProtocolError("context request has no examples")
    seen_ids = set()
    for example in examples:
        if not isinstance(example, dict) or set(example) != {"example_id", "input_ids"}:
            raise CandidateProtocolError("context example has an invalid schema")
        example_id = example["example_id"]
        input_ids = example["input_ids"]
        if (
            not isinstance(example_id, str)
            or not example_id
            or len(example_id) > 512
            or "\0" in example_id
            or example_id in seen_ids
        ):
            raise CandidateProtocolError("context example_id is invalid or duplicated")
        seen_ids.add(example_id)
        if not isinstance(input_ids, list) or len(input_ids) not in lengths:
            raise CandidateProtocolError("context example has an undeclared length")
        if not all(
            isinstance(token, int)
            and not isinstance(token, bool)
            and 0 <= token < vocab_size
            for token in input_ids
        ):
            raise CandidateProtocolError("context token is outside the vocabulary")
    return payload


def write_prediction_artifact(
    output_dir: Path,
    *,
    dataset_id: str,
    logits: Iterable[Sequence[Real]],
    examples: int,
    vocab_size: int,
) -> None:
    """Write the portable row-major float32 artifact required by the scorer."""

    output_dir.mkdir(parents=True, exist_ok=True)
    logits_path = output_dir / LOGITS_FILENAME
    rows = 0
    try:
        with logits_path.open("xb") as stream:
            for raw_row in logits:
                if hasattr(raw_row, "tolist"):
                    raw_row = raw_row.tolist()
                if not isinstance(raw_row, (list, tuple)) or len(raw_row) != vocab_size:
                    raise CandidateProtocolError(
                        "every prediction row must match the vocabulary size"
                    )
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(float(value))
                    for value in raw_row
                ):
                    raise CandidateProtocolError("prediction logits must be finite")
                values = array("f", (float(value) for value in raw_row))
                if sys.byteorder != "little":
                    values.byteswap()
                stream.write(values.tobytes())
                rows += 1
    except (OSError, OverflowError) as exc:
        raise CandidateProtocolError("prediction logits could not be written") from exc
    if rows != examples:
        raise CandidateProtocolError("prediction row count does not match the request")
    manifest = _expected_prediction_manifest(
        dataset_id=dataset_id,
        examples=examples,
        vocab_size=vocab_size,
    )
    try:
        (output_dir / PREDICTIONS_FILENAME).write_text(
            json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise CandidateProtocolError(
            "prediction manifest could not be written"
        ) from exc


def validate_prediction_artifact(
    output_dir: Path,
    *,
    dataset_id: str,
    examples: int,
    vocab_size: int,
) -> int:
    """Validate a Candidate output without requiring access to hidden targets."""

    try:
        entries = {entry.name: entry for entry in output_dir.iterdir()}
    except OSError as exc:
        raise CandidateProtocolError("prediction artifact is unavailable") from exc
    if set(entries) != {PREDICTIONS_FILENAME, LOGITS_FILENAME}:
        raise CandidateProtocolError("prediction artifact has unexpected entries")
    for entry in entries.values():
        metadata = entry.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise CandidateProtocolError("prediction entries must be regular files")
    storage_limit = prediction_storage_limit(
        examples=examples,
        vocab_size=vocab_size,
    )
    if (
        entries[PREDICTIONS_FILENAME].stat().st_size
        > storage_limit["manifest_max_bytes"]
    ):
        raise CandidateProtocolError("prediction manifest exceeds its byte limit")
    try:
        manifest = json.loads(entries[PREDICTIONS_FILENAME].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateProtocolError("prediction manifest is invalid") from exc
    _validate_prediction_manifest(
        manifest,
        dataset_id=dataset_id,
        examples=examples,
        vocab_size=vocab_size,
    )
    expected_bytes = storage_limit["logits_bytes"]
    if entries[LOGITS_FILENAME].stat().st_size != expected_bytes:
        raise CandidateProtocolError("prediction logits have the wrong byte length")
    if array("f").itemsize != 4:
        raise CandidateProtocolError("platform lacks IEEE-754 float32 arrays")
    try:
        with entries[LOGITS_FILENAME].open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                if len(chunk) % 4:
                    raise CandidateProtocolError("prediction logits are misaligned")
                values = array("f")
                values.frombytes(chunk)
                if sys.byteorder != "little":
                    values.byteswap()
                if any(not math.isfinite(value) for value in values):
                    raise CandidateProtocolError("prediction logits must be finite")
    except OSError as exc:
        raise CandidateProtocolError("prediction logits could not be read") from exc
    return expected_bytes + entries[PREDICTIONS_FILENAME].stat().st_size
