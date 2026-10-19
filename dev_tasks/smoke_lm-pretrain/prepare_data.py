"""Prepare a deterministic, sub-megabyte TinyStories development task.

Only Python's standard library is required. The source dataset revision and the
ordered sample digest are frozen so a changed upstream fails loudly instead of
silently creating a different task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from envcore.training_data import (
    TrainingDataError,
    verify_training_data,
    write_training_manifest,
)

DATASET = "karpathy/tinystories-gpt4-clean"
DATASET_REVISION = "0397e27157956705a0260709da3095bb9c43d6a7"
DATASET_LICENSE = "cdla-sharing-1.0"
DATASET_SERVER = "https://datasets-server.huggingface.co/rows"
CONFIG = "default"
SOURCE_SPLIT = "train"
DOCUMENT_COUNT = 1_200
PAGE_SIZE = 100
TRAIN_DOCUMENTS = 1_000
ANALYSIS_DOCUMENTS = 100
TEST_DOCUMENTS = 100
SOURCE_SHA256 = "0d17abc36c327b1acf37b2f172a76c77a5f4b099fb8608645f4149c8e987529e"
SEQUENCE_TARGET_TOKENS = 128
BOS_TOKEN_ID = 256
EVAL_CONTEXT_LENGTHS = (8, 16, 32, 64, 128)
EVAL_EXAMPLES = 512
EVALUATION_DATASET_SCHEMA_VERSION = 2
CANDIDATE_PROTOCOL = "offline_next_token_logits_v3"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train" / "train.json"
ANALYSIS_PATH = DATA_DIR / "private" / "analysis"
TEST_PATH = DATA_DIR / "private" / "test"
CONTEXTS_FILENAME = "contexts.json"
TARGETS_FILENAME = "targets.json"
METADATA_PATH = DATA_DIR / "metadata.json"


def _ordered_document_digest(documents: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in documents:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _download_page(offset: int) -> List[str]:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SOURCE_SPLIT,
            "offset": offset,
            "length": PAGE_SIZE,
        }
    )
    request = urllib.request.Request(
        f"{DATASET_SERVER}?{query}",
        headers={"User-Agent": "Envcore-smoke-lm-pretrain/0"},
    )
    last_error: Optional[BaseException] = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
            rows = payload["rows"]
            expected_indices = list(range(offset, offset + PAGE_SIZE))
            if [row["row_idx"] for row in rows] != expected_indices:
                raise RuntimeError("dataset server returned unexpected row indices")
            texts = [row["row"]["text"] for row in rows]
            if not all(isinstance(text, str) and text for text in texts):
                raise RuntimeError("dataset server returned invalid text rows")
            return texts
        except (KeyError, OSError, RuntimeError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(attempt)
    raise RuntimeError(
        f"failed to fetch TinyStories rows at offset {offset}: {last_error}"
    )


def download_documents() -> List[str]:
    documents: List[str] = []
    for offset in range(0, DOCUMENT_COUNT, PAGE_SIZE):
        print(f"Fetching TinyStories rows {offset}-{offset + PAGE_SIZE - 1}...")
        documents.extend(_download_page(offset))
    digest = _ordered_document_digest(documents)
    if digest != SOURCE_SHA256:
        raise RuntimeError(
            "TinyStories sample digest changed; expected "
            f"{SOURCE_SHA256}, received {digest}. The pinned task data must not "
            "silently change."
        )
    return documents


def make_sequences(
    documents: Iterable[str], *, max_target_tokens: Optional[int] = None
) -> List[List[int]]:
    sequences: List[List[int]] = []
    emitted = 0
    for text in documents:
        encoded = text.encode("utf-8")
        for start in range(0, len(encoded), SEQUENCE_TARGET_TOKENS):
            if max_target_tokens is not None and emitted >= max_target_tokens:
                return sequences
            limit = SEQUENCE_TARGET_TOKENS
            if max_target_tokens is not None:
                limit = min(limit, max_target_tokens - emitted)
            chunk = encoded[start : start + limit]
            if not chunk:
                continue
            sequences.append([BOS_TOKEN_ID, *chunk])
            emitted += len(chunk)
    return sequences


class EvaluationCollector:
    """Build balanced, non-overlapping, single-target evaluation examples."""

    def __init__(
        self,
        *,
        target_examples: int,
        context_lengths: Sequence[int],
        boundary_token_id: int,
    ) -> None:
        lengths = tuple(context_lengths)
        if (
            isinstance(target_examples, bool)
            or not isinstance(target_examples, int)
            or target_examples <= 0
            or not lengths
            or target_examples < len(lengths)
            or any(
                isinstance(length, bool) or not isinstance(length, int) or length <= 0
                for length in lengths
            )
            or tuple(sorted(set(lengths))) != lengths
        ):
            raise ValueError(
                "evaluation examples must cover every positive, ordered context length"
            )
        base, extra = divmod(target_examples, len(lengths))
        self.quotas = {
            length: base + (index < extra) for index, length in enumerate(lengths)
        }
        self.context_lengths = lengths
        self.boundary_token_id = boundary_token_id
        self.examples: List[Dict[str, object]] = []
        self.counts = {length: 0 for length in lengths}
        self._next_length_index = 0

    @property
    def complete(self) -> bool:
        return all(
            self.counts[length] == self.quotas[length]
            for length in self.context_lengths
        )

    def add_document(self, source_id: str, document_tokens: Sequence[int]) -> None:
        if self.complete:
            return
        tokens = list(document_tokens)
        start = 0
        while start < len(tokens) and not self.complete:
            selected_index = None
            for offset in range(len(self.context_lengths)):
                index = (self._next_length_index + offset) % len(self.context_lengths)
                length = self.context_lengths[index]
                if self.counts[length] < self.quotas[length] and start + length <= len(
                    tokens
                ):
                    selected_index = index
                    break
            if selected_index is None:
                return
            length = self.context_lengths[selected_index]
            end = start + length
            context = [self.boundary_token_id, *tokens[start : end - 1]]
            self.examples.append(
                {
                    "example_id": f"example-{len(self.examples):08d}",
                    "context": context,
                    "target": tokens[end - 1],
                    "source_id": source_id,
                    "source_start": start,
                    "source_end": end,
                }
            )
            self.counts[length] += 1
            self._next_length_index = (selected_index + 1) % len(self.context_lengths)
            start = end

    def context_payload(self, dataset_id: str) -> Dict[str, object]:
        if not self.complete:
            raise RuntimeError("source sample did not fill hidden evaluation examples")
        return {
            "schema_version": EVALUATION_DATASET_SCHEMA_VERSION,
            "candidate_protocol": CANDIDATE_PROTOCOL,
            "dataset_id": dataset_id,
            "vocab_size": 257,
            "context_lengths": list(self.context_lengths),
            "examples": [
                {
                    "example_id": example["example_id"],
                    "input_ids": example["context"],
                }
                for example in self.examples
            ],
        }

    def target_payload(
        self, dataset_id: str, contexts_sha256: str
    ) -> Dict[str, object]:
        if not self.complete:
            raise RuntimeError("source sample did not fill hidden evaluation examples")
        return {
            "schema_version": EVALUATION_DATASET_SCHEMA_VERSION,
            "candidate_protocol": CANDIDATE_PROTOCOL,
            "dataset_id": dataset_id,
            "contexts_sha256": contexts_sha256,
            "examples": [
                {
                    "example_id": example["example_id"],
                    "target": example["target"],
                    "source_id": example["source_id"],
                    "source_start": example["source_start"],
                    "source_end": example["source_end"],
                }
                for example in self.examples
            ],
        }

    def context_length_histogram(self) -> Dict[str, int]:
        return {str(length): self.counts[length] for length in self.context_lengths}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))
    path.chmod(0o600 if private else 0o644)


def _write_evaluation_split(
    path: Path, collector: EvaluationCollector, dataset_id: str
) -> Dict[str, str]:
    path.mkdir(parents=True, exist_ok=True)
    unexpected = {entry.name for entry in path.iterdir()} - {
        CONTEXTS_FILENAME,
        TARGETS_FILENAME,
    }
    if unexpected:
        raise RuntimeError(
            "refusing to overwrite unexpected hidden split entries: "
            + ", ".join(sorted(unexpected))
        )
    contexts_path = path / CONTEXTS_FILENAME
    targets_path = path / TARGETS_FILENAME
    _write_json(
        contexts_path,
        collector.context_payload(dataset_id),
        private=True,
    )
    contexts_sha256 = _sha256_file(contexts_path)
    _write_json(
        targets_path,
        collector.target_payload(dataset_id, contexts_sha256),
        private=True,
    )
    return {
        "contexts_sha256": contexts_sha256,
        "targets_sha256": _sha256_file(targets_path),
    }


def _existing_data_is_valid() -> bool:
    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        if metadata["source"]["revision"] != DATASET_REVISION:
            return False
        if metadata["source"]["ordered_sample_sha256"] != SOURCE_SHA256:
            return False
        if metadata["tokenizer"] != {
            "type": "utf8_bytes_with_bos",
            "vocab_size": 257,
            "bos_token_id": BOS_TOKEN_ID,
            "sequence_target_tokens": SEQUENCE_TARGET_TOKENS,
        }:
            return False
        if metadata["evaluation"] != {
            "candidate_protocol": CANDIDATE_PROTOCOL,
            "dataset_schema_version": EVALUATION_DATASET_SCHEMA_VERSION,
            "examples_per_split": EVAL_EXAMPLES,
            "context_lengths": list(EVAL_CONTEXT_LENGTHS),
            "construction": "balanced_non_overlapping_single_target_contexts",
        }:
            return False
        artifacts: Dict[str, str] = metadata["artifact_sha256"]
        verify_training_data(TRAIN_PATH.parent)
        return all(
            path.is_file() and _sha256_file(path) == artifacts[name]
            for name, path in {
                "train/train.json": TRAIN_PATH,
                "train/manifest.json": TRAIN_PATH.parent / "manifest.json",
                "private/analysis/contexts.json": ANALYSIS_PATH / CONTEXTS_FILENAME,
                "private/analysis/targets.json": ANALYSIS_PATH / TARGETS_FILENAME,
                "private/test/contexts.json": TEST_PATH / CONTEXTS_FILENAME,
                "private/test/targets.json": TEST_PATH / TARGETS_FILENAME,
            }.items()
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError, TrainingDataError):
        return False


def prepare(*, force: bool = False) -> Dict[str, object]:
    if not force and _existing_data_is_valid():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        print(f"Development task data is already ready at {DATA_DIR}")
        return metadata

    if TRAIN_PATH.parent.exists():
        unexpected = {entry.name for entry in TRAIN_PATH.parent.iterdir()} - {
            "train.json",
            "manifest.json",
        }
        if unexpected:
            raise RuntimeError(
                "unexpected public TRAIN entries; use a clean data directory: "
                + ", ".join(sorted(unexpected))
            )

    documents = download_documents()
    train_docs = documents[:TRAIN_DOCUMENTS]
    analysis_docs = documents[TRAIN_DOCUMENTS : TRAIN_DOCUMENTS + ANALYSIS_DOCUMENTS]
    test_docs = documents[-TEST_DOCUMENTS:]
    train_sequences = make_sequences(train_docs)
    analysis = EvaluationCollector(
        target_examples=EVAL_EXAMPLES,
        context_lengths=EVAL_CONTEXT_LENGTHS,
        boundary_token_id=BOS_TOKEN_ID,
    )
    test = EvaluationCollector(
        target_examples=EVAL_EXAMPLES,
        context_lengths=EVAL_CONTEXT_LENGTHS,
        boundary_token_id=BOS_TOKEN_ID,
    )
    for index, text in enumerate(analysis_docs, start=TRAIN_DOCUMENTS):
        analysis.add_document(f"tinystories:{index}", text.encode("utf-8"))
    for index, text in enumerate(
        test_docs,
        start=TRAIN_DOCUMENTS + ANALYSIS_DOCUMENTS,
    ):
        test.add_document(f"tinystories:{index}", text.encode("utf-8"))

    _write_json(TRAIN_PATH, {"sequences": train_sequences})
    analysis_artifacts = _write_evaluation_split(
        ANALYSIS_PATH,
        analysis,
        "tinystories-byte-lm-smoke-v1:offline-v3",
    )
    test_artifacts = _write_evaluation_split(
        TEST_PATH,
        test,
        "tinystories-byte-lm-smoke-v1:offline-v3",
    )

    metadata: Dict[str, object] = {
        "source": {
            "dataset": DATASET,
            "revision": DATASET_REVISION,
            "license": DATASET_LICENSE,
            "ordered_sample_sha256": SOURCE_SHA256,
            "documents": DOCUMENT_COUNT,
        },
        "tokenizer": {
            "type": "utf8_bytes_with_bos",
            "vocab_size": 257,
            "bos_token_id": BOS_TOKEN_ID,
            "sequence_target_tokens": SEQUENCE_TARGET_TOKENS,
        },
        "evaluation": {
            "candidate_protocol": CANDIDATE_PROTOCOL,
            "dataset_schema_version": EVALUATION_DATASET_SCHEMA_VERSION,
            "examples_per_split": EVAL_EXAMPLES,
            "context_lengths": list(EVAL_CONTEXT_LENGTHS),
            "construction": "balanced_non_overlapping_single_target_contexts",
        },
        "splits": {
            "train": {
                "documents": TRAIN_DOCUMENTS,
                "sequences": len(train_sequences),
                "target_tokens": sum(len(row) - 1 for row in train_sequences),
            },
            "analysis": {
                "documents": ANALYSIS_DOCUMENTS,
                "examples": len(analysis.examples),
                "target_predictions": len(analysis.examples),
                "context_length_histogram": analysis.context_length_histogram(),
            },
            "test": {
                "documents": TEST_DOCUMENTS,
                "examples": len(test.examples),
                "target_predictions": len(test.examples),
                "context_length_histogram": test.context_length_histogram(),
            },
        },
        "artifact_sha256": {
            "train/train.json": _sha256_file(TRAIN_PATH),
            "private/analysis/contexts.json": analysis_artifacts["contexts_sha256"],
            "private/analysis/targets.json": analysis_artifacts["targets_sha256"],
            "private/test/contexts.json": test_artifacts["contexts_sha256"],
            "private/test/targets.json": test_artifacts["targets_sha256"],
        },
    }
    write_training_manifest(
        TRAIN_PATH.parent,
        {
            "source": {
                "dataset": DATASET,
                "revision": DATASET_REVISION,
                "license": DATASET_LICENSE,
            },
            "tokenizer": metadata["tokenizer"],
            "train": metadata["splits"]["train"],
        },
    )
    metadata["artifact_sha256"]["train/manifest.json"] = _sha256_file(
        TRAIN_PATH.parent / "manifest.json"
    )
    _write_json(METADATA_PATH, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="download and rebuild even if verified data exists",
    )
    args = parser.parse_args()
    prepare(force=args.force)


if __name__ == "__main__":
    main()
