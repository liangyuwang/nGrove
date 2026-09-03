"""Prepare a deterministic, sub-megabyte TinyStories benchmark instance.

Only Python's standard library is required. The source dataset revision and the
ordered sample digest are frozen so a changed upstream fails loudly instead of
silently creating a different benchmark.
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
EVAL_TARGET_TOKENS = 8_192

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train" / "train.json"
ANALYSIS_PATH = DATA_DIR / "private" / "analysis.json"
TEST_PATH = DATA_DIR / "private" / "test.json"
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
        headers={"User-Agent": "AutoTrain-Bench-mini-lm/0"},
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
            f"{SOURCE_SHA256}, received {digest}. The pinned benchmark must not "
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
        artifacts: Dict[str, str] = metadata["artifact_sha256"]
        return all(
            path.is_file() and _sha256_file(path) == artifacts[path.name]
            for path in (TRAIN_PATH, ANALYSIS_PATH, TEST_PATH)
        )
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        return False


def prepare(*, force: bool = False) -> Dict[str, object]:
    if not force and _existing_data_is_valid():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        print(f"Mini benchmark data is already ready at {DATA_DIR}")
        return metadata

    documents = download_documents()
    train_docs = documents[:TRAIN_DOCUMENTS]
    analysis_docs = documents[TRAIN_DOCUMENTS : TRAIN_DOCUMENTS + ANALYSIS_DOCUMENTS]
    test_docs = documents[-TEST_DOCUMENTS:]
    train_sequences = make_sequences(train_docs)
    analysis_sequences = make_sequences(
        analysis_docs, max_target_tokens=EVAL_TARGET_TOKENS
    )
    test_sequences = make_sequences(test_docs, max_target_tokens=EVAL_TARGET_TOKENS)

    _write_json(TRAIN_PATH, {"sequences": train_sequences})
    _write_json(ANALYSIS_PATH, {"sequences": analysis_sequences}, private=True)
    _write_json(TEST_PATH, {"sequences": test_sequences}, private=True)

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
        "splits": {
            "train": {
                "documents": TRAIN_DOCUMENTS,
                "sequences": len(train_sequences),
                "target_tokens": sum(len(row) - 1 for row in train_sequences),
            },
            "analysis": {
                "documents": ANALYSIS_DOCUMENTS,
                "sequences": len(analysis_sequences),
                "target_tokens": sum(len(row) - 1 for row in analysis_sequences),
            },
            "test": {
                "documents": TEST_DOCUMENTS,
                "sequences": len(test_sequences),
                "target_tokens": sum(len(row) - 1 for row in test_sequences),
            },
        },
        "artifact_sha256": {
            path.name: _sha256_file(path)
            for path in (TRAIN_PATH, ANALYSIS_PATH, TEST_PATH)
        },
    }
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
