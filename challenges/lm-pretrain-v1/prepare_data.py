"""Build the lm-pretrain-v1 preview from pinned FineWeb-Edu Parquet files.

The default build downloads roughly 28.5 GB of source Parquet and emits about
10 billion GPT-2 tokens. Optional dependencies are deliberately lazy-loaded so
the deterministic sharding and split helpers can be tested without downloading
the corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from array import array
from datetime import date, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


DATASET = "HuggingFaceFW/fineweb-edu"
DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
UPSTREAM_SAMPLE = "sample/10BT"
TOKENIZER = "gpt2"
VOCAB_SIZE = 50_257
EOT_TOKEN_ID = 50_256
TARGET_TRAIN_TOKENS = 10_000_000_000
SHARD_TOKENS = 100_000_000
RAW_SHARD_DOCUMENTS = 100_000
PREVIEW_EVAL_TARGET_TOKENS = 4_096
EVAL_SEQUENCE_TARGETS = 256
SPLIT_BUCKETS = 4_096

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_RAW_DIR = DATA_DIR / "raw" / DATASET_REVISION
DEFAULT_OUTPUT_DIR = DATA_DIR / "generated"
BUILDER_ID = "autotrain-bench/lm-pretrain-v1"

class PreparationError(RuntimeError):
    pass


def _paths_overlap(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def _validate_existing_output(path: Path) -> None:
    """Allow --force only for a recognizable prior build of this task."""

    if not path.is_dir():
        raise PreparationError("--force target must be a generated data directory")
    manifest_path = path / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise PreparationError(
            "refusing --force because the target is not a recognizable "
            "lm-pretrain-v1 generated build"
        ) from exc
    builder = payload.get("builder")
    source = payload.get("source")
    tokenizer = payload.get("tokenizer")
    recognized = (
        isinstance(builder, dict)
        and builder.get("id", BUILDER_ID) == BUILDER_ID
        and isinstance(source, dict)
        and source.get("dataset") == DATASET
        and isinstance(tokenizer, dict)
        and tokenizer.get("name") == TOKENIZER
    )
    if not recognized:
        raise PreparationError(
            "refusing --force because the target belongs to another dataset build"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def tokenizer_fingerprint(encoding: Any) -> str:
    """Hash the complete GPT-2 merge table and special-token mapping."""

    try:
        mergeable_ranks = encoding._mergeable_ranks
        special_tokens = encoding._special_tokens
    except AttributeError as exc:
        raise PreparationError(
            "the installed tiktoken build does not expose canonical encoding assets"
        ) from exc
    digest = hashlib.sha256()
    for token, rank in sorted(mergeable_ranks.items(), key=lambda item: item[1]):
        digest.update(len(token).to_bytes(8, "big"))
        digest.update(token)
        digest.update(int(rank).to_bytes(8, "big"))
    for token, rank in sorted(special_tokens.items()):
        encoded = token.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(int(rank).to_bytes(8, "big"))
    return digest.hexdigest()


def preview_split(source_key: str) -> str:
    """Select small deterministic local-preview holdouts by pinned row identity."""

    bucket = int.from_bytes(
        hashlib.sha256(source_key.encode("utf-8")).digest()[:8], "big"
    ) % SPLIT_BUCKETS
    if bucket == 0:
        return "val"
    if bucket == 1:
        return "test"
    return "train"


class EvaluationCollector:
    def __init__(self, *, target_tokens: int, sequence_targets: int) -> None:
        if target_tokens <= 0 or sequence_targets <= 0:
            raise ValueError("evaluation token limits must be positive")
        self.target_tokens = target_tokens
        self.sequence_targets = sequence_targets
        self.sequences: List[List[int]] = []
        self.token_count = 0

    @property
    def complete(self) -> bool:
        return self.token_count >= self.target_tokens

    def add_document(self, document_tokens: Sequence[int]) -> None:
        if self.complete:
            return
        content = list(document_tokens[1:])
        for start in range(0, len(content), self.sequence_targets):
            remaining = self.target_tokens - self.token_count
            if remaining <= 0:
                return
            chunk = content[start : start + min(self.sequence_targets, remaining)]
            if not chunk:
                continue
            self.sequences.append([EOT_TOKEN_ID, *chunk])
            self.token_count += len(chunk)


class TokenShardWriter:
    """Write a document-aware little-endian uint16 token store."""

    def __init__(self, root: Path, *, maximum_tokens: int) -> None:
        if maximum_tokens <= 0:
            raise ValueError("maximum_tokens must be positive")
        self.root = root
        self.maximum_tokens = maximum_tokens
        self.root.mkdir(parents=True, exist_ok=False)
        self._stream = None
        self._digest: Optional[Any] = None
        self._shard_index = -1
        self._shard_tokens = 0
        self._shard_path: Optional[Path] = None
        self.shards: List[Dict[str, Any]] = []
        self.total_tokens = 0

    def _open_shard(self) -> None:
        self._shard_index += 1
        self._shard_tokens = 0
        self._shard_path = self.root / f"tokens-{self._shard_index:05d}.bin"
        self._stream = self._shard_path.open("wb")
        self._digest = hashlib.sha256()

    def _close_shard(self) -> None:
        if self._stream is None or self._shard_path is None or self._digest is None:
            return
        self._stream.close()
        self.shards.append(
            {
                "file": self._shard_path.name,
                "tokens": self._shard_tokens,
                "bytes": self._shard_tokens * 2,
                "sha256": self._digest.hexdigest(),
            }
        )
        self._stream = None
        self._digest = None
        self._shard_path = None

    @staticmethod
    def _encoded(tokens: Sequence[int]) -> bytes:
        if not tokens or any(
            not isinstance(token, int) or token < 0 or token > 65_535
            for token in tokens
        ):
            raise PreparationError("token IDs must be non-empty uint16 values")
        values = array("H", tokens)
        if sys.byteorder != "little":
            values.byteswap()
        return values.tobytes()

    def add_document(self, tokens: Sequence[int]) -> List[Dict[str, int]]:
        remaining = list(tokens)
        segments: List[Dict[str, int]] = []
        while remaining:
            if self._stream is None:
                self._open_shard()
            capacity = self.maximum_tokens - self._shard_tokens
            take = min(capacity, len(remaining))
            chunk = remaining[:take]
            del remaining[:take]
            offset = self._shard_tokens
            encoded = self._encoded(chunk)
            self._stream.write(encoded)
            self._digest.update(encoded)
            self._shard_tokens += take
            self.total_tokens += take
            segments.append(
                {
                    "shard": self._shard_index,
                    "token_offset": offset,
                    "token_count": take,
                }
            )
            if self._shard_tokens == self.maximum_tokens:
                self._close_shard()
        return segments

    def close(self) -> List[Dict[str, Any]]:
        self._close_shard()
        return list(self.shards)


class RawParquetWriter:
    """Preserve every public TRAIN document as text plus source metadata."""

    def __init__(
        self,
        root: Path,
        *,
        pyarrow: Any,
        parquet: Any,
        maximum_documents: int = RAW_SHARD_DOCUMENTS,
    ) -> None:
        self.root = root
        self.pyarrow = pyarrow
        self.parquet = parquet
        self.maximum_documents = maximum_documents
        self.root.mkdir(parents=True, exist_ok=False)
        self.schema = pyarrow.schema(
            [
                pyarrow.field("doc_id", pyarrow.string(), nullable=False),
                pyarrow.field("text", pyarrow.large_string(), nullable=False),
                pyarrow.field("url", pyarrow.string()),
                pyarrow.field("language", pyarrow.string()),
                pyarrow.field("score", pyarrow.float64()),
                pyarrow.field("metadata_json", pyarrow.large_string(), nullable=False),
            ]
        )
        self._records: List[Dict[str, Any]] = []
        self._shard_index = 0
        self._writer = None
        self._shard_path: Optional[Path] = None
        self._shard_documents = 0
        self.shards: List[Dict[str, Any]] = []
        self.total_documents = 0

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    def add_document(self, row: Dict[str, Any], source_key: str) -> None:
        metadata = {
            name: _json_scalar(value)
            for name, value in row.items()
            if name != "text" and not name.startswith("_")
        }
        self._records.append(
            {
                "doc_id": source_key,
                "text": row["text"],
                "url": _json_scalar(row.get("url")),
                "language": _json_scalar(row.get("language")),
                "score": self._optional_float(
                    row.get("score", row.get("int_score"))
                ),
                "metadata_json": json.dumps(
                    metadata, separators=(",", ":"), sort_keys=True
                ),
            }
        )
        self.total_documents += 1
        if len(self._records) >= 1_024:
            self._flush_records()

    def _open_shard(self) -> None:
        self._shard_path = self.root / f"documents-{self._shard_index:05d}.parquet"
        self._writer = self.parquet.ParquetWriter(
            self._shard_path,
            self.schema,
            compression="zstd",
        )
        self._shard_documents = 0

    def _close_shard(self) -> None:
        if self._writer is None or self._shard_path is None:
            return
        self._writer.close()
        self.shards.append(
            {
                "file": self._shard_path.name,
                "documents": self._shard_documents,
                "bytes": self._shard_path.stat().st_size,
                "sha256": _sha256_file(self._shard_path),
            }
        )
        self._shard_index += 1
        self._writer = None
        self._shard_path = None
        self._shard_documents = 0

    def _flush_records(self) -> None:
        while self._records:
            if self._writer is None:
                self._open_shard()
            capacity = self.maximum_documents - self._shard_documents
            count = min(capacity, len(self._records))
            records = self._records[:count]
            del self._records[:count]
            table = self.pyarrow.Table.from_pylist(records, schema=self.schema)
            self._writer.write_table(table)
            self._shard_documents += count
            if self._shard_documents == self.maximum_documents:
                self._close_shard()

    def close(self) -> List[Dict[str, Any]]:
        self._flush_records()
        self._close_shard()
        return list(self.shards)


def _load_dependencies() -> Tuple[Any, Any, Any, Any]:
    try:
        from huggingface_hub import snapshot_download
        import pyarrow
        import pyarrow.parquet as parquet
        import tiktoken
    except ImportError as exc:
        raise PreparationError(
            "data dependencies are missing; run: "
            "python -m pip install -r challenges/lm-pretrain-v1/data/requirements.txt"
        ) from exc
    return snapshot_download, pyarrow, parquet, tiktoken


def _download_source(snapshot_download: Any, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DATASET,
        repo_type="dataset",
        revision=DATASET_REVISION,
        allow_patterns=[f"{UPSTREAM_SAMPLE}/*.parquet"],
        local_dir=str(raw_dir),
    )
    return raw_dir


def _parquet_files(source_dir: Path) -> List[Path]:
    files = sorted(source_dir.rglob("*.parquet"))
    if not files:
        raise PreparationError(f"no Parquet files found beneath {source_dir}")
    return files


def iter_documents(files: Iterable[Path], parquet: Any) -> Iterator[Dict[str, Any]]:
    for path in files:
        parquet_file = parquet.ParquetFile(path)
        names = set(parquet_file.schema_arrow.names)
        if "text" not in names:
            raise PreparationError(f"FineWeb-Edu shard has no text column: {path}")
        columns = list(parquet_file.schema_arrow.names)
        row_offset = 0
        for batch in parquet_file.iter_batches(batch_size=256, columns=columns):
            for index, row in enumerate(batch.to_pylist()):
                text = row.get("text")
                if not isinstance(text, str) or not text:
                    continue
                row["_source_file"] = path.name
                row["_source_row"] = row_offset + index
                yield row
            row_offset += batch.num_rows


def _write_json(path: Path, payload: Any, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600 if private else 0o644)


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def prepare(
    *,
    source_dir: Optional[Path],
    output_dir: Path,
    target_train_tokens: int,
    shard_tokens: int,
    eval_target_tokens: int,
    eval_sequence_targets: int,
    accept_large_download: bool,
    force: bool,
) -> Dict[str, Any]:
    if min(
        target_train_tokens,
        shard_tokens,
        eval_target_tokens,
        eval_sequence_targets,
    ) <= 0:
        raise PreparationError("all token limits must be positive")
    if eval_sequence_targets > 2_048:
        raise PreparationError("evaluation sequences exceed task context_length=2048")
    configured_output = output_dir.expanduser()
    if configured_output.is_symlink():
        raise PreparationError("output directory may not be a symbolic link")
    output_dir = configured_output.resolve()
    source_root_argument = (
        source_dir.expanduser().resolve()
        if source_dir is not None
        else DEFAULT_RAW_DIR.resolve()
    )
    if _paths_overlap(output_dir, source_root_argument):
        raise PreparationError("output directory may not overlap the source directory")
    if output_dir.exists() and not force:
        raise PreparationError(
            f"output already exists: {output_dir}; pass --force to rebuild it"
        )
    if output_dir.exists():
        _validate_existing_output(output_dir)

    if source_dir is None:
        if not accept_large_download:
            raise PreparationError(
                "the pinned source download is roughly 28.5 GB; pass "
                "--accept-large-download or provide --source-dir"
            )
    snapshot_download, pyarrow, parquet, tiktoken = _load_dependencies()
    encoding = tiktoken.get_encoding(TOKENIZER)
    if encoding.n_vocab != VOCAB_SIZE or encoding.eot_token != EOT_TOKEN_ID:
        raise PreparationError("installed GPT-2 tokenizer has unexpected semantics")
    fingerprint = tokenizer_fingerprint(encoding)

    staging = output_dir.parent / f".{output_dir.name}.stage-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    public_dir = staging / "public"
    public_raw_dir = public_dir / "raw"
    tokenized_dir = public_dir / "tokenized"
    private_dir = staging / "private"
    writer: Optional[TokenShardWriter] = None
    raw_writer: Optional[RawParquetWriter] = None
    build_manifest: Dict[str, Any]

    try:
        if source_dir is None:
            source_root = _download_source(snapshot_download, DEFAULT_RAW_DIR)
        else:
            source_root = source_root_argument
        files = _parquet_files(source_root)

        writer = TokenShardWriter(tokenized_dir, maximum_tokens=shard_tokens)
        raw_writer = RawParquetWriter(
            public_raw_dir,
            pyarrow=pyarrow,
            parquet=parquet,
        )
        document_index_path = tokenized_dir / "documents.jsonl"
        val = EvaluationCollector(
            target_tokens=eval_target_tokens,
            sequence_targets=eval_sequence_targets,
        )
        test = EvaluationCollector(
            target_tokens=eval_target_tokens,
            sequence_targets=eval_sequence_targets,
        )
        train_documents = 0

        with document_index_path.open("w", encoding="utf-8") as document_index:
            for row in iter_documents(files, parquet):
                source_key = f"{row['_source_file']}:{row['_source_row']}"
                split = preview_split(source_key)
                if split == "val" and val.complete:
                    split = "train"
                elif split == "test" and test.complete:
                    split = "train"
                if split == "train" and writer.total_tokens >= target_train_tokens:
                    if val.complete and test.complete:
                        break
                    continue

                tokens = [EOT_TOKEN_ID, *encoding.encode_ordinary(row["text"])]
                if len(tokens) < 2:
                    continue
                if split == "val":
                    val.add_document(tokens)
                    continue
                if split == "test":
                    test.add_document(tokens)
                    continue

                raw_writer.add_document(row, source_key)
                segments = writer.add_document(tokens)
                metadata = {
                    "doc_id": source_key,
                    "token_count": len(tokens),
                    "segments": segments,
                    "source": {
                        name: _json_scalar(value)
                        for name, value in row.items()
                        if name != "text" and not name.startswith("_")
                    },
                }
                document_index.write(
                    json.dumps(metadata, separators=(",", ":"), sort_keys=True)
                    + "\n"
                )
                train_documents += 1
        shards = writer.close()
        raw_train_shards = raw_writer.close()
        if not val.complete or not test.complete:
            raise PreparationError("source sample did not fill preview VAL/TEST")

        _write_json(
            private_dir / "val.json",
            {"sequences": val.sequences},
            private=True,
        )
        _write_json(
            private_dir / "test.json",
            {"sequences": test.sequences},
            private=True,
        )
        source_files = [
            {
                "file": path.relative_to(source_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in files
        ]
        public_raw_shards = [
            {**shard, "file": f"raw/{shard['file']}"}
            for shard in raw_train_shards
        ]
        public_token_shards = [
            {**shard, "file": f"tokenized/{shard['file']}"}
            for shard in shards
        ]
        public_manifest = {
            "layout": {
                "raw": "TRAIN documents with original text and source metadata",
                "tokenized": "official GPT-2 token store with document index",
            },
            "source": {
                "dataset": DATASET,
                "revision": DATASET_REVISION,
                "upstream_sample": UPSTREAM_SAMPLE,
                "source_parquet_files": source_files,
            },
            "raw_train": {
                "format": "Parquet with text, selected columns, and metadata_json",
                "documents": raw_writer.total_documents,
                "shards": public_raw_shards,
            },
            "tokenizer": {
                "name": TOKENIZER,
                "vocab_size": VOCAB_SIZE,
                "document_separator_token_id": EOT_TOKEN_ID,
                "fingerprint_sha256": fingerprint,
                "tiktoken_version": _package_version("tiktoken"),
            },
            "tokenized_train": {
                "format": "little-endian uint16",
                "target_scale_tokens": target_train_tokens,
                "exact_tokens": writer.total_tokens,
                "target_scale_reached": writer.total_tokens >= target_train_tokens,
                "documents": train_documents,
                "document_index": (
                    f"tokenized/{document_index_path.name}"
                ),
                "document_index_format": "JSON Lines with shard token spans",
                "document_index_sha256": _sha256_file(document_index_path),
                "shards": public_token_shards,
            },
        }
        _write_json(tokenized_dir / "manifest.json", public_manifest)
        _write_json(public_dir / "manifest.json", public_manifest)
        build_manifest = {
            **public_manifest,
            "preview_hidden_evaluation": {
                "policy": "local-preview-only; replace for a verified release",
                "val_target_tokens": val.token_count,
                "test_target_tokens": test.token_count,
                "sequence_target_tokens": eval_sequence_targets,
                "val_sha256": _sha256_file(private_dir / "val.json"),
                "test_sha256": _sha256_file(private_dir / "test.json"),
            },
            "builder": {
                "id": BUILDER_ID,
                "huggingface_hub_version": _package_version("huggingface-hub"),
                "pyarrow_version": _package_version("pyarrow"),
            },
        }
        _write_json(staging / "manifest.json", build_manifest)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(str(staging), str(output_dir))
    except Exception:
        if writer is not None:
            writer.close()
        if raw_writer is not None:
            raw_writer.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(json.dumps(build_manifest, indent=2, sort_keys=True))
    return build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="existing directory containing the pinned sample/10BT Parquet files",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-train-tokens", type=int, default=TARGET_TRAIN_TOKENS
    )
    parser.add_argument("--shard-tokens", type=int, default=SHARD_TOKENS)
    parser.add_argument(
        "--eval-target-tokens", type=int, default=PREVIEW_EVAL_TARGET_TOKENS
    )
    parser.add_argument(
        "--eval-sequence-targets", type=int, default=EVAL_SEQUENCE_TARGETS
    )
    parser.add_argument(
        "--accept-large-download",
        action="store_true",
        help="acknowledge the roughly 28.5 GB pinned source download",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing generated build"
    )
    args = parser.parse_args()
    prepare(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        target_train_tokens=args.target_train_tokens,
        shard_tokens=args.shard_tokens,
        eval_target_tokens=args.eval_target_tokens,
        eval_sequence_targets=args.eval_sequence_targets,
        accept_large_download=args.accept_large_download,
        force=args.force,
    )


if __name__ == "__main__":
    main()
