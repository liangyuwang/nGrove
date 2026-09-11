"""Portable inventory and integrity checks for the public TRAIN directory.

The manifest binds every visible file except itself. Task builders own data
formats and provenance; this module only verifies file identities. Deployment
isolation must keep the verified data immutable throughout research.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, List, Mapping, Optional


TRAIN_MANIFEST_VERSION = 1
TRAIN_MANIFEST_FILENAME = "manifest.json"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class TrainingDataError(ValueError):
    """TRAIN is incomplete, changed, or does not match its declared inventory."""


def _signature(metadata: os.stat_result) -> tuple:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


@contextmanager
def _open_regular_file(path: Path):
    path = Path(path)
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise TrainingDataError(f"data file must be regular with one link: {path}")
        # Avoid following a replacement symlink or blocking on a replacement
        # FIFO where the host exposes these portable open flags. Descriptor
        # identity is checked before reading on every host.
        flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        flags |= getattr(os, "O_BINARY", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            if _signature(os.fstat(stream.fileno())) != _signature(before):
                raise TrainingDataError(f"data file changed before reading: {path}")
            yield stream
            if _signature(os.fstat(stream.fileno())) != _signature(before):
                raise TrainingDataError(f"data file changed while reading: {path}")
        if _signature(path.lstat()) != _signature(before):
            raise TrainingDataError(f"data file changed while reading: {path}")
    except OSError as exc:
        raise TrainingDataError(f"cannot read data file {path}: {exc}") from exc


def file_identity(path: Path) -> Dict[str, Any]:
    """Hash one regular file, detecting replacement or concurrent edits."""

    digest = hashlib.sha256()
    total = 0
    with _open_regular_file(path) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            total += len(chunk)
    return {"bytes": total, "sha256": digest.hexdigest()}


def _inventory(root: Path) -> Dict[str, tuple]:
    try:
        if root.is_symlink() or not root.is_dir():
            raise TrainingDataError("TRAIN root must be a directory")
        result = {}
        pending = [root]
        while pending:
            directory = pending.pop()
            for entry in sorted(directory.iterdir()):
                metadata = entry.lstat()
                relative = entry.relative_to(root).as_posix()
                if stat.S_ISDIR(metadata.st_mode):
                    pending.append(entry)
                elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                    result[relative] = _signature(metadata)
                else:
                    raise TrainingDataError(
                        f"TRAIN contains a link or special file: {relative}"
                    )
        return result
    except OSError as exc:
        raise TrainingDataError(f"cannot inventory TRAIN: {exc}") from exc


def _file_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(character in value for character in ("\\", ":", "\0"))
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or value == TRAIN_MANIFEST_FILENAME
    ):
        raise TrainingDataError("TRAIN inventory has an unsafe file path")
    return value


def _manifest_files(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("integrity"), dict):
        raise TrainingDataError(
            "TRAIN manifest has no integrity inventory; prepare the data again"
        )
    inventory = payload["integrity"]
    if (
        set(inventory) != {"schema_version", "algorithm", "files"}
        or isinstance(inventory["schema_version"], bool)
        or not isinstance(inventory["schema_version"], int)
        or inventory["schema_version"] != TRAIN_MANIFEST_VERSION
        or inventory["algorithm"] != "sha256"
    ):
        raise TrainingDataError("unsupported TRAIN integrity inventory")
    files = inventory["files"]
    if not isinstance(files, list) or not files:
        raise TrainingDataError("TRAIN inventory must contain files")
    names = set()
    for record in files:
        if not isinstance(record, dict) or set(record) != {"file", "bytes", "sha256"}:
            raise TrainingDataError("invalid TRAIN file record")
        name = _file_name(record["file"])
        if name in names:
            raise TrainingDataError(f"duplicate TRAIN inventory path: {name}")
        names.add(name)
        if (
            isinstance(record["bytes"], bool)
            or not isinstance(record["bytes"], int)
            or record["bytes"] < 0
            or not isinstance(record["sha256"], str)
            or _SHA256.fullmatch(record["sha256"]) is None
        ):
            raise TrainingDataError(f"invalid TRAIN file identity: {name}")
    return files


def verify_training_data(root: Path) -> Dict[str, Any]:
    """Verify the entire TRAIN view before task reveal, without reading hidden data."""

    root = Path(root)
    before = _inventory(root)
    manifest_path = root / TRAIN_MANIFEST_FILENAME
    try:
        with _open_regular_file(manifest_path) as stream:
            manifest_bytes = stream.read()
        payload = json.loads(manifest_bytes)
    except (OSError, ValueError) as exc:
        raise TrainingDataError(
            "TRAIN manifest is missing or invalid; prepare the data again"
        ) from exc
    files = _manifest_files(payload)
    expected_names = {item["file"] for item in files} | {TRAIN_MANIFEST_FILENAME}
    if set(before) != expected_names:
        missing = sorted(expected_names - set(before))
        extra = sorted(set(before) - expected_names)
        raise TrainingDataError(
            f"TRAIN file set mismatch: missing={missing}, extra={extra}"
        )
    total = 0
    for record in files:
        path = root.joinpath(*record["file"].split("/"))
        if before[record["file"]][4] != record["bytes"]:
            raise TrainingDataError(f"TRAIN byte length mismatch: {record['file']}")
        actual = file_identity(path)
        if actual != {key: record[key] for key in ("bytes", "sha256")}:
            raise TrainingDataError(f"TRAIN checksum mismatch: {record['file']}")
        total += actual["bytes"]
    if _inventory(root) != before:
        raise TrainingDataError("TRAIN changed during verification")
    return {
        "status": "verified",
        "manifest_schema_version": TRAIN_MANIFEST_VERSION,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "files": len(files),
        "bytes": total,
    }


def write_training_manifest(
    root: Path, metadata: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Bind task-owned public metadata and all prepared TRAIN file identities."""

    root = Path(root)
    before = _inventory(root)
    files = [
        {"file": name, **file_identity(root.joinpath(*name.split("/")))}
        for name in sorted(before)
        if name != TRAIN_MANIFEST_FILENAME
    ]
    payload = dict(metadata or {})
    payload["integrity"] = {
        "schema_version": TRAIN_MANIFEST_VERSION,
        "algorithm": "sha256",
        "files": files,
    }
    _manifest_files(payload)
    if _inventory(root) != before:
        raise TrainingDataError("TRAIN changed while creating its manifest")
    staged = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=root, prefix=".manifest-", delete=False
        ) as stream:
            staged = Path(stream.name)
            json.dump(payload, stream, separators=(",", ":"), sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        staged.chmod(0o644)
        os.replace(staged, root / TRAIN_MANIFEST_FILENAME)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
    return payload
