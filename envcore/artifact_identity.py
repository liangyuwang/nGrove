"""Release-owned, platform-neutral artifact identity algorithms."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path


FILE_IDENTITY_ALGORITHM = "sha256"
CANDIDATE_TREE_IDENTITY_ALGORITHM = "benchcore-tree-v2-sha256"
HIDDEN_BUNDLE_IDENTITY_ALGORITHM = "benchcore-hidden-bundle-v1-sha256"
HIDDEN_BUNDLE_FILENAMES = ("contexts.json", "targets.json")
_HIDDEN_BUNDLE_DOMAIN_SEPARATOR = b"benchcore-hidden-bundle-v1\0"


def hidden_bundle_sha256(path: Path) -> str:
    """Return the normative identity of one two-file hidden split bundle.

    The digest input is, in order, the domain separator above and then, for
    each filename in :data:`HIDDEN_BUNDLE_FILENAMES`, its unsigned 64-bit
    big-endian UTF-8 name length, UTF-8 name, unsigned 64-bit big-endian file
    size, and exact file bytes. The directory must contain only those two
    regular files and must not itself be a symlink.
    """

    bundle = Path(path)
    if bundle.is_symlink() or not bundle.is_dir():
        raise OSError("hidden dataset must be a directory")
    entries = {entry.name: entry for entry in bundle.iterdir()}
    if set(entries) != set(HIDDEN_BUNDLE_FILENAMES):
        raise OSError("hidden dataset must contain only contexts.json and targets.json")

    digest = hashlib.sha256()
    digest.update(_HIDDEN_BUNDLE_DOMAIN_SEPARATOR)
    for name in HIDDEN_BUNDLE_FILENAMES:
        item = entries[name]
        metadata = item.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError(f"hidden dataset entry is not a regular file: {name}")
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(metadata.st_size.to_bytes(8, "big"))
        bytes_read = 0
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                bytes_read += len(chunk)
                digest.update(chunk)
        if bytes_read != metadata.st_size or item.lstat().st_size != metadata.st_size:
            raise OSError(f"hidden dataset entry changed while hashing: {name}")
    return digest.hexdigest()
