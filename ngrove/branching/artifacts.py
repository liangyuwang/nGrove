"""Optional local content-addressed storage for quiescent research state.

This utility does not capture a live Agent or provider session. Runners supply
explicit exported context and choose research-owned directories; never snapshot
the trusted episode directory or a private trajectory export wholesale.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath

from .contracts import BranchError, components, encoded, identifier


class ArtifactStore:
    def __init__(self, root: Path, *, max_object_bytes: int = 256 * 1024 * 1024):
        if (
            isinstance(max_object_bytes, bool)
            or not isinstance(max_object_bytes, int)
            or max_object_bytes <= 0
        ):
            raise BranchError("invalid_object_limit")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.max_object_bytes = max_object_bytes

    def _path(self, digest: str) -> Path:
        ref = {"object": {"sha256": digest, "format": "bytes", "locator": "local"}}
        components(ref, ())
        return self.root / digest

    def put(self, data: bytes, *, format: str) -> dict:
        identifier(format)
        if len(data) > self.max_object_bytes:
            raise BranchError("object_too_large")
        digest = hashlib.sha256(data).hexdigest()
        path = self._path(digest)
        if path.exists():
            self.read({"format": format, "sha256": digest, "locator": digest})
        else:
            fd, temporary = tempfile.mkstemp(dir=self.root)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, 0o444)
                os.replace(temporary, path)
                if os.name == "posix":
                    directory_fd = os.open(self.root, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
            finally:
                Path(temporary).unlink(missing_ok=True)
        return {"format": format, "sha256": digest, "locator": digest}

    def put_json(self, value, *, format: str = "json-v1") -> dict:
        return self.put(encoded(value), format=format)

    def read(self, ref: dict) -> bytes:
        components({"object": ref}, ())
        if ref["locator"] != ref["sha256"]:
            raise BranchError("foreign_artifact_locator")
        path = self._path(ref["sha256"])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > self.max_object_bytes
        ):
            raise BranchError("artifact_unavailable")
        with path.open("rb") as stream:
            data = stream.read(self.max_object_bytes + 1)
        if hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise BranchError("artifact_identity_mismatch")
        return data

    def capture_directory(self, root: Path) -> dict:
        root = Path(root).resolve()
        if self.root == root or root in self.root.parents:
            raise BranchError("artifact_store_inside_snapshot")
        if not root.is_dir():
            raise BranchError("snapshot_directory_missing")
        files, directories = {}, []
        for parent, dirs, names in os.walk(root, followlinks=False):
            for name in sorted(dirs):
                path = Path(parent) / name
                if not stat.S_ISDIR(path.lstat().st_mode):
                    raise BranchError("snapshot_contains_link")
                directories.append(path.relative_to(root).as_posix())
            for name in sorted(names):
                path = Path(parent) / name
                before = path.lstat()
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                    raise BranchError("snapshot_requires_regular_files")
                if before.st_size > self.max_object_bytes:
                    raise BranchError("object_too_large")
                descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(descriptor, "rb") as stream:
                    opened = os.fstat(stream.fileno())
                    data = stream.read(self.max_object_bytes + 1)
                    after = os.fstat(stream.fileno())

                def signature(s):
                    return (
                        s.st_dev,
                        s.st_ino,
                        s.st_size,
                        s.st_mtime_ns,
                        s.st_ctime_ns,
                    )

                if signature(before) != signature(opened) or signature(
                    opened
                ) != signature(after):
                    raise BranchError("snapshot_changed_during_capture")
                relative = path.relative_to(root).as_posix()
                self._relative(relative)
                files[relative] = {
                    "object": self.put(data, format="bytes"),
                    "executable": bool(before.st_mode & 0o111),
                }
        for directory in directories:
            self._relative(directory)
        return self.put_json(
            {"files": files, "directories": sorted(directories)}, format="directory-v1"
        )

    @staticmethod
    def _relative(value):
        if (
            not isinstance(value, str)
            or not value
            or "\\" in value
            or ":" in value
            or "\0" in value
            or PurePosixPath(value).is_absolute()
            or any(part in {"", ".", ".."} for part in value.split("/"))
        ):
            raise BranchError("invalid_snapshot_path")
        return value

    def restore_directory(self, ref: dict, destination: Path):
        if ref["format"] != "directory-v1":
            raise BranchError("unsupported_directory_format")
        manifest = json.loads(self.read(ref))
        if not isinstance(manifest, dict) or set(manifest) != {"files", "directories"}:
            raise BranchError("invalid_directory_manifest")
        files, directories = manifest["files"], manifest["directories"]
        if not isinstance(files, dict) or not isinstance(directories, list):
            raise BranchError("invalid_directory_manifest")
        for name in list(files) + directories:
            self._relative(name)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        try:
            for name in directories:
                (destination / name).mkdir(parents=True, exist_ok=True)
            for name, value in files.items():
                if not isinstance(value, dict) or set(value) != {
                    "object",
                    "executable",
                }:
                    raise BranchError("invalid_directory_entry")
                path = destination / name
                path.parent.mkdir(parents=True, exist_ok=True)
                data = self.read(value["object"])
                with path.open("xb") as stream:
                    stream.write(data)
                path.chmod(0o700 if value["executable"] is True else 0o600)
        except BaseException:
            shutil.rmtree(destination)
            raise

    def copy_component(self, ref: dict, destination: ArtifactStore) -> dict:
        """Full physical copy to a different store, retaining canonical identities.

        Shared mode keeps the original immutable store; both modes must restore
        writable workspaces privately. Opaque/provider formats need their adapter.
        """
        data = self.read(ref)
        if ref["format"] == "directory-v1":
            manifest = json.loads(data)
            for entry in manifest["files"].values():
                self.copy_component(entry["object"], destination)
        return destination.put(data, format=ref["format"])
