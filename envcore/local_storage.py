"""Admission of deployment-provided storage for the bundled local Worker.

This module does not allocate disks, set quotas, or inspect the operator's
remote control machine. The Worker deployment supplies a dedicated, empty,
capacity-bounded filesystem in its own execution environment. Its lifecycle
and enforcement (including metadata overhead) belong to that deployment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from .config import operator_storage_limit
from .worker_protocol import WorkerFailure


class LocalStorageProtection:
    def __init__(self, root: Path, max_bytes: int) -> None:
        self.max_bytes = operator_storage_limit(max_bytes)
        if self.max_bytes is None:
            raise ValueError("storage protection requires a byte limit")
        self.root = root.expanduser().resolve()
        try:
            if (
                not self.root.is_dir()
                or self.root == self.root.parent
                or self.root.stat().st_dev == self.root.parent.stat().st_dev
            ):
                raise WorkerFailure(
                    "operator storage protection requires a dedicated bounded "
                    "filesystem mounted at the episode parent directory; an ordinary "
                    "directory or Docker volume is not sufficient"
                )
            self.device = self.root.stat().st_dev
            self.capacity = self._capacity()
            self._validate_capacity(self.capacity)
            # An existing episode must not consume another episode's allowance.
            # Some filesystems create this reserved directory at format time.
            for entry in self.root.iterdir():
                if (
                    entry.name == "lost+found"
                    and not entry.is_symlink()
                    and entry.is_dir()
                ):
                    continue
                raise WorkerFailure(
                    "operator storage filesystem must be dedicated and empty"
                )
        except OSError as exc:
            raise WorkerFailure(f"cannot inspect Worker storage: {exc}") from exc

    def _capacity(self) -> Dict[str, int]:
        try:
            usage = os.statvfs(self.root)
        except (AttributeError, OSError) as exc:
            raise WorkerFailure(
                "Worker cannot verify bounded filesystem capacity"
            ) from exc
        return {
            "filesystem_bytes": usage.f_blocks * usage.f_frsize,
            "filesystem_inodes": usage.f_files,
        }

    def _validate_capacity(self, capacity: Dict[str, int]) -> None:
        if (
            set(capacity) != {"filesystem_bytes", "filesystem_inodes"}
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in capacity.values()
            )
            or capacity["filesystem_bytes"] > self.max_bytes
        ):
            raise WorkerFailure(
                "Worker storage has no verifiable capacity within the requested "
                f"operator limit of {self.max_bytes} bytes; provision a bounded "
                "filesystem in the Worker environment or explicitly disable protection"
            )

    def verify_workload_view(self, capacity: Dict[str, int]) -> None:
        """The runtime must observe the same admitted filesystem, not its host disk."""

        self._validate_capacity(capacity)
        if capacity != self.capacity:
            raise WorkerFailure(
                "Worker and workload storage capacity observations disagree"
            )

    def check_paths(self, *paths: Path) -> None:
        """Reject escaped symlinks, nested filesystems, and replaced storage roots."""

        try:
            if (
                self.root.stat().st_dev != self.device
                or self._capacity() != self.capacity
            ):
                raise WorkerFailure("admitted Worker storage has changed")
            for path in paths:
                resolved = path.resolve()
                if not resolved.is_relative_to(self.root):
                    raise WorkerFailure(
                        "writable workload path escapes protected storage"
                    )
                # Check every existing ancestor: a nested mount may hide an
                # unbounded filesystem even when the final directory is new.
                for current in (resolved, *resolved.parents):
                    if current.exists() and current.stat().st_dev != self.device:
                        raise WorkerFailure(
                            "writable workload path uses another filesystem"
                        )
                    if current == self.root:
                        break
        except OSError as exc:
            raise WorkerFailure(
                f"cannot verify writable Worker storage: {exc}"
            ) from exc

    def manifest(self) -> Dict[str, Any]:
        return {
            "max_episode_storage_bytes": self.max_bytes,
            "mechanism": "deployment_bounded_filesystem",
            "verification": "worker_and_workload_capacity",
            "scope": "episode_directory",
            **self.capacity,
        }
