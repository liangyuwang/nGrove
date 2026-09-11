import json
import os
import subprocess
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchcore.local_storage import LocalStorageProtection
from benchcore.runtime import DockerRuntime, RuntimeFailure
from benchcore.worker_protocol import WorkerTeardownFailure


@contextmanager
def bounded_filesystem(root, *, size=8 * 2**20, inodes=1024):
    """Simulate mount admission only; real ENOSPC is an integration check."""
    original = Path.stat

    def stat(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if path == root.parent:
            fields = list(value)
            fields[2] += 1
            return os.stat_result(fields)
        return value

    usage = SimpleNamespace(f_blocks=size // 4096, f_frsize=4096, f_files=inodes)
    with patch.object(Path, "stat", stat), patch("os.statvfs", return_value=usage):
        yield usage


class StorageProtectionTests(unittest.TestCase):
    def test_ordinary_directories_are_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaisesRegex(RuntimeFailure, "dedicated bounded filesystem"):
                LocalStorageProtection(root, 100 * 2**30)
            self.assertEqual(list(root.iterdir()), [])

    def test_capacity_empty_volume_and_workload_view_must_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for size, inodes in ((16 * 2**20, 1024), (0, 1024), (4096, 0)):
                with self.subTest(size=size, inodes=inodes), bounded_filesystem(
                    root, size=size, inodes=inodes
                ):
                    with self.assertRaisesRegex(
                        RuntimeFailure, "no verifiable capacity"
                    ):
                        LocalStorageProtection(root, 8 * 2**20)
            with bounded_filesystem(root):
                storage = LocalStorageProtection(root, 8 * 2**20)
                storage.verify_workload_view(storage.capacity)
                with self.assertRaisesRegex(RuntimeFailure, "disagree"):
                    storage.verify_workload_view(
                        {"filesystem_bytes": 4 * 2**20, "filesystem_inodes": 1024}
                    )
                (root / "old-episode").mkdir()
                with self.assertRaisesRegex(RuntimeFailure, "dedicated and empty"):
                    LocalStorageProtection(root, 8 * 2**20)

    def test_paths_cannot_escape_or_outlive_the_admitted_filesystem(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with bounded_filesystem(root) as usage:
                storage = LocalStorageProtection(root, 8 * 2**20)
                storage.check_paths(
                    root / "episode/workspace", root / "episode/leases/one"
                )
                (root / "escape").symlink_to(root.parent, target_is_directory=True)
                with self.assertRaisesRegex(RuntimeFailure, "escapes"):
                    storage.check_paths(root / "escape/unbounded")
                usage.f_blocks *= 2
                with self.assertRaisesRegex(RuntimeFailure, "has changed"):
                    storage.check_paths(root / "episode")

    def test_worker_checks_actual_writable_subdirectories_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with bounded_filesystem(root):
                runtime = DockerRuntime(docker_binary="docker")
                runtime._operator_storage = LocalStorageProtection(root, 8 * 2**20)
                lease = root / "lease"
                lease.mkdir()
                (lease / "output").symlink_to(root.parent, target_is_directory=True)
                with self.assertRaisesRegex(RuntimeFailure, "escapes"):
                    runtime.start_training(
                        name="not-started",
                        task=None,
                        resources=None,
                        workspace=root / "workspace",
                        lease_dir=lease,
                    )
                with self.assertRaisesRegex(RuntimeFailure, "host_cli cannot confine"):
                    runtime.start_host_agent(
                        agent=None,
                        control_dir=root,
                        public_task_path=root,
                        stdout_path=root / "stdout",
                        stderr_path=root / "stderr",
                    )

    def test_unconfirmed_probe_teardown_retains_storage_and_rejects_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            runtime = DockerRuntime(docker_binary="docker")

            def run(args, **kwargs):
                if args[:2] == ["image", "inspect"]:
                    return subprocess.CompletedProcess(
                        args, 0, json.dumps(["sha256:" + "a" * 64, None]), ""
                    )
                raise RuntimeFailure("probe timed out")

            with bounded_filesystem(root), patch.object(
                runtime, "_run", side_effect=run
            ), patch.object(
                runtime,
                "stop_container",
                side_effect=WorkerTeardownFailure("still running"),
            ):
                with self.assertRaises(WorkerTeardownFailure):
                    runtime.configure_operator_storage(
                        root=root, max_bytes=8 * 2**20, probe_image="fixture"
                    )
            self.assertIsNone(runtime._operator_storage)
            self.assertEqual(len(list(root.glob(".atb-storage-*/identity"))), 1)

    def test_unwritable_volume_is_rejected_before_runtime_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            runtime = DockerRuntime(docker_binary="docker")
            with bounded_filesystem(root), patch(
                "benchcore.runtime.tempfile.mkdtemp",
                side_effect=PermissionError("read only"),
            ), patch.object(runtime, "_run") as run:
                with self.assertRaisesRegex(RuntimeFailure, "cannot write"):
                    runtime.configure_operator_storage(
                        root=root, max_bytes=8 * 2**20, probe_image="fixture"
                    )
                run.assert_not_called()
            self.assertIsNone(runtime._operator_storage)

    def test_docker_admission_verifies_shared_identity_and_cleans_up(self):
        for observation in (
            "valid",
            "different_identity",
            "different_capacity",
            "bad_json",
            "timeout",
        ):
            with self.subTest(
                observation=observation
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                runtime = DockerRuntime(docker_binary="docker")
                stopped = []

                def run(args, **kwargs):
                    if args[:2] == ["image", "inspect"]:
                        return subprocess.CompletedProcess(
                            args, 0, json.dumps(["sha256:" + "a" * 64, None]), ""
                        )
                    self.assertEqual(args[args.index("--network") + 1], "none")
                    self.assertEqual(args[args.index("--cap-drop") + 1], "ALL")
                    self.assertEqual(args[args.index("--pull") + 1], "never")
                    self.assertLessEqual(kwargs["timeout"], 20)
                    identity = next(root.glob(".atb-storage-*/identity")).read_text()
                    if observation == "timeout":
                        raise RuntimeFailure("timeout")
                    payload = {
                        "identity": identity
                        if observation != "different_identity"
                        else "wrong",
                        "capacity": {
                            "filesystem_bytes": (
                                4 if observation == "different_capacity" else 8
                            )
                            * 2**20,
                            "filesystem_inodes": 1024,
                        },
                    }
                    return subprocess.CompletedProcess(
                        args,
                        0,
                        "bad" if observation == "bad_json" else json.dumps(payload),
                        "",
                    )

                with bounded_filesystem(root), patch.object(
                    runtime, "_run", run
                ), patch.object(runtime, "stop_container", stopped.append):
                    if observation == "valid":
                        runtime.configure_operator_storage(
                            root=root, max_bytes=8 * 2**20, probe_image="fixture"
                        )
                        self.assertEqual(
                            runtime._operator_storage.manifest()[
                                "max_episode_storage_bytes"
                            ],
                            8 * 2**20,
                        )
                    else:
                        with self.assertRaises(RuntimeFailure):
                            runtime.configure_operator_storage(
                                root=root, max_bytes=8 * 2**20, probe_image="fixture"
                            )
                        self.assertIsNone(runtime._operator_storage)
                self.assertEqual(len(stopped), 1)
                self.assertEqual(list(root.iterdir()), [])

    def test_image_volumes_are_rejected_and_safe_images_are_pinned(self):
        runtime = DockerRuntime(docker_binary="docker")
        with patch.object(runtime, "_run") as run:
            self.assertEqual(runtime._storage_safe_image("mutable:tag"), "mutable:tag")
            run.assert_not_called()
        identity = "sha256:" + "a" * 64
        for volumes in (None, {}, {"/extra": {}}, {"/workspace/nested": {}}, []):
            with self.subTest(volumes=volumes), patch.object(
                runtime,
                "_run",
                return_value=subprocess.CompletedProcess(
                    [], 0, json.dumps([identity, volumes]), ""
                ),
            ) as run:
                if volumes is None or volumes == {}:
                    self.assertEqual(
                        runtime._storage_safe_image(
                            "mutable:tag", required=True, timeout_seconds=3
                        ),
                        identity,
                    )
                else:
                    with self.assertRaises(RuntimeFailure):
                        runtime._storage_safe_image(
                            "mutable:tag", required=True, timeout_seconds=3
                        )
                self.assertEqual(run.call_args.kwargs["timeout"], 3)
                self.assertEqual(run.call_args.args[0][:2], ["image", "inspect"])
        with patch.object(
            runtime,
            "_run",
            return_value=subprocess.CompletedProcess([], 0, "not JSON", ""),
        ):
            with self.assertRaisesRegex(RuntimeFailure, "cannot verify"):
                runtime._storage_safe_image("mutable:tag", required=True)


if __name__ == "__main__":
    unittest.main()
