import tempfile
import unittest
from pathlib import Path

from autotrain_bench.cli import (
    _initialize_agent_workspace,
    _snapshot_agent_workspace,
    _terminal_agent_action,
    build_parser,
)
from autotrain_bench.config import ConfigError


class WorkspaceSnapshotTests(unittest.TestCase):
    def test_default_episode_workspace_is_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "episode-workspace"
            _initialize_agent_workspace(None, destination)
            self.assertTrue(destination.is_dir())
            self.assertEqual(list(destination.iterdir()), [])

    def test_episode_workspace_is_an_independent_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "episode" / "workspace"
            source.mkdir()
            (source / "agent.py").write_text("initial\n", encoding="utf-8")
            (source / "agent-link.py").symlink_to("agent.py")

            _snapshot_agent_workspace(source, destination)
            (destination / "agent.py").write_text("changed\n", encoding="utf-8")

            self.assertEqual(
                (source / "agent.py").read_text(encoding="utf-8"), "initial\n"
            )
            self.assertTrue((destination / "agent-link.py").is_symlink())

    def test_episode_workspace_cannot_be_nested_in_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            with self.assertRaises(ConfigError):
                _snapshot_agent_workspace(source, source / "episode")

    def test_host_agent_can_exit_during_drain(self):
        self.assertEqual(
            _terminal_agent_action(
                runner="host_cli", running=True, drain_elapsed_seconds=29.9
            ),
            "wait",
        )
        self.assertEqual(
            _terminal_agent_action(
                runner="host_cli", running=False, drain_elapsed_seconds=29.9
            ),
            "exited",
        )

    def test_host_agent_drain_has_a_bounded_timeout(self):
        self.assertEqual(
            _terminal_agent_action(
                runner="host_cli", running=True, drain_elapsed_seconds=30.0
            ),
            "drain_timeout",
        )
        self.assertEqual(
            _terminal_agent_action(
                runner="container", running=True, drain_elapsed_seconds=3.0
            ),
            "drain_timeout",
        )

    def test_declare_incumbent_command_is_explicit(self):
        args = build_parser().parse_args(
            ["agent", "declare-incumbent", "sha256:candidate"]
        )
        self.assertEqual(args.agent_command, "declare-incumbent")
        self.assertEqual(args.candidate_id, "sha256:candidate")

    def test_training_resources_are_episode_arguments(self):
        args = build_parser().parse_args(
            [
                "start",
                "--task",
                "task.json",
                "--track",
                "A",
                "--agent-runner",
                "host_cli",
                "--agent-command",
                "example-agent",
                "--training-gpus",
                "2",
                "--training-gpu-model",
                "H100",
                "--training-device",
                "/dev/accel0",
            ]
        )
        self.assertEqual(args.training_gpus, 2)
        self.assertEqual(args.training_gpu_model, "H100")
        self.assertEqual(args.training_devices, ["/dev/accel0"])


if __name__ == "__main__":
    unittest.main()
