import tempfile
import unittest
from pathlib import Path

from autotrain_bench.cli import _snapshot_agent_workspace, _terminal_agent_action
from autotrain_bench.config import ConfigError


class WorkspaceSnapshotTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
