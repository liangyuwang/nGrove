import argparse
import io
import json
import math
import tempfile
import unittest
import uuid
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from autotrain_bench.cli import (
    _episode_exit_code,
    _initialize_agent_workspace,
    _resolve_challenge_run,
    _snapshot_agent_workspace,
    _terminal_agent_action,
    _validate_workspace_boundary,
    build_parser,
    command_report,
    command_start,
)
from autotrain_bench.config import ConfigError, TaskConfig


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tests/fixtures/task/task.json"
LM_PRETRAIN_PREVIEW = ROOT / "challenges/lm-pretrain-v1/challenge.json"


class WorkspaceSnapshotTests(unittest.TestCase):
    def test_episode_exit_code_requires_a_successful_test_outcome(self):
        self.assertEqual(_episode_exit_code({"outcome": "success"}), 0)
        self.assertEqual(_episode_exit_code({"outcome": "failed"}), 1)
        self.assertEqual(_episode_exit_code({"outcome": None}), 1)
        self.assertEqual(_episode_exit_code({"status": "finalized"}), 1)

    def test_report_migrates_a_legacy_success_without_rewriting_state(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory)
            original = {
                "protocol_version": 1,
                "status": "finalized",
                "final_candidate_id": "sha256:candidate",
                "test_nll": 1.5,
                "failure": None,
            }
            state_path = episode / "state.json"
            state_path.write_text(json.dumps(original), encoding="utf-8")
            output = io.StringIO()

            with redirect_stdout(output):
                return_code = command_report(argparse.Namespace(episode=episode))

            report = json.loads(output.getvalue())
            self.assertEqual(return_code, 0)
            self.assertEqual(report["result_schema_version"], 2)
            self.assertEqual(report["migrated_from_result_schema_version"], 1)
            self.assertEqual(report["outcome"], "success")
            self.assertEqual(report["test_nll"], 1.5)
            self.assertAlmostEqual(report["test_perplexity"], math.exp(1.5))
            self.assertEqual(
                json.loads(state_path.read_text(encoding="utf-8")), original
            )

    def test_report_migrates_a_legacy_unscored_result_to_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory)
            (episode / "state.json").write_text(
                json.dumps(
                    {
                        "protocol_version": 1,
                        "status": "finalized",
                        "final_candidate_id": None,
                        "test_nll": None,
                        "failure": None,
                    }
                ),
                encoding="utf-8",
            )
            output = io.StringIO()

            with redirect_stdout(output):
                command_report(argparse.Namespace(episode=episode))

            report = json.loads(output.getvalue())
            self.assertEqual(report["outcome"], "failed")
            self.assertEqual(report["failure"], "no_candidate")
            self.assertIsNone(report["test_nll"])
            self.assertIsNone(report["test_perplexity"])

    def test_report_rejects_an_unknown_result_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory)
            (episode / "state.json").write_text(
                json.dumps({"result_schema_version": 999}), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "unsupported result schema"):
                command_report(argparse.Namespace(episode=episode))

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

    def test_episode_workspace_must_be_outside_trusted_checkout(self):
        workspace = ROOT / ".autotrain" / "episodes" / "episode" / "workspace"
        with self.assertRaisesRegex(ConfigError, "outside the trusted"):
            _validate_workspace_boundary(
                workspace=workspace,
                task_source=TASK,
                agent_workspace=None,
            )

    def test_external_participant_project_instructions_are_outside_core_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "participant-project"
            (project / ".git").mkdir(parents=True)
            (project / "AGENTS.md").write_text(
                "participant instructions\n", encoding="utf-8"
            )
            workspace = project / "runs" / "episode" / "workspace"
            _validate_workspace_boundary(
                workspace=workspace,
                task_source=TASK,
                agent_workspace=None,
            )

    def test_dedicated_external_workspace_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "runs" / "episode" / "workspace"
            _validate_workspace_boundary(
                workspace=workspace,
                task_source=TASK,
                agent_workspace=None,
            )

    def test_participant_workspace_must_not_come_from_core(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "runs" / "episode" / "workspace"
            with self.assertRaisesRegex(ConfigError, "participant-owned"):
                _validate_workspace_boundary(
                    workspace=workspace,
                    task_source=TASK,
                    agent_workspace=ROOT / "participant-agent",
                )

    def test_participant_workspace_must_not_contain_the_core(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "runs" / "episode" / "workspace"
            with self.assertRaisesRegex(ConfigError, "must not contain"):
                _validate_workspace_boundary(
                    workspace=workspace,
                    task_source=TASK,
                    agent_workspace=ROOT.parent,
                )

    def test_non_git_task_directory_is_a_trusted_workspace_root(self):
        with tempfile.TemporaryDirectory() as directory:
            task_root = Path(directory) / "task-package"
            task_root.mkdir()
            task_source = task_root / "task.json"
            task_source.write_text("{}", encoding="utf-8")
            workspace = task_root / "episodes" / "run" / "workspace"

            with self.assertRaisesRegex(ConfigError, "outside the trusted"):
                _validate_workspace_boundary(
                    workspace=workspace,
                    task_source=task_source,
                    agent_workspace=None,
                )

    def test_participant_owned_instructions_are_allowed_in_snapshot_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "participant"
            source.mkdir()
            (source / "AGENTS.md").write_text(
                "participant instructions\n", encoding="utf-8"
            )
            workspace = root / "runs" / "episode" / "workspace"

            _validate_workspace_boundary(
                workspace=workspace,
                task_source=TASK,
                agent_workspace=source,
            )

    def test_start_rejects_core_workspace_before_worker_or_directory_creation(self):
        episode = ROOT / ".autotrain" / f"preflight-{uuid.uuid4().hex}"
        args = build_parser().parse_args(
            [
                "start",
                "--task",
                str(TASK),
                "--track",
                "A",
                "--episode-dir",
                str(episode),
                "--agent-runner",
                "host_cli",
                "--agent-command",
                "example-agent",
            ]
        )

        with patch("autotrain_bench.cli.DockerRuntime") as runtime:
            with self.assertRaisesRegex(ConfigError, "outside the trusted"):
                command_start(args)
        runtime.assert_not_called()
        self.assertFalse(episode.exists())

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
                "--episodes-root",
                "/dedicated/episodes",
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

    def test_challenge_profiles_replace_ad_hoc_track_and_hardware_flags(self):
        args = build_parser().parse_args(
            [
                "start",
                "--challenge",
                "challenge.json",
                "--track-config",
                "preview-b-6x1h",
                "--runtime-profile",
                "h100-sxm-80gb-x4",
                "--agent-runner",
                "host_cli",
                "--agent-command",
                "example-agent",
                "--episodes-root",
                "/dedicated/episodes",
            ]
        )
        self.assertIsNone(args.track)
        self.assertIsNone(args.task)
        self.assertEqual(args.track_config, "preview-b-6x1h")
        self.assertEqual(args.runtime_profile, "h100-sxm-80gb-x4")
        self.assertEqual(args.training_gpus, 0)

        task = replace(
            TaskConfig.load(TASK),
            task_id="lm-pretrain-v1",
            task_kind="benchmark_challenge",
        ).validate()
        args.challenge = LM_PRETRAIN_PREVIEW
        resolved = _resolve_challenge_run(args, task)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.track, "B")
        self.assertEqual(resolved.task.track_b_max_leases, 6)
        self.assertEqual(resolved.resources.nvidia_gpu_count, 4)

        args.training_gpus = 4
        with self.assertRaisesRegex(ConfigError, "may not override"):
            _resolve_challenge_run(args, task)

    def test_challenge_info_has_a_dedicated_command(self):
        args = build_parser().parse_args(
            ["challenge-info", "--challenge", "challenge.json"]
        )
        self.assertEqual(args.challenge, Path("challenge.json"))

    def test_start_requires_one_explicit_episode_destination(self):
        argv = [
            "start",
            "--task",
            "task.json",
            "--track",
            "A",
            "--agent-runner",
            "host_cli",
            "--agent-command",
            "example-agent",
        ]
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(argv)

        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(
                    [
                        *argv,
                        "--episode-dir",
                        "/exact/episode",
                        "--episodes-root",
                        "/episodes",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
