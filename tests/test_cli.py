import argparse
import io
import json
import tempfile
import unittest
import uuid
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from autotrain_bench.cli import (
    _episode_exit_code,
    _initialize_agent_workspace,
    _report_cleanup_failures,
    _resolve_challenge_run,
    _run_cleanup_steps,
    _stop_and_record_container_agent,
    _stop_and_record_startup_container_agent,
    _snapshot_agent_workspace,
    _terminal_agent_action,
    _validate_workspace_boundary,
    build_parser,
    command_report,
    command_start,
)
from autotrain_bench.config import ConfigError, TaskConfig
from autotrain_bench.runtime import DockerRuntime, RuntimeFailure


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tests/fixtures/task/task.json"
LM_PRETRAIN_PREVIEW = ROOT / "challenges/lm-pretrain-v1/challenge.json"


class WorkspaceSnapshotTests(unittest.TestCase):
    def test_episode_exit_code_requires_a_successful_test_outcome(self):
        self.assertEqual(
            _episode_exit_code({"status": "finalized", "outcome": "success"}), 0
        )
        self.assertEqual(_episode_exit_code({"outcome": "success"}), 1)
        self.assertEqual(_episode_exit_code({"outcome": "failed"}), 1)
        self.assertEqual(_episode_exit_code({"outcome": None}), 1)
        self.assertEqual(_episode_exit_code({"status": "finalized"}), 1)
        self.assertEqual(
            _episode_exit_code(
                {
                    "status": "finalized",
                    "outcome": "success",
                    "deployment_teardown_failures": ["network_teardown_failure"],
                }
            ),
            1,
        )

    def test_removed_agent_mode_flag_reports_the_egress_migration(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "start",
                "--task",
                str(TASK),
                "--track",
                "A",
                "--episodes-root",
                "/tmp/does-not-matter",
                "--agent-runner",
                "container",
                "--agent-image",
                "example:test",
                "--agent-command",
                "example",
                "--agent-mode",
                "sandboxed",
            ]
        )
        self.assertEqual(args.agent_mode, "sandboxed")
        with self.assertRaisesRegex(ConfigError, "--agent-egress"):
            command_start(args)

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

    def test_keyboard_interrupt_during_agent_start_runs_compensating_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory) / "episode"
            args = build_parser().parse_args(
                [
                    "start",
                    "--task",
                    str(TASK),
                    "--track",
                    "B",
                    "--episode-dir",
                    str(episode),
                    "--agent-runner",
                    "container",
                    "--agent-image",
                    f"example/agent@sha256:{'a' * 64}",
                    "--agent-command",
                    "example-agent",
                ]
            )
            runtime = DockerRuntime()

            with patch("autotrain_bench.cli.DockerRuntime", return_value=runtime):
                with patch.object(runtime, "prepare"):
                    with patch.object(
                        runtime,
                        "environment_manifest",
                        return_value={},
                    ):
                        with patch.object(
                            runtime,
                            "start_agent",
                            side_effect=KeyboardInterrupt(),
                        ):
                            with patch.object(runtime, "stop_container") as stop:
                                with self.assertRaises(KeyboardInterrupt):
                                    command_start(args)

            stop.assert_called_once()
            state = json.loads((episode / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "infrastructure_failure")
            self.assertEqual(state["failure"], "supervisor_interrupted")
            self.assertEqual(
                state["agent_system"]["termination_reason"],
                "supervisor_interrupted",
            )
            self.assertTrue(state["agent_control"]["cleanup_complete"])

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

    def test_container_termination_is_recorded_only_after_verified_stop(self):
        calls = []

        class Runtime:
            def container_running(self, name):
                calls.append(("running", name))
                return True

            def container_exit_code(self, name):
                raise AssertionError(f"unexpected exit-code query for {name}")

            def stop_container(self, name):
                calls.append(("stop", name))

        class Supervisor:
            def record_agent_execution(self, **metadata):
                calls.append(("record", metadata))

            def infrastructure_failed(self, *args, **kwargs):
                raise AssertionError((args, kwargs))

        _stop_and_record_container_agent(
            runtime=Runtime(),
            supervisor=Supervisor(),
            name="agent",
            drain_seconds=3.0,
        )
        self.assertEqual([call[0] for call in calls], ["running", "stop", "record"])
        self.assertEqual(calls[-1][1]["termination_reason"], "drain_timeout")

    def test_unverified_container_stop_is_not_recorded(self):
        calls = []

        class Runtime:
            def container_running(self, name):
                del name
                return True

            def stop_container(self, name):
                del name
                raise RuntimeFailure("private Docker daemon detail")

        class Supervisor:
            def record_agent_execution(self, **metadata):
                raise AssertionError(metadata)

        with self.assertRaisesRegex(RuntimeFailure, "private Docker daemon detail"):
            _stop_and_record_container_agent(
                runtime=Runtime(),
                supervisor=Supervisor(),
                name="agent",
                drain_seconds=3.0,
            )
        self.assertEqual(calls, [])

    def test_startup_agent_is_recorded_only_after_confirmed_absence(self):
        calls = []

        class Runtime:
            def stop_container(self, name):
                calls.append(("stop", name))

        class Supervisor:
            def record_agent_execution(self, **metadata):
                calls.append(("record", metadata))

        _stop_and_record_startup_container_agent(
            runtime=Runtime(),
            supervisor=Supervisor(),
            name="uncertain-agent",
        )

        self.assertEqual([call[0] for call in calls], ["stop", "record"])
        self.assertEqual(calls[-1][1]["termination_reason"], "episode_startup_failure")

    def test_startup_agent_records_operator_interrupt_after_confirmed_absence(self):
        calls = []

        class Runtime:
            def stop_container(self, name):
                calls.append(("stop", name))

        class Supervisor:
            def record_agent_execution(self, **metadata):
                calls.append(("record", metadata))

        _stop_and_record_startup_container_agent(
            runtime=Runtime(),
            supervisor=Supervisor(),
            name="uncertain-agent",
            termination_reason="supervisor_interrupted",
        )

        self.assertEqual([call[0] for call in calls], ["stop", "record"])
        self.assertEqual(calls[-1][1]["termination_reason"], "supervisor_interrupted")

    def test_cleanup_steps_continue_after_an_earlier_failure(self):
        calls = []

        def fail_lease():
            calls.append("lease")
            raise RuntimeFailure("lease still running")

        def stop_agent():
            calls.append("agent")

        def stop_gateway():
            calls.append("gateway")

        failures = _run_cleanup_steps(
            [
                ("training_lease_termination_failure", fail_lease),
                ("agent_termination_failure", stop_agent),
                ("inference_gateway_termination_failure", stop_gateway),
            ]
        )

        self.assertEqual(calls, ["lease", "agent", "gateway"])
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0][0], "training_lease_termination_failure")
        self.assertIn("lease still running", failures[0][1])

    def test_cleanup_steps_continue_after_keyboard_interrupt(self):
        calls = []

        def interrupted_cleanup():
            calls.append("interrupted")
            raise KeyboardInterrupt()

        def later_cleanup():
            calls.append("later")

        failures = _run_cleanup_steps(
            [
                ("interrupted_cleanup", interrupted_cleanup),
                ("later_cleanup", later_cleanup),
            ]
        )

        self.assertEqual(calls, ["interrupted", "later"])
        self.assertEqual(failures, [("interrupted_cleanup", "KeyboardInterrupt: ")])

    def test_cleanup_steps_treat_an_explicit_false_result_as_incomplete(self):
        calls = []

        def incomplete_control_cleanup():
            calls.append("control")
            return False

        def remove_network():
            calls.append("network")

        failures = _run_cleanup_steps(
            [
                ("agent_control_teardown_failure", incomplete_control_cleanup),
                ("runtime_network_teardown_failure", remove_network),
            ]
        )

        self.assertEqual(calls, ["control", "network"])
        self.assertEqual(
            failures,
            [("agent_control_teardown_failure", "cleanup reported incomplete")],
        )

    def test_cleanup_failures_use_the_terminal_safe_supervisor_path(self):
        calls = []

        class Supervisor:
            def deployment_teardown_failed(self, code, *, private_detail=None):
                calls.append((code, private_detail))

            def infrastructure_failed(self, *args, **kwargs):
                raise AssertionError((args, kwargs))

        _report_cleanup_failures(
            Supervisor(),
            [
                (
                    "agent_termination_failure",
                    "RuntimeFailure: untrusted container remains",
                ),
                (
                    "runtime_network_teardown_failure",
                    "RuntimeFailure: network remains",
                ),
            ],
        )

        self.assertEqual(calls[0][0], "agent_termination_failure")
        self.assertIn("untrusted container remains", calls[0][1])
        self.assertEqual(calls[1][0], "runtime_network_teardown_failure")
        self.assertIn("network remains", calls[1][1])

    def test_cleanup_failure_reporting_attempts_every_record_after_an_error(self):
        calls = []

        class Supervisor:
            def deployment_teardown_failed(self, code, *, private_detail=None):
                calls.append((code, private_detail))
                if code == "first_failure":
                    raise OSError("state persistence failed")

        with self.assertRaisesRegex(OSError, "state persistence failed"):
            _report_cleanup_failures(
                Supervisor(),
                [
                    ("first_failure", "first detail"),
                    ("second_failure", "second detail"),
                ],
            )

        self.assertEqual(
            calls,
            [
                ("first_failure", "first detail"),
                ("second_failure", "second detail"),
            ],
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
