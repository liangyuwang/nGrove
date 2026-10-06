import argparse
import io
import json
import shutil
import tempfile
import unittest
import uuid
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from envcore.cli import (
    _episode_exit_code,
    _initialize_agent_workspace,
    _report_cleanup_failures,
    _resolve_challenge_run,
    _run_cleanup_steps,
    _stop_and_record_container_agent,
    _stop_and_record_host_agent,
    _stop_and_record_startup_container_agent,
    _snapshot_agent_workspace,
    _terminal_agent_action,
    _validate_workspace_boundary,
    build_parser,
    command_report,
    command_prepare,
    command_start,
    command_worker_recover,
)
from envcore.config import ConfigError, TaskConfig
from envcore.host_agent import summarize_host_agent_trace
from envcore.runtime import DockerRuntime, RuntimeFailure
from envcore.training_data import TrainingDataError, verify_training_data
from envcore.results import normalize_result_report


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tests/fixtures/task/task.json"
LM_PRETRAIN_PREVIEW = ROOT / "challenges/lm-pretrain/challenge.json"


class WorkspaceSnapshotTests(unittest.TestCase):
    def test_manual_recovery_timeout_is_forwarded_and_incomplete_cleanup_fails(self):
        report = {
            "cleanup_complete": False,
            "deadline_exceeded": True,
            "resumed": False,
        }
        args = build_parser().parse_args(
            [
                "worker-recover",
                "--state-dir",
                "/unused-worker",
                "--timeout-seconds",
                "45",
            ]
        )
        output = io.StringIO()
        with patch("envcore.cli.DockerRuntime") as runtime, redirect_stdout(output):
            runtime.return_value.recover_resources.return_value = report
            self.assertEqual(command_worker_recover(args), 2)
        runtime.return_value.recover_resources.assert_called_once_with(
            Path("/unused-worker"), timeout_seconds=45
        )
        self.assertEqual(json.loads(output.getvalue()), report)
        defaults = build_parser().parse_args(
            ["worker-recover", "--state-dir", "/unused-worker"]
        )
        self.assertEqual(defaults.timeout_seconds, 120)

    def test_invalid_manual_recovery_timeout_is_rejected_before_runtime_setup(self):
        for value in ("nan", "inf", "0", "-1"):
            args = build_parser().parse_args(
                [
                    "worker-recover",
                    "--state-dir",
                    "/unused-worker",
                    f"--timeout-seconds={value}",
                ]
            )
            with self.subTest(value=value), patch(
                "envcore.cli.DockerRuntime"
            ) as runtime:
                with self.assertRaisesRegex(RuntimeFailure, "finite positive"):
                    command_worker_recover(args)
                runtime.assert_not_called()

    def test_inactive_invalid_defaults_do_not_block_start(self):
        for track, time_default, storage_default, flags in (
            ("B", 1800, "invalid", ["--no-operator-storage-limit"]),
            ("B", 1800, "invalid", ["--operator-max-storage-bytes", "4096"]),
            ("B", "invalid", 2**30, ["--operator-max-research-seconds", "60"]),
            ("B", "invalid", 2**30, ["--no-operator-research-limit"]),
            ("A", "invalid", 2**30, []),
        ):
            with self.subTest(
                track=track, flags=flags
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "task/task.json"
                source.parent.mkdir()
                source.with_name("operator-defaults.json").write_text(
                    json.dumps(
                        {
                            "track_b_max_research_seconds": time_default,
                            "max_episode_storage_bytes": storage_default,
                        }
                    )
                )
                config = replace(TaskConfig.load(TASK), source=source)
                episode = root / "episode"
                args = build_parser().parse_args(
                    [
                        "start",
                        "--task",
                        str(TASK),
                        "--track",
                        track,
                        "--episode-dir",
                        str(episode),
                        "--agent-runner",
                        "container",
                        "--agent-image",
                        "fixture@sha256:" + "a" * 64,
                        "--agent-command",
                        "participant-agent",
                        *flags,
                    ]
                )
                with patch("envcore.cli.TaskConfig.load", return_value=config), patch(
                    "envcore.cli.DockerRuntime"
                ) as runtime, redirect_stdout(io.StringIO()):
                    runtime.return_value.prepare.side_effect = ConfigError(
                        "reached Worker preparation"
                    )
                    with self.assertRaisesRegex(
                        ConfigError, "reached Worker preparation"
                    ):
                        command_start(args)
                    runtime.return_value.prepare.assert_called_once()
                    runtime.return_value.start_agent.assert_not_called()
                self.assertFalse(episode.exists())

    def test_storage_admission_is_private_and_precedes_episode_writes(self):
        for folder, expected in (
            ("challenges/lm-pretrain", 100 * 2**30),
            ("dev_tasks/smoke_lm-pretrain", 2 * 2**30),
        ):
            for track in ("A", "B"):
                for flags, limit in (
                    ([], expected),
                    (["--operator-max-storage-bytes", "1048576"], 1048576),
                    (["--no-operator-storage-limit"], None),
                ):
                    with self.subTest(
                        folder=folder, track=track, flags=flags
                    ), tempfile.TemporaryDirectory() as directory:
                        episode = Path(directory) / "episode"
                        config = replace(
                            TaskConfig.load(TASK), source=ROOT / folder / "task.json"
                        )
                        public_before = config.public_manifest(track)
                        args = build_parser().parse_args(
                            [
                                "start",
                                "--task",
                                str(TASK),
                                "--track",
                                track,
                                "--episode-dir",
                                str(episode),
                                "--agent-runner",
                                "container",
                                "--agent-image",
                                "fixture@sha256:" + "a" * 64,
                                "--agent-command",
                                "participant-agent",
                                *flags,
                            ]
                        )
                        output = io.StringIO()

                        def admit(**kwargs):
                            self.assertFalse(episode.exists())
                            self.assertEqual(kwargs["root"], episode.parent.resolve())
                            self.assertEqual(kwargs["max_bytes"], limit)
                            self.assertIn(
                                "not sent to the Tested Agent", output.getvalue()
                            )
                            self.assertIn(str(limit), output.getvalue())
                            self.assertIn(
                                "task default" if not flags else "run override",
                                output.getvalue(),
                            )
                            self.assertIn(
                                "Change it before launch with --operator-max-storage-bytes BYTES",
                                output.getvalue(),
                            )
                            self.assertIn(
                                "--no-operator-storage-limit", output.getvalue()
                            )

                        with patch(
                            "envcore.cli.TaskConfig.load", return_value=config
                        ), patch(
                            "envcore.cli.DockerRuntime"
                        ) as runtime, redirect_stdout(output):
                            runtime.return_value.configure_operator_storage.side_effect = admit
                            runtime.return_value.prepare.side_effect = ConfigError(
                                "stop after admission"
                            )
                            with self.assertRaisesRegex(
                                ConfigError, "stop after admission"
                            ):
                                command_start(args)
                            self.assertEqual(
                                runtime.return_value.configure_operator_storage.call_count,
                                int(limit is not None),
                            )
                            runtime.return_value.start_agent.assert_not_called()
                        self.assertFalse(episode.exists())
                        self.assertEqual(config.public_manifest(track), public_before)

    def test_storage_rejection_never_launches_or_creates_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory) / "episode"
            for runner in ("host_cli", "container"):
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
                        runner,
                        "--agent-command",
                        "participant-agent",
                        "--operator-max-storage-bytes",
                        "1048576",
                    ]
                )
                with patch("envcore.cli.DockerRuntime") as runtime, redirect_stdout(
                    io.StringIO()
                ):
                    runtime.return_value.configure_operator_storage.side_effect = (
                        RuntimeFailure("unbounded Worker storage")
                    )
                    with self.assertRaises((ConfigError, RuntimeFailure)):
                        command_start(args)
                    runtime.return_value.prepare.assert_not_called()
                    runtime.return_value.start_agent.assert_not_called()
                    runtime.return_value.start_host_agent.assert_not_called()
                self.assertFalse(episode.exists())

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "start",
                    "--task",
                    str(TASK),
                    "--track",
                    "A",
                    "--episode-dir",
                    "/unused",
                    "--agent-runner",
                    "host_cli",
                    "--agent-command",
                    "participant-agent",
                    "--operator-max-storage-bytes",
                    "1024",
                    "--no-operator-storage-limit",
                ]
            )

    def test_start_resolves_bundled_operator_defaults_and_announces_before_preflight(
        self,
    ):
        for folder, lane, flags, expected in (
            ("challenges/lm-pretrain", "preview-b-6x1h", [], 86400),
            ("dev_tasks/smoke_lm-pretrain", "smoke-b-3x75s", [], 1800),
            (
                "dev_tasks/smoke_lm-pretrain",
                "smoke-b-3x75s",
                ["--operator-max-research-seconds", "60"],
                60,
            ),
            (
                "dev_tasks/smoke_lm-pretrain",
                "smoke-b-3x75s",
                ["--no-operator-research-limit"],
                None,
            ),
            ("dev_tasks/smoke_lm-pretrain", "smoke-a-5m", [], None),
        ):
            with self.subTest(
                folder=folder, lane=lane, flags=flags
            ), tempfile.TemporaryDirectory() as directory:
                root = ROOT / folder
                payload = json.loads((root / "task.json").read_text())
                config = replace(
                    TaskConfig.load(TASK),
                    source=root / "task.json",
                    task_id=payload["task_id"],
                )
                episode = Path(directory) / "episode"
                args = build_parser().parse_args(
                    [
                        "start",
                        "--challenge",
                        str(root / "challenge.json"),
                        "--track-config",
                        lane,
                        "--runtime-profile",
                        "a100-sxm4-80gb-x4",
                        "--episode-dir",
                        str(episode),
                        "--agent-runner",
                        "host_cli",
                        "--agent-command",
                        "participant-agent",
                        "--no-operator-storage-limit",
                        *flags,
                    ]
                )
                output = io.StringIO()

                def check_notice(*args, **kwargs):
                    notice = output.getvalue()
                    if "-a-" in lane:
                        self.assertNotIn("operator research-time protection", notice)
                    else:
                        self.assertIn("extremely high", notice)
                        self.assertIn("token consumption", notice)
                        self.assertIn("not sent to the Tested Agent", notice)
                        if expected is not None:
                            self.assertIn(f"{expected} seconds", notice)
                            self.assertIn("--no-operator-research-limit", notice)
                        else:
                            self.assertIn("protection is disabled", notice)

                with patch("envcore.cli.TaskConfig.load", return_value=config), patch(
                    "envcore.cli.DockerRuntime"
                ) as runtime, patch(
                    "envcore.cli.Supervisor",
                    side_effect=ConfigError("stop after preflight"),
                ) as supervisor, redirect_stdout(output):
                    runtime.return_value.prepare.side_effect = check_notice
                    with self.assertRaisesRegex(ConfigError, "stop after preflight"):
                        command_start(args)
                    runtime.return_value.prepare.assert_called_once()
                    self.assertEqual(
                        supervisor.call_args.kwargs["operator_max_research_seconds"],
                        expected,
                    )
                    runtime.return_value.start_agent.assert_not_called()
                    runtime.return_value.start_host_agent.assert_not_called()
                self.assertFalse(episode.exists())

    def test_operator_limit_override_and_disable_are_mutually_exclusive(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "start",
                    "--task",
                    str(TASK),
                    "--track",
                    "B",
                    "--agent-runner",
                    "host_cli",
                    "--agent-command",
                    "participant-agent",
                    "--operator-max-research-seconds",
                    "30",
                    "--no-operator-research-limit",
                ]
            )

    def test_operator_limit_rejects_invalid_values_and_track_a_before_writes(self):
        for track, value in (
            ("B", "0"),
            ("B", "-1"),
            ("B", "nan"),
            ("B", "inf"),
            ("A", "30"),
            ("A", None),
        ):
            with self.subTest(
                track=track, value=value
            ), tempfile.TemporaryDirectory() as directory:
                episode = Path(directory) / "episode"
                args = build_parser().parse_args(
                    [
                        "start",
                        "--task",
                        str(TASK),
                        "--track",
                        track,
                        "--episode-dir",
                        str(episode),
                        "--agent-runner",
                        "host_cli",
                        "--agent-command",
                        "participant-agent",
                        f"--operator-max-research-seconds={value}"
                        if value is not None
                        else "--no-operator-research-limit",
                    ]
                )
                with patch("envcore.cli.DockerRuntime") as runtime:
                    with self.assertRaises(ConfigError):
                        command_start(args)
                    runtime.assert_not_called()
                self.assertFalse(episode.exists())

    def test_operator_guard_stops_local_runners_and_gateway_without_evaluation(self):
        # Exercise the real launcher/state machine with fake workload boundaries;
        # no Agent, Docker daemon, inference endpoint, or hidden data is executed.
        for runner in ("container", "host_cli"):
            running = set()
            stops = []

            class LocalRuntime(DockerRuntime):
                def enable_recovery(self, **kwargs):
                    pass  # Workloads and daemon boundaries are faked in this test.

                def prepare(self, *args, **kwargs):
                    pass

                def environment_manifest(self, *args, **kwargs):
                    return {}

                def start_agent(self, *, name, **kwargs):
                    running.add(name)

                def start_host_agent(self, *, stdout_path, **kwargs):
                    stdout_path.write_text("")

                    class Host:
                        running = True

                        def summarize_usage(self):
                            return summarize_host_agent_trace(self.stdout_path)

                        def stop(self):
                            self.running = False
                            stops.append("host")
                            return 0

                    host = Host()
                    host.stdout_path = stdout_path
                    return host

                def create_network(self, *args, **kwargs):
                    pass

                def remove_network(self, *args, **kwargs):
                    pass

                def start_inference_gateway(self, *, name, **kwargs):
                    running.add(name)

                def container_running(self, name):
                    return name in running

                def container_exit_code(self, name):
                    return None

                def container_logs(self, name, **kwargs):
                    return ""

                def stop_container(self, name):
                    stops.append(name)
                    running.discard(name)

                def evaluate_candidate(self, **kwargs):
                    raise AssertionError("operator stop must not run hidden evaluation")

            with self.subTest(
                runner=runner
            ), tempfile.TemporaryDirectory() as directory:
                episode = Path(directory) / "episode"
                argv = [
                    "start",
                    "--task",
                    str(TASK),
                    "--track",
                    "B",
                    "--episode-dir",
                    str(episode),
                    "--agent-runner",
                    runner,
                    "--agent-command",
                    "participant-agent",
                    "--operator-max-research-seconds",
                    "0.05",
                ]
                if runner == "container":
                    argv += [
                        "--agent-image",
                        "participant:test",
                        "--inference-base-url",
                        "https://inference.invalid",
                        "--inference-model",
                        "registered-model",
                    ]
                args = build_parser().parse_args(argv)
                with patch("envcore.cli.DockerRuntime", LocalRuntime), redirect_stdout(
                    io.StringIO()
                ):
                    self.assertEqual(command_start(args), 1)
                state = json.loads((episode / "state.json").read_text())
                report = normalize_result_report(state)
                self.assertEqual(report["outcome"], "aborted")
                self.assertEqual(
                    report["agent_system"]["termination_reason"],
                    "operator_time_limit_reached",
                )
                self.assertTrue(report["agent_control"]["cleanup_complete"])
                self.assertEqual(report["deployment_teardown_failures"], [])
                self.assertIsNone(report["test_status"])
                self.assertIsNone(report["analysis_status"])
                self.assertFalse(running)
                if runner == "container":
                    self.assertTrue(stops[0].endswith("-inference"))
                else:
                    self.assertIn("host", stops)

    def test_prepare_rejects_invalid_train_before_runtime_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            config = TaskConfig.load(TASK)
            train = Path(directory) / "train"
            shutil.copytree(config.train_dir, train)
            (train / "train.json").write_bytes(b"corrupt")
            args = build_parser().parse_args(["prepare", "--task", str(TASK)])
            with patch(
                "envcore.cli.TaskConfig.load",
                return_value=replace(config, train_dir=train),
            ):
                with patch("envcore.cli.DockerRuntime") as runtime:
                    with self.assertRaises(TrainingDataError):
                        command_prepare(args)
                    runtime.assert_not_called()

    def test_start_rechecks_train_and_cleans_up_before_agent_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episode"
            config = TaskConfig.load(TASK)
            train = root / "train"
            shutil.copytree(config.train_dir, train)
            (train / "train.json").write_bytes(b"corrupt since prepare")
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
            with patch(
                "envcore.cli.TaskConfig.load",
                return_value=replace(config, train_dir=train),
            ):
                with patch("envcore.cli.DockerRuntime") as runtime:
                    with self.assertRaises(TrainingDataError):
                        command_start(args)
                    runtime.assert_not_called()
            self.assertFalse(episode.exists())

    def test_start_passes_verified_data_evidence_to_core_before_reveal(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory) / "episode"
            config = TaskConfig.load(TASK)
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
                    "example:test-agent",
                    "--agent-command",
                    "example-agent",
                ]
            )
            with patch(
                "envcore.cli.verify_training_data", wraps=verify_training_data
            ) as verify, patch("envcore.cli.DockerRuntime") as runtime, patch(
                "envcore.cli.Supervisor",
                side_effect=RuntimeError("stop before task reveal"),
            ) as supervisor:
                with self.assertRaisesRegex(RuntimeError, "stop before task reveal"):
                    command_start(args)
                verify.assert_called_once_with(config.train_dir)
                self.assertEqual(
                    supervisor.call_args.kwargs["training_data_verification"],
                    verify_training_data(config.train_dir),
                )
                runtime.return_value.start_agent.assert_not_called()
            self.assertFalse(episode.exists())

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

        with patch("envcore.cli.DockerRuntime") as runtime:
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

            with patch("envcore.cli.DockerRuntime", return_value=runtime):
                with patch.object(runtime, "prepare"), patch.object(
                    runtime, "enable_recovery"
                ):
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

    def test_host_termination_records_capture_evidence_after_stop(self):
        calls = []
        usage = {
            "usage_complete": False,
            "total_tokens": 25,
            "log_capture": {"stdout": {"complete": False}},
        }

        class Host:
            def stop(self):
                calls.append("stop")
                return 0

            def summarize_usage(self):
                calls.append("usage")
                return usage

        supervisor = Mock()
        supervisor.record_agent_execution.side_effect = lambda **kwargs: calls.append(
            "record"
        )
        _stop_and_record_host_agent(
            supervisor=supervisor,
            host_agent=Host(),
            termination_reason="exited",
            drain_seconds=0.5,
        )
        self.assertEqual(calls, ["stop", "usage", "record"])
        supervisor.record_agent_execution.assert_called_once_with(
            exit_code=0, usage=usage, termination_reason="exited", drain_seconds=0.5
        )

    def test_host_termination_failure_does_not_record_an_exit(self):
        host = Mock()
        host.stop.side_effect = RuntimeError("termination unconfirmed")
        supervisor = Mock()
        with self.assertRaisesRegex(RuntimeError, "termination unconfirmed"):
            _stop_and_record_host_agent(
                supervisor=supervisor,
                host_agent=host,
                termination_reason="exited",
                drain_seconds=0.5,
            )
        host.summarize_usage.assert_not_called()
        supervisor.record_agent_execution.assert_not_called()

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
            task_id="lm-pretrain",
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
