"""Run-owned environments, benchmark separation and public package boundaries."""

import configparser
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from envcore.agent_api_bundle import publish_agent_api_bundle
from envcore.challenge import ChallengeRelease
from envcore.cli import build_parser, command_start
from envcore.config import ConfigError, TaskConfig, TrainingResources
from envcore.environment import EnvironmentConfig, validate_environment_record
from envcore.results import ResultSchemaError, normalize_result_report
from envcore.run_config import resolve_environment_run
from envcore.supervisor import ProtocolError
from envcore.worker_protocol import (
    ResourceEnvelope,
    ResourceRequirement,
    EnforcementLevel,
    ResearchTaskSpec,
    WorkloadRequest,
    assess_resource_allocation,
    EvaluationResourceFailure,
)
from tests import test_supervisor
from tests.test_runtime import CapturingDockerRuntime


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tests/fixtures/task/task.json"
CHALLENGE = ROOT / "challenges/lm-pretrain/challenge.json"


def cpu_config():
    return {
        "schema_version": 1,
        "name": "local-cpu",
        "runtime_profile": {"host_count": 1, "accelerator": None},
    }


class EnvironmentTests(unittest.TestCase):
    def test_gpu_scaling_is_run_owned_and_does_not_modify_the_release(self):
        before = CHALLENGE.read_bytes()
        release = ChallengeRelease.load(CHALLENGE)
        task = replace(
            TaskConfig.load(TASK), source=release.task_path, task_id=release.task_id
        )
        profile = release.runtime_profiles["h100-sxm-80gb-x1"].public_manifest()
        identities = set()
        for count in (1, 4, 16):
            profile["accelerator"]["count"] = count
            environment = EnvironmentConfig.from_manifest(
                {
                    "schema_version": 1,
                    "name": "scaling",
                    "runtime_profile": profile,
                }
            )
            run = resolve_environment_run(
                task,
                environment=environment,
                release=release,
                track_config_id="preview-a-8h",
            )
            self.assertEqual(run.resources.nvidia_gpu_count, count)
            self.assertEqual(run.task.track_a_total_seconds, 28800)
            self.assertFalse(run.task.leaderboard_eligible)
            identities.add(environment.configuration_sha256)
        self.assertEqual(len(identities), 3)
        self.assertEqual(
            release.runtime_profiles["h100-sxm-80gb-x1"].accelerator_count, 1
        )
        self.assertEqual(CHALLENGE.read_bytes(), before)
        with self.assertRaisesRegex(ConfigError, "unknown runtime profile"):
            release.resolve(
                task,
                track_config_id="preview-a-8h",
                runtime_profile_id="scaling",
            )

    def test_config_identity_ignores_labels_but_binds_evaluation_resources(self):
        original = EnvironmentConfig.from_manifest(cpu_config())
        renamed = cpu_config()
        renamed["name"] = "renamed"
        renamed["runtime_profile"]["label"] = "Another label"
        self.assertEqual(
            original.configuration_sha256,
            EnvironmentConfig.from_manifest(renamed).configuration_sha256,
        )
        renamed["runtime_profile"]["evaluation_resource_contract"] = ResourceEnvelope(
            requirements=(
                ResourceRequirement(
                    "cpu.quota_cores", "exact", 2, EnforcementLevel.BEST_EFFORT
                ),
            ),
        ).public_manifest()
        self.assertNotEqual(
            original.configuration_sha256,
            EnvironmentConfig.from_manifest(renamed).configuration_sha256,
        )

    def test_typos_versions_and_mutated_records_fail_closed(self):
        for change in (
            lambda x: x.update(schema_version=True),
            lambda x: x.update(schema_version=2),
            lambda x: x.update(leaderboard_eligible=True),
            lambda x: x["runtime_profile"].update(gpu_count=4),
        ):
            raw = cpu_config()
            change(raw)
            with self.subTest(raw=raw), self.assertRaises(ConfigError):
                EnvironmentConfig.from_manifest(raw)
        record = EnvironmentConfig.from_manifest(cpu_config()).record()
        record["configuration_sha256"] = "0" * 64
        with self.assertRaisesRegex(ConfigError, "identity mismatch"):
            validate_environment_record(record)

    def test_benchmark_rejects_custom_environment_before_data_or_worker_setup(self):
        args = build_parser().parse_args(
            [
                "start",
                "--mode",
                "benchmark",
                "--task",
                str(TASK),
                "--track",
                "A",
                "--environment",
                "/does-not-exist.json",
                "--episode-dir",
                "/unused",
                "--agent-runner",
                "container",
                "--agent-command",
                "agent",
            ]
        )
        with patch("envcore.cli.DockerRuntime") as worker, patch(
            "envcore.cli.verify_training_data"
        ) as verify:
            with self.assertRaisesRegex(ConfigError, "benchmark mode cannot override"):
                command_start(args)
            worker.assert_not_called()
            verify.assert_not_called()

    def test_cli_delivers_environment_resources_before_task_reveal(self):
        release = ChallengeRelease.load(CHALLENGE)
        raw = {
            "schema_version": 1,
            "name": "custom",
            "runtime_profile": release.runtime_profiles[
                "h100-sxm-80gb-x1"
            ].public_manifest(),
        }
        raw["runtime_profile"]["accelerator"]["count"] = 16
        raw["budget"] = {"track": "A", "total_seconds": 7200}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "environment.json"
            path.write_text(json.dumps(raw))
            args = build_parser().parse_args(
                [
                    "start",
                    "--task",
                    str(TASK),
                    "--track",
                    "A",
                    "--environment",
                    str(path),
                    "--episode-dir",
                    str(root / "episode"),
                    "--agent-runner",
                    "container",
                    "--agent-image",
                    "agent@sha256:" + "a" * 64,
                    "--agent-command",
                    "participant-agent",
                ]
            )
            with patch("envcore.cli.DockerRuntime") as runtime, patch(
                "envcore.cli.Supervisor"
            ) as supervisor, redirect_stdout(io.StringIO()):
                runtime.return_value.prepare.side_effect = ConfigError(
                    "preparation reached"
                )
                with self.assertRaisesRegex(ConfigError, "preparation reached"):
                    command_start(args)
                self.assertEqual(
                    runtime.return_value.prepare.call_args.kwargs[
                        "resources"
                    ].nvidia_gpu_count,
                    16,
                )
                self.assertEqual(
                    runtime.return_value.prepare.call_args.args[
                        0
                    ].track_a_total_seconds,
                    7200,
                )
                self.assertEqual(
                    runtime.return_value.prepare.call_args.kwargs["agent"].environment,
                    [],
                )
                supervisor.assert_not_called()
                self.assertFalse((root / "episode").exists())

    def test_environment_info_needs_no_worker_or_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env.json"
            path.write_text(json.dumps(cpu_config()))
            args = build_parser().parse_args(
                ["environment-info", "--environment", str(path)]
            )
            with patch("envcore.cli.DockerRuntime") as worker, redirect_stdout(
                io.StringIO()
            ) as out:
                self.assertEqual(args.function(args), 0)
            record = json.loads(out.getvalue())
            self.assertEqual(
                validate_environment_record(record).profile.accelerator_count, 0
            )
            worker.assert_not_called()

    def test_environment_uses_task_budget_without_mutating_eligibility(self):
        task = replace(
            TaskConfig.load(TASK),
            task_kind="benchmark_challenge",
            leaderboard_eligible=True,
        )
        run = resolve_environment_run(
            task, environment=EnvironmentConfig.from_manifest(cpu_config()), track="B"
        )
        self.assertTrue(task.leaderboard_eligible)
        self.assertFalse(run.task.leaderboard_eligible)
        self.assertEqual(run.task.track_b_max_leases, task.track_b_max_leases)
        self.assertEqual(
            run.environment.budget.public_budget(), task.public_manifest("B")["budget"]
        )
        cuda = replace(task, evaluator_device="cuda")
        with self.assertRaisesRegex(ConfigError, "requires.*training-gpus"):
            resolve_environment_run(
                cuda,
                environment=EnvironmentConfig.from_manifest(cpu_config()),
                track="A",
            )

    def test_time_scaling_is_fingerprinted_and_preserves_task_and_release_files(self):
        task_bytes, release_bytes = TASK.read_bytes(), CHALLENGE.read_bytes()
        release = ChallengeRelease.load(CHALLENGE)
        task = replace(
            TaskConfig.load(TASK), source=release.task_path, task_id=release.task_id
        )
        fingerprints = set()
        for seconds in (3600, 7200, 28800):
            raw = cpu_config()
            raw["budget"] = {"track": "A", "total_seconds": seconds}
            environment = EnvironmentConfig.from_manifest(raw)
            run = resolve_environment_run(
                task,
                environment=environment,
                release=release,
                track_config_id="preview-a-8h",
            )
            self.assertEqual(run.task.track_a_total_seconds, seconds)
            self.assertEqual(run.environment, environment)
            self.assertEqual(
                validate_environment_record(environment.record()), environment
            )
            fingerprints.add(run.environment.configuration_sha256)
        self.assertEqual(len(fingerprints), 3)
        self.assertEqual(TASK.read_bytes(), task_bytes)
        self.assertEqual(CHALLENGE.read_bytes(), release_bytes)
        self.assertEqual(release.track_configs["preview-a-8h"].total_seconds, 28800)

    def test_omitted_budget_is_frozen_from_the_selected_challenge_budget(self):
        release = ChallengeRelease.load(CHALLENGE)
        task = replace(
            TaskConfig.load(TASK), source=release.task_path, task_id=release.task_id
        )
        inherited = resolve_environment_run(
            task,
            environment=EnvironmentConfig.from_manifest(cpu_config()),
            release=release,
            track_config_id="preview-a-8h",
        )
        raw = cpu_config()
        raw["budget"] = {"track": "A", "total_seconds": 28800}
        explicit = resolve_environment_run(
            task,
            environment=EnvironmentConfig.from_manifest(raw),
            release=release,
            track_config_id="preview-a-8h",
        )
        self.assertEqual(inherited.environment, explicit.environment)
        self.assertNotEqual(task.track_a_total_seconds, 28800)

    def test_invalid_and_wrong_track_budgets_fail_before_execution(self):
        for budget in (
            None,
            {},
            {"track": "C", "total_seconds": 1},
            {"track": "A", "total_seconds": True},
            {"track": "A", "total_seconds": 0},
            {"track": "A", "total_seconds": -1},
            {"track": "A", "total_seconds": 1.5},
            {"track": "A", "total_seconds": 10**400},
            {"track": "A", "total_seconds": 1, "lease_seconds": 1},
            {"track": "B", "max_leases": 2},
            {"track": "B", "max_leases": 2, "lease_seconds": 0},
            {"track": "B", "max_leases": True, "lease_seconds": 1},
            {"track": "B", "max_leases": 2, "lease_seconds": 1, "total_seconds": 2},
        ):
            raw = cpu_config()
            raw["budget"] = budget
            with self.subTest(budget=budget), self.assertRaises(ConfigError):
                EnvironmentConfig.from_manifest(raw)
        raw = cpu_config()
        raw["budget"] = {"track": "B", "max_leases": 3, "lease_seconds": 4}
        with self.assertRaisesRegex(ConfigError, "budget track differs"):
            resolve_environment_run(
                TaskConfig.load(TASK),
                environment=EnvironmentConfig.from_manifest(raw),
                track="A",
            )
        raw["budget"]["max_leases"] = 9
        with self.assertRaisesRegex(ConfigError, "max_candidates"):
            resolve_environment_run(
                TaskConfig.load(TASK),
                environment=EnvironmentConfig.from_manifest(raw),
                track="B",
            )

    def test_track_a_deadline_uses_the_frozen_environment_budget(self):
        fixture = test_supervisor.SupervisorTests()
        fixture.setUp()
        try:
            path = fixture.episode.parent / "environment.json"
            raw = cpu_config()
            raw["budget"] = {"track": "A", "total_seconds": 3}
            path.write_text(json.dumps(raw))
            run = resolve_environment_run(
                fixture.config,
                environment=EnvironmentConfig.load(path),
                track="A",
            )
            supervisor = fixture.supervisor(
                track="A",
                config=run.task,
                environment=run.environment,
            )
            # Editing the input file after task reveal must not extend a run.
            raw["budget"]["total_seconds"] = 1000
            path.write_text(json.dumps(raw))
            public = json.loads(supervisor.public_task_path.read_text())
            self.assertEqual(public["budget"], {"total_wall_clock_seconds": 3})
            self.assertEqual(public["environment"], run.environment.record())
            supervisor.start_lease()
            fixture.clock.advance(3)
            supervisor.enforce_deadlines()
            self.assertEqual(
                supervisor.public_status()["remaining_wall_clock_seconds"], 0
            )
            self.assertNotEqual(supervisor.state["status"], "active")
            self.assertIsNone(supervisor.state["active_lease"])
            self.assertFalse(fixture.worker.running)
            self.assertEqual(supervisor.state["environment"], run.environment.record())
        finally:
            fixture.tearDown()

    def test_track_b_environment_enforces_lease_count_and_duration(self):
        fixture = test_supervisor.SupervisorTests()
        fixture.setUp()
        try:
            raw = cpu_config()
            raw["budget"] = {"track": "B", "max_leases": 3, "lease_seconds": 4}
            run = resolve_environment_run(
                fixture.config,
                environment=EnvironmentConfig.from_manifest(raw),
                track="B",
            )
            supervisor = fixture.supervisor(
                config=run.task, environment=run.environment
            )
            self.assertEqual(fixture.config.track_b_max_leases, 2)
            self.assertEqual(fixture.config.track_b_lease_seconds, 10)
            self.assertEqual(
                supervisor.public_status()["budget"],
                {"max_leases": 3, "max_seconds_per_lease": 4},
            )
            with self.assertRaisesRegex(ProtocolError, "at most 4s"):
                supervisor.start_lease(5)
            for _ in range(3):
                supervisor.start_lease()
                self.assertEqual(
                    supervisor.public_status()["active_lease"]["remaining_seconds"], 4
                )
                fixture.clock.advance(4)
                supervisor.enforce_deadlines()
                self.assertIsNone(supervisor.state["active_lease"])
            self.assertEqual(supervisor.public_status()["leases_remaining"], 0)
            with self.assertRaises(ProtocolError):
                supervisor.start_lease()
        finally:
            fixture.tearDown()

    def test_supervisor_rejects_a_budget_different_from_its_execution_config(self):
        fixture = test_supervisor.SupervisorTests()
        fixture.setUp()
        try:
            raw = cpu_config()
            raw["budget"] = {"track": "B", "max_leases": 3, "lease_seconds": 4}
            with self.assertRaisesRegex(ValueError, "time budget differs"):
                fixture.supervisor(environment=EnvironmentConfig.from_manifest(raw))
        finally:
            fixture.tearDown()

    def test_environment_survives_the_full_supervisor_lifecycle(self):
        # Exercise the same task protocol with a fake Worker, never an allocation.
        fixture = test_supervisor.SupervisorTests()
        fixture.setUp()
        try:
            environment = EnvironmentConfig.from_manifest(cpu_config())
            supervisor = fixture.supervisor(environment=environment)
            environment = environment.resolve_budget(fixture.config, "B")
            supervisor.start_lease()
            committed = supervisor.commit_lease("/output/candidate")
            supervisor.declare_incumbent(committed["candidate_id"])
            supervisor.finish()
            fixture.finalize_after_agent_stop(supervisor)
            report = normalize_result_report(supervisor.state)
            self.assertEqual(report["environment"], environment.record())
            self.assertIsNone(report["comparison_key"])
            self.assertFalse(report["leaderboard_eligible"])
            self.assertEqual(fixture.worker.evaluations, ["test", "analysis"])
            for field, value in (
                ("leaderboard_eligible", True),
                ("comparison_key", {"fake": "lane"}),
                ("result_schema_version", 15),
                ("budget", {"max_leases": 5, "max_seconds_per_lease": 10}),
                ("budget", {"max_leases": 2.0, "max_seconds_per_lease": 10}),
                ("track", "A"),
                (
                    "resource_envelope",
                    ResourceEnvelope(requirements=()).public_manifest(),
                ),
            ):
                changed = copy.deepcopy(report)
                changed[field] = value
                with self.subTest(field=field), self.assertRaises(ResultSchemaError):
                    normalize_result_report(changed)
        finally:
            fixture.tearDown()

    def test_training_cpu_memory_caps_are_applied_verified_and_receipted(self):
        environment = EnvironmentConfig.from_manifest(cpu_config())
        baseline = environment.resource_envelope()
        envelope = replace(
            baseline,
            requirements=baseline.requirements
            + (
                ResourceRequirement(
                    "cpu.quota_cores", "exact", 2.5, EnforcementLevel.BEST_EFFORT
                ),
                ResourceRequirement(
                    "memory.limit_bytes",
                    "exact",
                    64 * 1024**2,
                    EnforcementLevel.BEST_EFFORT,
                ),
            ),
        )
        for mismatch in (False, True):
            with self.subTest(
                mismatch=mismatch
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                api = publish_agent_api_bundle(root / "api")
                task = ResearchTaskSpec.from_task(TaskConfig.load(TASK), api)
                runtime = CapturingDockerRuntime()
                original = runtime._run

                def run(args, **kwargs):
                    if args[0] == "info":
                        return subprocess.CompletedProcess(
                            args, 0, "[true,true,true]", ""
                        )
                    if args[0] == "inspect" and "{{json .HostConfig}}" in args:
                        return subprocess.CompletedProcess(
                            args,
                            0,
                            json.dumps(
                                {
                                    "NanoCpus": 1 if mismatch else 2500000000,
                                    "Memory": 64 * 1024**2,
                                    "MemorySwap": 64 * 1024**2,
                                }
                            ),
                            "",
                        )
                    return original(args, **kwargs)

                workload = runtime.resolve_workload(
                    episode_id="episode",
                    workload=WorkloadRequest("training", task.training_image),
                    timeout_seconds=20,
                )
                with patch.object(runtime, "_run", side_effect=run), patch.object(
                    runtime, "terminate_lease"
                ) as stop:

                    def start():
                        return runtime.start_lease(
                            episode_id="episode",
                            lease_id="lease-1",
                            duration_seconds=30,
                            task=task,
                            workspace=root,
                            resources=TrainingResources(),
                            resource_envelope=envelope,
                            lease_dir=root / "lease",
                            workload=workload,
                        )

                    if mismatch:
                        with self.assertRaises(EvaluationResourceFailure):
                            start()
                        stop.assert_called_once()
                    else:
                        receipt = start()
                        assessment = assess_resource_allocation(
                            envelope,
                            receipt,
                            episode_id="episode",
                            lease_id="lease-1",
                            worker_id="local-reference-docker",
                        )
                        self.assertTrue(assessment.satisfied)
                        launch = next(
                            call for call in runtime.calls if call[0] == "run"
                        )
                        self.assertEqual(launch[launch.index("--cpus") + 1], "2.5")
                        self.assertEqual(
                            launch[launch.index("--memory") + 1], str(64 * 1024**2)
                        )
                        stop.assert_not_called()

    def test_distribution_exposes_one_package_and_command(self):
        config = configparser.ConfigParser()
        config.read(ROOT / "setup.cfg")
        self.assertEqual(config["metadata"]["name"], "envcore")
        self.assertEqual(config["options.packages.find"]["include"], "envcore*")
        self.assertEqual(
            config["options.entry_points"]["console_scripts"].strip(),
            "envcore = envcore.cli:main",
        )

    def test_public_bundle_supports_envcore_without_trusted_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = publish_agent_api_bundle(root / "api")
            self.assertEqual({path.name for path in bundle.iterdir()}, {"envcore"})
            code = """import importlib.util
from envcore.candidate_protocol import prediction_storage_limit
assert prediction_storage_limit(examples=1, vocab_size=257)['max_bytes'] > 0
assert importlib.util.find_spec('envcore.supervisor') is None
"""
            for command in (
                ["-c", code],
                ["-m", "envcore.cli", "agent", "--help"],
            ):
                # Editable installs add import hooks via site-packages that can
                # resolve trusted Core modules outside the public bundle.
                # Test the bundle with only its PYTHONPATH and the standard library.
                completed = subprocess.run(
                    [sys.executable, "-S", *command],
                    cwd=root,
                    env={**os.environ, "PYTHONPATH": str(bundle)},
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
