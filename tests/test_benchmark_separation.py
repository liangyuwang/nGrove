"""Benchmark metadata is emitted only when a release is explicitly selected."""

import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from envcore.benchmark import BenchmarkRelease
from envcore.cli import _resolve_start_run, build_parser
from envcore.config import ConfigError
from envcore.environment import EnvironmentConfig
from envcore.operator_status import format_operator_status, read_operator_status
from envcore.recording import read_recording
from envcore.results import ResultSchemaError, normalize_result_report
from envcore.supervisor import Supervisor
from envcore.training_data import verify_training_data
from tests import test_supervisor


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tests/fixtures/task/task.json"
BENCHMARK = ROOT / "dev_tasks/smoke_lm-pretrain/benchmark.json"
BENCHMARK_FIELDS = {
    "acceptance",
    "benchmark",
    "policy_compliant",
    "evaluator_policy_compliant",
    "official_elapsed_seconds",
    "benchmark_id",
    "benchmark_release",
    "benchmark_manifest_sha256",
    "comparison_key",
    "leaderboard_eligible",
    "leaderboard_eligibility_violations",
    "formal_release_verification",
    "formal_release_verified",
    "trust_satisfied",
    "conformance_satisfied",
}


class BenchmarkSeparationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_supervisor.SupervisorTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def assertResearchMetadata(self, value, path="root"):
        """Inspect nested execution evidence and recorded responses as well."""
        if isinstance(value, dict):
            for key, child in value.items():
                self.assertNotIn(key, BENCHMARK_FIELDS, f"{path}.{key}")
                self.assertResearchMetadata(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                self.assertResearchMetadata(child, f"{path}[{index}]")
        elif isinstance(value, str):
            # Metric contracts must describe task measurements independently
            # of the optional release that might later compare those results.
            self.assertNotEqual(value, "benchmark_release", path)

    def cli(self, *arguments):
        args = build_parser().parse_args(list(arguments))
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(args.function(args), 0)
        return output.getvalue()

    def assertResearchSurfaces(self, supervisor):
        for name, payload in (
            ("state", supervisor.state),
            ("persisted_state", json.loads(supervisor.state_path.read_text())),
            ("task", json.loads(supervisor.public_task_path.read_text())),
            ("status", supervisor.public_status()),
            ("normalized_report", normalize_result_report(supervisor.state)),
            ("recording", read_recording(supervisor.episode_dir)),
            (
                "cli_report",
                json.loads(
                    self.cli("report", "--episode", str(supervisor.episode_dir))
                ),
            ),
            (
                "operator_status",
                json.loads(
                    self.cli(
                        "status", "--episode", str(supervisor.episode_dir), "--json"
                    )
                ),
            ),
        ):
            with self.subTest(surface=name):
                self.assertResearchMetadata(payload, name)

    def test_plain_task_resolution_has_no_benchmark_configuration(self):
        args = build_parser().parse_args(
            [
                "start",
                "--task",
                str(TASK),
                "--track",
                "B",
                "--episode-dir",
                str(self.fixture.episode),
                "--agent-runner",
                "host_cli",
                "--agent-command",
                "participant-agent",
            ]
        )
        run, benchmark = _resolve_start_run(args)
        self.assertIsNone(benchmark)
        self.assertFalse(hasattr(run.task, "leaderboard_eligible"))
        self.assertResearchMetadata(run.task.public_manifest("B"))

    def test_status_response_and_recording_have_only_research_metadata(self):
        supervisor = self.fixture.supervisor()
        response = supervisor.handle({"op": "status"})
        self.assertEqual(response["status"], "active")
        recording = read_recording(supervisor.episode_dir)
        returned = [
            event for event in recording["events"] if event["type"] == "ACTION_RETURNED"
        ]
        self.assertEqual(len(returned), 1)
        payload = recording["payloads"][returned[0]["payload"]["sha256"]]
        self.assertEqual(payload, {"ok": True, "result": response})
        self.assertEqual(recording["issues"], [])
        self.assertResearchSurfaces(supervisor)

    def test_full_research_lifecycle_keeps_artifact_evaluation_and_execution_evidence(
        self,
    ):
        supervisor = self.fixture.supervisor()
        supervisor.handle({"op": "status"})
        self.fixture.clock.advance(2)
        supervisor.start_lease()
        self.fixture.clock.advance(3)
        committed = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(committed["candidate_id"])
        supervisor.handle({"op": "finish"})
        self.fixture.finalize_after_agent_stop(supervisor)
        supervisor.handle({"op": "status"})

        self.assertEqual(supervisor.state["status"], "finalized")
        self.assertEqual(supervisor.state["outcome"], "success")
        self.assertEqual(self.fixture.worker.evaluations, ["test", "analysis"])
        self.assertTrue(supervisor.state["agent_control"]["cleanup_complete"])
        self.assertEqual(len(supervisor.state["candidate_commits"]), 1)
        self.assertEqual(len(supervisor.state["resource_allocations"]), 1)
        timing = supervisor.state["timing_summary"]
        self.assertEqual(timing["research_elapsed_seconds"], 5)
        self.assertEqual(timing["training_lease_seconds"], 3)
        self.assertEqual(timing["agent_and_worker_seconds"], 2)
        execution = supervisor.state["agent_system"]["execution_evidence"]
        self.assertIn("enforcement", execution)
        self.assertIn("limitations", execution)
        self.assertResearchSurfaces(supervisor)

    def test_operator_stop_needs_no_leaderboard_flag_to_validate(self):
        supervisor = self.fixture.supervisor(operator_max_research_seconds=1)
        supervisor.start_lease()
        self.fixture.clock.advance(1)
        self.assertTrue(supervisor.enforce_operator_limit())
        supervisor.terminate_worker_leases_for_teardown()
        supervisor.record_agent_execution(
            exit_code=0, termination_reason="operator_time_limit_reached"
        )
        self.assertTrue(supervisor.cleanup_agent_control())

        self.assertEqual(supervisor.state["outcome"], "aborted")
        self.assertEqual(supervisor.state["failure"], "operator_time_limit_reached")
        self.assertIsNone(supervisor.state["test_status"])
        self.assertFalse(self.fixture.worker.running)
        self.assertEqual(self.fixture.worker.evaluations, [])
        self.assertResearchSurfaces(supervisor)

    def test_teardown_failure_records_execution_failure_without_eligibility(self):
        supervisor = self.fixture.supervisor()
        supervisor.deployment_teardown_failed("research_worker_teardown_failure")

        self.assertEqual(supervisor.state["status"], "infrastructure_failure")
        self.assertIn(
            "research_worker_teardown_failure",
            supervisor.state["deployment_teardown_failures"],
        )
        self.assertIn(
            "deployment_teardown_incomplete", supervisor.state["execution_issues"]
        )
        self.assertNotIn("policy_compliant", supervisor.state)
        supervisor.handle({"op": "status"})
        self.assertResearchSurfaces(supervisor)

    def test_prepare_and_operator_text_do_not_advertise_comparison_or_eligibility(self):
        with patch("envcore.cli.DockerRuntime") as runtime:
            runtime.return_value.describe_capabilities.return_value = (
                self.fixture.worker.describe_capabilities()
            )
            runtime.return_value.environment_manifest.return_value = {"backend": "fake"}
            output = self.cli("prepare", "--task", str(TASK), "--no-pull")
            runtime.return_value.prepare.assert_called_once()
        self.assertEqual(json.loads(output)["status"], "ready")
        self.assertResearchMetadata(json.loads(output))

        supervisor = self.fixture.supervisor()
        rendered = self.cli("status", "--episode", str(supervisor.episode_dir))
        self.assertIn("Core snapshot:", rendered)
        for text in (output, rendered):
            self.assertNotIn("leaderboard", text.casefold())
            self.assertNotIn("comparison", text.casefold())
            self.assertNotIn("benchmark", text.casefold())

    def test_explicit_preview_release_keeps_benchmark_metadata_without_a_lane(self):
        release = BenchmarkRelease.load(BENCHMARK)
        task = replace(self.fixture.config, task_id=release.task_id)
        environment = EnvironmentConfig.from_manifest(
            {
                "schema_version": 1,
                "name": "custom-cpu",
                "runtime_profile": {"host_count": 1, "accelerator": None},
                "budget": {"track": "B", "max_leases": 2, "lease_seconds": 4},
            }
        )
        run = release.resolve(
            task,
            track_config_id="smoke-b-3x75s",
            runtime_profile_id="cpu-local",
            environment=environment,
        )
        self.assertFalse(run.release_manifest["leaderboard_eligible"])
        self.assertIsNone(run.comparison_key)
        supervisor = self.fixture.supervisor(
            config=run.task,
            environment=run.environment,
            benchmark_release=run.release_manifest,
        )
        response = supervisor.handle({"op": "status"})
        for payload in (
            supervisor.state,
            response,
            json.loads(supervisor.public_task_path.read_text()),
            normalize_result_report(supervisor.state),
        ):
            self.assertNotIn("comparison_key", payload)
            self.assertNotIn("leaderboard_eligible", payload)
            self.assertIn("comparison_key", payload["benchmark"])
            self.assertFalse(payload["benchmark"]["leaderboard_eligible"])
        self.assertEqual(
            supervisor.state["benchmark"]["release"]["benchmark_id"],
            release.benchmark_id,
        )
        view = read_operator_status(supervisor.episode_dir)
        self.assertIn("leaderboard eligible: false", format_operator_status(view))
        self.assertIn("Benchmark comparison key:", format_operator_status(view))

    def test_ordinary_workload_identity_is_recorded_without_organizer_acceptance(self):
        supervisor = self.fixture.supervisor()
        supervisor.start_lease()
        evidence = supervisor.state["workload_resolutions"]["training"]
        self.assertIn("effective_sha256", evidence)
        self.assertIn("effective_format", evidence)
        self.assertIn("worker_id", evidence)
        self.assertNotIn("acceptance", evidence)
        recording = read_recording(supervisor.episode_dir)
        events = [
            event
            for event in recording["events"]
            if event["type"] == "WORKLOAD_RESOLVED"
        ]
        self.assertEqual(len(events), 1)
        self.assertNotIn("acceptance", events[0])
        for acceptance in ("unverified", "canonical_identity", "organizer_receipt"):
            tampered = copy.deepcopy(supervisor.state)
            tampered["workload_resolutions"]["training"]["acceptance"] = acceptance
            with self.subTest(acceptance=acceptance), self.assertRaises(
                ResultSchemaError
            ):
                normalize_result_report(tampered)

    def test_direct_formal_entry_validates_task_before_creating_episode(self):
        formal, _ = self.fixture.formal_supervisor(begin=False)
        root = self.fixture.episode.parent
        cases = [
            (
                "development_smoke",
                {"task_kind": "development_smoke"},
                "development smoke",
            ),
            *(
                (
                    field,
                    {field: "example/mutable:latest"},
                    "digest-pinned images",
                )
                for field in (
                    "training_image",
                    "candidate_image",
                    "evaluator_image",
                    "inference_gateway_image",
                )
            ),
        ]
        for name, changes, error in cases:
            with self.subTest(task_change=name):
                config = replace(formal.config, **changes)
                self.fixture.episode = root / f"research-{name}"
                ordinary = self.fixture.supervisor(config=config, begin=False)
                self.assertEqual(ordinary.state["status"], "created")
                self.assertNotIn("benchmark", ordinary.state)

                episode = root / f"rejected-formal-{name}"
                with patch.object(
                    self.fixture.worker,
                    "describe_capabilities",
                    wraps=self.fixture.worker.describe_capabilities,
                ) as worker:
                    with self.assertRaisesRegex(ConfigError, error):
                        Supervisor(
                            config=config,
                            agent=formal.agent,
                            resources=self.fixture.resources,
                            training_data_verification=verify_training_data(
                                config.train_dir
                            ),
                            track="B",
                            episode_dir=episode,
                            agent_api_dir=formal.agent_api_dir,
                            agent_execution=formal.agent_execution,
                            research_worker=self.fixture.worker,
                            evaluation_worker=self.fixture.worker,
                            benchmark=formal.benchmark,
                        )
                    worker.assert_not_called()
                self.assertFalse(episode.exists())


if __name__ == "__main__":
    unittest.main()
