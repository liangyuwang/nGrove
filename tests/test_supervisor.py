import hashlib
import errno
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from autotrain_bench.agent_control import AgentControlStepResult
from autotrain_bench.agent_api_bundle import publish_agent_api_bundle
from autotrain_bench.challenge import ReleaseVerificationReceipt
from autotrain_bench.config import (
    AgentConfig,
    InferenceConfig,
    TaskConfig,
    TrainingResources,
)
from autotrain_bench.client import Client
from autotrain_bench.local_mailbox import (
    MAX_MAILBOX_RESPONSE_BYTES,
    LocalFileMailbox,
    write_untrusted_response,
)
from autotrain_bench.local_agent_execution import describe_local_agent_execution
from autotrain_bench.supervisor import (
    MAX_COMMAND_STREAM_BYTES,
    ProtocolError,
    Supervisor,
    _core_identity,
)
from autotrain_bench.worker_protocol import (
    BASELINE_REQUIREMENTS,
    GPU_DEVICE_ISOLATION,
    HIDDEN_DATASET_ISOLATION,
    Capability,
    CommandResult,
    ConformanceReceipt,
    EnforcementLevel,
    LeaseState,
    TrustLevel,
    WorkerDescriptor,
    WorkerFailure,
    WorkerTeardownFailure,
)


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tests/fixtures/task/task.json"


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class FakeWorker:
    def __init__(self):
        self.running = set()
        self.fail_start = False
        self.evaluations = []
        self.evaluation_timeouts = []
        self.evaluation_hook = None
        self.evaluation_failure = None
        self.evaluation_exception = None
        self.val_nlls = []
        self.test_nll = 1.5
        self.capability_overrides = {}
        self.command_result = None
        self.trust_level = TrustLevel.DEVELOPMENT
        self.conformance = None
        self.roles = ("research", "evaluation")
        self.research_tasks = []
        self.hidden_splits = []

    def describe_capabilities(self):
        names = [requirement.name for requirement in BASELINE_REQUIREMENTS]
        names.append(GPU_DEVICE_ISOLATION)
        capabilities = {name: Capability(EnforcementLevel.ENFORCED) for name in names}
        capabilities.update(self.capability_overrides)
        return WorkerDescriptor(
            worker_id="test-worker",
            trust_level=self.trust_level,
            roles=self.roles,
            capabilities=capabilities,
            conformance=self.conformance,
        )

    def start_lease(
        self,
        *,
        episode_id,
        lease_id,
        duration_seconds,
        task,
        workspace,
        resources,
        lease_dir,
    ):
        del duration_seconds, resources
        if hasattr(task, "val_path") or hasattr(task, "test_path"):
            raise AssertionError("Research Worker received a hidden split path")
        self.research_tasks.append(task)
        if self.fail_start:
            detail = (
                self.fail_start
                if isinstance(self.fail_start, str)
                else "simulated infrastructure failure"
            )
            self.fail_start = False
            raise WorkerFailure(detail)
        output = lease_dir / "output" / "candidate"
        output.mkdir(parents=True)
        shutil.copy2(
            workspace / "candidate/candidate.py",
            output / "candidate.py",
        )
        self.running.add((episode_id, lease_id))

    def exec_lease(self, *, episode_id, lease_id, argv, timeout_seconds):
        del timeout_seconds
        self.assert_running(episode_id, lease_id)
        if self.command_result is not None:
            return self.command_result
        return CommandResult(0, "executed " + " ".join(argv), "")

    def assert_running(self, episode_id, lease_id):
        if (episode_id, lease_id) not in self.running:
            raise AssertionError(f"lease is not running: {lease_id}")

    def lease_state(self, *, episode_id, lease_id):
        if (episode_id, lease_id) in self.running:
            return LeaseState.RUNNING
        return LeaseState.STOPPED

    def terminate_lease(self, *, episode_id, lease_id):
        self.running.discard((episode_id, lease_id))

    def evaluate_candidate(
        self,
        *,
        episode_id,
        task,
        resources,
        candidate_dir,
        hidden_split,
        output_dir,
        timeout_seconds=None,
    ):
        del episode_id, task, resources, candidate_dir, output_dir
        split = hidden_split.name
        self.hidden_splits.append(hidden_split)
        self.evaluations.append(split)
        self.evaluation_timeouts.append(timeout_seconds)
        if self.evaluation_exception is not None:
            raise self.evaluation_exception
        if self.evaluation_hook is not None:
            self.evaluation_hook(timeout_seconds)
        if self.evaluation_failure is not None:
            raise WorkerFailure(self.evaluation_failure)
        if split == "val" and self.val_nlls:
            return self.val_nlls.pop(0)
        return 1.25 if split == "val" else self.test_nll


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        workspace = root / "workspace"
        workspace.mkdir()
        candidate = workspace / "candidate"
        candidate.mkdir()
        (candidate / "candidate.py").write_text(
            "def load(candidate_dir, device): return None\n"
            "def next_token_logits(model, input_ids): return []\n",
            encoding="utf-8",
        )
        self.config = TaskConfig.load(TASK)
        self.agent = AgentConfig(
            workspace=workspace,
            runner="container",
            command=["python", "/workspace/agent.py"],
            initial_workspace="participant_snapshot",
            image="python:test-agent",
            mode="sandboxed",
        )
        self.resources = TrainingResources()
        self.workspace = workspace
        self.worker = FakeWorker()
        self.clock = FakeClock()
        self.episode = root / "episode"

    def tearDown(self):
        self.temporary.cleanup()

    def supervisor(
        self,
        track="B",
        config=None,
        agent=None,
        comparison_key=None,
        challenge_release=None,
        release_verification=None,
        agent_execution=None,
    ):
        resolved_agent = agent or self.agent
        agent_api_dir = publish_agent_api_bundle(self.episode / "public" / "agent_api")
        self.mailbox = LocalFileMailbox(self.episode / "control")
        self.mailbox.initialize()
        supervisor = Supervisor(
            config=config or self.config,
            agent=resolved_agent,
            resources=self.resources,
            track=track,
            episode_dir=self.episode,
            agent_api_dir=agent_api_dir,
            agent_execution=(
                agent_execution or describe_local_agent_execution(resolved_agent)
            ),
            research_worker=self.worker,
            evaluation_worker=self.worker,
            agent_control_transport=self.mailbox,
            comparison_key=comparison_key,
            challenge_release=challenge_release,
            release_verification=release_verification,
            monotonic=self.clock,
            wall_time=lambda: 1_700_000_000.0,
        )
        supervisor.begin()
        return supervisor

    def finalize_after_agent_stop(self, supervisor):
        supervisor.record_agent_execution(exit_code=0, termination_reason="exited")
        self.assertTrue(supervisor.cleanup_agent_control())
        return supervisor.public_status()

    def test_existing_episode_state_is_rejected_before_mutation(self):
        self.episode.mkdir()
        (self.episode / "state.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already contains state"):
            Supervisor(
                config=self.config,
                agent=self.agent,
                resources=self.resources,
                track="B",
                episode_dir=self.episode,
                agent_api_dir=self.episode / "public" / "agent_api",
                agent_execution=describe_local_agent_execution(self.agent),
                research_worker=self.worker,
                evaluation_worker=self.worker,
            )
        self.assertFalse((self.episode / "control").exists())

    def test_comparison_key_and_release_are_recorded_publicly(self):
        comparison_key = {
            "challenge_id": "lm-pretrain-v1",
            "track": "B",
            "track_config": "preview-b-6x1h",
            "runtime_profile": "h100-sxm-80gb-x4",
        }
        release = {
            "challenge_id": "lm-pretrain-v1",
            "release_stage": "preview",
            "leaderboard_eligible": False,
        }
        supervisor = self.supervisor(
            comparison_key=comparison_key,
            challenge_release=release,
        )
        self.assertEqual(supervisor.state["comparison_key"], comparison_key)
        self.assertFalse(supervisor.state["leaderboard_eligible"])
        self.assertEqual(supervisor.public_status()["comparison_key"], comparison_key)
        public_task = json.loads(
            supervisor.public_task_path.read_text(encoding="utf-8")
        )
        self.assertEqual(public_task["comparison_key"], comparison_key)
        self.assertEqual(public_task["challenge_release"], release)

    def test_leaderboard_eligibility_requires_the_complete_formal_contract(self):
        digest = "@sha256:" + "a" * 64
        formal_task_dir = self.episode.parent / "formal-task"
        formal_task_dir.mkdir()
        formal_task = formal_task_dir / "task.json"
        formal_task.write_bytes(self.config.source.read_bytes())
        (formal_task_dir / "POLICY.md").write_text(
            "# Frozen formal policy\n", encoding="utf-8"
        )
        formal_train_dir = formal_task_dir / "train"
        formal_train_dir.mkdir()
        (formal_train_dir / "manifest.json").write_text(
            '{"frozen":true}\n', encoding="utf-8"
        )
        config = replace(
            self.config,
            source=formal_task,
            train_dir=formal_train_dir,
            task_kind="benchmark_challenge",
            leaderboard_eligible=True,
            training_image="example/training" + digest,
            candidate_image="example/candidate" + digest,
            evaluator_image="example/evaluator" + digest,
            inference_gateway_image="example/gateway" + digest,
        )
        agent = replace(self.agent, image="example/agent" + digest)
        self.worker.trust_level = TrustLevel.CERTIFIED
        self.worker.conformance = ConformanceReceipt("worker-v2", True)
        comparison_key = {
            "challenge_id": "test-byte-lm-v1",
            "track": "B",
            "track_config": "b-final",
            "runtime_profile": "h100-final",
        }
        release = {
            "challenge_id": "test-byte-lm-v1",
            "challenge_manifest_sha256": "b" * 64,
            "task_manifest_sha256": hashlib.sha256(
                config.source.read_bytes()
            ).hexdigest(),
            "release_stage": "official",
            "leaderboard_eligible": True,
            "track_config_id": "b-final",
            "runtime_profile_id": "h100-final",
            "track_config": {"provisional": False},
            "runtime_profile": {"provisional": False},
        }
        artifacts = {
            "challenge_manifest_sha256": release["challenge_manifest_sha256"],
            "task_manifest_sha256": release["task_manifest_sha256"],
            "policy_sha256": hashlib.sha256(
                (config.source.parent / "POLICY.md").read_bytes()
            ).hexdigest(),
            "evaluator_sha256": hashlib.sha256(
                config.evaluator_entrypoint.read_bytes()
            ).hexdigest(),
            "train_manifest_sha256": hashlib.sha256(
                (config.train_dir / "manifest.json").read_bytes()
            ).hexdigest(),
            "val_sha256": hashlib.sha256(config.val_path.read_bytes()).hexdigest(),
            "test_sha256": hashlib.sha256(config.test_path.read_bytes()).hexdigest(),
        }
        core_revision = "d" * 40
        receipt = ReleaseVerificationReceipt(
            authority="test-organizer",
            release_id="test-byte-lm-v1",
            core_revision=core_revision,
            artifact_sha256=artifacts,
        )
        with patch(
            "autotrain_bench.supervisor._core_identity",
            return_value={"revision": core_revision, "dirty": False},
        ):
            supervisor = self.supervisor(
                config=config,
                agent=agent,
                agent_execution=replace(
                    describe_local_agent_execution(agent),
                    enforcement=EnforcementLevel.ENFORCED,
                    trust_level=TrustLevel.CERTIFIED,
                    limitations=(),
                ),
                comparison_key=comparison_key,
                challenge_release=release,
                release_verification=receipt,
            )
        self.assertTrue(supervisor.state["leaderboard_eligible"])
        self.assertEqual(supervisor.state["leaderboard_eligibility_violations"], [])

    def test_agent_execution_identity_must_match_the_selected_image(self):
        evidence = replace(
            describe_local_agent_execution(self.agent),
            workload_identity="different-agent@sha256:" + "a" * 64,
        )

        with self.assertRaisesRegex(ValueError, "workload identity"):
            self.supervisor(agent_execution=evidence)

    def test_mutable_agent_image_cannot_be_declared_immutable(self):
        evidence = replace(
            describe_local_agent_execution(self.agent),
            immutable_workload_identity=True,
        )

        with self.assertRaisesRegex(ValueError, "mutable image"):
            self.supervisor(agent_execution=evidence)

    def test_dirty_core_and_mismatched_release_cannot_be_formal(self):
        digest = "@sha256:" + "a" * 64
        config = replace(
            self.config,
            task_kind="benchmark_challenge",
            leaderboard_eligible=True,
            training_image="example/training" + digest,
            candidate_image="example/candidate" + digest,
            evaluator_image="example/evaluator" + digest,
            inference_gateway_image="example/gateway" + digest,
        )
        agent = replace(self.agent, image="example/agent" + digest)
        self.worker.trust_level = TrustLevel.CERTIFIED
        self.worker.conformance = ConformanceReceipt("worker-v2", True)
        with patch(
            "autotrain_bench.supervisor._core_identity",
            return_value={"revision": "d" * 40, "dirty": True},
        ):
            supervisor = self.supervisor(
                config=config,
                agent=agent,
                comparison_key={
                    "challenge_id": "test-byte-lm-v1",
                    "track": "B",
                    "track_config": "b-final",
                    "runtime_profile": "h100-final",
                },
                challenge_release={
                    "challenge_id": "test-byte-lm-v1",
                    "challenge_manifest_sha256": "b" * 64,
                    "task_manifest_sha256": "c" * 64,
                    "release_stage": "official",
                    "leaderboard_eligible": True,
                    "track_config_id": "b-final",
                    "runtime_profile_id": "h100-final",
                    "track_config": {"provisional": False},
                    "runtime_profile": {"provisional": False},
                },
            )
        self.assertFalse(supervisor.state["leaderboard_eligible"])
        self.assertIn(
            "task_manifest_identity_mismatch",
            supervisor.state["leaderboard_eligibility_violations"],
        )
        self.assertIn(
            "core_worktree_not_clean",
            supervisor.state["leaderboard_eligibility_violations"],
        )
        self.assertIn(
            "formal_release_verification_missing",
            supervisor.state["leaderboard_eligibility_violations"],
        )

    def test_core_identity_is_recomputed_for_each_episode(self):
        revision = "d" * 40
        results = [
            subprocess.CompletedProcess([], 0, revision + "\n", ""),
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, revision + "\n", ""),
            subprocess.CompletedProcess([], 0, " M changed.py\n", ""),
        ]
        with patch(
            "autotrain_bench.supervisor.subprocess.run", side_effect=results
        ) as run:
            first = _core_identity()
            second = _core_identity()

        self.assertFalse(first["dirty"])
        self.assertTrue(second["dirty"])
        self.assertEqual(run.call_count, 4)

    def test_worker_role_specs_do_not_cross_hidden_split_boundaries(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        candidate = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(candidate["candidate_id"])
        supervisor.finish()

        self.assertEqual(self.worker.research_tasks[0].train_dir, self.config.train_dir)
        self.assertFalse(hasattr(self.worker.research_tasks[0], "val_path"))
        self.assertEqual(
            {
                path.name
                for path in (
                    self.worker.research_tasks[0].public_api_dir / "autotrain_bench"
                ).iterdir()
            },
            {
                "__init__.py",
                "cli.py",
                "client.py",
                "candidate_protocol.py",
                "candidate_preflight.py",
            },
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.worker.research_tasks[0].public_api_dir)
        smoke = subprocess.run(
            [sys.executable, "-m", "autotrain_bench.cli", "agent", "--help"],
            cwd=self.workspace,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(smoke.returncode, 0, smoke.stderr)
        self.assertIn("lease-start", smoke.stdout)
        self.assertEqual(
            [(item.name, item.path) for item in self.worker.hidden_splits],
            [("val", self.config.val_path), ("test", self.config.test_path)],
        )

    def test_track_b_interactive_lease_commit_and_finish(self):
        supervisor = self.supervisor()
        self.assertEqual(supervisor.state["agent_system"]["network"], "disabled")
        self.assertFalse(supervisor.state["agent_system"]["policy_compliant"])
        self.assertFalse(supervisor.state["policy_compliant"])
        self.assertEqual(
            supervisor.state["policy_violations"],
            ["agent_writable_storage_unbounded"],
        )
        self.assertEqual(
            supervisor.state["workers"]["research"]["worker_id"],
            "test-worker",
        )
        self.assertEqual(
            supervisor.state["workers"]["evaluation"]["worker_id"],
            "test-worker",
        )
        self.assertTrue(
            supervisor.state["workers"]["research"]["assessment"][
                "capabilities_satisfied"
            ]
        )
        self.assertFalse(
            supervisor.state["workers"]["research"]["assessment"][
                "leaderboard_eligible"
            ]
        )
        self.assertEqual(
            supervisor.state["evaluation_system"]["backend"],
            "isolated_candidate",
        )
        self.assertTrue(supervisor.state["evaluation_system"]["policy_compliant"])
        self.assertEqual(
            len(supervisor.state["agent_system"]["initial_workspace_sha256"]), 64
        )
        public_task = json.loads(
            supervisor.public_task_path.read_text(encoding="utf-8")
        )
        self.assertEqual(
            public_task["agent_environment"]["initial_workspace"],
            "participant_snapshot",
        )
        self.assertEqual(public_task["agent_environment"]["control"], "file_mailbox")
        self.assertEqual(
            public_task["training_environment"]["resources"]["nvidia_gpu_count"],
            0,
        )
        self.assertEqual(
            public_task["policy"]["violations"],
            ["agent_writable_storage_unbounded"],
        )
        lease = supervisor.start_lease()
        self.assertEqual(lease["lease_id"], "lease-0001")
        self.assertNotIn("container_name", supervisor.state["active_lease"])
        command = supervisor.exec_lease(["python", "train.py"])
        self.assertEqual(command["exit_code"], 0)
        committed = supervisor.commit_lease("/output/candidate")
        self.assertEqual(committed["val_nll"], 1.25)
        self.assertIsNone(supervisor.state["active_lease"])
        self.assertIsNone(supervisor.state["incumbent_id"])
        declared = supervisor.declare_incumbent(committed["candidate_id"])
        self.assertEqual(declared["incumbent_id"], committed["candidate_id"])

        with self.assertRaises(ProtocolError):
            supervisor.commit_lease("/output/candidate")

        supervisor.start_lease()
        supervisor.stop_lease()
        with self.assertRaises(ProtocolError):
            supervisor.start_lease()

        staged = supervisor.finish()
        self.assertEqual(staged["status"], "closed")
        self.assertIsNone(staged["outcome"])
        self.assertIsNone(staged["test_nll"])
        self.assertIsNone(supervisor.state["agent_control"]["cleanup_complete"])
        self.assertNotIn(
            "EPISODE_FINALIZED",
            supervisor.events_path.read_text(encoding="utf-8"),
        )

        final = self.finalize_after_agent_stop(supervisor)
        self.assertEqual(final["status"], "finalized")
        self.assertEqual(final["result_schema_version"], 2)
        self.assertEqual(final["outcome"], "success")
        self.assertEqual(final["test_nll"], 1.5)
        self.assertAlmostEqual(final["test_perplexity"], math.exp(1.5))
        self.assertIsNone(final["failure"])
        self.assertTrue(final["agent_finish_observed"])
        self.assertEqual(self.worker.evaluations, ["val", "test"])
        with self.assertRaises(ProtocolError):
            supervisor.start_lease()

        events = [
            json.loads(line)
            for line in supervisor.events_path.read_text(encoding="utf-8").splitlines()
        ]
        types = [event["type"] for event in events]
        self.assertLess(types.index("VAL_EVALUATED"), types.index("INCUMBENT_DECLARED"))
        self.assertLess(types.index("EPISODE_CLOSED"), types.index("TEST_EVALUATED"))
        self.assertLess(
            types.index("AGENT_PROCESS_EXITED"),
            types.index("AGENT_CONTROL_CLEANED"),
        )
        self.assertLess(
            types.index("AGENT_CONTROL_CLEANED"), types.index("EPISODE_FINALIZED")
        )
        self.assertEqual(types.count("EPISODE_FINALIZED"), 1)

    def test_finish_without_an_incumbent_is_a_failed_experiment(self):
        supervisor = self.supervisor()

        supervisor.finish()
        final = self.finalize_after_agent_stop(supervisor)

        self.assertEqual(final["status"], "finalized")
        self.assertEqual(final["outcome"], "failed")
        self.assertEqual(final["failure"], "no_candidate")
        self.assertIsNone(final["test_nll"])
        self.assertIsNone(final["test_perplexity"])
        self.assertEqual(self.worker.evaluations, [])

    def test_failed_test_evaluation_has_no_metric(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        committed = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(committed["candidate_id"])
        self.worker.evaluation_failure = "private evaluator detail"

        supervisor.finish()
        final = self.finalize_after_agent_stop(supervisor)

        self.assertEqual(final["status"], "finalized")
        self.assertEqual(final["outcome"], "failed")
        self.assertEqual(final["failure"], "test_evaluation_failed")
        self.assertIsNone(final["test_nll"])
        self.assertIsNone(final["test_perplexity"])

    def test_test_worker_teardown_failure_is_infrastructure_failure(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        committed = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(committed["candidate_id"])
        private_detail = "/organizer/private/evaluator-stop.log"
        self.worker.evaluation_exception = WorkerTeardownFailure(private_detail)

        result = supervisor.finish()

        self.assertEqual(result["status"], "infrastructure_failure")
        self.assertEqual(result["failure"], "evaluation_worker_teardown_failure")
        self.assertEqual(
            result["deployment_teardown_failures"],
            ["evaluation_worker_teardown_failure"],
        )
        self.assertNotIn(private_detail, json.dumps(result))
        self.assertIn(
            private_detail,
            supervisor.events_path.read_text(encoding="utf-8"),
        )

    def test_unexpected_test_exception_reaches_an_explicit_terminal_state(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        committed = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(committed["candidate_id"])

        def fail_test(_timeout):
            raise RuntimeError("simulated coordinator failure")

        self.worker.evaluation_hook = fail_test
        supervisor.finish()
        final = self.finalize_after_agent_stop(supervisor)

        self.assertEqual(final["status"], "infrastructure_failure")
        self.assertEqual(final["outcome"], "failed")
        self.assertEqual(final["failure"], "test_evaluation_infrastructure_failure")
        self.assertIsNone(final["test_nll"])
        self.assertIsNone(final["test_perplexity"])

    def test_nonfinite_test_nll_is_a_failed_experiment(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        committed = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(committed["candidate_id"])
        self.worker.test_nll = float("nan")

        supervisor.finish()
        final = self.finalize_after_agent_stop(supervisor)

        self.assertEqual(final["outcome"], "failed")
        self.assertEqual(final["failure"], "invalid_test_metric")
        self.assertIsNone(final["test_nll"])
        self.assertIsNone(final["test_perplexity"])

    def test_perplexity_overflow_is_a_failed_experiment(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        committed = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(committed["candidate_id"])
        self.worker.test_nll = 710.0

        supervisor.finish()
        final = self.finalize_after_agent_stop(supervisor)

        self.assertEqual(final["outcome"], "failed")
        self.assertEqual(final["failure"], "perplexity_overflow")
        self.assertIsNone(final["test_nll"])
        self.assertIsNone(final["test_perplexity"])

    def test_agent_text_cannot_turn_local_encoding_errors_into_infrastructure(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        for invalid_argv in (["python", "bad\0argument"], ["python", "\ud800"]):
            with self.subTest(argv=repr(invalid_argv)):
                with self.assertRaises(ProtocolError) as caught:
                    supervisor.exec_lease(invalid_argv)
                self.assertEqual(caught.exception.code, "invalid_command")
                self.assertEqual(supervisor.state["status"], "active")

        with self.assertRaises(ProtocolError) as caught:
            supervisor.commit_lease("/output/bad\0candidate")
        self.assertEqual(caught.exception.code, "invalid_candidate")
        self.assertEqual(supervisor.state["status"], "active")

    def test_command_timeout_must_be_finite(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        for invalid_timeout in (float("nan"), float("inf")):
            with self.subTest(timeout=invalid_timeout):
                with self.assertRaises(ProtocolError) as caught:
                    supervisor.exec_lease(["python", "train.py"], invalid_timeout)
                self.assertEqual(caught.exception.code, "invalid_timeout")
                self.assertEqual(supervisor.state["status"], "active")

    def test_timed_out_command_is_serialized_and_returned_through_mailbox(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        self.worker.command_result = CommandResult(
            124,
            "partial stdout: \ufffd",
            "partial stderr: \ufffd",
            timed_out=True,
        )
        request = {
            "id": "timeout-command",
            "op": "lease_exec",
            "argv": ["python", "train.py"],
        }
        (self.mailbox.inbox / "timeout-command.json").write_text(
            json.dumps(request), encoding="utf-8"
        )

        self.assertTrue(supervisor.process_one_request())

        response = json.loads(
            (self.mailbox.outbox / "timeout-command.json").read_text(encoding="utf-8")
        )
        self.assertTrue(response["ok"])
        self.assertEqual(response["result"]["exit_code"], 124)
        self.assertEqual(response["result"]["stdout"], "partial stdout: \ufffd")
        self.assertEqual(response["result"]["stderr"], "partial stderr: \ufffd")
        self.assertTrue(response["result"]["timed_out"])
        self.assertFalse(response["result"]["lease_active"])

        command_log = json.loads(
            (supervisor.runs_dir / "lease-0001" / "commands.jsonl").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(command_log["stdout"], "partial stdout: \ufffd")
        self.assertEqual(command_log["stderr"], "partial stderr: \ufffd")
        self.assertTrue(command_log["timed_out"])

        events = [
            json.loads(line)
            for line in supervisor.events_path.read_text(encoding="utf-8").splitlines()
        ]
        types = [event["type"] for event in events]
        self.assertIn("LEASE_COMMAND_FINISHED", types)
        self.assertIn("LEASE_ENDED", types)
        self.assertNotIn("SUPERVISOR_ERROR", types)
        self.assertIsNone(supervisor.state["active_lease"])

    def test_command_output_is_bounded_before_logging_or_mailbox_response(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        self.worker.command_result = CommandResult(
            0,
            "x" * (MAX_COMMAND_STREAM_BYTES + 100),
            "",
        )

        result = supervisor.exec_lease(["python", "train.py"])

        self.assertTrue(result["stdout_truncated"])
        self.assertLessEqual(
            len(result["stdout"].encode("utf-8")), MAX_COMMAND_STREAM_BYTES
        )

    def test_json_escaping_cannot_replace_a_completed_command_response(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        self.worker.command_result = CommandResult(
            0,
            "\x00" * MAX_COMMAND_STREAM_BYTES,
            "",
        )
        request = {
            "id": "large-command",
            "op": "lease_exec",
            "argv": ["python", "non-idempotent.py"],
        }
        (self.mailbox.inbox / "large-command.json").write_text(
            json.dumps(request), encoding="utf-8"
        )

        self.assertTrue(supervisor.process_one_request())

        response_path = self.mailbox.outbox / "large-command.json"
        response = json.loads(response_path.read_text(encoding="utf-8"))
        self.assertTrue(response["ok"])
        self.assertEqual(response["result"]["exit_code"], 0)
        self.assertTrue(response["result"]["stdout_truncated"])
        self.assertLessEqual(response_path.stat().st_size, MAX_MAILBOX_RESPONSE_BYTES)

    def test_surrogate_output_cannot_make_a_completed_command_ambiguous(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        self.worker.command_result = CommandResult(0, "\ud800", "")
        request = {
            "id": "surrogate-command",
            "op": "lease_exec",
            "argv": ["python", "non-idempotent.py"],
        }
        (self.mailbox.inbox / "surrogate-command.json").write_text(
            json.dumps(request), encoding="utf-8"
        )

        self.assertTrue(supervisor.process_one_request())

        response = json.loads(
            (self.mailbox.outbox / "surrogate-command.json").read_text(encoding="utf-8")
        )
        self.assertTrue(response["ok"])
        self.assertEqual(response["result"]["exit_code"], 0)
        self.assertEqual(response["result"]["stdout"], "?")
        self.assertTrue(response["result"]["stdout_truncated"])
        self.assertEqual(supervisor.state["active_lease"]["exec_count"], 1)

    def test_official_elapsed_time_freezes_before_agent_drain(self):
        supervisor = self.supervisor()
        self.clock.advance(4)
        supervisor.finish()
        self.assertEqual(supervisor.public_status()["elapsed_seconds"], 4.0)
        self.clock.advance(30)
        self.assertEqual(supervisor.public_status()["elapsed_seconds"], 4.0)
        self.assertEqual(supervisor.state["elapsed_seconds"], 4.0)

    def test_infrastructure_failure_does_not_consume_lease(self):
        supervisor = self.supervisor()
        self.worker.fail_start = True
        with self.assertRaises(ProtocolError) as caught:
            supervisor.start_lease()
        self.assertEqual(caught.exception.code, "infrastructure_failure")
        self.assertEqual(supervisor.state["leases_used"], 0)
        lease = supervisor.start_lease()
        self.assertEqual(lease["lease_id"], "lease-0002")
        self.assertEqual(supervisor.state["leases_used"], 1)

    def test_lease_cleanup_intent_is_durable_before_worker_start(self):
        supervisor = self.supervisor()
        original_start = self.worker.start_lease

        def observe_intent(**kwargs):
            persisted = json.loads(supervisor.state_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["lease_serial"], 1)
            self.assertEqual(persisted["pending_lease_teardowns"], [kwargs["lease_id"]])
            original_start(**kwargs)

        with patch.object(self.worker, "start_lease", side_effect=observe_intent):
            supervisor.start_lease()

        self.assertEqual(supervisor.state["pending_lease_teardowns"], [])

    def test_failed_lease_directory_cleanup_is_a_teardown_failure(self):
        supervisor = self.supervisor()
        self.worker.fail_start = True
        private_detail = "/trusted/episode/lease directory is busy"

        with patch(
            "autotrain_bench.supervisor.shutil.rmtree",
            side_effect=OSError(private_detail),
        ):
            with self.assertRaises(ProtocolError) as caught:
                supervisor.start_lease()

        self.assertEqual(caught.exception.code, "infrastructure_failure")
        public = supervisor.public_status()
        self.assertEqual(public["status"], "infrastructure_failure")
        self.assertEqual(
            public["deployment_teardown_failures"],
            ["research_lease_directory_cleanup_failure"],
        )
        self.assertNotIn(private_detail, json.dumps(public))

    def test_lease_start_teardown_failure_is_terminal(self):
        supervisor = self.supervisor()
        private_detail = "/organizer/private/partial-lease.log"

        with patch.object(
            self.worker,
            "start_lease",
            side_effect=WorkerTeardownFailure(private_detail),
        ):
            with self.assertRaises(ProtocolError) as caught:
                supervisor.start_lease()

        self.assertEqual(caught.exception.code, "infrastructure_failure")
        self.assertEqual(str(caught.exception), "research worker teardown failed")
        public = supervisor.public_status()
        self.assertEqual(public["status"], "infrastructure_failure")
        self.assertEqual(
            public["deployment_teardown_failures"],
            ["research_worker_teardown_failure"],
        )
        self.assertNotIn(private_detail, json.dumps(public))
        self.assertIn(
            private_detail,
            supervisor.events_path.read_text(encoding="utf-8"),
        )

    def test_lease_exec_teardown_failure_is_terminal(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        private_detail = "/organizer/private/lease-timeout.log"

        with patch.object(
            self.worker,
            "exec_lease",
            side_effect=WorkerTeardownFailure(private_detail),
        ):
            with self.assertRaises(ProtocolError) as caught:
                supervisor.exec_lease(["python", "train.py"])

        self.assertEqual(caught.exception.code, "infrastructure_failure")
        self.assertEqual(str(caught.exception), "research worker teardown failed")
        public = supervisor.public_status()
        self.assertEqual(public["status"], "infrastructure_failure")
        self.assertIsNone(public["active_lease"])
        self.assertEqual(
            public["deployment_teardown_failures"],
            ["research_worker_teardown_failure"],
        )
        self.assertNotIn(private_detail, json.dumps(public))
        self.assertIn(
            private_detail,
            supervisor.events_path.read_text(encoding="utf-8"),
        )

    def test_lease_execution_failure_is_explicit_and_terminal(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        private_detail = "/scheduler/private/exec.log"

        with patch.object(
            self.worker,
            "exec_lease",
            side_effect=WorkerFailure(private_detail),
        ):
            with self.assertRaises(ProtocolError) as caught:
                supervisor.exec_lease(["python", "train.py"])

        self.assertEqual(caught.exception.code, "infrastructure_failure")
        self.assertEqual(str(caught.exception), "research worker execution failed")
        self.assertEqual(supervisor.state["status"], "infrastructure_failure")
        self.assertEqual(
            supervisor.state["failure"], "research_worker_execution_failure"
        )
        self.assertNotIn(private_detail, json.dumps(supervisor.public_status()))

    def test_lease_state_failure_after_exec_is_explicit_and_terminal(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        private_detail = "/scheduler/private/state.log"

        with patch.object(
            self.worker,
            "lease_state",
            side_effect=WorkerFailure(private_detail),
        ):
            with self.assertRaises(ProtocolError) as caught:
                supervisor.exec_lease(["python", "train.py"])

        self.assertEqual(caught.exception.code, "infrastructure_failure")
        self.assertEqual(str(caught.exception), "research worker state query failed")
        self.assertEqual(supervisor.state["status"], "infrastructure_failure")
        self.assertEqual(supervisor.state["failure"], "research_worker_state_failure")
        self.assertNotIn(private_detail, json.dumps(supervisor.public_status()))

    def test_lease_termination_teardown_failure_is_recorded(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        private_detail = "/organizer/private/lease-stop.log"

        with patch.object(
            self.worker,
            "terminate_lease",
            side_effect=[WorkerTeardownFailure(private_detail), None],
        ):
            with self.assertRaises(WorkerTeardownFailure):
                supervisor.stop_lease()

        public = supervisor.public_status()
        self.assertEqual(public["status"], "infrastructure_failure")
        self.assertIsNone(public["active_lease"])
        self.assertEqual(
            public["deployment_teardown_failures"],
            ["research_worker_teardown_failure"],
        )
        self.assertNotIn(private_detail, json.dumps(public))
        self.assertIn(
            private_detail,
            supervisor.events_path.read_text(encoding="utf-8"),
        )

    def test_infrastructure_failure_is_terminal_when_lease_stop_fails(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        private_detail = "/organizer/private/worker-stop.log"

        with patch.object(
            self.worker,
            "terminate_lease",
            side_effect=WorkerFailure(private_detail),
        ):
            supervisor.infrastructure_failed("benchmark_monitor_failure")

        self.assertEqual(supervisor.state["status"], "infrastructure_failure")
        self.assertEqual(supervisor.state["failure"], "benchmark_monitor_failure")
        self.assertIsNotNone(supervisor.state["active_lease"])
        self.assertEqual(
            supervisor.state["deployment_teardown_failures"],
            ["research_worker_teardown_failure"],
        )
        events = supervisor.events_path.read_text(encoding="utf-8")
        self.assertIn("LEASE_TERMINATION_FAILED", events)
        self.assertIn(private_detail, events)
        self.assertNotIn(private_detail, json.dumps(supervisor.public_status()))

    def test_lease_reconciliation_failure_is_not_mislabeled_as_teardown(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        original_save = supervisor._save_state
        attempts = 0

        def fail_first_reconciliation_save():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("transient state persistence failure")
            original_save()

        with patch.object(
            supervisor,
            "_save_state",
            side_effect=fail_first_reconciliation_save,
        ):
            supervisor.infrastructure_failed("benchmark_monitor_failure")

        self.assertIsNone(supervisor.state["active_lease"])
        self.assertEqual(supervisor.state["deployment_teardown_failures"], [])
        events = supervisor.events_path.read_text(encoding="utf-8")
        self.assertIn("lease_reconciliation_error", events)
        self.assertNotIn(
            "transient state persistence failure",
            json.dumps(supervisor.public_status()),
        )

    def test_successful_teardown_retry_reconciles_lease_at_frozen_time(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        self.clock.advance(4)

        with patch.object(
            self.worker,
            "terminate_lease",
            side_effect=[WorkerFailure("first stop failed"), None],
        ):
            supervisor.infrastructure_failed("benchmark_monitor_failure")
            self.clock.advance(30)
            self.assertTrue(supervisor.terminate_worker_leases_for_teardown())

        self.assertIsNone(supervisor.state["active_lease"])
        self.assertEqual(supervisor.state["last_lease_id"], "lease-0001")
        self.assertEqual(
            supervisor.public_status()["timing"]["training_lease_seconds"], 4
        )
        self.assertIsNone(supervisor.public_status()["active_lease"])

    def test_uncertain_lease_start_is_retained_for_teardown_retry(self):
        supervisor = self.supervisor()
        private_detail = "/organizer/private/partial-lease.log"

        with patch.object(
            self.worker,
            "start_lease",
            side_effect=WorkerTeardownFailure(private_detail),
        ):
            with self.assertRaises(ProtocolError) as caught:
                supervisor.start_lease()

        self.assertEqual(caught.exception.code, "infrastructure_failure")
        self.assertEqual(supervisor.state["pending_lease_teardowns"], ["lease-0001"])
        self.assertTrue(supervisor.terminate_worker_leases_for_teardown())
        self.assertEqual(supervisor.state["pending_lease_teardowns"], [])
        persisted = json.loads(supervisor.state_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["pending_lease_teardowns"], [])

    def test_terminal_teardown_failure_is_recorded_without_reclassifying_dnf(self):
        supervisor = self.supervisor()
        supervisor.agent_exited_without_finish()
        private_detail = "/organizer/private/docker-agent.log"

        supervisor.deployment_teardown_failed(
            "agent_termination_failure",
            private_detail=private_detail,
        )

        public = supervisor.public_status()
        self.assertEqual(public["status"], "dnf")
        self.assertEqual(
            public["failure"], "agent exited without finishing the episode"
        )
        self.assertEqual(
            public["deployment_teardown_failures"],
            ["agent_termination_failure"],
        )
        self.assertFalse(public["policy_compliant"])
        self.assertFalse(public["leaderboard_eligible"])
        self.assertIn("deployment_teardown_incomplete", public["policy_violations"])
        events = supervisor.events_path.read_text(encoding="utf-8")
        self.assertIn("DEPLOYMENT_TEARDOWN_FAILED", events)
        self.assertIn(private_detail, events)
        self.assertNotIn(private_detail, json.dumps(public))

    def test_startup_cleanup_failure_is_appended_after_infrastructure_failure(self):
        supervisor = self.supervisor()
        supervisor.infrastructure_failed("episode_startup_failure")

        supervisor.deployment_teardown_failed(
            "agent_termination_failure",
            private_detail="private container cleanup detail",
        )

        public = supervisor.public_status()
        self.assertEqual(public["status"], "infrastructure_failure")
        self.assertEqual(public["failure"], "episode_startup_failure")
        self.assertEqual(
            public["deployment_teardown_failures"],
            ["agent_termination_failure"],
        )

    def test_lease_start_failure_does_not_expose_worker_detail(self):
        supervisor = self.supervisor()
        private_detail = "/organizer/private/slurm/job-42.log"
        self.worker.fail_start = private_detail
        (self.mailbox.inbox / "lease-start.json").write_text(
            json.dumps({"op": "lease_start"}), encoding="utf-8"
        )

        self.assertTrue(supervisor.process_one_request())

        response_text = (self.mailbox.outbox / "lease-start.json").read_text(
            encoding="utf-8"
        )
        self.assertNotIn(private_detail, response_text)
        response = json.loads(response_text)
        self.assertEqual(response["error"]["code"], "infrastructure_failure")
        self.assertEqual(response["error"]["message"], "training lease startup failed")
        self.assertEqual(supervisor.state["status"], "active")
        self.assertIn(
            private_detail,
            supervisor.events_path.read_text(encoding="utf-8"),
        )

    def test_evaluation_worker_capability_failure_is_role_scoped(self):
        self.worker.capability_overrides[HIDDEN_DATASET_ISOLATION] = Capability(
            EnforcementLevel.BEST_EFFORT
        )
        supervisor = self.supervisor()
        self.assertTrue(
            supervisor.state["workers"]["research"]["assessment"][
                "capabilities_satisfied"
            ]
        )
        self.assertFalse(
            supervisor.state["workers"]["evaluation"]["assessment"][
                "capabilities_satisfied"
            ]
        )
        self.assertIn(
            "evaluation_worker_capability_insufficient:hidden_dataset_isolation",
            supervisor.state["policy_violations"],
        )

    def test_worker_descriptor_must_declare_its_assigned_roles(self):
        self.worker.roles = ("research",)
        supervisor = self.supervisor()
        self.assertFalse(supervisor.state["policy_compliant"])
        self.assertIn(
            "evaluation_worker_role_missing",
            supervisor.state["policy_violations"],
        )

    def test_commit_accepts_only_stopped_lease_output(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        with self.assertRaises(ProtocolError) as caught:
            supervisor.commit_lease("/workspace/candidate")
        self.assertEqual(caught.exception.code, "invalid_candidate")
        self.assertIsNone(supervisor.state["active_lease"])
        self.assertEqual(supervisor.state["candidates"], {})

    def test_candidate_snapshot_happens_after_the_worker_stops_the_lease(self):
        supervisor = self.supervisor()
        lease = supervisor.start_lease()
        original_snapshot = supervisor._snapshot_candidate

        def assert_stopped(source):
            self.assertEqual(
                self.worker.lease_state(
                    episode_id=supervisor.state["episode_id"],
                    lease_id=lease["lease_id"],
                ),
                LeaseState.STOPPED,
            )
            return original_snapshot(source)

        with patch.object(
            supervisor, "_snapshot_candidate", side_effect=assert_stopped
        ):
            committed = supervisor.commit_lease("/output/candidate")

        snapshot = (
            supervisor.candidates_dir / committed["candidate_id"].split(":", 1)[1]
        )
        source = supervisor.runs_dir / lease["lease_id"] / "output/candidate"
        (source / "candidate.py").write_text("changed after commit\n", encoding="utf-8")
        self.assertNotEqual(
            (snapshot / "candidate.py").read_text(encoding="utf-8"),
            (source / "candidate.py").read_text(encoding="utf-8"),
        )

    def test_candidate_output_symlink_is_rejected_after_lease_stop(self):
        supervisor = self.supervisor()
        lease = supervisor.start_lease()
        candidate = supervisor.runs_dir / lease["lease_id"] / "output" / "candidate"
        (candidate / "candidate.py").unlink()
        (candidate / "candidate.py").symlink_to(self.config.val_path)
        with self.assertRaises(ProtocolError) as caught:
            supervisor.commit_lease("/output/candidate")
        self.assertEqual(caught.exception.code, "invalid_candidate")
        self.assertIsNone(supervisor.state["active_lease"])

    def test_candidate_output_hard_link_is_rejected_after_lease_stop(self):
        supervisor = self.supervisor()
        lease = supervisor.start_lease()
        candidate_file = (
            supervisor.runs_dir / lease["lease_id"] / "output/candidate/candidate.py"
        )
        candidate_file.unlink()
        os.link(self.workspace / "candidate/candidate.py", candidate_file)
        with self.assertRaises(ProtocolError) as caught:
            supervisor.commit_lease("/output/candidate")
        self.assertEqual(caught.exception.code, "invalid_candidate")
        self.assertIsNone(supervisor.state["active_lease"])

    def test_unreadable_candidate_is_not_an_infrastructure_failure(self):
        supervisor = self.supervisor()
        lease = supervisor.start_lease()
        candidate_file = (
            supervisor.runs_dir / lease["lease_id"] / "output/candidate/candidate.py"
        )
        original_open = Path.open

        def fail_candidate_read(path, *args, **kwargs):
            if path == candidate_file and args and args[0] == "rb":
                raise PermissionError(
                    errno.EACCES, "participant-controlled permissions"
                )
            return original_open(path, *args, **kwargs)

        with patch("pathlib.Path.open", new=fail_candidate_read):
            with self.assertRaises(ProtocolError) as caught:
                supervisor.commit_lease("/output/candidate")

        self.assertEqual(caught.exception.code, "invalid_candidate")
        self.assertEqual(supervisor.state["status"], "active")
        self.assertIsNone(supervisor.state["active_lease"])

    def test_candidate_storage_io_failure_is_not_charged_to_the_agent(self):
        supervisor = self.supervisor()
        lease = supervisor.start_lease()
        candidate_file = (
            supervisor.runs_dir / lease["lease_id"] / "output/candidate/candidate.py"
        )
        original_open = Path.open
        private_detail = "/trusted/storage/read failed"

        def fail_candidate_read(path, *args, **kwargs):
            if path == candidate_file and args and args[0] == "rb":
                raise OSError(errno.EIO, private_detail)
            return original_open(path, *args, **kwargs)

        with patch("pathlib.Path.open", new=fail_candidate_read):
            with self.assertRaises(ProtocolError) as caught:
                supervisor.commit_lease("/output/candidate")

        self.assertEqual(caught.exception.code, "infrastructure_failure")
        self.assertEqual(
            supervisor.state["failure"],
            "candidate_inspection_infrastructure_failure",
        )
        self.assertNotIn(private_detail, json.dumps(supervisor.public_status()))

    def test_candidate_rejects_a_path_without_stable_utf8_identity(self):
        supervisor = self.supervisor()
        lease = supervisor.start_lease()
        candidate_dir = supervisor.runs_dir / lease["lease_id"] / "output/candidate"
        invalid_path = candidate_dir / "invalid-\udcff"

        with patch.object(
            supervisor,
            "_tree_entries",
            return_value=iter([(invalid_path, candidate_dir.stat())]),
        ):
            with self.assertRaises(ProtocolError) as caught:
                supervisor.commit_lease("/output/candidate")

        self.assertEqual(caught.exception.code, "invalid_candidate")
        self.assertEqual(supervisor.state["status"], "active")

    def test_candidate_snapshot_storage_failure_is_explicit_infrastructure(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        private_detail = "/trusted/storage/candidates is unavailable"

        with patch.object(
            supervisor,
            "_snapshot_candidate",
            side_effect=OSError(private_detail),
        ):
            with self.assertRaises(ProtocolError) as caught:
                supervisor.commit_lease("/output/candidate")

        self.assertEqual(caught.exception.code, "infrastructure_failure")
        self.assertEqual(
            str(caught.exception), "candidate snapshot infrastructure failed"
        )
        self.assertEqual(supervisor.state["status"], "infrastructure_failure")
        self.assertEqual(
            supervisor.state["failure"],
            "candidate_snapshot_infrastructure_failure",
        )
        self.assertNotIn(private_detail, json.dumps(supervisor.public_status()))

    def test_candidate_identity_includes_empty_directories_and_modes(self):
        first = self.workspace / "hash-first"
        second = self.workspace / "hash-second"
        first.mkdir()
        second.mkdir()
        (first / "candidate.py").write_text("pass\n", encoding="utf-8")
        (second / "candidate.py").write_text("pass\n", encoding="utf-8")
        (first / "behavior-flag").mkdir()
        self.assertNotEqual(Supervisor._tree_hash(first), Supervisor._tree_hash(second))

        original = Supervisor._tree_hash(second)
        (second / "candidate.py").chmod(0o755)
        self.assertNotEqual(original, Supervisor._tree_hash(second))

    def test_candidate_identity_separates_file_content_from_the_next_entry(self):
        first = self.workspace / "collision-first"
        second = self.workspace / "collision-second"
        first.mkdir()
        second.mkdir()
        (second / "a").write_bytes(b"x")
        (second / "b").write_bytes(b"")
        mode = stat.S_IMODE((second / "b").stat().st_mode).to_bytes(4, "big")
        forged_next_entry = b"file\0" + (1).to_bytes(8, "big") + b"b" + mode
        (first / "a").write_bytes(b"x" + forged_next_entry)

        self.assertNotEqual(Supervisor._tree_hash(first), Supervisor._tree_hash(second))

    def test_candidate_entry_limit_counts_directories(self):
        config = replace(self.config, candidate_max_files=1)
        supervisor = self.supervisor(config=config)
        lease = supervisor.start_lease()
        candidate = supervisor.runs_dir / lease["lease_id"] / "output/candidate"
        (candidate / "empty-directory").mkdir()
        with self.assertRaisesRegex(ProtocolError, "artifact limit"):
            supervisor.commit_lease("/output/candidate")

    def test_failed_candidate_snapshot_removes_its_staging_directory(self):
        supervisor = self.supervisor()
        source = self.workspace / "candidate"

        with patch.object(
            Supervisor,
            "_tree_hash",
            side_effect=OSError("simulated candidate hashing failure"),
        ):
            with self.assertRaisesRegex(OSError, "hashing failure"):
                supervisor._snapshot_candidate(source)

        self.assertEqual(list(supervisor.candidates_dir.glob(".stage-*")), [])

    def test_duplicate_snapshot_cleans_read_only_staging_directories(self):
        supervisor = self.supervisor()
        source = self.workspace / "read-only-candidate"
        nested = source / "nested"
        nested.mkdir(parents=True)
        (source / "candidate.py").write_text("pass\n", encoding="utf-8")
        (nested / "weights.bin").write_bytes(b"weights")
        nested.chmod(0o555)
        try:
            first = supervisor._snapshot_candidate(source)
            second = supervisor._snapshot_candidate(source)
        finally:
            nested.chmod(0o700)

        self.assertEqual(first["candidate_id"], second["candidate_id"])
        self.assertEqual(list(supervisor.candidates_dir.glob(".stage-*")), [])

    def test_timing_report_separates_lease_use_and_unused_track_a_budget(self):
        supervisor = self.supervisor(
            track="A", config=replace(self.config, track_a_total_seconds=100)
        )
        supervisor.start_lease(10)
        self.clock.advance(3)
        candidate = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(candidate["candidate_id"])
        final = supervisor.finish()
        timing = final["timing"]
        self.assertEqual(timing["official_elapsed_seconds"], 3.0)
        self.assertEqual(timing["training_lease_seconds"], 3.0)
        self.assertEqual(timing["validation_evaluation_seconds"], 0.0)
        self.assertEqual(timing["agent_and_worker_seconds"], 0.0)
        self.assertEqual(timing["unused_budget_seconds"], 97.0)

    def test_track_a_deadline_revokes_lease_and_closes_episode(self):
        config = replace(self.config, track_a_total_seconds=5)
        supervisor = self.supervisor(track="A", config=config)
        supervisor.start_lease()
        self.clock.advance(6)
        self.assertTrue(supervisor.enforce_deadlines())
        self.assertEqual(supervisor.state["status"], "closed")
        self.assertIsNone(supervisor.state["outcome"])
        self.finalize_after_agent_stop(supervisor)
        self.assertEqual(supervisor.state["status"], "finalized")
        self.assertEqual(supervisor.state["outcome"], "failed")
        self.assertEqual(supervisor.state["failure"], "no_candidate")
        self.assertEqual(supervisor.state["elapsed_seconds"], 5.0)
        self.assertIsNone(supervisor.state["active_lease"])
        self.assertFalse(self.worker.running)

    def test_track_a_val_uses_remaining_time_and_discards_late_result(self):
        config = replace(
            self.config,
            track_a_total_seconds=5,
            evaluator_timeout_seconds=30,
        )
        supervisor = self.supervisor(track="A", config=config)
        supervisor.start_lease(4)
        self.clock.advance(4)
        self.worker.evaluation_hook = lambda timeout: self.clock.advance(2)

        committed = supervisor.commit_lease("/output/candidate")

        self.assertEqual(self.worker.evaluation_timeouts[0], 1.0)
        self.assertIsNone(committed["val_nll"])
        self.assertEqual(committed["evaluation_error"], "track_a_deadline")
        self.assertEqual(supervisor.state["status"], "closed")
        self.assertIsNone(supervisor.state["outcome"])
        self.finalize_after_agent_stop(supervisor)
        self.assertEqual(supervisor.state["status"], "finalized")
        self.assertEqual(supervisor.state["outcome"], "failed")
        self.assertEqual(supervisor.state["failure"], "no_candidate")
        self.assertEqual(supervisor.state["closure_reason"], "track_a_deadline")
        self.assertEqual(supervisor.state["elapsed_seconds"], 5.0)
        self.assertEqual(
            supervisor.state["timing"]["validation_evaluation_seconds"], 1.0
        )
        self.assertIsNone(supervisor.state["final_candidate_id"])

    def test_validation_failures_expose_only_a_fixed_error_code(self):
        supervisor = self.supervisor()
        self.worker.evaluation_failure = "hidden input_ids: [0, 1, 2]"
        supervisor.start_lease()

        committed = supervisor.commit_lease("/output/candidate")

        self.assertEqual(committed["evaluation_error"], "evaluation_failed")
        public = json.dumps(supervisor.public_status())
        events = supervisor.events_path.read_text(encoding="utf-8")
        self.assertNotIn("input_ids", public)
        self.assertNotIn("input_ids", events)

    def test_val_worker_teardown_failure_terminates_the_episode(self):
        supervisor = self.supervisor()
        private_detail = "/organizer/private/evaluation-network.log"
        self.worker.evaluation_exception = WorkerTeardownFailure(private_detail)
        supervisor.start_lease()

        with self.assertRaises(ProtocolError) as caught:
            supervisor.commit_lease("/output/candidate")

        self.assertEqual(caught.exception.code, "infrastructure_failure")
        self.assertEqual(str(caught.exception), "evaluation worker teardown failed")
        public = supervisor.public_status()
        self.assertEqual(public["status"], "infrastructure_failure")
        self.assertEqual(public["failure"], "evaluation_worker_teardown_failure")
        self.assertEqual(
            public["deployment_teardown_failures"],
            ["evaluation_worker_teardown_failure"],
        )
        self.assertNotIn(private_detail, json.dumps(public))
        self.assertIn(
            private_detail,
            supervisor.events_path.read_text(encoding="utf-8"),
        )

    def test_val_teardown_failure_persists_evaluation_time(self):
        supervisor = self.supervisor()
        supervisor.start_lease()

        def fail_after_evaluation_time(**kwargs):
            del kwargs
            self.clock.advance(3)
            raise WorkerTeardownFailure("private evaluator teardown detail")

        with patch.object(
            self.worker,
            "evaluate_candidate",
            side_effect=fail_after_evaluation_time,
        ):
            with self.assertRaises(ProtocolError):
                supervisor.commit_lease("/output/candidate")

        persisted = json.loads(supervisor.state_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["timing"]["validation_evaluation_seconds"], 3.0)

    def test_track_a_deadline_uses_only_agent_declared_incumbent(self):
        config = replace(self.config, track_a_total_seconds=100)
        supervisor = self.supervisor(track="A", config=config)
        self.worker.val_nlls = [1.0, 2.0]

        with self.assertRaises(ProtocolError) as unknown:
            supervisor.declare_incumbent("sha256:missing")
        self.assertEqual(unknown.exception.code, "unknown_candidate")

        supervisor.start_lease(10)
        first = supervisor.commit_lease("/output/candidate")
        self.assertIsNone(supervisor.state["incumbent_id"])
        supervisor.declare_incumbent(first["candidate_id"])

        (self.workspace / "candidate/candidate.py").write_text(
            "def load(candidate_dir, device): return None\n"
            "def next_token_logits(model, input_ids): return []\n"
            "# a distinct later candidate\n",
            encoding="utf-8",
        )
        supervisor.start_lease(10)
        later = supervisor.commit_lease("/output/candidate")
        self.assertNotEqual(first["candidate_id"], later["candidate_id"])
        self.assertEqual(supervisor.state["incumbent_id"], first["candidate_id"])

        self.clock.advance(101)
        self.assertTrue(supervisor.enforce_deadlines())
        self.assertEqual(supervisor.state["final_candidate_id"], first["candidate_id"])
        self.assertEqual(supervisor.state["status"], "closed")
        self.assertIsNone(supervisor.state["test_nll"])
        self.finalize_after_agent_stop(supervisor)
        self.assertEqual(supervisor.state["test_nll"], 1.5)
        self.assertAlmostEqual(supervisor.state["test_perplexity"], math.exp(1.5))

    def test_agent_mailbox_exposes_only_public_status(self):
        supervisor = self.supervisor()
        self.assertEqual(supervisor.episode_dir.stat().st_mode & 0o777, 0o700)
        self.assertEqual(self.mailbox.control_dir.stat().st_mode & 0o777, 0o700)
        result = {}

        def request_status():
            result.update(Client(self.mailbox.control_dir, poll_seconds=0.001).status())

        thread = threading.Thread(target=request_status)
        thread.start()
        deadline = time.monotonic() + 2
        while thread.is_alive() and time.monotonic() < deadline:
            supervisor.process_one_request()
            time.sleep(0.001)
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["status"], "active")
        self.assertEqual(result["task_kind"], "development_smoke")
        self.assertFalse(result["leaderboard_eligible"])
        self.assertEqual(result["agent_network"], "disabled")
        self.assertEqual(result["agent_mode"], "sandboxed")
        serialized = json.dumps(result)
        self.assertNotIn(str(self.config.val_path), serialized)
        self.assertNotIn(str(self.config.test_path), serialized)

    def test_semantic_handler_does_not_require_mailbox_traffic(self):
        agent_api_dir = publish_agent_api_bundle(self.episode / "public" / "agent_api")
        supervisor = Supervisor(
            config=self.config,
            agent=self.agent,
            resources=self.resources,
            track="B",
            episode_dir=self.episode,
            agent_api_dir=agent_api_dir,
            agent_execution=describe_local_agent_execution(self.agent),
            research_worker=self.worker,
            evaluation_worker=self.worker,
            monotonic=self.clock,
            wall_time=lambda: 1_700_000_000.0,
        )
        supervisor.begin()

        result = supervisor.handle({"op": "status"})

        self.assertEqual(result["status"], "active")
        self.assertFalse((self.episode / "control").exists())
        public_task = json.loads(
            supervisor.public_task_path.read_text(encoding="utf-8")
        )
        self.assertIsNone(public_task["agent_environment"]["control"])

        staged = supervisor.handle({"op": "finish"})
        self.assertEqual(staged["status"], "closed")
        with self.assertRaisesRegex(RuntimeError, "termination must be recorded"):
            supervisor.cleanup_agent_control()
        supervisor.record_agent_execution(exit_code=0, termination_reason="exited")
        self.assertEqual(supervisor.state["status"], "closed")
        self.assertTrue(supervisor.cleanup_agent_control())
        self.assertEqual(supervisor.state["status"], "finalized")
        self.assertEqual(supervisor.state["failure"], "no_candidate")

    def test_control_cleanup_is_rejected_until_agent_termination_is_recorded(self):
        supervisor = self.supervisor()
        supervisor.finish()

        with self.assertRaisesRegex(RuntimeError, "termination must be recorded"):
            supervisor.cleanup_agent_control()
        with self.assertRaisesRegex(ValueError, "termination reason"):
            supervisor.record_agent_execution(exit_code=0, termination_reason="")

        late_entry = self.mailbox.inbox / "late.tmp"
        late_entry.write_text("late Agent output", encoding="utf-8")
        supervisor.record_agent_execution(exit_code=0, termination_reason="exited")
        self.assertTrue(supervisor.cleanup_agent_control())

        self.assertFalse(late_entry.exists())
        self.assertEqual(supervisor.state["status"], "finalized")
        self.assertTrue(supervisor.state["agent_control"]["cleanup_complete"])

    def test_agent_termination_requires_episode_closure_and_is_immutable(self):
        supervisor = self.supervisor()

        with self.assertRaisesRegex(RuntimeError, "closure must be recorded"):
            supervisor.record_agent_execution(
                exit_code=0,
                termination_reason="exited",
            )

        supervisor.finish()
        supervisor.record_agent_execution(
            exit_code=0,
            termination_reason="exited",
            drain_seconds=1.0,
        )
        supervisor.record_agent_execution(
            exit_code=0,
            termination_reason="exited",
            drain_seconds=1.0,
        )
        with self.assertRaisesRegex(RuntimeError, "metadata is immutable"):
            supervisor.record_agent_execution(
                exit_code=1,
                termination_reason="exited",
                drain_seconds=1.0,
            )

    def test_completed_control_cleanup_is_idempotent(self):
        supervisor = self.supervisor()
        supervisor.finish()
        supervisor.record_agent_execution(exit_code=0, termination_reason="exited")

        with patch.object(
            self.mailbox,
            "cleanup",
            wraps=self.mailbox.cleanup,
        ) as cleanup:
            self.assertTrue(supervisor.cleanup_agent_control())
            self.assertTrue(supervisor.cleanup_agent_control())

        self.assertEqual(cleanup.call_count, 1)

    def test_mailbox_rejects_agent_created_request_symlinks(self):
        supervisor = self.supervisor()
        request = self.mailbox.inbox / "attack.json"
        request.symlink_to(supervisor.state_path)
        self.assertTrue(supervisor.process_one_request())
        response = json.loads(
            (self.mailbox.outbox / "attack.json").read_text(encoding="utf-8")
        )
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "invalid_request")
        self.assertTrue(supervisor.state_path.is_file())
        self.assertEqual(supervisor.state["status"], "dnf")
        self.assertEqual(supervisor.state["failure"], "agent_control_policy_violation")
        self.assertEqual(list(self.mailbox.rejected.iterdir()), [])

    def test_mailbox_directory_causes_policy_failure_and_is_cleaned_after_stop(self):
        supervisor = self.supervisor()
        request = self.mailbox.inbox / "directory.json"
        request.mkdir()
        (request / "nested").write_text("untrusted", encoding="utf-8")

        self.assertTrue(supervisor.process_one_request())

        response = json.loads(
            (self.mailbox.outbox / "directory.json").read_text(encoding="utf-8")
        )
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "invalid_request")
        self.assertFalse(request.exists())
        self.assertEqual(len(list(self.mailbox.rejected.iterdir())), 1)
        self.assertEqual(supervisor.state["status"], "dnf")
        self.assertEqual(supervisor.state["outcome"], "failed")
        self.assertEqual(supervisor.state["failure"], "agent_control_policy_violation")

        supervisor.record_agent_execution(exit_code=0, termination_reason="exited")
        self.assertTrue(supervisor.cleanup_agent_control())

        self.assertEqual(list(self.mailbox.inbox.iterdir()), [])
        self.assertEqual(list(self.mailbox.outbox.iterdir()), [])
        self.assertEqual(list(self.mailbox.rejected.iterdir()), [])
        events = supervisor.events_path.read_text(encoding="utf-8")
        self.assertIn("MAILBOX_REQUEST_REJECTED", events)
        self.assertIn("AGENT_CONTROL_CLEANED", events)
        self.assertNotIn("EPISODE_INFRASTRUCTURE_FAILURE", events)

    def test_mailbox_cleanup_covers_outbox_and_records_incomplete_teardown(self):
        supervisor = self.supervisor()
        supervisor.agent_exited_without_finish()
        supervisor.record_agent_execution(exit_code=1, termination_reason="exited")
        (self.mailbox.inbox / "stale.json").write_text("{}", encoding="utf-8")
        (self.mailbox.outbox / "stale.json").write_text("{}", encoding="utf-8")
        rejected = self.mailbox.rejected / "tree.rejected"
        rejected.mkdir()
        (rejected / "nested").write_text("untrusted", encoding="utf-8")

        with patch(
            "autotrain_bench.local_mailbox.shutil.rmtree",
            side_effect=OSError("simulated cross-uid cleanup failure"),
        ):
            self.assertFalse(supervisor.cleanup_agent_control())

        self.assertEqual(list(self.mailbox.inbox.iterdir()), [])
        self.assertEqual(list(self.mailbox.outbox.iterdir()), [])
        self.assertTrue(rejected.exists())
        self.assertFalse(supervisor.state["agent_control"]["cleanup_complete"])
        self.assertEqual(supervisor.state["agent_control"]["cleanup_removed"], 2)
        self.assertEqual(supervisor.state["agent_control"]["cleanup_failures"], 1)
        self.assertFalse(supervisor.state["policy_compliant"])
        self.assertFalse(supervisor.state["leaderboard_eligible"])
        self.assertIn(
            "agent_control_cleanup_incomplete",
            supervisor.state["policy_violations"],
        )
        self.assertEqual(
            supervisor.state["deployment_teardown_failures"],
            ["agent_control_teardown_failure"],
        )
        self.assertNotIn(
            "EPISODE_INFRASTRUCTURE_FAILURE",
            supervisor.events_path.read_text(encoding="utf-8"),
        )

    def test_mailbox_pending_limit_is_an_agent_policy_failure(self):
        supervisor = self.supervisor()
        self.mailbox.maximum_pending_entries = 1
        (self.mailbox.inbox / "one.tmp").write_text("", encoding="utf-8")
        (self.mailbox.outbox / "two.tmp").write_text("", encoding="utf-8")

        self.assertTrue(supervisor.process_one_request())

        self.assertEqual(supervisor.state["status"], "dnf")
        self.assertEqual(supervisor.state["outcome"], "failed")
        self.assertEqual(supervisor.state["failure"], "agent_control_policy_violation")
        events = supervisor.events_path.read_text(encoding="utf-8")
        self.assertIn("mailbox_entry_limit", events)
        self.assertNotIn("EPISODE_INFRASTRUCTURE_FAILURE", events)

    def test_trusted_handler_os_error_becomes_infrastructure_failure(self):
        supervisor = self.supervisor()
        private_detail = "/organizer/private/test.bin"
        (self.mailbox.inbox / "lease-start.json").write_text(
            json.dumps({"op": "lease_start"}),
            encoding="utf-8",
        )
        original_save = supervisor._save_state
        failed_once = False

        def fail_once():
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                raise OSError(private_detail)
            return original_save()

        with patch.object(supervisor, "_save_state", side_effect=fail_once):
            self.assertTrue(supervisor.process_one_request())

        response = json.loads(
            (self.mailbox.outbox / "lease-start.json").read_text(encoding="utf-8")
        )
        self.assertEqual(response["error"]["code"], "infrastructure_failure")
        self.assertNotIn(private_detail, json.dumps(response))
        self.assertEqual(supervisor.state["status"], "infrastructure_failure")
        self.assertEqual(supervisor.state["outcome"], "failed")
        self.assertEqual(supervisor.state["failure"], "agent_control_handler_failure")

        (self.mailbox.inbox / "status-after-failure.json").write_text(
            json.dumps({"op": "status"}),
            encoding="utf-8",
        )
        self.assertTrue(supervisor.process_one_request())
        later_response = (self.mailbox.outbox / "status-after-failure.json").read_text(
            encoding="utf-8"
        )
        self.assertNotIn(private_detail, later_response)
        self.assertEqual(
            json.loads(later_response)["result"]["failure"],
            "agent_control_handler_failure",
        )
        self.assertIn(
            "EPISODE_INFRASTRUCTURE_FAILURE",
            supervisor.events_path.read_text(encoding="utf-8"),
        )
        self.assertIn(
            private_detail,
            supervisor.events_path.read_text(encoding="utf-8"),
        )

    def test_mailbox_read_io_error_becomes_transport_infrastructure_failure(self):
        supervisor = self.supervisor()
        private_detail = "/organizer/private/mailbox.db"
        (self.mailbox.inbox / "status.json").write_text(
            json.dumps({"op": "status"}), encoding="utf-8"
        )

        with patch.object(
            self.mailbox,
            "read_request",
            side_effect=OSError(private_detail),
        ):
            self.assertTrue(supervisor.process_one_request())

        response_text = (self.mailbox.outbox / "status.json").read_text(
            encoding="utf-8"
        )
        self.assertNotIn(private_detail, response_text)
        self.assertEqual(
            json.loads(response_text)["error"]["code"], "infrastructure_failure"
        )
        self.assertEqual(supervisor.state["status"], "infrastructure_failure")
        self.assertEqual(supervisor.state["failure"], "agent_control_transport_failure")

    def test_control_failure_code_without_detail_is_not_ignored(self):
        supervisor = self.supervisor()
        result = AgentControlStepResult(
            handled=True,
            infrastructure_failure_code="synthetic_transport_failure",
        )

        with patch.object(self.mailbox, "process_one", return_value=result):
            self.assertTrue(supervisor.process_one_request())

        self.assertEqual(supervisor.state["status"], "infrastructure_failure")
        self.assertEqual(supervisor.state["failure"], "synthetic_transport_failure")

    def test_control_failure_detail_requires_a_code(self):
        with self.assertRaisesRegex(ValueError, "detail requires a failure code"):
            AgentControlStepResult(
                handled=True,
                infrastructure_failure="private detail",
            )

    def test_regular_artifact_cleanup_failure_invalidates_a_successful_run(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        candidate = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(candidate["candidate_id"])
        supervisor.finish()
        supervisor.record_agent_execution(exit_code=0, termination_reason="exited")
        stale = self.mailbox.outbox / "stale.json"
        stale.write_text("{}", encoding="utf-8")

        original_unlink = Path.unlink

        def fail_stale(path, *args, **kwargs):
            if path == stale:
                raise OSError("simulated trusted teardown failure")
            return original_unlink(path, *args, **kwargs)

        with patch("pathlib.Path.unlink", new=fail_stale):
            self.assertFalse(supervisor.cleanup_agent_control())

        self.assertEqual(supervisor.state["status"], "infrastructure_failure")
        self.assertEqual(supervisor.state["outcome"], "failed")
        self.assertIsNone(supervisor.state["test_nll"])
        self.assertIsNone(supervisor.state["test_perplexity"])
        self.assertEqual(
            supervisor.state["agent_control"]["cleanup_infrastructure_failures"],
            1,
        )
        self.assertNotIn(
            "EPISODE_FINALIZED",
            supervisor.events_path.read_text(encoding="utf-8"),
        )

    def test_final_result_persistence_failure_cannot_return_success(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        candidate = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(candidate["candidate_id"])
        supervisor.finish()
        supervisor.record_agent_execution(exit_code=0, termination_reason="exited")
        original_save = supervisor._save_state
        save_attempts = 0

        def fail_first_publication_save():
            nonlocal save_attempts
            save_attempts += 1
            if save_attempts == 1:
                raise OSError("simulated final state persistence failure")
            original_save()

        with patch.object(
            supervisor,
            "_save_state",
            side_effect=fail_first_publication_save,
        ):
            with self.assertRaisesRegex(OSError, "final state persistence"):
                supervisor.cleanup_agent_control()

        self.assertEqual(supervisor.state["status"], "closed")
        self.assertIsNone(supervisor.state["outcome"])
        persisted = json.loads(supervisor.state_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["status"], "closed")
        self.assertIsNone(persisted["outcome"])
        self.assertNotIn(
            "EPISODE_FINALIZED",
            supervisor.events_path.read_text(encoding="utf-8"),
        )

        supervisor.deployment_teardown_failed(
            "agent_control_teardown_failure",
            private_detail="simulated final state persistence failure",
        )
        self.assertEqual(supervisor.state["status"], "infrastructure_failure")
        self.assertEqual(supervisor.state["outcome"], "failed")

    def test_non_regular_cleanup_entry_is_an_agent_policy_failure(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        candidate = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(candidate["candidate_id"])
        supervisor.finish()
        supervisor.record_agent_execution(exit_code=0, termination_reason="exited")
        unseen = self.mailbox.inbox / "unseen.tmp"
        unseen.mkdir()
        (unseen / "nested").write_text("untrusted", encoding="utf-8")

        self.assertTrue(supervisor.cleanup_agent_control())

        self.assertEqual(supervisor.state["status"], "dnf")
        self.assertEqual(supervisor.state["outcome"], "failed")
        self.assertIsNone(supervisor.state["test_nll"])
        self.assertIsNone(supervisor.state["test_perplexity"])
        self.assertEqual(
            supervisor.state["agent_control"]["cleanup_policy_violations"],
            1,
        )
        self.assertNotIn(
            "EPISODE_INFRASTRUCTURE_FAILURE",
            supervisor.events_path.read_text(encoding="utf-8"),
        )
        self.assertNotIn(
            "EPISODE_FINALIZED",
            supervisor.events_path.read_text(encoding="utf-8"),
        )

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO requires os.mkfifo")
    def test_mailbox_fifo_is_removed_and_causes_policy_failure(self):
        supervisor = self.supervisor()
        request = self.mailbox.inbox / "fifo.json"
        os.mkfifo(request)

        self.assertTrue(supervisor.process_one_request())

        response = json.loads(
            (self.mailbox.outbox / "fifo.json").read_text(encoding="utf-8")
        )
        self.assertEqual(response["error"]["code"], "invalid_request")
        self.assertEqual(supervisor.state["status"], "dnf")
        self.assertEqual(supervisor.state["failure"], "agent_control_policy_violation")
        self.assertFalse(request.exists())
        self.assertEqual(list(self.mailbox.rejected.iterdir()), [])

    def test_unquarantinable_mailbox_entry_is_an_agent_failure(self):
        supervisor = self.supervisor()
        (self.mailbox.inbox / "directory.json").mkdir()

        with patch.object(self.mailbox, "quarantine", return_value=(False, True)):
            self.assertTrue(supervisor.process_one_request())

        self.assertEqual(supervisor.state["status"], "dnf")
        self.assertEqual(supervisor.state["outcome"], "failed")
        self.assertEqual(supervisor.state["failure"], "agent_control_policy_violation")
        events = supervisor.events_path.read_text(encoding="utf-8")
        self.assertNotIn("EPISODE_INFRASTRUCTURE_FAILURE", events)

    def test_mailbox_response_has_a_portable_non_dir_fd_path(self):
        self.supervisor()
        with patch(
            "autotrain_bench.local_mailbox.directory_fd_operations_available",
            return_value=False,
        ):
            write_untrusted_response(
                self.mailbox.outbox,
                "portable",
                {"ok": True, "result": {"status": "active"}},
            )

        response = json.loads(
            (self.mailbox.outbox / "portable.json").read_text(encoding="utf-8")
        )
        self.assertTrue(response["ok"])

    def test_invalid_mailbox_filename_is_untrusted_input_not_infrastructure(self):
        supervisor = self.supervisor()
        (self.mailbox.inbox / "é.json").write_text(
            json.dumps({"op": "status"}), encoding="utf-8"
        )

        self.assertTrue(supervisor.process_one_request())

        self.assertEqual(supervisor.state["status"], "active")
        events = supervisor.events_path.read_text(encoding="utf-8")
        self.assertIn("MAILBOX_REQUEST_REJECTED", events)
        self.assertNotIn("EPISODE_INFRASTRUCTURE_FAILURE", events)

    def test_agent_created_outbox_conflict_is_a_policy_failure(self):
        supervisor = self.supervisor()
        (self.mailbox.inbox / "conflict.json").write_text(
            json.dumps({"op": "status"}), encoding="utf-8"
        )
        (self.mailbox.outbox / "conflict.json").mkdir()

        self.assertTrue(supervisor.process_one_request())

        self.assertEqual(supervisor.state["status"], "dnf")
        self.assertEqual(supervisor.state["failure"], "agent_control_policy_violation")
        events = supervisor.events_path.read_text(encoding="utf-8")
        self.assertIn("MAILBOX_RESPONSE_REJECTED", events)
        self.assertNotIn("EPISODE_INFRASTRUCTURE_FAILURE", events)

    def test_host_cli_runner_is_recorded_as_non_compliant(self):
        agent = AgentConfig(
            workspace=self.workspace,
            runner="host_cli",
            command=["example-agent"],
            initial_workspace="participant_snapshot",
        )
        supervisor = self.supervisor(agent=agent)
        self.assertEqual(supervisor.state["agent_system"]["runner"], "host_cli")
        self.assertIsNone(supervisor.state["agent_system"]["image"])
        self.assertEqual(supervisor.state["agent_system"]["network"], "host_managed")
        self.assertFalse(supervisor.state["agent_system"]["policy_compliant"])
        usage = {"tool": "codex", "total_tokens": 42}
        supervisor.finish()
        supervisor.record_agent_execution(
            exit_code=0,
            usage=usage,
            termination_reason="exited",
            drain_seconds=2.5,
        )
        self.assertEqual(supervisor.state["agent_system"]["exit_code"], 0)
        self.assertEqual(
            supervisor.state["agent_system"]["termination_reason"], "exited"
        )
        self.assertTrue(supervisor.state["agent_system"]["finish_observed"])
        self.assertEqual(supervisor.state["agent_system"]["drain_seconds"], 2.5)
        self.assertEqual(supervisor.state["agent_usage"], usage)
        status = supervisor.public_status()
        self.assertEqual(status["agent_runner"], "host_cli")
        self.assertIsNone(status["agent_mode"])
        self.assertEqual(status["agent_usage"], usage)
        self.assertEqual(status["agent_exit_code"], 0)
        self.assertEqual(status["agent_termination_reason"], "exited")
        self.assertTrue(status["agent_finish_observed"])
        self.assertEqual(status["agent_drain_seconds"], 2.5)
        self.assertFalse(status["policy_compliant"])
        self.assertIn("agent_execution_boundary", status["policy_violations"])
        self.assertEqual(
            status["policy_violations"],
            ["agent_execution_boundary", "agent_writable_storage_unbounded"],
        )

    def test_registered_inference_usage_is_summarized_from_trusted_trace(self):
        inference = InferenceConfig(
            protocol="openai_compatible",
            base_url="https://api.example.com",
            api_key_env=None,
            allowed_models=("researcher-a",),
            allowed_paths=("/v1/responses",),
            timeout_seconds=30,
        )
        agent = replace(self.agent, inference=inference)
        supervisor = self.supervisor(agent=agent)

        records = [
            {
                "type": "INFERENCE_STARTED",
                "request_id": "request-1",
                "model": "researcher-a",
                "path": "/v1/responses",
            },
            {
                "type": "INFERENCE_COMPLETED",
                "request_id": "request-1",
                "model": "researcher-a",
                "path": "/v1/responses",
                "status": 200,
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "total_tokens": 10,
                },
            },
        ]
        supervisor.inference_trace_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        summary = supervisor.refresh_inference_usage()
        self.assertEqual(summary["model_calls"], 1)
        self.assertEqual(summary["total_tokens"], 10)
        self.assertTrue(summary["usage_complete"])
        state = json.loads(supervisor.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["inference_usage"], summary)


if __name__ == "__main__":
    unittest.main()
