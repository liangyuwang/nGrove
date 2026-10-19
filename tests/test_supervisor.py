import hashlib
import copy
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

from envcore.agent_control import AgentControlStepResult
from envcore.agent_api_bundle import publish_agent_api_bundle
from envcore.benchmark import (
    FORMAL_ARTIFACT_DIGEST_FIELDS,
    FORMAL_ARTIFACT_IDENTITY_ALGORITHMS,
    BenchmarkRelease,
    ReleaseVerificationReceipt,
)
from envcore.artifact_identity import (
    HIDDEN_BUNDLE_IDENTITY_ALGORITHM,
    hidden_bundle_sha256,
)
from envcore.config import (
    AgentConfig,
    InferenceConfig,
    TaskConfig,
    TrainingResources,
)
from envcore.client import Client
from envcore.local_mailbox import (
    MAX_MAILBOX_RESPONSE_BYTES,
    LocalFileMailbox,
    write_untrusted_response,
)
from envcore.local_agent_execution import describe_local_agent_execution
from envcore.gateway_execution import (
    GatewayArtifactIdentity,
    GatewayExecutionReceipt,
)
from envcore.local_run_guard import LocalResearchGuard
from envcore.results import (
    RESULT_SCHEMA_VERSION,
    ResultSchemaError,
    normalize_result_report,
)
from envcore.training_data import verify_training_data, write_training_manifest
from envcore.supervisor import (
    MAX_COMMAND_STREAM_BYTES,
    MAX_INCUMBENT_DECLARATIONS,
    ProtocolError,
    Supervisor,
    _core_identity,
)
from envcore.worker_protocol import (
    BASELINE_REQUIREMENTS,
    CANDIDATE_PREDICTION_STORAGE_QUOTA,
    CONFORMANCE_RECEIPT_VERSION,
    GPU_DEVICE_ISOLATION,
    HIDDEN_DATASET_IDENTITY_VERIFICATION,
    HIDDEN_DATASET_ISOLATION,
    Capability,
    CandidateEvaluationResult,
    CommandResult,
    ConformanceReceipt,
    EnforcementLevel,
    HiddenDatasetIdentityMismatch,
    HiddenDatasetRequest,
    HiddenDatasetResolutionFailure,
    LeaseState,
    ResourceAllocationReceipt,
    ResourceObservation,
    EvaluationResourceReceipt,
    ResourcePropertyEvidence,
    ResolvedHiddenDataset,
    TrustLevel,
    WORKER_CONFORMANCE_SUITE_VERSION,
    ResolvedWorkload,
    WorkloadRequest,
    WorkloadResolutionAcceptance,
    WorkloadResolutionFailure,
    WorkerDescriptor,
    WorkerFailure,
    WorkerTeardownFailure,
    v0_resource_envelope,
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
        self.evaluation_failures = {}
        self.evaluation_exception = None
        self.evaluation_exceptions = {}
        self.analysis_nlls = []
        self.test_nll = 1.5
        self.capability_overrides = {}
        self.command_result = None
        self.trust_level = TrustLevel.DEVELOPMENT
        self.roles = ("research", "evaluation")
        self.research_tasks = []
        self.hidden_datasets = []
        self.hidden_dataset_requests = []
        self.hidden_dataset_digests = {
            name: hashlib.sha256(f"{name}-dataset".encode()).hexdigest()
            for name in ("analysis", "test")
        }
        self.hidden_dataset_resolution_failures = {}
        self.hidden_dataset_resolution_hook = None
        self.hidden_dataset_resolution_timeouts = []
        self.workload_requests = []
        self.workload_resolutions = {}
        self.workload_resolution_failures = {}
        self.workload_resolution_exception = None
        self.workload_resolution_hook = None
        self.workload_resolution_timeouts = []
        self.lease_workloads = []
        self.resource_envelopes = []
        self.resource_receipt_override = None
        self.evaluation_workloads = []
        # Simulates benchmark-owned preparation inside start_lease: queue wait,
        # image materialization, workload startup.
        self.start_lease_delay = 0.0
        self.clock = None

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
        resource_envelope,
        lease_dir,
        workload,
    ):
        del duration_seconds, resources
        if not isinstance(workload, ResolvedWorkload):
            raise AssertionError("Research Worker did not receive a workload binding")
        self.lease_workloads.append(workload)
        self.resource_envelopes.append(resource_envelope)
        if self.start_lease_delay and self.clock is not None:
            self.clock.advance(self.start_lease_delay)
        if hasattr(task, "analysis_path") or hasattr(task, "test_path"):
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
        if self.resource_receipt_override is not None:
            return self.resource_receipt_override
        return ResourceAllocationReceipt(
            episode_id=episode_id,
            lease_id=lease_id,
            worker_id="test-worker",
            resource_request_sha256=resource_envelope.sha256,
            properties=tuple(
                ResourcePropertyEvidence(
                    name=requirement.name,
                    allocator_allocated=ResourceObservation(
                        "observed",
                        requirement.value,
                        "test-allocator",
                    ),
                    workload_visible=ResourceObservation("not_applicable"),
                    enforcement_level=EnforcementLevel.ENFORCED,
                    enforcement_authority="test-worker",
                )
                for requirement in resource_envelope.requirements
            ),
            receipt_id=f"receipt:{lease_id}",
        )

    def resolve_workload(self, *, episode_id, workload, timeout_seconds):
        if not isinstance(workload, WorkloadRequest):
            raise AssertionError("Worker received an invalid workload request")
        self.workload_requests.append(workload)
        self.workload_resolution_timeouts.append(timeout_seconds)
        if self.workload_resolution_hook is not None:
            self.workload_resolution_hook(workload, timeout_seconds)
        if self.workload_resolution_exception is not None:
            raise self.workload_resolution_exception
        failure = self.workload_resolution_failures.get(workload.role)
        if failure is not None:
            raise WorkloadResolutionFailure(failure)
        resolved = self.workload_resolutions.get(workload.role)
        if resolved is not None:
            return resolved
        declared = (
            workload.requested_identity.rsplit("@sha256:", 1)[1]
            if "@sha256:" in workload.requested_identity
            else None
        )
        if (
            declared is not None
            and len(declared) == 64
            and all(character in "0123456789abcdef" for character in declared)
        ):
            digest = declared
        else:
            digest = hashlib.sha256(workload.requested_identity.encode()).hexdigest()
        return ResolvedWorkload(
            role=workload.role,
            requested_identity=workload.requested_identity,
            effective_format="oci",
            effective_sha256=digest,
            transformation_sha256=None,
            resolution_id=f"{episode_id}:{workload.role}",
        )

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

    def resolve_hidden_dataset(
        self,
        *,
        episode_id,
        task,
        hidden_dataset,
        timeout_seconds,
    ):
        del task
        self.hidden_dataset_requests.append(hidden_dataset)
        self.hidden_dataset_resolution_timeouts.append(timeout_seconds)
        if self.hidden_dataset_resolution_hook is not None:
            self.hidden_dataset_resolution_hook(hidden_dataset)
        failure = self.hidden_dataset_resolution_failures.get(hidden_dataset.name)
        if failure is not None:
            raise HiddenDatasetResolutionFailure(failure)
        resolved = ResolvedHiddenDataset(
            hidden_dataset.name,
            self.hidden_dataset_digests[hidden_dataset.name],
            f"{episode_id}:{hidden_dataset.name}:{len(self.hidden_dataset_requests)}",
        )
        if (
            hidden_dataset.expected_sha256 is not None
            and hidden_dataset.expected_sha256 != resolved.sha256
        ):
            raise HiddenDatasetIdentityMismatch(resolved)
        return resolved

    def evaluate_candidate(
        self,
        *,
        episode_id,
        task,
        resources,
        candidate_dir,
        hidden_dataset,
        output_dir,
        candidate_workload,
        evaluator_workload,
        evaluation_id,
        resource_envelope,
        timeout_seconds=None,
    ):
        del resources, candidate_dir, output_dir
        if not isinstance(candidate_workload, ResolvedWorkload):
            raise AssertionError("Candidate workload is not resolved")
        if not isinstance(evaluator_workload, ResolvedWorkload):
            raise AssertionError("Evaluator workload is not resolved")
        self.evaluation_workloads.append((candidate_workload, evaluator_workload))
        split = hidden_dataset.name
        self.hidden_datasets.append(hidden_dataset)
        self.evaluations.append(split)
        self.evaluation_timeouts.append(timeout_seconds)
        if self.evaluation_exception is not None:
            raise self.evaluation_exception
        if split in self.evaluation_exceptions:
            raise self.evaluation_exceptions[split]
        if self.evaluation_hook is not None:
            self.evaluation_hook(split, timeout_seconds)
        if split in self.evaluation_failures:
            raise WorkerFailure(self.evaluation_failures[split])
        if self.evaluation_failure is not None:
            raise WorkerFailure(self.evaluation_failure)
        if split == "analysis" and self.analysis_nlls:
            nll = self.analysis_nlls.pop(0)
        else:
            nll = 1.25 if split == "analysis" else self.test_nll
        performance_score = (
            100.0
            if task.vocab_size == 1
            else 100.0 * max(0.0, min(1.0, 1.0 - nll / math.log(task.vocab_size)))
        )
        return CandidateEvaluationResult(
            performance_score=performance_score,
            raw_metrics={"mean_nll": nll},
            candidate_wall_seconds=2.0,
            task_count=3,
            input_tokens=6,
            prediction_bytes=64,
            resource_receipt=(
                EvaluationResourceReceipt(
                    episode_id=episode_id,
                    evaluation_id=evaluation_id,
                    worker_id="test-worker",
                    resource_request_sha256=resource_envelope.sha256,
                    receipt_id="receipt:" + evaluation_id,
                    properties=tuple(
                        ResourcePropertyEvidence(
                            name=item.name,
                            allocator_allocated=ResourceObservation("unavailable"),
                            workload_visible=ResourceObservation(
                                "observed", item.value, "test-worker"
                            ),
                            enforcement_level=EnforcementLevel.ENFORCED,
                            enforcement_authority="test-worker",
                        )
                        for item in resource_envelope.requirements
                    ),
                )
                if resource_envelope is not None
                else None
            ),
        )


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        workspace = root / "workspace"
        workspace.mkdir()
        candidate = workspace / "candidate"
        candidate.mkdir()
        (candidate / "candidate.py").write_text(
            "def run_offline(candidate_dir, contexts_path, output_dir, device): "
            "return None\n",
            encoding="utf-8",
        )
        self.config = TaskConfig.load(TASK)
        self.agent = AgentConfig(
            workspace=workspace,
            runner="container",
            command=["python", "/workspace/agent.py"],
            initial_workspace="participant_snapshot",
            image="python:test-agent",
            egress="denied",
        )
        self.resources = TrainingResources()
        self.workspace = workspace
        self.worker = FakeWorker()
        self.worker.hidden_dataset_digests = {
            "analysis": hidden_bundle_sha256(self.config.analysis_path),
            "test": hidden_bundle_sha256(self.config.test_path),
        }
        self.clock = FakeClock()
        self.episode = root / "episode"

    def tearDown(self):
        self.temporary.cleanup()

    def test_verification_evidence_must_bind_train_before_task_reveal(self):
        valid = verify_training_data(self.config.train_dir)
        root = self.episode.parent
        for index, invalid in enumerate(
            (
                {},
                {**valid, "status": "not_verified"},
                {**valid, "manifest_sha256": "0" * 64},
            )
        ):
            with self.subTest(evidence=invalid):
                self.episode = root / f"episode-{index}"
                with patch.object(self.worker, "describe_capabilities") as worker:
                    with self.assertRaisesRegex(ValueError, "verification must bind"):
                        self.supervisor(training_data_verification=invalid)
                    worker.assert_not_called()
                self.assertFalse((self.episode / "public/task.json").exists())
                self.assertFalse((self.episode / "state.json").exists())

    def test_verified_identity_is_recorded_without_reading_dataset_files(self):
        supplied = verify_training_data(self.config.train_dir)
        # Only the manifest is accessible here. A deployment can verify and
        # expose the actual dataset on the Worker side.
        train = self.episode.parent / "train-manifest-only"
        train.mkdir()
        (train / "manifest.json").write_bytes(
            (self.config.train_dir / "manifest.json").read_bytes()
        )
        config = replace(
            self.config,
            train_dir=train,
            analysis_path=self.episode.parent / "absent-analysis",
            test_path=self.episode.parent / "absent-test",
        )
        with patch(
            "envcore.training_data.verify_training_data",
            side_effect=AssertionError("Core must not scan TRAIN"),
        ):
            supervisor = self.supervisor(
                config=config, training_data_verification=supplied
            )
        receipt = supervisor.state["training_data_verification"]
        self.assertEqual(receipt, supplied)
        supplied["status"] = "changed by caller"
        self.assertEqual(receipt["status"], "verified")
        self.assertEqual(receipt["files"], 1)
        self.assertEqual(
            supervisor.state["artifact_identities"]["train_manifest_sha256"],
            receipt["manifest_sha256"],
        )

    def supervisor(
        self,
        track="B",
        config=None,
        agent=None,
        comparison_key=None,
        benchmark_release=None,
        release_verification=None,
        agent_execution=None,
        worker_conformance=None,
        training_data_verification=None,
        operator_max_research_seconds=None,
        evaluation_resource_envelope=None,
        environment=None,
        recording=None,
        begin=True,
    ):
        resolved_config = config or self.config
        if training_data_verification is None:
            training_data_verification = verify_training_data(resolved_config.train_dir)
        resolved_agent = agent or self.agent
        agent_api_dir = publish_agent_api_bundle(self.episode / "public" / "agent_api")
        self.mailbox = LocalFileMailbox(self.episode / "control")
        self.mailbox.initialize()
        supervisor = Supervisor(
            config=resolved_config,
            agent=resolved_agent,
            resources=self.resources,
            training_data_verification=training_data_verification,
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
            benchmark_release=benchmark_release,
            release_verification=release_verification,
            research_worker_conformance=worker_conformance,
            evaluation_worker_conformance=worker_conformance,
            monotonic=self.clock,
            wall_time=lambda: 1_700_000_000.0,
            operator_max_research_seconds=operator_max_research_seconds,
            evaluation_resource_envelope=evaluation_resource_envelope,
            environment=environment,
            recording=recording,
        )
        if begin:
            supervisor.begin()
        return supervisor

    def test_operator_limit_is_private_and_does_not_change_track_b_budget(self):
        supervisor = self.supervisor(operator_max_research_seconds=17)
        task = json.loads(supervisor.public_task_path.read_text())
        self.assertEqual(task["budget"], self.config.public_manifest("B")["budget"])
        self.assertNotIn("operator", json.dumps(task))
        self.assertNotIn("operator", json.dumps(supervisor.public_status()))
        self.assertEqual(
            supervisor.state["operator_limits"]["max_research_seconds"], 17
        )
        lease = supervisor.start_lease()
        self.assertEqual(lease["duration_seconds"], self.config.track_b_lease_seconds)

    def test_operator_limit_defaults_to_disabled(self):
        supervisor = self.supervisor()
        self.clock.advance(1_000_000)
        self.assertIsNone(supervisor.operator_limit_remaining())
        self.assertFalse(supervisor.enforce_deadlines())
        self.assertEqual(supervisor.state["status"], "active")

    def test_operator_limit_aborts_at_boundary_without_hidden_evaluation(self):
        supervisor = self.supervisor(operator_max_research_seconds=10)
        supervisor.start_lease()
        commit = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(commit["candidate_id"])
        lease = supervisor.start_lease()
        self.clock.advance(9.9)
        self.assertFalse(supervisor.enforce_operator_limit())
        self.clock.advance(0.1)
        self.assertTrue(supervisor.enforce_operator_limit())
        self.assertFalse(supervisor.enforce_operator_limit())
        state = supervisor.state
        self.assertEqual(
            (state["status"], state["outcome"]), ("operator_stopped", "aborted")
        )
        self.assertEqual(state["failure"], "operator_time_limit_reached")
        self.assertIsNotNone(state["research_sealed_at"])
        self.assertEqual(state["active_lease"]["lease_id"], lease["lease_id"])
        self.assertIsNone(state["final_candidate_id"])
        self.assertIsNone(state["test_status"])
        self.assertFalse(state["leaderboard_eligible"])
        with self.assertRaises(ProtocolError):
            supervisor.handle({"op": "finish"})
        supervisor.terminate_worker_leases_for_teardown()
        supervisor.record_agent_execution(
            exit_code=0, termination_reason="operator_time_limit_reached"
        )
        supervisor.cleanup_agent_control()
        report = normalize_result_report(supervisor.state)
        self.assertEqual(report["outcome"], "aborted")
        self.assertEqual(self.worker.hidden_dataset_requests, [])
        self.assertEqual(self.worker.evaluations, [])
        events = supervisor.events_path.read_text()
        self.assertEqual(events.count('"type": "EPISODE_OPERATOR_STOPPED"'), 1)

    def test_operator_limit_includes_refunded_startup_and_rejects_late_ack(self):
        self.worker.clock = self.clock
        self.worker.start_lease_delay = 11
        supervisor = self.supervisor(operator_max_research_seconds=10)
        with self.assertRaisesRegex(ProtocolError, "closed"):
            supervisor.handle({"op": "lease_start"})
        self.assertEqual(supervisor.state["status"], "operator_stopped")
        self.assertEqual(supervisor.state["timing"]["benchmark_overhead_seconds"], 11)
        self.assertEqual(supervisor._elapsed(), 0)
        supervisor.terminate_worker_leases_for_teardown()
        self.assertFalse(self.worker.running)

    def test_operator_limit_does_not_interrupt_postmortem_after_normal_finish(self):
        supervisor = self.supervisor(operator_max_research_seconds=10)
        supervisor.start_lease()
        commit = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(commit["candidate_id"])
        supervisor.finish()
        self.clock.advance(100)
        self.assertIsNone(supervisor.operator_limit_remaining())
        self.assertFalse(supervisor.enforce_operator_limit())
        result = self.finalize_after_agent_stop(supervisor)
        self.assertEqual(result["outcome"], "success")

    def test_operator_stop_preserves_cause_when_termination_fails(self):
        supervisor = self.supervisor(operator_max_research_seconds=1)
        supervisor.start_lease()
        self.clock.advance(1)
        supervisor.infrastructure_failed("worker_died_during_operator_stop")
        with patch.object(
            self.worker,
            "terminate_lease",
            side_effect=WorkerTeardownFailure("still running"),
        ):
            with self.assertRaises(WorkerTeardownFailure):
                supervisor.terminate_worker_leases_for_teardown()
        supervisor.deployment_teardown_failed("research_lease_cleanup_failure")
        self.assertEqual(supervisor.state["status"], "operator_stopped")
        self.assertEqual(supervisor.state["outcome"], "aborted")
        self.assertEqual(supervisor.state["failure"], "operator_time_limit_reached")
        self.assertFalse(supervisor.state["policy_compliant"])
        self.assertIsNotNone(supervisor.state["active_lease"])

    def test_local_guard_revokes_a_workload_while_core_waits_inside_worker(self):
        supervisor = self.supervisor(operator_max_research_seconds=10)
        lease = supervisor.start_lease()
        revoked = threading.Event()

        def terminate():
            self.worker.terminate_lease(
                episode_id=supervisor.state["episode_id"], lease_id=lease["lease_id"]
            )
            revoked.set()
            return []

        def blocked_worker(**kwargs):
            self.clock.advance(10)
            self.assertTrue(revoked.wait(2), "guard waited for the Coordinator lock")
            raise WorkerFailure("revoked while waiting for a training command")

        guard = LocalResearchGuard(supervisor.operator_limit_remaining, terminate)
        guard.start()
        try:
            with patch.object(self.worker, "exec_lease", side_effect=blocked_worker):
                with self.assertRaises(ProtocolError):
                    supervisor.handle({"op": "lease_exec", "argv": ["long-command"]})
        finally:
            self.assertEqual(guard.close(), [])
        self.assertEqual(supervisor.state["outcome"], "aborted")
        self.assertEqual(supervisor.state["failure"], "operator_time_limit_reached")
        self.assertFalse(self.worker.running)

    def finalize_after_agent_stop(self, supervisor):
        supervisor.record_agent_execution(exit_code=0, termination_reason="exited")
        self.assertTrue(supervisor.cleanup_agent_control())
        if supervisor.state["status"] == "closed":
            supervisor.evaluate_postmortem()
        if supervisor.state["status"] == "closed":
            return supervisor.publish_postmortem_result()
        return supervisor.public_status()

    def test_episode_protocol_id_is_independent_of_directory_name(self):
        self.episode = self.episode.parent / "episode with spaces and é"
        supervisor = self.supervisor()

        self.assertRegex(supervisor.state["episode_id"], r"^episode:[0-9a-f]{32}$")
        granted = supervisor.start_lease()

        self.assertEqual(granted["lease_id"], "lease-0001")
        self.assertEqual(
            supervisor.state["resource_allocations"][0]["receipt"]["episode_id"],
            supervisor.state["episode_id"],
        )

    def test_existing_episode_state_is_rejected_before_mutation(self):
        self.episode.mkdir()
        (self.episode / "state.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already contains state"):
            Supervisor(
                config=self.config,
                agent=self.agent,
                resources=self.resources,
                training_data_verification=verify_training_data(self.config.train_dir),
                track="B",
                episode_dir=self.episode,
                agent_api_dir=self.episode / "public" / "agent_api",
                agent_execution=describe_local_agent_execution(self.agent),
                research_worker=self.worker,
                evaluation_worker=self.worker,
            )
        self.assertFalse((self.episode / "control").exists())

    def test_public_release_contains_only_declared_identifiers(self):
        comparison_key = {
            "benchmark_id": "lm-pretrain",
            "track": "B",
            "track_config": "preview-b-6x1h",
            "runtime_profile": "h100-sxm-80gb-x4",
        }
        public_release = {
            "benchmark_id": "lm-pretrain",
            "release_stage": "preview",
            "leaderboard_eligible": False,
            "track_config_id": comparison_key["track_config"],
            "runtime_profile_id": comparison_key["runtime_profile"],
        }
        release = {
            **public_release,
            "benchmark_manifest_sha256": "b" * 64,
            "task_manifest_sha256": hashlib.sha256(
                self.config.source.read_bytes()
            ).hexdigest(),
            "private_evaluation": {"context_lengths": [1, 3]},
            "track_config": {
                "track": "B",
                "max_leases": self.config.track_b_max_leases,
                "lease_seconds": self.config.track_b_lease_seconds,
                "private_extension": "not-for-the-agent",
            },
            "runtime_profile": {"private_extension": "not-for-the-agent"},
            "resource_envelope": v0_resource_envelope(
                self.resources, minimum_enforcement=EnforcementLevel.BEST_EFFORT
            ).public_manifest(),
        }
        supervisor = self.supervisor(
            comparison_key=comparison_key,
            benchmark_release=release,
        )
        self.assertEqual(supervisor.state["comparison_key"], comparison_key)
        self.assertFalse(supervisor.state["leaderboard_eligible"])
        self.assertEqual(supervisor.public_status()["comparison_key"], comparison_key)
        public_task = json.loads(
            supervisor.public_task_path.read_text(encoding="utf-8")
        )
        self.assertEqual(public_task["comparison_key"], comparison_key)
        self.assertEqual(public_task["benchmark_release"], public_release)
        self.assertEqual(supervisor.state["benchmark_release"], release)

    def test_full_agent_manifest_does_not_identify_hidden_length_choices(self):
        root = self.episode.parent
        task_path = root / "task.json"
        benchmark_path = root / "benchmark.json"
        task_payload = json.loads(self.config.source.read_text(encoding="utf-8"))
        task_payload["paths"] = {
            name: str((self.config.source.parent / value).resolve())
            for name, value in task_payload["paths"].items()
        }
        task_payload["evaluation"]["entrypoint"] = str(self.config.evaluator_entrypoint)
        benchmark_payload = json.loads(
            (ROOT / "dev_tasks/smoke_lm-pretrain/benchmark.json").read_text(
                encoding="utf-8"
            )
        )
        benchmark_payload["task_id"] = self.config.task_id
        benchmark_payload["benchmark_id"] = "hidden-lengths-regression"
        benchmark_path.write_text(json.dumps(benchmark_payload), encoding="utf-8")
        public_manifests = {}
        private_digests = set()
        for index, lengths in enumerate(([1], [2], [1, 3], [1, 2, 3])):
            task_payload["evaluation"]["context_lengths"] = lengths
            task_path.write_text(json.dumps(task_payload), encoding="utf-8")
            task = TaskConfig.load(task_path)
            release = BenchmarkRelease.load(benchmark_path)
            private_digests.add(release.task_manifest_sha256)
            for track_config in ("smoke-a-5m", "smoke-b-3x75s"):
                with self.subTest(lengths=lengths, track_config=track_config):
                    resolved = release.resolve(
                        task,
                        track_config_id=track_config,
                        runtime_profile_id="cpu-local",
                    )
                    self.episode = root / f"episode-{track_config}-{index}"
                    supervisor = self.supervisor(
                        track=resolved.track,
                        config=resolved.task,
                        comparison_key=resolved.comparison_key.public_manifest(),
                        benchmark_release=resolved.release_manifest,
                    )
                    public = json.loads(
                        supervisor.public_task_path.read_text(encoding="utf-8")
                    )
                    self.assertEqual(
                        public, public_manifests.setdefault(track_config, public)
                    )
                    audit = json.loads(
                        supervisor.state_path.read_text(encoding="utf-8")
                    )
                    for record in (
                        audit["benchmark_release"],
                        audit["artifact_identities"],
                    ):
                        self.assertEqual(
                            record["task_manifest_sha256"],
                            release.task_manifest_sha256,
                        )
                    self.assertNotIn(
                        release.task_manifest_sha256,
                        json.dumps(supervisor.public_status()),
                    )
        self.assertEqual(len(private_digests), 4)
        for public in public_manifests.values():
            serialized = json.dumps(public)
            for digest in private_digests:
                self.assertNotIn(digest, serialized)

    def formal_supervisor(self, *, inference=None, gateway_artifact=None, begin=True):
        """A trusted test deployment with every formal contract input supplied."""
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
        (formal_train_dir / "train.json").write_bytes(
            (self.config.train_dir / "train.json").read_bytes()
        )
        write_training_manifest(formal_train_dir, {"frozen": True})
        config = replace(
            self.config,
            source=formal_task,
            train_dir=formal_train_dir,
            task_kind="benchmark",
            leaderboard_eligible=True,
            training_image="example/training" + digest,
            candidate_image="example/candidate" + digest,
            evaluator_image="example/evaluator" + digest,
            inference_gateway_image="example/gateway" + digest,
        )
        agent = replace(self.agent, image="example/agent" + digest, inference=inference)
        self.worker.trust_level = TrustLevel.CERTIFIED
        worker_conformance = ConformanceReceipt(
            receipt_version=CONFORMANCE_RECEIPT_VERSION,
            suite_version=WORKER_CONFORMANCE_SUITE_VERSION,
            worker_id="test-worker",
            authority="test-organizer",
            report_sha256="e" * 64,
            passed=True,
        )
        comparison_key = {
            "benchmark_id": "test-byte-lm-v1",
            "track": "B",
            "track_config": "b-final",
            "runtime_profile": "h100-final",
        }
        release = {
            "benchmark_id": "test-byte-lm-v1",
            "benchmark_manifest_sha256": "b" * 64,
            "task_manifest_sha256": hashlib.sha256(
                config.source.read_bytes()
            ).hexdigest(),
            "release_stage": "official",
            "leaderboard_eligible": True,
            "track_config_id": "b-final",
            "runtime_profile_id": "h100-final",
            "track_config": {
                "provisional": False,
                "track": "B",
                "max_leases": 2,
                "lease_seconds": 10,
            },
            "runtime_profile": {"provisional": False},
        }
        release["resource_envelope"] = v0_resource_envelope(
            self.resources,
            minimum_enforcement=EnforcementLevel.ENFORCED,
        ).public_manifest()
        release["evaluation_resource_envelope"] = release["resource_envelope"]
        artifacts = {
            "benchmark_manifest_sha256": release["benchmark_manifest_sha256"],
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
            "analysis_sha256": hidden_bundle_sha256(config.analysis_path),
            "test_sha256": hidden_bundle_sha256(config.test_path),
        }
        core_revision = "d" * 40
        receipt = ReleaseVerificationReceipt(
            authority="test-organizer",
            release_id="test-byte-lm-v1",
            core_revision=core_revision,
            artifact_sha256=artifacts,
            artifact_identity_algorithms=FORMAL_ARTIFACT_IDENTITY_ALGORITHMS,
            inference_gateway_artifact=gateway_artifact,
        )
        with patch(
            "envcore.supervisor._core_identity",
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
                benchmark_release=release,
                release_verification=receipt,
                worker_conformance=worker_conformance,
                begin=begin,
            )
        return supervisor, artifacts

    def gateway_supervisor(self, *, expected_artifact=True):
        artifact = GatewayArtifactIdentity("test-gateway-bundle-v1", "f" * 64)
        supervisor, _ = self.formal_supervisor(
            inference=InferenceConfig(
                protocol="openai_compatible",
                base_url="https://private.inference.example",
                api_key_env="TEST_GATEWAY_KEY",
                allowed_models=("model",),
                allowed_paths=("/v1/responses",),
                timeout_seconds=30,
            ),
            gateway_artifact=artifact if expected_artifact else None,
            begin=False,
        )
        receipt = GatewayExecutionReceipt(
            receipt_id="gateway:execution-1",
            request_sha256=supervisor.gateway_execution_request.sha256,
            source="test-trusted-launcher",
            artifact=artifact,
            enforcement=EnforcementLevel.ENFORCED,
            trust_level=TrustLevel.CERTIFIED,
        )
        return supervisor, receipt

    def finalize_gateway_supervisor(self, supervisor):
        supervisor.begin()
        supervisor.start_lease()
        candidate = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(candidate["candidate_id"])
        supervisor.finish()
        return self.finalize_after_agent_stop(supervisor)

    def test_formal_gateway_receipt_binds_execution_and_preserves_test_eligibility(
        self,
    ):
        supervisor, receipt = self.gateway_supervisor()
        supervisor.record_gateway_execution(receipt)
        public_task = supervisor.public_task_path.read_text()
        public_status = json.dumps(supervisor.public_status())
        for private_value in (
            receipt.request_sha256,
            receipt.source,
            receipt.artifact.sha256,
            "private.inference.example",
            "TEST_GATEWAY_KEY",
            supervisor.gateway_execution_request.inference_registration_sha256,
            supervisor.gateway_execution_request.inference_policy_sha256,
        ):
            self.assertNotIn(private_value, public_task)
            self.assertNotIn(private_value, public_status)
        self.assertEqual(
            json.loads(supervisor.state_path.read_text())["gateway_execution"][
                "receipt"
            ],
            receipt.private_manifest(),
        )
        final = self.finalize_gateway_supervisor(supervisor)
        self.assertTrue(final["leaderboard_eligible"])
        self.assertEqual(final["leaderboard_eligibility_violations"], [])
        self.assertEqual(
            normalize_result_report(supervisor.state)["gateway_execution"],
            supervisor.state["gateway_execution"],
        )
        for changes in (
            {"enforcement": "best_effort"},
            {"trust_level": "development"},
            {"limitations": ["mutable_code"]},
            {"artifact": {"format": receipt.artifact.format, "sha256": "e" * 64}},
        ):
            payload = copy.deepcopy(supervisor.state)
            payload["gateway_execution"]["receipt"].update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(
                ResultSchemaError, "Gateway evidence is incomplete"
            ):
                normalize_result_report(payload)
        for field in ("receipt", "release_artifact"):
            payload = copy.deepcopy(supervisor.state)
            if field == "receipt":
                payload["gateway_execution"]["receipt"] = None
            else:
                payload["formal_release_verification"]["inference_gateway_artifact"] = (
                    None
                )
            with self.subTest(field=field), self.assertRaisesRegex(
                ResultSchemaError, "Gateway evidence is incomplete"
            ):
                normalize_result_report(payload)
        for field, value in (
            ("protocol", "another-protocol"),
            ("allowed_models", ["another-model"]),
            ("allowed_paths", ["/v1/unregistered"]),
        ):
            payload = copy.deepcopy(supervisor.state)
            payload["agent_system"]["inference"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ResultSchemaError, "Gateway inference policy does not match"
            ):
                normalize_result_report(payload)

    def test_gateway_receipt_cannot_be_replayed_or_changed_after_registration(self):
        supervisor, receipt = self.gateway_supervisor()
        other_request = replace(
            supervisor.gateway_execution_request, episode_id="episode:other"
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            supervisor.record_gateway_execution(
                replace(receipt, request_sha256=other_request.sha256)
            )
        self.assertIsNone(supervisor.state["gateway_execution"]["receipt"])
        supervisor.record_gateway_execution(receipt)
        with self.assertRaisesRegex(ValueError, "already fixed"):
            supervisor.record_gateway_execution(receipt)
        supervisor.begin()
        with self.assertRaisesRegex(ProtocolError, "precede task reveal"):
            supervisor.record_gateway_execution(receipt)

    def test_gateway_receipt_cannot_be_added_after_a_missing_evidence_run_begins(self):
        supervisor, receipt = self.gateway_supervisor()
        supervisor.begin()
        with self.assertRaisesRegex(ProtocolError, "precede task reveal"):
            supervisor.record_gateway_execution(receipt)
        self.assertIn(
            "inference_gateway_workload_evidence_missing",
            supervisor.state["leaderboard_eligibility_violations"],
        )

    def test_gateway_without_frozen_release_artifact_remains_ineligible(self):
        supervisor, receipt = self.gateway_supervisor(expected_artifact=False)
        supervisor.record_gateway_execution(receipt)
        final = self.finalize_gateway_supervisor(supervisor)
        self.assertFalse(final["leaderboard_eligible"])
        self.assertEqual(
            final["leaderboard_eligibility_violations"],
            ["inference_gateway_release_artifact_missing"],
        )
        self.assertEqual(final["test_status"], "evaluated")

    def test_gateway_artifact_substitution_remains_ineligible(self):
        supervisor, receipt = self.gateway_supervisor()
        receipt = replace(receipt, artifact=replace(receipt.artifact, sha256="e" * 64))
        supervisor.record_gateway_execution(receipt)
        final = self.finalize_gateway_supervisor(supervisor)
        self.assertFalse(final["leaderboard_eligible"])
        self.assertEqual(
            final["leaderboard_eligibility_violations"],
            ["inference_gateway_artifact_mismatch"],
        )

    def test_gateway_development_observation_cannot_certify_a_formal_run(self):
        supervisor, receipt = self.gateway_supervisor()
        supervisor.record_gateway_execution(
            replace(receipt, trust_level=TrustLevel.DEVELOPMENT)
        )
        final = self.finalize_gateway_supervisor(supervisor)
        self.assertFalse(final["leaderboard_eligible"])
        self.assertEqual(
            final["leaderboard_eligibility_violations"],
            ["inference_gateway_execution_not_certified"],
        )

    def test_gateway_receipt_is_rejected_when_inference_is_disabled(self):
        supervisor = self.supervisor(begin=False)
        self.assertIsNone(supervisor.gateway_execution_request)
        self.assertIsNone(supervisor.state["gateway_execution"])
        with self.assertRaisesRegex(ValueError, "not enabled"):
            supervisor.record_gateway_execution(None)

    def test_leaderboard_eligibility_requires_the_complete_formal_contract(self):
        supervisor, artifacts = self.formal_supervisor()
        self.assertFalse(supervisor.state["leaderboard_eligible"])
        self.assertEqual(
            supervisor.state["leaderboard_eligibility_violations"],
            [
                "formal_test_evaluation_pending",
                "workload_resolution_pending:training",
                "workload_resolution_pending:candidate",
                "workload_resolution_pending:evaluator",
                "resource_allocation_pending",
                "evaluation_resource_allocation_pending",
            ],
        )
        public_task = json.loads(
            supervisor.public_task_path.read_text(encoding="utf-8")
        )
        self.assertFalse(public_task["formal_release_verified"])

        supervisor.start_lease()
        candidate = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(candidate["candidate_id"])
        supervisor.finish()
        final = self.finalize_after_agent_stop(supervisor)

        self.assertTrue(final["leaderboard_eligible"])
        self.assertEqual(final["leaderboard_eligibility_violations"], [])
        self.assertEqual(
            [
                (request.name, request.expected_sha256)
                for request in self.worker.hidden_dataset_requests
            ],
            [
                ("test", artifacts["test_sha256"]),
                ("analysis", artifacts["analysis_sha256"]),
            ],
        )

    def test_agent_execution_identity_must_match_the_selected_image(self):
        evidence = replace(
            describe_local_agent_execution(self.agent),
            workload_identity="different-agent@sha256:" + "a" * 64,
        )

        with self.assertRaisesRegex(ValueError, "workload identity"):
            self.supervisor(agent_execution=evidence)

    def test_derived_workload_requires_organizer_acceptance_for_formal_use(self):
        digest = "@sha256:" + "a" * 64
        config = replace(
            self.config,
            task_kind="benchmark",
            leaderboard_eligible=True,
            training_image="example/training" + digest,
            candidate_image="example/candidate" + digest,
            evaluator_image="example/evaluator" + digest,
            inference_gateway_image="example/gateway" + digest,
        )
        self.worker.workload_resolutions["training"] = ResolvedWorkload(
            role="training",
            requested_identity=config.training_image,
            effective_format="sif",
            effective_sha256="b" * 64,
            transformation_sha256="c" * 64,
            resolution_id="derived-training",
        )
        supervisor = self.supervisor(config=config)

        with self.assertRaisesRegex(
            ProtocolError, "training workload resolution failed"
        ):
            supervisor.start_lease()

        self.assertEqual(supervisor.state["status"], "infrastructure_failure")
        self.assertEqual(
            supervisor.state["workload_resolutions"]["training"]["acceptance"],
            "unverified",
        )
        self.assertIn(
            "workload_resolution_not_accepted:training",
            supervisor.state["leaderboard_eligibility_violations"],
        )
        self.assertEqual(self.worker.lease_workloads, [])

    def test_trusted_release_receipt_accepts_a_derived_workload(self):
        digest = "@sha256:" + "a" * 64
        config = replace(
            self.config,
            task_kind="benchmark",
            leaderboard_eligible=True,
            training_image="example/training" + digest,
            candidate_image="example/candidate" + digest,
            evaluator_image="example/evaluator" + digest,
            inference_gateway_image="example/gateway" + digest,
        )
        resolved = ResolvedWorkload(
            role="training",
            requested_identity=config.training_image,
            effective_format="sif",
            effective_sha256="b" * 64,
            transformation_sha256="c" * 64,
            resolution_id="derived-training",
        )
        self.worker.workload_resolutions["training"] = resolved
        acceptance = WorkloadResolutionAcceptance(
            role=resolved.role,
            requested_identity=resolved.requested_identity,
            effective_format=resolved.effective_format,
            effective_sha256=resolved.effective_sha256,
            transformation_sha256=resolved.transformation_sha256,
        )
        receipt = ReleaseVerificationReceipt(
            authority="test-organizer",
            release_id="test-byte-lm-v1",
            core_revision="d" * 40,
            artifact_sha256={
                field: "a" * 64 for field in FORMAL_ARTIFACT_DIGEST_FIELDS
            },
            artifact_identity_algorithms=FORMAL_ARTIFACT_IDENTITY_ALGORITHMS,
            workload_resolution_acceptance={"training": acceptance},
        )
        supervisor = self.supervisor(config=config, release_verification=receipt)

        supervisor.start_lease()

        self.assertEqual(self.worker.lease_workloads, [resolved])
        self.assertEqual(
            supervisor.state["workload_resolutions"]["training"]["acceptance"],
            "organizer_receipt",
        )
        self.assertNotIn(
            "workload_resolution_pending:training",
            supervisor.state["leaderboard_eligibility_violations"],
        )

    def test_evaluation_workloads_are_not_resolved_before_research_teardown(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        candidate = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(candidate["candidate_id"])
        supervisor.finish()

        self.assertEqual(
            [request.role for request in self.worker.workload_requests],
            ["training"],
        )

        self.finalize_after_agent_stop(supervisor)

        self.assertEqual(
            [request.role for request in self.worker.workload_requests],
            ["training", "candidate", "evaluator"],
        )

    def test_test_operation_passes_only_remaining_deadline_to_evaluator(self):
        config = replace(self.config, evaluator_timeout_seconds=30)
        supervisor = self.supervisor(config=config)

        def resolve_workload(workload, timeout_seconds):
            self.assertGreater(timeout_seconds, 0)
            if workload.role == "candidate":
                self.clock.advance(5)
            elif workload.role == "evaluator":
                self.clock.advance(7)

        def resolve_hidden(hidden_dataset):
            if hidden_dataset.name == "test":
                self.clock.advance(3)

        self.worker.workload_resolution_hook = resolve_workload
        self.worker.hidden_dataset_resolution_hook = resolve_hidden
        supervisor.start_lease()
        candidate = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(candidate["candidate_id"])
        supervisor.finish()

        final = self.finalize_after_agent_stop(supervisor)

        self.assertEqual(final["outcome"], "success")
        self.assertEqual(self.worker.evaluation_timeouts[0], 15.0)
        self.assertEqual(
            self.worker.hidden_dataset_resolution_timeouts[0],
            18.0,
        )

    def test_workload_resolution_overrun_fails_before_test_execution(self):
        config = replace(self.config, evaluator_timeout_seconds=30)
        supervisor = self.supervisor(config=config)

        def overrun(workload, timeout_seconds):
            if workload.role == "candidate":
                self.clock.advance(timeout_seconds + 1)

        self.worker.workload_resolution_hook = overrun
        supervisor.start_lease()
        candidate = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(candidate["candidate_id"])
        supervisor.finish()

        final = self.finalize_after_agent_stop(supervisor)

        self.assertEqual(final["status"], "infrastructure_failure")
        self.assertEqual(final["test_status"], "failed")
        self.assertEqual(final["failure"], "evaluation_workload_resolution_failure")
        self.assertEqual(self.worker.evaluations, [])

    def test_late_test_metric_is_not_accepted(self):
        config = replace(self.config, evaluator_timeout_seconds=30)
        supervisor = self.supervisor(config=config)

        def finish_late(split, timeout_seconds):
            if split == "test":
                self.clock.advance(timeout_seconds + 1)

        self.worker.evaluation_hook = finish_late
        supervisor.start_lease()
        candidate = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(candidate["candidate_id"])
        supervisor.finish()

        final = self.finalize_after_agent_stop(supervisor)

        self.assertEqual(final["status"], "finalized")
        self.assertEqual(final["outcome"], "failed")
        self.assertEqual(final["failure"], "test_evaluation_failed")
        self.assertIsNone(final["test_performance_score"])

    def test_formal_inference_fails_closed_without_gateway_execution_evidence(self):
        digest = "@sha256:" + "a" * 64
        config = replace(
            self.config,
            task_kind="benchmark",
            leaderboard_eligible=True,
            training_image="example/training" + digest,
            candidate_image="example/candidate" + digest,
            evaluator_image="example/evaluator" + digest,
            inference_gateway_image="example/gateway" + digest,
        )
        agent = replace(
            self.agent,
            image="example/agent" + digest,
            inference=InferenceConfig(
                protocol="openai_compatible",
                base_url="https://inference.example",
                api_key_env=None,
                allowed_models=("model",),
                allowed_paths=("/v1/responses",),
                timeout_seconds=30,
            ),
        )

        supervisor = self.supervisor(config=config, agent=agent)

        self.assertIn(
            "inference_gateway_workload_evidence_missing",
            supervisor.state["leaderboard_eligibility_violations"],
        )

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
            task_kind="benchmark",
            leaderboard_eligible=True,
            training_image="example/training" + digest,
            candidate_image="example/candidate" + digest,
            evaluator_image="example/evaluator" + digest,
            inference_gateway_image="example/gateway" + digest,
        )
        agent = replace(self.agent, image="example/agent" + digest)
        self.worker.trust_level = TrustLevel.CERTIFIED
        worker_conformance = ConformanceReceipt(
            receipt_version=CONFORMANCE_RECEIPT_VERSION,
            suite_version=WORKER_CONFORMANCE_SUITE_VERSION,
            worker_id="test-worker",
            authority="test-organizer",
            report_sha256="e" * 64,
            passed=True,
        )
        with patch(
            "envcore.supervisor._core_identity",
            return_value={"revision": "d" * 40, "dirty": True},
        ):
            supervisor = self.supervisor(
                config=config,
                agent=agent,
                comparison_key={
                    "benchmark_id": "test-byte-lm-v1",
                    "track": "B",
                    "track_config": "b-final",
                    "runtime_profile": "h100-final",
                },
                benchmark_release={
                    "benchmark_id": "test-byte-lm-v1",
                    "benchmark_manifest_sha256": "b" * 64,
                    "task_manifest_sha256": "c" * 64,
                    "release_stage": "official",
                    "leaderboard_eligible": True,
                    "track_config_id": "b-final",
                    "runtime_profile_id": "h100-final",
                    "track_config": {
                        "provisional": False,
                        "track": "B",
                        "max_leases": 2,
                        "lease_seconds": 10,
                    },
                    "runtime_profile": {"provisional": False},
                    "resource_envelope": v0_resource_envelope(
                        self.resources, minimum_enforcement=EnforcementLevel.ENFORCED
                    ).public_manifest(),
                },
                worker_conformance=worker_conformance,
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
        with patch("envcore.supervisor.subprocess.run", side_effect=results) as run:
            first = _core_identity()
            second = _core_identity()

        self.assertFalse(first["dirty"])
        self.assertTrue(second["dirty"])
        self.assertEqual(run.call_count, 4)

    def test_worker_role_specs_do_not_cross_hidden_dataset_boundaries(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        candidate = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(candidate["candidate_id"])
        supervisor.finish()
        self.finalize_after_agent_stop(supervisor)

        self.assertEqual(self.worker.research_tasks[0].train_dir, self.config.train_dir)
        self.assertFalse(hasattr(self.worker.research_tasks[0], "analysis_path"))
        self.assertEqual(
            {
                path.name
                for path in (
                    self.worker.research_tasks[0].public_api_dir / "envcore"
                ).iterdir()
            },
            {
                "__init__.py",
                "cli.py",
                "client.py",
                "candidate_protocol.py",
            },
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.worker.research_tasks[0].public_api_dir)
        smoke = subprocess.run(
            [sys.executable, "-m", "envcore.cli", "agent", "--help"],
            cwd=self.workspace,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(smoke.returncode, 0, smoke.stderr)
        self.assertIn("lease-start", smoke.stdout)
        self.assertEqual(
            [item.name for item in self.worker.hidden_dataset_requests],
            ["test", "analysis"],
        )
        self.assertTrue(
            all(
                not hasattr(item, "path")
                for item in self.worker.hidden_dataset_requests
                + self.worker.hidden_datasets
            )
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
            public_task["incumbent_declaration_limit"], MAX_INCUMBENT_DECLARATIONS
        )
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
        self.assertNotIn("val_nll", committed)
        self.assertNotIn("analysis_performance_score", committed)
        self.assertEqual(self.worker.evaluations, [])
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
        self.assertIsNone(staged["test_performance_score"])
        self.assertIsNone(supervisor.state["agent_control"]["cleanup_complete"])
        self.assertNotIn(
            "RESULT_PUBLICATION_PREPARED",
            supervisor.events_path.read_text(encoding="utf-8"),
        )

        final = self.finalize_after_agent_stop(supervisor)
        self.assertEqual(final["status"], "finalized")
        self.assertEqual(final["result_schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(final["outcome"], "success")
        self.assertEqual(final["test_performance_score"], 0.0)
        self.assertEqual(final["test_raw_metrics"]["mean_nll"], 1.5)
        self.assertEqual(
            final["test_cost"],
            {
                "time_per_task": 2.0 / 3.0,
                "candidate_wall_seconds": 2.0,
                "task_count": 3,
                "input_tokens": 6,
                "prediction_bytes": 64,
                "tasks_per_second": 1.5,
                "input_tokens_per_second": 3.0,
            },
        )
        self.assertEqual(
            final["analysis_trajectory"][0]["analysis_cost"],
            final["test_cost"],
        )
        self.assertIsNone(final["failure"])
        self.assertTrue(final["agent_finish_observed"])
        self.assertEqual(self.worker.evaluations, ["test", "analysis"])
        with self.assertRaises(ProtocolError):
            supervisor.start_lease()

        events = [
            json.loads(line)
            for line in supervisor.events_path.read_text(encoding="utf-8").splitlines()
        ]
        types = [event["type"] for event in events]
        self.assertLess(
            types.index("INCUMBENT_DECLARED"), types.index("RESEARCH_SEALED")
        )
        self.assertLess(
            types.index("AGENT_PROCESS_EXITED"),
            types.index("AGENT_CONTROL_CLEANED"),
        )
        self.assertLess(
            types.index("AGENT_CONTROL_CLEANED"),
            types.index("HIDDEN_DATASET_RESOLVED"),
        )
        self.assertLess(
            types.index("HIDDEN_DATASET_RESOLVED"),
            types.index("TEST_EVALUATED"),
        )
        self.assertLess(
            types.index("TEST_EVALUATED"), types.index("ANALYSIS_EVALUATED")
        )
        self.assertLess(
            types.index("ANALYSIS_EVALUATED"),
            types.index("RESULT_PUBLICATION_PREPARED"),
        )
        self.assertEqual(types.count("RESULT_PUBLICATION_PREPARED"), 1)
        prepared = next(
            event for event in events if event["type"] == "RESULT_PUBLICATION_PREPARED"
        )
        self.assertEqual(
            supervisor.state["result_publication"],
            {
                "publication_id": prepared["publication_id"],
                "prepared_event_seq": prepared["seq"],
            },
        )

    def test_hidden_data_can_be_materialized_only_after_research_teardown(self):
        hidden = self.episode.parent / "late-hidden"
        config = replace(
            self.config,
            analysis_path=hidden / "analysis",
            test_path=hidden / "test",
        ).validate()
        supervisor = self.supervisor(config=config)
        supervisor.start_lease()
        committed = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(committed["candidate_id"])
        supervisor.finish()

        self.assertFalse(hidden.exists())
        self.assertEqual(self.worker.evaluations, [])
        with self.assertRaisesRegex(RuntimeError, "termination must precede"):
            supervisor.evaluate_postmortem()
        self.assertEqual(self.worker.hidden_dataset_requests, [])

        supervisor.record_agent_execution(exit_code=0, termination_reason="exited")
        self.assertTrue(supervisor.cleanup_agent_control())

        def materialize(request):
            path = hidden / request.name
            path.mkdir(parents=True)
            token = 1 if request.name == "analysis" else 2
            (path / "contexts.json").write_text(
                json.dumps({"contexts": [[0]]}),
                encoding="utf-8",
            )
            (path / "targets.json").write_text(
                json.dumps({"targets": [token]}), encoding="utf-8"
            )
            self.worker.hidden_dataset_digests[request.name] = hidden_bundle_sha256(
                path
            )

        self.worker.hidden_dataset_resolution_hook = materialize

        staged = supervisor.evaluate_postmortem()
        self.assertEqual(staged["status"], "closed")
        final = supervisor.publish_postmortem_result()

        self.assertEqual(final["status"], "finalized")
        self.assertEqual(self.worker.evaluations, ["test", "analysis"])
        self.assertTrue(config.analysis_path.is_dir())
        self.assertTrue(config.test_path.is_dir())
        for path in (config.analysis_path, config.test_path):
            self.assertTrue((path / "contexts.json").is_file())
            self.assertTrue((path / "targets.json").is_file())
        identities = supervisor.state["artifact_identities"]["hidden_evaluation"]
        self.assertEqual(set(identities), {"analysis_sha256", "test_sha256"})
        self.assertTrue(all(identities.values()))
        self.assertEqual(
            supervisor.state["artifact_identity_algorithms"]["hidden_evaluation"],
            HIDDEN_BUNDLE_IDENTITY_ALGORITHM,
        )

    def test_postmortem_result_is_not_published_before_deployment_cleanup(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        committed = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(committed["candidate_id"])
        supervisor.finish()
        supervisor.record_agent_execution(exit_code=0, termination_reason="exited")
        self.assertTrue(supervisor.cleanup_agent_control())

        staged = supervisor.evaluate_postmortem()

        self.assertEqual(staged["status"], "closed")
        self.assertIsNone(staged["outcome"])
        self.assertIsNotNone(supervisor.state["pending_result"])

        supervisor.deployment_teardown_failed(
            "evaluation_private_storage_teardown_failure"
        )

        self.assertEqual(supervisor.state["status"], "infrastructure_failure")
        self.assertIsNone(supervisor.state["result_publication"])

    def test_test_dataset_identity_mismatch_fails_before_candidate_evaluation(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        committed = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(committed["candidate_id"])
        supervisor.finish()
        supervisor.record_agent_execution(exit_code=0, termination_reason="exited")
        self.assertTrue(supervisor.cleanup_agent_control())

        with patch.object(
            supervisor,
            "_hidden_dataset_request",
            return_value=HiddenDatasetRequest("test", "a" * 64),
        ):
            final = supervisor.evaluate_postmortem()

        self.assertEqual(final["status"], "infrastructure_failure")
        self.assertEqual(final["failure"], "hidden_dataset_identity_mismatch")
        self.assertEqual(self.worker.evaluations, [])
        events = supervisor.events_path.read_text(encoding="utf-8")
        self.assertIn("HIDDEN_DATASET_IDENTITY_MISMATCH", events)

    def test_test_resolution_failure_exposes_only_a_stable_code(self):
        private_detail = "/organizer/private/test-download.log"
        self.worker.hidden_dataset_resolution_failures["test"] = private_detail
        supervisor = self.supervisor()
        supervisor.start_lease()
        committed = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(committed["candidate_id"])
        supervisor.finish()

        final = self.finalize_after_agent_stop(supervisor)

        self.assertEqual(final["status"], "infrastructure_failure")
        self.assertEqual(final["failure"], "hidden_dataset_resolution_failure")
        self.assertNotIn(private_detail, json.dumps(final))
        self.assertIn(
            private_detail,
            supervisor.events_path.read_text(encoding="utf-8"),
        )
        self.assertEqual(self.worker.evaluations, [])

    def test_invalid_hidden_dataset_receipt_is_infrastructure_failure(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        committed = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(committed["candidate_id"])
        supervisor.finish()
        supervisor.record_agent_execution(exit_code=0, termination_reason="exited")
        self.assertTrue(supervisor.cleanup_agent_control())

        with patch.object(
            self.worker,
            "resolve_hidden_dataset",
            return_value={"path": "/organizer/private/test.json"},
        ):
            final = supervisor.evaluate_postmortem()

        self.assertEqual(final["status"], "infrastructure_failure")
        self.assertEqual(final["failure"], "hidden_dataset_receipt_invalid")
        self.assertNotIn("/organizer/private", json.dumps(final))
        self.assertEqual(self.worker.evaluations, [])

    def test_unsubstantiated_hidden_mismatch_is_terminal(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        committed = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(committed["candidate_id"])
        supervisor.finish()
        receipt = ResolvedHiddenDataset(
            "test",
            self.worker.hidden_dataset_digests["test"],
            "unsubstantiated-mismatch",
        )

        with patch.object(
            self.worker,
            "resolve_hidden_dataset",
            side_effect=HiddenDatasetIdentityMismatch(receipt),
        ):
            final = self.finalize_after_agent_stop(supervisor)

        self.assertEqual(final["status"], "infrastructure_failure")
        self.assertEqual(final["test_status"], "failed")
        self.assertEqual(final["failure"], "hidden_dataset_receipt_invalid")
        self.assertEqual(self.worker.evaluations, [])

    def test_hidden_mismatch_matching_prior_identity_is_invalid(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        committed = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(committed["candidate_id"])
        supervisor.finish()
        supervisor.state["artifact_identities"]["hidden_evaluation"]["test_sha256"] = (
            self.worker.hidden_dataset_digests["test"]
        )
        receipt = ResolvedHiddenDataset(
            "test",
            self.worker.hidden_dataset_digests["test"],
            "matching-mismatch",
        )

        with patch.object(
            self.worker,
            "resolve_hidden_dataset",
            side_effect=HiddenDatasetIdentityMismatch(receipt),
        ):
            final = self.finalize_after_agent_stop(supervisor)

        self.assertEqual(final["status"], "infrastructure_failure")
        self.assertEqual(final["test_status"], "failed")
        self.assertEqual(final["failure"], "hidden_dataset_receipt_invalid")
        self.assertEqual(self.worker.evaluations, [])

    def test_hidden_resolution_overrun_is_terminal(self):
        config = replace(self.config, evaluator_timeout_seconds=30)
        supervisor = self.supervisor(config=config)
        supervisor.start_lease()
        committed = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(committed["candidate_id"])
        supervisor.finish()

        def overrun(hidden_dataset):
            if hidden_dataset.name == "test":
                self.clock.advance(31)

        self.worker.hidden_dataset_resolution_hook = overrun
        final = self.finalize_after_agent_stop(supervisor)

        self.assertEqual(final["status"], "infrastructure_failure")
        self.assertEqual(final["test_status"], "failed")
        self.assertEqual(final["failure"], "hidden_dataset_resolution_failure")
        self.assertEqual(self.worker.evaluations, [])

    def test_unexpected_resolver_failure_marks_worker_state_unknown(self):
        private_detail = "/organizer/private/mount-helper.log"
        supervisor = self.supervisor()
        supervisor.start_lease()
        committed = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(committed["candidate_id"])
        supervisor.finish()
        supervisor.record_agent_execution(exit_code=0, termination_reason="exited")
        self.assertTrue(supervisor.cleanup_agent_control())

        with patch.object(
            self.worker,
            "resolve_hidden_dataset",
            side_effect=RuntimeError(private_detail),
        ):
            final = supervisor.evaluate_postmortem()

        self.assertEqual(final["status"], "infrastructure_failure")
        self.assertEqual(final["failure"], "evaluation_worker_state_unknown")
        self.assertEqual(
            final["deployment_teardown_failures"],
            ["evaluation_worker_state_unknown"],
        )
        self.assertNotIn(private_detail, json.dumps(final))
        self.assertIn(
            private_detail,
            supervisor.events_path.read_text(encoding="utf-8"),
        )
        self.assertEqual(self.worker.evaluations, [])

    def test_finish_without_an_incumbent_is_a_failed_experiment(self):
        supervisor = self.supervisor()

        supervisor.finish()
        final = self.finalize_after_agent_stop(supervisor)

        self.assertEqual(final["status"], "finalized")
        self.assertEqual(final["outcome"], "failed")
        self.assertEqual(final["failure"], "no_candidate")
        self.assertIsNone(final["test_performance_score"])
        self.assertIsNone(final["test_raw_metrics"])
        self.assertEqual(self.worker.evaluations, [])

    def test_committed_candidates_without_incumbent_receive_postmortem_analysis(self):
        supervisor = self.supervisor()
        candidates = []
        for index in range(2):
            lease = supervisor.start_lease()
            output = supervisor.runs_dir / lease["lease_id"] / "output/candidate"
            (output / "weights.bin").write_bytes(bytes([index]))
            candidates.append(
                supervisor.commit_lease("/output/candidate")["candidate_id"]
            )
        supervisor.finish()

        self.assertEqual(self.worker.evaluations, [])
        self.assertEqual(self.worker.hidden_dataset_requests, [])
        self.assertEqual(
            [request.role for request in self.worker.workload_requests], ["training"]
        )
        with self.assertRaisesRegex(RuntimeError, "termination must precede"):
            supervisor.evaluate_postmortem()

        def assert_sealed(workload, _timeout):
            self.assertEqual(supervisor.state["status"], "closed")
            self.assertIsNotNone(supervisor.state["agent_system"]["termination_reason"])
            self.assertTrue(supervisor.state["agent_control"]["cleanup_complete"])
            self.assertFalse(self.worker.running)

        self.worker.workload_resolution_hook = assert_sealed
        final = self.finalize_after_agent_stop(supervisor)

        self.assertEqual(final["status"], "finalized")
        self.assertEqual(final["outcome"], "failed")
        self.assertEqual(final["failure"], "no_candidate")
        self.assertIsNone(final["final_candidate_id"])
        self.assertIsNone(final["test_performance_score"])
        self.assertIsNone(final["test_raw_metrics"])
        self.assertIsNone(final["test_cost"])
        self.assertEqual(final["incumbent_declarations"], [])
        self.assertTrue(final["analysis_complete"])
        self.assertEqual(final["analysis_status"], "complete")
        self.assertIsNone(final["analysis_failure"])
        self.assertEqual(final["analysis_selected_points"], 2)
        self.assertEqual(final["analysis_evaluated_points"], 2)
        self.assertEqual(
            [point["candidate_id"] for point in final["analysis_trajectory"]],
            candidates,
        )
        self.assertTrue(
            all(
                point["analysis_status"] == "evaluated"
                for point in final["analysis_trajectory"]
            )
        )
        self.assertEqual(self.worker.evaluations, ["analysis", "analysis"])
        self.assertEqual(
            [request.name for request in self.worker.hidden_dataset_requests],
            ["analysis"],
        )
        self.assertEqual(
            [request.role for request in self.worker.workload_requests],
            ["training", "candidate", "evaluator"],
        )
        self.assertEqual(final["deployment_teardown_failures"], [])
        normalize_result_report(supervisor.state)

    def test_analysis_workload_resolution_failure_preserves_no_candidate_result(self):
        private_detail = "/organizer/private/workload-resolution.log"
        for role in ("candidate", "evaluator"):
            with self.subTest(role=role):
                self.episode = self.episode.parent / f"episode-{role}"
                supervisor = self.supervisor()
                supervisor.start_lease()
                supervisor.commit_lease("/output/candidate")
                supervisor.finish()
                self.worker.workload_resolution_failures = {role: private_detail}

                final = self.finalize_after_agent_stop(supervisor)

                self.assertEqual(final["status"], "finalized")
                self.assertEqual(final["failure"], "no_candidate")
                self.assertEqual(final["test_status"], "failed")
                self.assertEqual(final["analysis_status"], "incomplete")
                self.assertEqual(
                    final["analysis_failure"], "analysis_workload_resolution_failure"
                )
                self.assertEqual(
                    final["analysis_trajectory"][0]["analysis_status"],
                    "infrastructure_failure",
                )
                self.assertEqual(
                    final["analysis_trajectory"][0]["analysis_error"],
                    "workload_resolution_failure",
                )
                self.assertEqual(final["analysis_evaluated_points"], 0)
                self.assertEqual(final["deployment_teardown_failures"], [])
                self.assertEqual(self.worker.evaluations, [])
                self.assertNotIn(private_detail, json.dumps(final))
                self.assertIn(private_detail, supervisor.events_path.read_text())
                normalize_result_report(supervisor.state)

    def test_analysis_workload_resolution_shares_candidate_evaluation_deadline(self):
        supervisor = self.supervisor(
            config=replace(self.config, evaluator_timeout_seconds=30)
        )
        supervisor.start_lease()
        supervisor.commit_lease("/output/candidate")
        supervisor.finish()

        def resolve_workload(workload, _timeout):
            self.clock.advance(5 if workload.role == "candidate" else 7)

        self.worker.workload_resolution_hook = resolve_workload
        self.worker.evaluation_hook = lambda _split, _timeout: self.clock.advance(2)
        final = self.finalize_after_agent_stop(supervisor)

        self.assertTrue(final["analysis_complete"])
        self.assertEqual(final["failure"], "no_candidate")
        self.assertEqual(self.worker.workload_resolution_timeouts[-2:], [30, 25])
        self.assertEqual(self.worker.evaluation_timeouts, [18])
        self.assertEqual(final["timing"]["analysis_evaluation_seconds"], 14)
        self.assertEqual(final["timing"]["test_evaluation_seconds"], 0)

    def test_analysis_workload_resolution_overrun_prevents_evaluation(self):
        for role in ("candidate", "evaluator"):
            with self.subTest(role=role):
                self.episode = self.episode.parent / f"episode-{role}"
                supervisor = self.supervisor(
                    config=replace(self.config, evaluator_timeout_seconds=30)
                )
                supervisor.start_lease()
                supervisor.commit_lease("/output/candidate")
                supervisor.finish()

                def overrun(workload, timeout_seconds):
                    if workload.role == role:
                        self.clock.advance(timeout_seconds + 1)

                self.worker.workload_resolution_hook = overrun
                final = self.finalize_after_agent_stop(supervisor)

                self.assertEqual(final["failure"], "no_candidate")
                self.assertEqual(final["analysis_status"], "incomplete")
                self.assertEqual(
                    final["analysis_failure"], "analysis_workload_resolution_failure"
                )
                self.assertEqual(final["timing"]["analysis_evaluation_seconds"], 31)
                self.assertEqual(final["deployment_teardown_failures"], [])
                self.assertEqual(self.worker.evaluations, [])

    def test_unexpected_analysis_workload_resolution_failure_keeps_teardown_evidence(
        self,
    ):
        private_detail = "/organizer/private/workload-process.log"
        supervisor = self.supervisor()
        supervisor.start_lease()
        supervisor.commit_lease("/output/candidate")
        supervisor.finish()
        self.worker.workload_resolution_exception = RuntimeError(private_detail)

        final = self.finalize_after_agent_stop(supervisor)

        self.assertEqual(final["failure"], "no_candidate")
        self.assertEqual(final["analysis_failure"], "analysis_worker_teardown_failure")
        self.assertEqual(
            final["deployment_teardown_failures"],
            ["evaluation_worker_teardown_failure"],
        )
        self.assertFalse(final["policy_compliant"])
        self.assertEqual(self.worker.evaluations, [])
        self.assertNotIn(private_detail, json.dumps(final))
        self.assertIn(
            "workload_resolver_state_unknown", supervisor.events_path.read_text()
        )

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
        self.assertIsNone(final["test_performance_score"])
        self.assertIsNone(final["test_raw_metrics"])

    def test_test_worker_teardown_failure_is_infrastructure_failure(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        committed = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(committed["candidate_id"])
        private_detail = "/organizer/private/evaluator-stop.log"
        self.worker.evaluation_exception = WorkerTeardownFailure(private_detail)

        supervisor.finish()
        result = self.finalize_after_agent_stop(supervisor)

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

        def fail_test(split, _timeout):
            if split == "test":
                raise RuntimeError("simulated coordinator failure")

        self.worker.evaluation_hook = fail_test
        supervisor.finish()
        final = self.finalize_after_agent_stop(supervisor)

        self.assertEqual(final["status"], "infrastructure_failure")
        self.assertEqual(final["outcome"], "failed")
        self.assertEqual(final["failure"], "test_evaluation_infrastructure_failure")
        self.assertIsNone(final["test_performance_score"])
        self.assertIsNone(final["test_raw_metrics"])

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
        self.assertIsNone(final["test_performance_score"])
        self.assertIsNone(final["test_raw_metrics"])

    def test_large_finite_nll_still_produces_a_finite_score(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        committed = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(committed["candidate_id"])
        self.worker.test_nll = 710.0

        supervisor.finish()
        final = self.finalize_after_agent_stop(supervisor)

        self.assertEqual(final["outcome"], "success")
        self.assertEqual(
            final["test_performance_score"],
            0.0,
        )
        self.assertEqual(final["test_raw_metrics"], {"mean_nll": 710.0})

    def test_nonzero_performance_score_is_persisted_in_the_final_result(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        committed = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(committed["candidate_id"])
        self.worker.test_nll = 0.5

        supervisor.finish()
        final = self.finalize_after_agent_stop(supervisor)
        persisted = json.loads(supervisor.state_path.read_text(encoding="utf-8"))

        self.assertEqual(final["outcome"], "success")
        self.assertAlmostEqual(
            final["test_performance_score"],
            54.488038668658135,
        )
        self.assertEqual(final["test_raw_metrics"], {"mean_nll": 0.5})
        self.assertEqual(
            persisted["test_performance_score"],
            final["test_performance_score"],
        )
        self.assertEqual(persisted["test_raw_metrics"], {"mean_nll": 0.5})

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
            return original_start(**kwargs)

        with patch.object(self.worker, "start_lease", side_effect=observe_intent):
            supervisor.start_lease()

        self.assertEqual(supervisor.state["pending_lease_teardowns"], [])

    def test_accepted_resource_receipt_is_bound_to_the_active_lease(self):
        supervisor = self.supervisor()

        lease = supervisor.start_lease()

        self.assertTrue(lease["resource_allocation_verified"])
        active = supervisor.state["active_lease"]
        allocation = supervisor.state["resource_allocations"][0]
        self.assertTrue(allocation["accepted"])
        self.assertEqual(
            active["resource_allocation_receipt_id"],
            allocation["receipt"]["receipt_id"],
        )
        self.assertEqual(
            allocation["receipt"]["resource_request_sha256"],
            supervisor.resource_envelope.sha256,
        )

    def test_resource_mismatch_revokes_the_started_workload_and_fails_closed(self):
        supervisor = self.supervisor()
        envelope = supervisor.resource_envelope
        self.worker.resource_receipt_override = ResourceAllocationReceipt(
            episode_id=supervisor.state["episode_id"],
            lease_id="lease-0001",
            worker_id="test-worker",
            resource_request_sha256=envelope.sha256,
            properties=tuple(
                ResourcePropertyEvidence(
                    name=requirement.name,
                    allocator_allocated=ResourceObservation(
                        "observed",
                        2 if requirement.name == "host.count" else requirement.value,
                        "test-allocator",
                    ),
                    workload_visible=ResourceObservation("not_applicable"),
                    enforcement_level=EnforcementLevel.ENFORCED,
                    enforcement_authority="test-worker",
                )
                for requirement in envelope.requirements
            ),
            receipt_id="mismatched-allocation",
        )

        with self.assertRaisesRegex(ProtocolError, "evidence was rejected"):
            supervisor.start_lease()

        self.assertEqual(supervisor.state["status"], "infrastructure_failure")
        self.assertEqual(supervisor.state["failure"], "resource_allocation_mismatch")
        self.assertEqual(supervisor.state["leases_used"], 0)
        self.assertEqual(supervisor.state["pending_lease_teardowns"], [])
        self.assertEqual(self.worker.running, set())
        self.assertFalse(supervisor.state["resource_allocations"][0]["accepted"])

    def test_resource_receipt_from_another_worker_is_rejected_and_revoked(self):
        supervisor = self.supervisor()
        envelope = supervisor.resource_envelope
        self.worker.resource_receipt_override = ResourceAllocationReceipt(
            episode_id=supervisor.state["episode_id"],
            lease_id="lease-0001",
            worker_id="different-worker",
            resource_request_sha256=envelope.sha256,
            properties=(),
            receipt_id="wrong-worker-allocation",
        )

        with self.assertRaisesRegex(ProtocolError, "evidence was rejected"):
            supervisor.start_lease()

        self.assertEqual(
            supervisor.state["failure"],
            "resource_allocation_receipt_invalid",
        )
        self.assertEqual(self.worker.running, set())
        self.assertEqual(supervisor.state["pending_lease_teardowns"], [])

    def test_failed_lease_directory_cleanup_is_a_teardown_failure(self):
        supervisor = self.supervisor()
        self.worker.fail_start = True
        private_detail = "/trusted/episode/lease directory is busy"

        with patch(
            "envcore.supervisor.shutil.rmtree",
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

    def test_hidden_dataset_identity_verification_is_a_required_capability(self):
        self.worker.capability_overrides[HIDDEN_DATASET_IDENTITY_VERIFICATION] = (
            Capability(EnforcementLevel.UNVERIFIED)
        )

        supervisor = self.supervisor()

        self.assertIn(
            "evaluation_worker_capability_insufficient:"
            "hidden_dataset_identity_verification",
            supervisor.state["policy_violations"],
        )

    def test_prediction_storage_quota_is_a_required_capability(self):
        self.worker.capability_overrides[CANDIDATE_PREDICTION_STORAGE_QUOTA] = (
            Capability(EnforcementLevel.UNSUPPORTED, "unbounded bind mount")
        )

        supervisor = self.supervisor()

        self.assertIn(
            "evaluation_worker_capability_insufficient:"
            "candidate_prediction_storage_quota",
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
        (candidate / "candidate.py").symlink_to(self.config.analysis_path)
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
        self.assertEqual(timing["analysis_evaluation_seconds"], 0.0)
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

    def test_track_a_commit_never_runs_hidden_evaluation(self):
        config = replace(
            self.config,
            track_a_total_seconds=5,
            evaluator_timeout_seconds=30,
        )
        supervisor = self.supervisor(track="A", config=config)
        supervisor.start_lease(4)
        self.clock.advance(4)
        committed = supervisor.commit_lease("/output/candidate")

        self.assertEqual(self.worker.evaluation_timeouts, [])
        self.assertNotIn("val_nll", committed)
        self.assertNotIn("analysis_performance_score", committed)
        self.assertEqual(supervisor.state["status"], "active")
        self.clock.advance(2)
        self.assertTrue(supervisor.enforce_deadlines())
        self.finalize_after_agent_stop(supervisor)
        self.assertEqual(supervisor.state["status"], "finalized")
        self.assertEqual(supervisor.state["outcome"], "failed")
        self.assertEqual(supervisor.state["failure"], "no_candidate")
        self.assertEqual(supervisor.state["closure_reason"], "track_a_deadline")
        self.assertEqual(supervisor.state["elapsed_seconds"], 5.0)
        self.assertEqual(supervisor.state["timing"]["analysis_evaluation_seconds"], 0.0)
        self.assertIsNone(supervisor.state["final_candidate_id"])
        self.assertTrue(supervisor.state["analysis_complete"])
        self.assertEqual(supervisor.state["analysis_evaluated_points"], 1)
        self.assertEqual(self.worker.evaluations, ["analysis"])
        self.assertEqual(supervisor.state["deployment_teardown_failures"], [])

    def test_analysis_failures_expose_only_a_fixed_error_code(self):
        supervisor = self.supervisor()
        self.worker.evaluation_failures["analysis"] = "hidden input_ids: [0, 1, 2]"
        supervisor.start_lease()

        committed = supervisor.commit_lease("/output/candidate")
        self.assertNotIn("evaluation_error", committed)
        supervisor.declare_incumbent(committed["candidate_id"])
        supervisor.finish()
        final = self.finalize_after_agent_stop(supervisor)

        self.assertEqual(
            final["analysis_trajectory"][0]["analysis_error"],
            "evaluation_failed",
        )
        public = json.dumps(supervisor.public_status())
        events = supervisor.events_path.read_text(encoding="utf-8")
        self.assertNotIn("input_ids", public)
        self.assertNotIn("input_ids", events)

    def test_missing_analysis_is_diagnostic_and_does_not_erase_test(self):
        missing_analysis = self.episode.parent / "not-materialized-analysis.json"
        private_detail = "private organizer source is unavailable"
        supervisor = self.supervisor(
            config=replace(self.config, analysis_path=missing_analysis).validate()
        )
        self.worker.hidden_dataset_resolution_failures["analysis"] = private_detail
        supervisor.start_lease()
        committed = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(committed["candidate_id"])
        supervisor.finish()

        final = self.finalize_after_agent_stop(supervisor)

        self.assertEqual(final["status"], "finalized")
        self.assertEqual(final["outcome"], "success")
        self.assertEqual(final["test_status"], "evaluated")
        self.assertEqual(final["test_raw_metrics"]["mean_nll"], 1.5)
        self.assertEqual(final["analysis_status"], "incomplete")
        self.assertEqual(final["analysis_failure"], "hidden_dataset_resolution_failure")
        self.assertFalse(final["analysis_identity_verified"])
        self.assertEqual(final["deployment_teardown_failures"], [])
        self.assertEqual(self.worker.evaluations, ["test"])
        self.assertNotIn(private_detail, json.dumps(final))
        self.assertIn(
            private_detail,
            supervisor.events_path.read_text(encoding="utf-8"),
        )

    def test_analysis_worker_teardown_failure_preserves_test_measurement(self):
        supervisor = self.supervisor()
        private_detail = "/organizer/private/evaluation-network.log"
        self.worker.evaluation_exceptions["analysis"] = WorkerTeardownFailure(
            private_detail
        )
        supervisor.start_lease()

        committed = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(committed["candidate_id"])
        supervisor.finish()
        public = self.finalize_after_agent_stop(supervisor)
        self.assertEqual(public["status"], "finalized")
        self.assertEqual(public["outcome"], "success")
        self.assertEqual(public["test_status"], "evaluated")
        self.assertEqual(public["test_raw_metrics"]["mean_nll"], 1.5)
        self.assertEqual(public["analysis_status"], "incomplete")
        self.assertEqual(public["analysis_failure"], "analysis_worker_teardown_failure")
        self.assertEqual(
            public["deployment_teardown_failures"],
            ["evaluation_worker_teardown_failure"],
        )
        self.assertFalse(supervisor.state["policy_compliant"])
        self.assertFalse(supervisor.state["leaderboard_eligible"])
        self.assertNotIn(private_detail, json.dumps(public))
        self.assertIn(
            private_detail,
            supervisor.events_path.read_text(encoding="utf-8"),
        )

    def test_analysis_teardown_failure_persists_evaluation_time(self):
        supervisor = self.supervisor()
        supervisor.start_lease()

        def fail_after_evaluation_time(split, timeout_seconds):
            del timeout_seconds
            if split == "analysis":
                self.clock.advance(3)
                raise WorkerTeardownFailure("private evaluator teardown detail")

        committed = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(committed["candidate_id"])
        supervisor.finish()
        supervisor.record_agent_execution(exit_code=0, termination_reason="exited")
        self.assertTrue(supervisor.cleanup_agent_control())
        self.worker.evaluation_hook = fail_after_evaluation_time
        supervisor.evaluate_postmortem()

        persisted = json.loads(supervisor.state_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["timing"]["analysis_evaluation_seconds"], 3.0)
        self.assertEqual(persisted["test_status"], "evaluated")
        self.assertEqual(persisted["analysis_status"], "incomplete")

    def test_track_a_deadline_uses_only_agent_declared_incumbent(self):
        config = replace(self.config, track_a_total_seconds=100)
        supervisor = self.supervisor(track="A", config=config)
        self.worker.analysis_nlls = [1.0, 2.0]

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
        self.assertIsNone(supervisor.state["test_performance_score"])
        self.finalize_after_agent_stop(supervisor)
        self.assertEqual(supervisor.state["test_raw_metrics"]["mean_nll"], 1.5)

    def test_incumbent_history_is_ordered_by_supervisor_events(self):
        supervisor = self.supervisor()

        supervisor.start_lease()
        first = supervisor.commit_lease("/output/candidate")
        self.clock.advance(1)
        supervisor.declare_incumbent(first["candidate_id"])

        (self.workspace / "candidate/candidate.py").write_text(
            "def load(candidate_dir, device): return None\n"
            "def next_token_logits(model, input_ids): return []\n"
            "# second candidate\n",
            encoding="utf-8",
        )
        supervisor.start_lease()
        second = supervisor.commit_lease("/output/candidate")
        self.clock.advance(2)
        supervisor.declare_incumbent(second["candidate_id"])
        self.clock.advance(3)
        supervisor.declare_incumbent(first["candidate_id"])

        events = [
            json.loads(line)
            for line in supervisor.events_path.read_text(encoding="utf-8").splitlines()
        ]
        declarations = [
            event for event in events if event["type"] == "INCUMBENT_DECLARED"
        ]
        self.assertEqual(
            [event["candidate_id"] for event in declarations],
            [first["candidate_id"], second["candidate_id"], first["candidate_id"]],
        )
        self.assertEqual(
            [event["previous_candidate_id"] for event in declarations],
            [None, first["candidate_id"], second["candidate_id"]],
        )
        self.assertEqual(
            [event["seq"] for event in declarations],
            sorted(event["seq"] for event in declarations),
        )
        self.assertEqual(
            [event["elapsed_seconds"] for event in declarations],
            sorted(event["elapsed_seconds"] for event in declarations),
        )
        self.assertTrue(all("timestamp" in event for event in declarations))
        self.assertEqual(
            [item["event_seq"] for item in supervisor.state["incumbent_declarations"]],
            [event["seq"] for event in declarations],
        )

        supervisor.finish()
        final = self.finalize_after_agent_stop(supervisor)
        self.assertEqual(
            [item["candidate_id"] for item in final["incumbent_declarations"]],
            [first["candidate_id"], second["candidate_id"], first["candidate_id"]],
        )
        self.assertEqual(
            [item["candidate_id"] for item in final["analysis_trajectory"]],
            [item["candidate_id"] for item in final["candidate_commits"]],
        )
        declarations_by_candidate = {
            first["candidate_id"]: [declarations[0]["seq"], declarations[2]["seq"]],
            second["candidate_id"]: [declarations[1]["seq"]],
        }
        for point in final["analysis_trajectory"]:
            self.assertEqual(
                point["declaration_event_seqs"],
                declarations_by_candidate[point["candidate_id"]],
            )

    def test_incumbent_declaration_history_has_a_public_safety_bound(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        candidate = supervisor.commit_lease("/output/candidate")
        supervisor.state["incumbent_declaration_limit"] = 1
        supervisor.declare_incumbent(candidate["candidate_id"])

        with self.assertRaises(ProtocolError) as rejected:
            supervisor.declare_incumbent(candidate["candidate_id"])

        self.assertEqual(rejected.exception.code, "incumbent_declaration_limit_reached")
        self.assertEqual(len(supervisor.state["incumbent_declarations"]), 1)

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
        self.assertEqual(result["agent_egress"], "denied")
        serialized = json.dumps(result)
        self.assertNotIn(str(self.config.analysis_path), serialized)
        self.assertNotIn(str(self.config.test_path), serialized)

    def test_semantic_handler_does_not_require_mailbox_traffic(self):
        agent_api_dir = publish_agent_api_bundle(self.episode / "public" / "agent_api")
        supervisor = Supervisor(
            config=self.config,
            agent=self.agent,
            resources=self.resources,
            training_data_verification=verify_training_data(self.config.train_dir),
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
        supervisor.evaluate_postmortem()
        supervisor.publish_postmortem_result()
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
        supervisor.evaluate_postmortem()
        supervisor.publish_postmortem_result()
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
            "envcore.local_mailbox.shutil.rmtree",
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
        self.assertIsNone(supervisor.state["test_performance_score"])
        self.assertIsNone(supervisor.state["test_raw_metrics"])
        self.assertEqual(
            supervisor.state["agent_control"]["cleanup_infrastructure_failures"],
            1,
        )
        self.assertNotIn(
            "RESULT_PUBLICATION_PREPARED",
            supervisor.events_path.read_text(encoding="utf-8"),
        )

    def test_final_result_persistence_failure_cannot_return_success(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        candidate = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(candidate["candidate_id"])
        supervisor.finish()
        supervisor.record_agent_execution(exit_code=0, termination_reason="exited")
        self.assertTrue(supervisor.cleanup_agent_control())
        original_save = supervisor._save_state
        save_attempts = 0

        def fail_first_publication_save():
            nonlocal save_attempts
            if supervisor.state["status"] == "finalized":
                save_attempts += 1
            if supervisor.state["status"] == "finalized" and save_attempts == 1:
                raise OSError("simulated final state persistence failure")
            original_save()

        supervisor.evaluate_postmortem()
        with patch.object(
            supervisor,
            "_save_state",
            side_effect=fail_first_publication_save,
        ):
            with self.assertRaisesRegex(OSError, "final state persistence"):
                supervisor.publish_postmortem_result()

        self.assertEqual(supervisor.state["status"], "closed")
        self.assertIsNone(supervisor.state["outcome"])
        persisted = json.loads(supervisor.state_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["status"], "closed")
        self.assertIsNone(persisted["outcome"])
        self.assertNotIn(
            "EPISODE_FINALIZED",
            supervisor.events_path.read_text(encoding="utf-8"),
        )
        self.assertIsNone(supervisor.state["result_publication"])

        supervisor.deployment_teardown_failed(
            "agent_control_teardown_failure",
            private_detail="simulated final state persistence failure",
        )
        self.assertEqual(supervisor.state["status"], "infrastructure_failure")
        self.assertEqual(supervisor.state["outcome"], "failed")
        events = [
            json.loads(line)
            for line in supervisor.events_path.read_text(encoding="utf-8").splitlines()
        ]
        sequences = [event["seq"] for event in events]
        self.assertEqual(sequences, sorted(sequences))
        self.assertEqual(len(sequences), len(set(sequences)))
        self.assertEqual(
            [
                event["type"]
                for event in events
                if event["type"] == "RESULT_PUBLICATION_PREPARED"
            ],
            ["RESULT_PUBLICATION_PREPARED"],
        )
        self.assertNotIn("EPISODE_FAILED", [event["type"] for event in events])
        self.assertNotIn("EPISODE_FINALIZED", [event["type"] for event in events])

    def test_final_result_persistence_retry_reuses_prepared_event(self):
        supervisor = self.supervisor()
        supervisor.finish()
        supervisor.record_agent_execution(exit_code=0, termination_reason="exited")
        self.assertTrue(supervisor.cleanup_agent_control())
        original_save = supervisor._save_state

        def fail_publication_save():
            if supervisor.state["status"] == "finalized":
                raise OSError("simulated final state persistence failure")
            original_save()

        supervisor.evaluate_postmortem()
        with patch.object(supervisor, "_save_state", side_effect=fail_publication_save):
            with self.assertRaisesRegex(OSError, "final state persistence"):
                supervisor.publish_postmortem_result()

        self.assertEqual(supervisor.state["status"], "closed")
        supervisor.publish_postmortem_result()
        self.assertEqual(supervisor.state["status"], "finalized")
        self.assertEqual(supervisor.state["outcome"], "failed")
        events = [
            json.loads(line)
            for line in supervisor.events_path.read_text(encoding="utf-8").splitlines()
        ]
        prepared = [
            event for event in events if event["type"] == "RESULT_PUBLICATION_PREPARED"
        ]
        self.assertEqual(len(prepared), 1)
        self.assertEqual(
            supervisor.state["result_publication"]["prepared_event_seq"],
            prepared[0]["seq"],
        )

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
        self.assertIsNone(supervisor.state["test_performance_score"])
        self.assertIsNone(supervisor.state["test_raw_metrics"])
        self.assertEqual(
            supervisor.state["agent_control"]["cleanup_policy_violations"],
            1,
        )
        self.assertNotIn(
            "EPISODE_INFRASTRUCTURE_FAILURE",
            supervisor.events_path.read_text(encoding="utf-8"),
        )
        self.assertNotIn(
            "RESULT_PUBLICATION_PREPARED",
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
            "envcore.local_mailbox.directory_fd_operations_available",
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
        self.assertIsNone(status["agent_egress"])
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

    def test_slow_worker_startup_does_not_shorten_the_track_a_lease_it_grants(self):
        """Worker preparation is benchmark-owned and must not bill the Agent.

        The fixture budget is 100s. The Agent spends 10s thinking, then asks for
        a lease while the Worker needs 40s to materialize the workload. The
        Agent must still receive the full 90s it was told it has, and every
        Agent-visible clock must agree about it.
        """

        self.worker.clock = self.clock
        supervisor = self.supervisor(track="A")
        self.clock.advance(10)

        self.worker.start_lease_delay = 40
        granted = supervisor.start_lease()
        status = supervisor.public_status()

        # The Agent plans against this number, so it must be the truth.
        self.assertEqual(granted["duration_seconds"], 90)

        # Worker preparation is not participant time.
        self.assertEqual(status["elapsed_seconds"], 10)
        self.assertEqual(status["remaining_wall_clock_seconds"], 90)
        self.assertEqual(
            status["timing"]["benchmark_overhead_seconds"],
            40,
        )

        # The lease-scoped clock and the episode-scoped clock must not disagree.
        self.assertLessEqual(
            status["active_lease"]["remaining_seconds"],
            status["remaining_wall_clock_seconds"],
        )

        # The lease really gets its full duration: it is still alive one second
        # before the granted deadline, and the episode has not closed early.
        self.clock.advance(89)
        self.assertFalse(supervisor.enforce_deadlines())
        self.assertIsNotNone(supervisor.public_status()["active_lease"])

    def test_failed_slow_worker_startup_still_refunds_its_preparation_time(self):
        """A Lease that never started must not bill its queue wait either."""

        self.worker.clock = self.clock
        supervisor = self.supervisor(track="A")
        self.clock.advance(10)

        self.worker.start_lease_delay = 40
        self.worker.fail_start = True
        with self.assertRaises(ProtocolError) as raised:
            supervisor.start_lease()
        self.assertEqual(raised.exception.code, "infrastructure_failure")

        status = supervisor.public_status()
        self.assertEqual(status["timing"]["benchmark_overhead_seconds"], 40)
        self.assertEqual(status["elapsed_seconds"], 10)
        self.assertEqual(status["remaining_wall_clock_seconds"], 90)
        # A Lease that never started is not consumed.
        self.assertEqual(status["leases_used"], 0)

    def test_a_release_may_declare_that_worker_startup_is_participant_time(self):
        """The accounting boundary is the release's choice, not the runtime's."""

        charged = replace(self.config, benchmark_owned_preparation=())
        self.worker.clock = self.clock
        supervisor = self.supervisor(track="A", config=charged)
        self.clock.advance(10)

        self.worker.start_lease_delay = 40
        granted = supervisor.start_lease()
        status = supervisor.public_status()

        # Nothing is refunded, so the Agent is charged for the 40s of startup...
        self.assertEqual(status["timing"]["benchmark_overhead_seconds"], 0)
        self.assertEqual(status["elapsed_seconds"], 50)
        self.assertEqual(status["remaining_wall_clock_seconds"], 50)
        # ...and the granted duration is clamped to what actually remains, so
        # the Agent is still never promised time it cannot have.
        self.assertEqual(granted["duration_seconds"], 50)
        self.assertLessEqual(
            status["active_lease"]["remaining_seconds"],
            status["remaining_wall_clock_seconds"],
        )


if __name__ == "__main__":
    unittest.main()
