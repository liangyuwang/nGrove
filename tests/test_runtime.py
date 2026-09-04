import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from autotrain_bench.artifact_identity import hidden_bundle_sha256
from autotrain_bench.config import (
    AgentConfig,
    InferenceConfig,
    TaskConfig,
    TrainingResources,
)
from autotrain_bench.host_agent import (
    HostAgentProcess,
    HostAgentStartupFailure,
    _BoundedLogSink,
    start_host_agent_process,
    summarize_host_agent_trace,
)
from autotrain_bench.runtime import DockerRuntime, RuntimeFailure
from autotrain_bench.worker_protocol import (
    CANDIDATE_PREDICTION_ARTIFACT_LIFECYCLE,
    CANDIDATE_PREDICTION_STORAGE_QUOTA,
    DEADLINE_TERMINATION,
    CandidateEvaluationResult,
    EnforcementLevel,
    EvaluationTaskSpec,
    HiddenDatasetRequest,
    HiddenDatasetResolutionFailure,
    ResearchTaskSpec,
    WorkloadRequest,
    WorkloadResolutionFailure,
    TrustLevel,
    WorkerTeardownFailure,
    v0_resource_envelope,
)


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tests/fixtures/task/task.json"
RESOLUTION_TIMEOUT = 30.0


def write_hidden_bundle(path: Path, *, marker: str = "original") -> Path:
    path.mkdir(parents=True)
    (path / "contexts.json").write_text(
        json.dumps({"marker": marker, "visible": True}), encoding="utf-8"
    )
    (path / "targets.json").write_text(
        json.dumps({"marker": marker, "hidden": True}), encoding="utf-8"
    )
    return path


def publish_agent_api(public_task: Path) -> None:
    package = public_task.parent / "agent_api/autotrain_bench"
    package.mkdir(parents=True)
    (package / "client.py").write_text("class Client: pass\n", encoding="utf-8")
    (package / "candidate_preflight.py").write_text(
        "def main(): pass\n", encoding="utf-8"
    )


def resolve_evaluation_workloads(runtime, task, episode_id):
    return {
        "candidate_workload": runtime.resolve_workload(
            episode_id=episode_id,
            workload=WorkloadRequest("candidate", task.candidate_image),
            timeout_seconds=RESOLUTION_TIMEOUT,
        ),
        "evaluator_workload": runtime.resolve_workload(
            episode_id=episode_id,
            workload=WorkloadRequest("evaluator", task.evaluator_image),
            timeout_seconds=RESOLUTION_TIMEOUT,
        ),
    }


class CapturingDockerRuntime(DockerRuntime):
    def __init__(self, *, monotonic=None, hidden_datasets=None):
        super().__init__(
            docker_binary="docker",
            monotonic=monotonic,
            hidden_datasets=hidden_datasets,
        )
        self.calls = []
        self.call_kwargs = []

    def _run(self, args, **kwargs):
        self.calls.append(list(args))
        self.call_kwargs.append(dict(kwargs))
        if args[:2] == ["image", "inspect"] and "{{.Id}}" in args:
            return subprocess.CompletedProcess(args, 0, "sha256:" + "a" * 64 + "\n", "")
        if args and args[0] == "wait":
            return subprocess.CompletedProcess(args, 0, "0\n", "")
        return subprocess.CompletedProcess(args, 0, "container-id\n", "")


class RuntimeBoundaryTests(unittest.TestCase):
    def test_local_gpu_receipt_reports_workload_visible_values_but_not_exact_binding(
        self,
    ):
        class GpuRuntime(CapturingDockerRuntime):
            def _run(self, args, **kwargs):
                if args and args[0] == "exec" and "nvidia-smi" in args:
                    self.calls.append(list(args))
                    self.call_kwargs.append(dict(kwargs))
                    return subprocess.CompletedProcess(
                        args,
                        0,
                        "NVIDIA H100 80GB HBM3, 81559\n",
                        "",
                    )
                return super()._run(args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public_task = root / "task.json"
            public_task.write_text("{}", encoding="utf-8")
            publish_agent_api(public_task)
            reference = "example/training@sha256:" + "a" * 64
            task = replace(
                ResearchTaskSpec.from_task(
                    TaskConfig.load(TASK), public_task.parent / "agent_api"
                ),
                training_image=reference,
            )
            resources = TrainingResources(
                nvidia_gpu_count=1,
                required_gpu_model="H100",
                minimum_gpu_memory_mib=80000,
            )
            envelope = v0_resource_envelope(
                resources,
                minimum_enforcement=EnforcementLevel.BEST_EFFORT,
            )
            runtime = GpuRuntime()
            workload = runtime.resolve_workload(
                episode_id="episode-1",
                workload=WorkloadRequest("training", reference),
                timeout_seconds=RESOLUTION_TIMEOUT,
            )

            receipt = runtime.start_lease(
                episode_id="episode-1",
                lease_id="lease-0001",
                duration_seconds=60,
                task=task,
                workspace=root,
                resources=resources,
                resource_envelope=envelope,
                lease_dir=root / "lease",
                workload=workload,
            )

            evidence = {item.name: item for item in receipt.properties}
            self.assertEqual(
                evidence["accelerator.model"].workload_visible.value,
                ("NVIDIA H100 80GB HBM3",),
            )
            self.assertEqual(
                evidence["accelerator.count"].enforcement_level,
                EnforcementLevel.BEST_EFFORT,
            )

    def test_local_reference_worker_is_explicitly_development_only(self):
        descriptor = DockerRuntime(docker_binary="docker").describe_capabilities()
        self.assertEqual(descriptor.trust_level, TrustLevel.DEVELOPMENT)
        self.assertFalse(hasattr(descriptor, "conformance"))
        self.assertEqual(
            descriptor.capabilities[DEADLINE_TERMINATION].level,
            EnforcementLevel.BEST_EFFORT,
        )
        self.assertEqual(
            descriptor.capabilities[CANDIDATE_PREDICTION_STORAGE_QUOTA].level,
            EnforcementLevel.UNSUPPORTED,
        )
        self.assertIn(
            "no hard byte or inode quota",
            descriptor.capabilities[CANDIDATE_PREDICTION_STORAGE_QUOTA].mechanism,
        )
        self.assertEqual(
            descriptor.capabilities[CANDIDATE_PREDICTION_ARTIFACT_LIFECYCLE].level,
            EnforcementLevel.ENFORCED,
        )

    def test_hidden_dataset_source_is_resolved_only_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "private" / "test"
            runtime = CapturingDockerRuntime(hidden_datasets={"test": source})
            request = HiddenDatasetRequest("test")
            task = EvaluationTaskSpec.from_task(TaskConfig.load(TASK))

            self.assertFalse(source.exists())
            with self.assertRaises(HiddenDatasetResolutionFailure):
                runtime.resolve_hidden_dataset(
                    episode_id="episode-1",
                    task=task,
                    hidden_dataset=request,
                    timeout_seconds=RESOLUTION_TIMEOUT,
                )

            write_hidden_bundle(source)
            resolved = runtime.resolve_hidden_dataset(
                episode_id="episode-1",
                task=task,
                hidden_dataset=request,
                timeout_seconds=RESOLUTION_TIMEOUT,
            )

            self.assertEqual(
                resolved.sha256,
                hidden_bundle_sha256(source),
            )
            self.assertFalse(hasattr(resolved, "path"))

    def test_workload_resolution_is_episode_bound_and_drives_the_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public_task = root / "task.json"
            public_task.write_text("{}", encoding="utf-8")
            publish_agent_api(public_task)
            reference = "example/training@sha256:" + "a" * 64
            task = replace(
                ResearchTaskSpec.from_task(
                    TaskConfig.load(TASK), public_task.parent / "agent_api"
                ),
                training_image=reference,
            )
            runtime = CapturingDockerRuntime()
            resolved = runtime.resolve_workload(
                episode_id="episode-1",
                workload=WorkloadRequest("training", reference),
                timeout_seconds=RESOLUTION_TIMEOUT,
            )

            with self.assertRaises(WorkloadResolutionFailure):
                runtime.start_lease(
                    episode_id="episode-2",
                    lease_id="lease-0001",
                    duration_seconds=60,
                    task=task,
                    workspace=root,
                    resources=TrainingResources(),
                    lease_dir=root / "wrong-lease",
                    workload=resolved,
                )
            self.assertEqual(runtime.calls, [])

            receipt = runtime.start_lease(
                episode_id="episode-1",
                lease_id="lease-0001",
                duration_seconds=60,
                task=task,
                workspace=root,
                resources=TrainingResources(),
                lease_dir=root / "lease",
                workload=resolved,
            )

            launch = next(call for call in runtime.calls if call[0] == "run")
            self.assertIn(reference, launch)
            self.assertNotIn(resolved.resolution_id, launch)
            self.assertEqual(receipt.episode_id, "episode-1")
            self.assertEqual(receipt.lease_id, "lease-0001")
            self.assertEqual(
                {item.name for item in receipt.properties},
                {"host.count", "accelerator.count"},
            )

    def test_mutable_preview_tag_is_frozen_to_image_id_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public_task = root / "task.json"
            public_task.write_text("{}", encoding="utf-8")
            publish_agent_api(public_task)
            task = ResearchTaskSpec.from_task(
                TaskConfig.load(TASK), public_task.parent / "agent_api"
            )
            runtime = CapturingDockerRuntime()
            resolved = runtime.resolve_workload(
                episode_id="episode",
                workload=WorkloadRequest("training", task.training_image),
                timeout_seconds=RESOLUTION_TIMEOUT,
            )
            runtime.start_lease(
                episode_id="episode",
                lease_id="lease-0001",
                duration_seconds=60,
                task=task,
                workspace=root,
                resources=TrainingResources(),
                lease_dir=root / "lease",
                workload=resolved,
            )

            launch = next(call for call in runtime.calls if call[0] == "run")
            self.assertEqual(resolved.effective_format, "oci-image-config")
            self.assertIn("sha256:" + "a" * 64, launch)
            self.assertNotIn(task.training_image, launch)

    def test_workload_inspect_failure_preserves_resolution_failure_type(self):
        runtime = DockerRuntime(docker_binary="docker")

        with patch.object(
            runtime,
            "_run",
            side_effect=RuntimeFailure("private Docker diagnostic"),
        ):
            with self.assertRaisesRegex(
                WorkloadResolutionFailure,
                "private Docker diagnostic",
            ):
                runtime.resolve_workload(
                    episode_id="episode",
                    workload=WorkloadRequest("training", "mutable:latest"),
                    timeout_seconds=RESOLUTION_TIMEOUT,
                )

    def test_workload_inspect_does_not_downgrade_unknown_teardown_state(self):
        runtime = DockerRuntime(docker_binary="docker")

        with patch.object(
            runtime,
            "_run",
            side_effect=WorkerTeardownFailure("inspection helper still running"),
        ):
            with self.assertRaisesRegex(
                WorkerTeardownFailure,
                "still running",
            ):
                runtime.resolve_workload(
                    episode_id="episode",
                    workload=WorkloadRequest("training", "mutable:latest"),
                    timeout_seconds=RESOLUTION_TIMEOUT,
                )

    def test_resolved_hidden_dataset_uses_the_receipted_immutable_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_hidden_bundle(Path(directory) / "test")
            original = {
                name: (source / name).read_bytes()
                for name in ("contexts.json", "targets.json")
            }
            runtime = CapturingDockerRuntime(hidden_datasets={"test": source})
            task = EvaluationTaskSpec.from_task(TaskConfig.load(TASK))
            resolved = runtime.resolve_hidden_dataset(
                episode_id="episode-1",
                task=task,
                hidden_dataset=HiddenDatasetRequest("test"),
                timeout_seconds=RESOLUTION_TIMEOUT,
            )
            workloads = resolve_evaluation_workloads(runtime, task, "episode-1")
            runtime.calls.clear()
            (source / "contexts.json").write_text(
                '{"marker":"mutated"}', encoding="utf-8"
            )

            evaluated_bytes = []

            def evaluate_snapshot(**kwargs):
                evaluated_bytes.append(
                    {
                        name: (kwargs["data_path"] / name).read_bytes()
                        for name in ("contexts.json", "targets.json")
                    }
                )
                self.assertNotEqual(kwargs["data_path"], source)
                return CandidateEvaluationResult(
                    72.5,
                    {"mean_nll": 1.25},
                    2.0,
                    2,
                    3,
                    64,
                )

            with patch.object(
                runtime,
                "_run_isolated_evaluator",
                side_effect=evaluate_snapshot,
            ):
                result = runtime.evaluate_candidate(
                    episode_id="episode-1",
                    task=task,
                    resources=TrainingResources(),
                    candidate_dir=TASK.parent / "agent",
                    hidden_dataset=resolved,
                    output_dir=Path(directory) / "output",
                    **workloads,
                )

            self.assertEqual(result.performance_score, 72.5)
            self.assertEqual(evaluated_bytes, [original])
            snapshot = runtime._resolved_hidden_datasets[
                ("episode-1", resolved.resolution_id)
            ][2]
            self.assertEqual(resolved.sha256, hidden_bundle_sha256(snapshot))

            snapshot_root = runtime._hidden_snapshot_root
            self.assertIsNotNone(snapshot_root)
            runtime.close_private_materializations()
            self.assertFalse(snapshot_root.exists())
            self.assertEqual(runtime._resolved_hidden_datasets, {})

    def test_private_hidden_snapshots_wait_for_confirmed_evaluation_teardown(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_hidden_bundle(Path(directory) / "test")
            runtime = CapturingDockerRuntime(hidden_datasets={"test": source})
            task = EvaluationTaskSpec.from_task(TaskConfig.load(TASK))
            runtime.resolve_hidden_dataset(
                episode_id="episode",
                task=task,
                hidden_dataset=HiddenDatasetRequest("test"),
                timeout_seconds=RESOLUTION_TIMEOUT,
            )
            runtime._pending_evaluations[("episode", "test", "candidate")] = (
                "candidate",
                "evaluator",
                Path(directory) / "predictions",
            )

            with self.assertRaisesRegex(
                WorkerTeardownFailure,
                "teardown is unconfirmed",
            ):
                runtime.close_private_materializations()

            self.assertTrue(runtime._hidden_snapshot_root.exists())
            runtime._pending_evaluations.clear()
            runtime.close_private_materializations()

    def test_hidden_dataset_resolution_is_bound_to_one_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_hidden_bundle(Path(directory) / "test")
            runtime = CapturingDockerRuntime(hidden_datasets={"test": source})
            task = EvaluationTaskSpec.from_task(TaskConfig.load(TASK))
            resolved = runtime.resolve_hidden_dataset(
                episode_id="episode-1",
                task=task,
                hidden_dataset=HiddenDatasetRequest("test"),
                timeout_seconds=RESOLUTION_TIMEOUT,
            )
            workloads = resolve_evaluation_workloads(runtime, task, "episode-1")
            runtime.calls.clear()

            with self.assertRaises(HiddenDatasetResolutionFailure):
                runtime.evaluate_candidate(
                    episode_id="episode-2",
                    task=task,
                    resources=TrainingResources(),
                    candidate_dir=TASK.parent / "agent",
                    hidden_dataset=resolved,
                    output_dir=Path(directory) / "output",
                    **workloads,
                )

            self.assertEqual(runtime.calls, [])

    def test_stop_container_requires_confirmed_absence(self):
        runtime = DockerRuntime(docker_binary="docker")
        removal = subprocess.CompletedProcess(
            ["docker", "rm"], 1, "", "daemon refused removal"
        )
        still_present = subprocess.CompletedProcess(
            ["docker", "container", "ls"], 0, "agent\n", ""
        )

        with patch.object(runtime, "_run", side_effect=[removal, still_present]):
            with self.assertRaisesRegex(RuntimeFailure, "daemon refused removal"):
                runtime.stop_container("agent")

    def test_runtime_execution_error_has_a_typed_failure(self):
        runtime = DockerRuntime(docker_binary="docker")

        with patch(
            "autotrain_bench.runtime.subprocess.run",
            side_effect=OSError("runtime executable disappeared"),
        ):
            with self.assertRaisesRegex(RuntimeFailure, "failed to execute"):
                runtime.stop_container("agent")

    def test_container_state_query_does_not_treat_daemon_failure_as_exit(self):
        runtime = DockerRuntime(docker_binary="docker")

        with patch.object(
            runtime,
            "_run",
            side_effect=RuntimeFailure("Docker daemon is unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeFailure, "daemon is unavailable"):
                runtime.container_running("agent")

    def test_stop_container_accepts_an_already_absent_container(self):
        runtime = DockerRuntime(docker_binary="docker")
        removal = subprocess.CompletedProcess(
            ["docker", "rm"], 1, "", "No such container"
        )
        absent = subprocess.CompletedProcess(
            ["docker", "container", "ls"], 0, "another-container\n", ""
        )

        with patch.object(runtime, "_run", side_effect=[removal, absent]):
            runtime.stop_container("agent")

    def test_remove_network_requires_confirmed_absence(self):
        runtime = DockerRuntime(docker_binary="docker")
        removal = subprocess.CompletedProcess(
            ["docker", "network", "rm"], 1, "", "network has active endpoints"
        )
        still_present = subprocess.CompletedProcess(
            ["docker", "network", "ls"], 0, "evaluation-network\n", ""
        )

        with patch.object(runtime, "_run", side_effect=[removal, still_present]):
            with self.assertRaisesRegex(RuntimeFailure, "active endpoints"):
                runtime.remove_network("evaluation-network")

    def test_failed_lease_stop_keeps_the_runtime_lease_handle(self):
        runtime = DockerRuntime(docker_binary="docker")
        runtime._worker_leases[("episode", "lease")] = "training"

        with patch.object(
            runtime,
            "stop_container",
            side_effect=RuntimeFailure("termination not confirmed"),
        ):
            with self.assertRaisesRegex(WorkerTeardownFailure, "not confirmed"):
                runtime.terminate_lease(episode_id="episode", lease_id="lease")

        self.assertEqual(
            runtime._worker_leases[("episode", "lease")],
            "training",
        )

    def test_uncertain_lease_start_keeps_a_retryable_runtime_handle(self):
        runtime = DockerRuntime(docker_binary="docker")
        task = ResearchTaskSpec.from_task(
            TaskConfig.load(TASK), TASK.parent / "agent-api"
        )
        task = replace(
            task,
            training_image="example/training@sha256:" + "a" * 64,
        )
        workload = runtime.resolve_workload(
            episode_id="episode",
            workload=WorkloadRequest("training", task.training_image),
            timeout_seconds=RESOLUTION_TIMEOUT,
        )

        with patch.object(
            runtime,
            "start_training",
            side_effect=RuntimeFailure("Docker client timed out after create"),
        ):
            with patch.object(
                runtime,
                "stop_container",
                side_effect=[
                    RuntimeFailure("container absence not confirmed"),
                    None,
                ],
            ) as stop:
                with self.assertRaisesRegex(
                    WorkerTeardownFailure, "cleanup was not confirmed"
                ):
                    runtime.start_lease(
                        episode_id="episode",
                        lease_id="lease",
                        duration_seconds=60,
                        task=task,
                        workspace=TASK.parent,
                        resources=TrainingResources(),
                        lease_dir=TASK.parent / "lease",
                        workload=workload,
                    )

                self.assertIn(("episode", "lease"), runtime._worker_leases)
                runtime.terminate_lease(episode_id="episode", lease_id="lease")

        self.assertEqual(stop.call_count, 2)
        self.assertNotIn(("episode", "lease"), runtime._worker_leases)

    def test_fresh_runtime_can_reconstruct_a_lease_cleanup_handle(self):
        runtime = DockerRuntime(docker_binary="docker")
        expected = runtime._worker_lease_runtime_name("episode", "lease-0001")

        with patch.object(runtime, "stop_container") as stop:
            runtime.terminate_lease(episode_id="episode", lease_id="lease-0001")

        stop.assert_called_once_with(expected)

    def test_host_agent_is_stopped_if_log_thread_startup_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            control = root / "control"
            workspace.mkdir()
            control.mkdir()
            public_task = root / "task.json"
            public_task.write_text("{}", encoding="utf-8")
            created = []
            popen = subprocess.Popen

            def launch(*args, **kwargs):
                process = popen(*args, **kwargs)
                created.append(process)
                return process

            with patch(
                "autotrain_bench.host_agent.subprocess.Popen", side_effect=launch
            ):
                with patch(
                    "autotrain_bench.host_agent.threading.Thread.start",
                    side_effect=RuntimeError("thread startup failed"),
                ):
                    with self.assertRaises(HostAgentStartupFailure) as raised:
                        start_host_agent_process(
                            command=[
                                sys.executable,
                                "-c",
                                "import time; time.sleep(30)",
                            ],
                            workspace=workspace,
                            control_dir=control,
                            public_task_path=public_task,
                            package_root=ROOT,
                            forwarded_environment=(),
                            stdout_path=root / "stdout.log",
                            stderr_path=root / "stderr.log",
                        )

            self.assertEqual(len(created), 1)
            self.assertIsNotNone(created[0].poll())
            self.assertIs(raised.exception.handle.process, created[0])
            self.assertEqual(
                str(raised.exception.startup_error), "thread startup failed"
            )
            self.assertIsNone(raised.exception.cleanup_error)

    def test_host_agent_closes_first_log_if_second_log_setup_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            control = root / "control"
            workspace.mkdir()
            control.mkdir()
            public_task = root / "task.json"
            public_task.write_text("{}", encoding="utf-8")
            stdout_sink = _BoundedLogSink(root / "stdout.log")

            with patch(
                "autotrain_bench.host_agent._BoundedLogSink",
                side_effect=[stdout_sink, OSError("stderr log unavailable")],
            ):
                with self.assertRaisesRegex(OSError, "stderr log unavailable"):
                    start_host_agent_process(
                        command=["agent"],
                        workspace=workspace,
                        control_dir=control,
                        public_task_path=public_task,
                        package_root=ROOT,
                        forwarded_environment=(),
                        stdout_path=root / "stdout.log",
                        stderr_path=root / "stderr.log",
                    )

            self.assertTrue(stdout_sink.stream.closed)

    def test_host_agent_start_failure_preserves_handle_if_termination_fails(self):
        class UncertainProcess:
            def __init__(self):
                self.pid = 12345
                self.stdout = io.BytesIO()
                self.stderr = io.BytesIO()
                self.returncode = None

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            control = root / "control"
            workspace.mkdir()
            control.mkdir()
            public_task = root / "task.json"
            public_task.write_text("{}", encoding="utf-8")
            process = UncertainProcess()
            termination_error = subprocess.TimeoutExpired("host Agent", 3)

            with patch(
                "autotrain_bench.host_agent.subprocess.Popen",
                return_value=process,
            ):
                with patch(
                    "autotrain_bench.host_agent.threading.Thread.start",
                    side_effect=RuntimeError("thread startup failed"),
                ):
                    with patch.object(
                        HostAgentProcess,
                        "stop",
                        side_effect=termination_error,
                    ):
                        with self.assertRaises(HostAgentStartupFailure) as raised:
                            start_host_agent_process(
                                command=["agent"],
                                workspace=workspace,
                                control_dir=control,
                                public_task_path=public_task,
                                package_root=ROOT,
                                forwarded_environment=(),
                                stdout_path=root / "stdout.log",
                                stderr_path=root / "stderr.log",
                            )

            failure = raised.exception
            self.assertIs(failure.handle.process, process)
            self.assertEqual(str(failure.startup_error), "thread startup failed")
            self.assertIs(failure.cleanup_error, termination_error)
            self.assertTrue(failure.handle.running)
            failure.handle.close_logs()

    def test_host_agent_stop_closes_unstarted_log_resources_on_timeout(self):
        class UncertainProcess:
            def __init__(self):
                self.pid = 12345
                self.stdout = io.BytesIO()
                self.stderr = io.BytesIO()
                self.returncode = None

            def poll(self):
                return None

            def wait(self, timeout):
                raise subprocess.TimeoutExpired("host Agent", timeout)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = UncertainProcess()
            stdout_sink = _BoundedLogSink(root / "stdout.log")
            stderr_sink = _BoundedLogSink(root / "stderr.log")
            handle = HostAgentProcess(
                process=process,
                stdout_path=root / "stdout.log",
                stderr_path=root / "stderr.log",
                _stdout_sink=stdout_sink,
                _stderr_sink=stderr_sink,
            )

            with patch("autotrain_bench.host_agent.os.killpg") as killpg:
                with self.assertRaises(subprocess.TimeoutExpired):
                    handle.stop(timeout_seconds=0.01)

            self.assertEqual(killpg.call_count, 2)
            self.assertTrue(process.stdout.closed)
            self.assertTrue(process.stderr.closed)
            self.assertTrue(stdout_sink.stream.closed)
            self.assertTrue(stderr_sink.stream.closed)
            self.assertTrue(handle.running)

    def test_runtime_profile_checks_model_count_and_memory_together(self):
        resources = TrainingResources(
            nvidia_gpu_count=2,
            required_gpu_model="H100",
            minimum_gpu_memory_mib=80000,
        )
        inventory = subprocess.CompletedProcess(
            ["nvidia-smi"],
            0,
            "NVIDIA H100 80GB HBM3, 81559\n"
            "NVIDIA H100 80GB HBM3, 81559\n"
            "NVIDIA A100-SXM4-80GB, 81920\n",
            "",
        )
        runtime = DockerRuntime(docker_binary="docker")
        with patch("autotrain_bench.runtime.shutil.which", return_value="nvidia-smi"):
            with patch(
                "autotrain_bench.runtime.subprocess.run", return_value=inventory
            ):
                runtime.validate_hardware(resources)

        undersized = subprocess.CompletedProcess(
            ["nvidia-smi"],
            0,
            "NVIDIA H100 80GB HBM3, 81559\n" "NVIDIA H100 PCIe, 65536\n",
            "",
        )
        with patch("autotrain_bench.runtime.shutil.which", return_value="nvidia-smi"):
            with patch(
                "autotrain_bench.runtime.subprocess.run", return_value=undersized
            ):
                with self.assertRaisesRegex(RuntimeFailure, "at least 80000 MiB"):
                    runtime.validate_hardware(resources)

    def test_training_timeout_decodes_partial_byte_output(self):
        runtime = DockerRuntime(docker_binary="docker")

        class TimedOutProcess:
            def __init__(self):
                self.stdout = io.BytesIO(b"partial stdout: \xff")
                self.stderr = io.BytesIO(b"partial stderr: \xfe")
                self.returncode = None
                self.wait_count = 0

            def wait(self, timeout):
                del timeout
                self.wait_count += 1
                if self.wait_count == 1:
                    raise subprocess.TimeoutExpired(["docker", "exec"], 5)
                self.returncode = -9
                return self.returncode

            def poll(self):
                return self.returncode

            def kill(self):
                self.returncode = -9

        with patch(
            "autotrain_bench.runtime.subprocess.Popen",
            return_value=TimedOutProcess(),
        ):
            with patch.object(runtime, "stop_container") as stop_container:
                result = runtime.exec_training(
                    name="training",
                    argv=["python", "train.py"],
                    timeout_seconds=5,
                )

        stop_container.assert_called_once_with("training")
        self.assertEqual(result.exit_code, 124)
        self.assertEqual(result.stdout, "partial stdout: \ufffd")
        self.assertEqual(result.stderr, "partial stderr: \ufffd")
        self.assertIsInstance(result.stdout, str)
        self.assertIsInstance(result.stderr, str)
        self.assertTrue(result.timed_out)

    def test_training_timeout_kills_exec_client_when_container_stop_fails(self):
        runtime = DockerRuntime(docker_binary="docker")

        class TimedOutProcess:
            def __init__(self):
                self.stdout = io.BytesIO(b"")
                self.stderr = io.BytesIO(b"")
                self.returncode = None
                self.wait_count = 0
                self.killed = False

            def wait(self, timeout):
                del timeout
                self.wait_count += 1
                if self.wait_count == 1:
                    raise subprocess.TimeoutExpired(["docker", "exec"], 5)
                return self.returncode

            def poll(self):
                return self.returncode

            def kill(self):
                self.killed = True
                self.returncode = -9

        process = TimedOutProcess()
        with patch("autotrain_bench.runtime.subprocess.Popen", return_value=process):
            with patch.object(
                runtime,
                "stop_container",
                side_effect=RuntimeFailure("container remains"),
            ):
                with self.assertRaisesRegex(RuntimeFailure, "runtime teardown failed"):
                    runtime.exec_training(
                        name="training",
                        argv=["python", "train.py"],
                        timeout_seconds=5,
                    )

        self.assertTrue(process.killed)
        self.assertEqual(process.wait_count, 2)

    def test_training_output_is_drained_with_a_bounded_capture(self):
        class CompletedProcess:
            def __init__(self):
                self.stdout = io.BytesIO(b"x" * (2 * 1024 * 1024))
                self.stderr = io.BytesIO(b"")
                self.returncode = None

            def wait(self, timeout):
                del timeout
                self.returncode = 0
                return 0

            def poll(self):
                return self.returncode

            def kill(self):
                self.returncode = -9

        runtime = DockerRuntime(docker_binary="docker")
        with patch(
            "autotrain_bench.runtime.subprocess.Popen",
            return_value=CompletedProcess(),
        ):
            result = runtime.exec_training(
                name="training",
                argv=["python", "train.py"],
                timeout_seconds=5,
            )
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(result.stdout_truncated)
        self.assertLessEqual(len(result.stdout.encode("utf-8")), 1024 * 1024)

    def test_task_preparation_has_no_implicit_agent_image(self):
        config = TaskConfig.load(TASK)
        runtime = CapturingDockerRuntime()
        runtime.prepare(config)
        serialized = "\n".join(" ".join(call) for call in runtime.calls)
        self.assertNotIn("test-agent", serialized)
        self.assertNotIn("test-gateway", serialized)

    def test_host_cli_runner_uses_workspace_and_a_minimal_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            control = root / "control"
            workspace.mkdir()
            control.mkdir()
            public_task = root / "task.json"
            public_task.write_text("{}", encoding="utf-8")
            stdout_path = root / "agent.stdout.log"
            stderr_path = root / "agent.stderr.log"
            script = (
                "import json, os; "
                "print(json.dumps({'cwd': os.getcwd(), "
                "'control': os.environ.get('AUTOTRAIN_BENCH_CONTROL_DIR'), "
                "'visible': os.environ.get('VISIBLE'), "
                "'leaked': os.environ.get('SHOULD_NOT_LEAK')}))"
            )
            agent = AgentConfig(
                workspace=workspace,
                runner="host_cli",
                command=[sys.executable, "-c", script],
                initial_workspace="participant_snapshot",
                environment=["VISIBLE"],
            )
            runtime = DockerRuntime(docker_binary="docker")
            with patch.dict(
                os.environ,
                {"VISIBLE": "yes", "SHOULD_NOT_LEAK": "secret"},
                clear=False,
            ):
                handle = runtime.start_host_agent(
                    agent=agent,
                    control_dir=control,
                    public_task_path=public_task,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
                handle.process.wait(timeout=5)
                self.assertEqual(handle.stop(), 0)
            payload = json.loads(stdout_path.read_text(encoding="utf-8"))
            self.assertEqual(Path(payload["cwd"]).resolve(), workspace.resolve())
            self.assertEqual(Path(payload["control"]).resolve(), control.resolve())
            self.assertEqual(payload["visible"], "yes")
            self.assertIsNone(payload["leaked"])
            self.assertEqual(stdout_path.stat().st_mode & 0o777, 0o600)

    def test_host_agent_jsonl_usage_is_summarized(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "agent.stdout.log"
            records = [
                {
                    "type": "autotrain_bench.agent_metadata",
                    "tool": "codex",
                    "tool_version": "codex-cli 1.2.3",
                    "model": "test-model",
                    "reasoning_effort": "high",
                },
                {
                    "type": "item.started",
                    "item": {"type": "command_execution"},
                },
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 20,
                        "cached_input_tokens": 12,
                        "output_tokens": 5,
                        "reasoning_output_tokens": 2,
                    },
                },
            ]
            trace.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            summary = summarize_host_agent_trace(trace)
            self.assertEqual(summary["tool"], "codex")
            self.assertEqual(summary["model"], "test-model")
            self.assertEqual(summary["total_tokens"], 25)
            self.assertEqual(summary["command_executions"], 1)
            self.assertTrue(summary["usage_complete"])

    def test_tool_independent_host_agent_usage_is_summarized(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "agent.stdout.log"
            records = [
                {
                    "type": "autotrain_bench.agent_metadata",
                    "tool": "example-cli",
                    "tool_version": "4.2",
                    "model": "example-model",
                },
                {
                    "type": "autotrain_bench.agent_usage",
                    "usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 25,
                        "output_tokens": 30,
                        "reasoning_output_tokens": 4,
                    },
                    "usage_complete": True,
                },
            ]
            trace.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            summary = summarize_host_agent_trace(trace)
            self.assertEqual(summary["tool"], "example-cli")
            self.assertEqual(summary["model"], "example-model")
            self.assertEqual(summary["input_tokens"], 100)
            self.assertEqual(summary["cached_input_tokens"], 25)
            self.assertEqual(summary["output_tokens"], 30)
            self.assertEqual(summary["total_tokens"], 130)
            self.assertTrue(summary["usage_complete"])

    def test_host_agent_without_usage_records_is_marked_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "agent.stdout.log"
            trace.write_text("ordinary CLI output\n", encoding="utf-8")
            summary = summarize_host_agent_trace(trace)
            self.assertFalse(summary["usage_complete"])

    def test_host_agent_log_truncation_may_split_utf8_without_breaking_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "agent.stdout.log"
            sink = _BoundedLogSink(trace)
            sink.limit = 3
            sink.drain(io.BytesIO("😀".encode("utf-8")))

            summary = summarize_host_agent_trace(trace)

            self.assertFalse(summary["usage_complete"])
            self.assertGreater(summary["malformed_records"], 0)

    def test_agent_mount_does_not_expose_repository_or_private_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            control = root / "control"
            workspace.mkdir()
            control.mkdir()
            public_task = root / "task.json"
            public_task.write_text("{}", encoding="utf-8")
            publish_agent_api(public_task)
            config = TaskConfig.load(TASK)
            agent = AgentConfig(
                workspace=workspace,
                runner="container",
                command=["python", "/workspace/agent.py"],
                initial_workspace="participant_snapshot",
                image="python:test-agent",
                egress="denied",
            )
            runtime = CapturingDockerRuntime()
            runtime.start_agent(
                name="agent",
                config=config,
                agent=agent,
                control_dir=control,
                public_task_path=public_task,
            )
            command = "\n".join(runtime.calls[0])
            self.assertIn("dst=/opt/agent-api,readonly", command)
            self.assertNotIn("dst=/opt/autotrain_bench", command)
            self.assertIn("dst=/control/inbox", command)
            self.assertIn("dst=/control/outbox", command)
            self.assertNotIn("dst=/control\n", command)
            self.assertIn("/agent-home:rw,nosuid,size=1g,mode=1777", command)
            self.assertIn("HOME=/agent-home", command)
            self.assertNotIn("reference_agent", command)
            self.assertNotIn(str(TASK.parent / "private"), command)
            self.assertNotIn(str(config.analysis_path), command)
            self.assertNotIn(str(config.test_path), command)

    def test_egress_denied_inference_uses_trusted_sidecar_and_internal_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            control = root / "control"
            lease = root / "lease"
            audit = root / "audit"
            workspace.mkdir()
            control.mkdir()
            public_task = root / "task.json"
            public_task.write_text("{}", encoding="utf-8")
            publish_agent_api(public_task)
            inference = InferenceConfig(
                protocol="openai_compatible",
                base_url="https://api.openai.com",
                api_key_env="OPENAI_API_KEY",
                allowed_models=("researcher-a",),
                allowed_paths=("/v1/responses",),
                timeout_seconds=300,
            )
            config = TaskConfig.load(TASK)
            agent = AgentConfig(
                workspace=workspace,
                runner="container",
                command=["node", "/workspace/agent.js"],
                initial_workspace="participant_snapshot",
                image="python:test-agent",
                egress="denied",
                inference=inference,
            )
            runtime = CapturingDockerRuntime()
            runtime.create_network("agent-net", internal=True)
            runtime.create_network("egress-net", internal=False)
            with patch.dict(os.environ, {"OPENAI_API_KEY": "host-secret"}):
                runtime.start_inference_gateway(
                    name="gateway",
                    agent_network="agent-net",
                    egress_network="egress-net",
                    config=config,
                    agent=agent,
                    audit_dir=audit,
                )
            runtime.start_agent(
                name="agent",
                config=config,
                agent=agent,
                control_dir=control,
                public_task_path=public_task,
                inference_network="agent-net",
            )
            runtime.start_training(
                name="training",
                task=ResearchTaskSpec.from_task(
                    config, public_task.parent / "agent_api"
                ),
                workspace=workspace,
                resources=TrainingResources(),
                lease_dir=lease,
            )

            self.assertIn("--internal", runtime.calls[0])
            run_calls = [call for call in runtime.calls if call and call[0] == "run"]
            gateway_command = next(
                call
                for call in run_calls
                if call[call.index("--name") + 1] == "gateway"
            )
            agent_command = next(
                call for call in run_calls if call[call.index("--name") + 1] == "agent"
            )
            training_command = next(
                call
                for call in run_calls
                if call[call.index("--name") + 1] == "training"
            )
            self.assertEqual(
                gateway_command[gateway_command.index("--network") + 1], "egress-net"
            )
            self.assertEqual(
                agent_command[agent_command.index("--network") + 1], "agent-net"
            )
            self.assertEqual(
                training_command[training_command.index("--network") + 1], "none"
            )
            serialized_agent = "\n".join(agent_command)
            serialized_gateway = "\n".join(gateway_command)
            self.assertIn(
                "OPENAI_BASE_URL=http://inference-gateway:8765/v1",
                serialized_agent,
            )
            self.assertNotIn("autotrain_bench.agent_entrypoint", serialized_agent)
            self.assertEqual(agent_command[-2:], ["node", "/workspace/agent.js"])
            self.assertNotIn("host-secret", serialized_agent)
            self.assertNotIn("api.openai.com", serialized_agent)
            self.assertIn("api.openai.com", serialized_gateway)
            self.assertNotIn("host-secret", serialized_gateway)
            self.assertIn("OPENAI_API_KEY", gateway_command)
            self.assertNotIn("OPENAI_API_KEY", "\n".join(training_command))
            self.assertIn(
                [
                    "network",
                    "connect",
                    "--alias",
                    "inference-gateway",
                    "agent-net",
                    "gateway",
                ],
                runtime.calls,
            )

    def test_unrestricted_dev_egress_has_bridge_network_and_is_not_training_network(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            control = root / "control"
            lease = root / "lease"
            workspace.mkdir()
            control.mkdir()
            public_task = root / "task.json"
            public_task.write_text("{}", encoding="utf-8")
            publish_agent_api(public_task)
            config = TaskConfig.load(TASK)
            agent = AgentConfig(
                workspace=workspace,
                runner="container",
                command=["python", "/workspace/agent.py"],
                initial_workspace="participant_snapshot",
                image="python:test-agent",
                egress="unrestricted-dev",
            )
            runtime = CapturingDockerRuntime()
            runtime.start_agent(
                name="agent",
                config=config,
                agent=agent,
                control_dir=control,
                public_task_path=public_task,
            )
            runtime.start_training(
                name="training",
                task=ResearchTaskSpec.from_task(
                    config, public_task.parent / "agent_api"
                ),
                workspace=workspace,
                resources=TrainingResources(),
                lease_dir=lease,
            )
            agent_command, training_command = runtime.calls
            self.assertEqual(
                agent_command[agent_command.index("--network") + 1], "bridge"
            )
            self.assertEqual(
                training_command[training_command.index("--network") + 1], "none"
            )

    def test_training_devices_come_from_episode_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            public_task = root / "task.json"
            public_task.write_text("{}", encoding="utf-8")
            publish_agent_api(public_task)
            runtime = CapturingDockerRuntime()
            runtime.start_training(
                name="training",
                task=ResearchTaskSpec.from_task(
                    TaskConfig.load(TASK), public_task.parent / "agent_api"
                ),
                workspace=workspace,
                resources=TrainingResources(
                    nvidia_gpu_count=2,
                    devices=["/dev/accel0"],
                ),
                lease_dir=root / "lease",
            )
            command = runtime.calls[0]
            self.assertEqual(command[command.index("--gpus") + 1], "2")
            self.assertEqual(command[command.index("--device") + 1], "/dev/accel0")

    def test_evaluator_and_candidate_use_separate_internal_containers(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                TaskConfig.load(TASK),
                evaluator_device="cuda",
            ).validate()
            resources = TrainingResources(nvidia_gpu_count=1)
            runtime = CapturingDockerRuntime(monotonic=lambda: 100.0)
            output = Path(directory) / "output"
            with self.assertRaises(RuntimeFailure):
                runtime.run_evaluator(
                    config=config,
                    resources=resources,
                    candidate_dir=TASK.parent / "agent",
                    split="analysis",
                    output_dir=output,
                    timeout_seconds=7,
                )
            run_calls = [call for call in runtime.calls if call and call[0] == "run"]
            candidate = next(
                call
                for call in run_calls
                if "candidate_offline_runner" in " ".join(call)
            )
            evaluators = [
                call for call in run_calls if "/opt/docker-scorer-adapter.py" in call
            ]
            self.assertEqual(len(evaluators), 2)
            validation, evaluator = evaluators
            self.assertIn("dst=/candidate,readonly", "\n".join(candidate))
            self.assertIn("dst=/opt/candidate-api,readonly", "\n".join(candidate))
            self.assertNotIn(str(ROOT / "autotrain_bench"), "\n".join(candidate))
            self.assertNotIn(str(config.analysis_path), "\n".join(candidate))
            self.assertIn(
                "dst=/eval-input/contexts.json,readonly", "\n".join(candidate)
            )
            self.assertNotIn("targets.json", "\n".join(candidate))
            self.assertIn("dst=/predictions", "\n".join(candidate))
            self.assertNotIn("dst=/candidate", "\n".join(evaluator))
            evaluator_command = "\n".join(evaluator)
            self.assertNotIn(str(config.analysis_path), evaluator_command)
            self.assertIn("autotrain-bench-hidden-", evaluator_command)
            self.assertIn("dst=/eval-data/contexts.json,readonly", evaluator_command)
            self.assertIn("dst=/eval-data/targets.json,readonly", evaluator_command)
            self.assertIn(str(config.evaluator_entrypoint), "\n".join(evaluator))
            lengths_index = evaluator.index("--context-lengths")
            self.assertEqual(
                evaluator[lengths_index + 1 : lengths_index + 4],
                ["1", "2", "3"],
            )
            self.assertIn("--gpus", candidate)
            self.assertNotIn("--gpus", evaluator)
            self.assertEqual(candidate[candidate.index("--network") + 1], "none")
            self.assertEqual(evaluator[evaluator.index("--network") + 1], "none")
            self.assertEqual(validation[validation.index("--mode") + 1], "validate")
            self.assertEqual(evaluator[evaluator.index("--mode") + 1], "score")
            evaluator_index = next(
                index
                for index, call in enumerate(runtime.calls)
                if "/opt/docker-scorer-adapter.py" in call
            )
            self.assertEqual(runtime.call_kwargs[evaluator_index]["timeout"], 7)

    def test_evaluator_deadline_includes_validation_and_candidate_serving(self):
        class FakeClock:
            value = 100.0

            def __call__(self):
                return self.value

            def advance(self, seconds):
                self.value += seconds

        clock = FakeClock()

        class AdvancingRuntime(CapturingDockerRuntime):
            def _run(self, args, **kwargs):
                result = super()._run(args, **kwargs)
                if "--mode" in args and args[args.index("--mode") + 1] == "validate":
                    clock.advance(2)
                elif "candidate_offline_runner" in " ".join(args):
                    clock.advance(3)
                return result

        with tempfile.TemporaryDirectory() as directory:
            runtime = AdvancingRuntime(monotonic=clock)
            config = TaskConfig.load(TASK)
            with self.assertRaises(RuntimeFailure):
                runtime.run_evaluator(
                    config=config,
                    resources=TrainingResources(),
                    candidate_dir=TASK.parent / "agent",
                    split="analysis",
                    output_dir=Path(directory) / "output",
                    timeout_seconds=7,
                )

            validation_index = next(
                index
                for index, call in enumerate(runtime.calls)
                if "--mode" in call and call[call.index("--mode") + 1] == "validate"
            )
            candidate_index = next(
                index
                for index, call in enumerate(runtime.calls)
                if "candidate_offline_runner" in " ".join(call)
            )
            evaluator_index = next(
                index
                for index, call in enumerate(runtime.calls)
                if "--mode" in call and call[call.index("--mode") + 1] == "score"
            )
            self.assertEqual(runtime.call_kwargs[validation_index]["timeout"], 7)
            self.assertEqual(runtime.call_kwargs[candidate_index]["timeout"], 5)
            self.assertEqual(runtime.call_kwargs[evaluator_index]["timeout"], 2)

    def test_serving_clock_excludes_trusted_validation_and_scoring(self):
        class FakeClock:
            value = 100.0

            def __call__(self):
                return self.value

            def advance(self, seconds):
                self.value += seconds

        clock = FakeClock()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation"

            class ResultRuntime(CapturingDockerRuntime):
                def _run(self, args, **kwargs):
                    result = super()._run(args, **kwargs)
                    joined = " ".join(args)
                    if "--mode" in args:
                        mode = args[args.index("--mode") + 1]
                        if mode == "validate":
                            clock.advance(5)
                        else:
                            output.mkdir(parents=True, exist_ok=True)
                            (output / "analysis.json").write_text(
                                json.dumps(
                                    {
                                        "performance_score": 72.5,
                                        "raw_metrics": {"mean_nll": 1.25},
                                        "examples": 3,
                                        "input_tokens": 6,
                                        "prediction_bytes": 64,
                                    }
                                ),
                                encoding="utf-8",
                            )
                            clock.advance(7)
                    elif "candidate_offline_runner" in joined:
                        clock.advance(3)
                    return result

            runtime = ResultRuntime(monotonic=clock)
            result = runtime.run_evaluator(
                config=TaskConfig.load(TASK),
                resources=TrainingResources(),
                candidate_dir=TASK.parent / "agent",
                split="analysis",
                output_dir=output,
                timeout_seconds=20,
            )

            self.assertEqual(result.performance_score, 72.5)
            self.assertEqual(result.raw_metrics["mean_nll"], 1.25)
            self.assertEqual(result.candidate_wall_seconds, 3.0)
            self.assertEqual(result.task_count, 3)
            self.assertEqual(result.input_tokens, 6)
            self.assertEqual(result.prediction_bytes, 64)
            candidate = next(
                call
                for call in runtime.calls
                if "candidate_offline_runner" in " ".join(call)
            )
            self.assertNotIn("--rm", candidate)
            self.assertEqual(list(output.glob("predictions-*")), [])

    def test_evaluator_teardown_attempts_every_isolated_resource(self):
        class FailingTeardownRuntime(CapturingDockerRuntime):
            def __init__(self):
                super().__init__()
                self.teardown = []

            def stop_container(self, name):
                role = "candidate" if "candidate" in name else "evaluator"
                self.teardown.append(role)
                if role == "evaluator":
                    raise RuntimeFailure("evaluator termination not confirmed")

        with tempfile.TemporaryDirectory() as directory:
            runtime = FailingTeardownRuntime()
            with self.assertRaisesRegex(
                WorkerTeardownFailure, "runtime teardown failed"
            ):
                runtime.run_evaluator(
                    config=TaskConfig.load(TASK),
                    resources=TrainingResources(),
                    candidate_dir=TASK.parent / "agent",
                    split="analysis",
                    output_dir=Path(directory) / "output",
                )
            self.assertEqual(runtime.teardown, ["candidate", "evaluator"])
            self.assertEqual(len(runtime._pending_evaluations), 1)
            prediction_dir = next(iter(runtime._pending_evaluations.values()))[2]
            self.assertTrue(prediction_dir.exists())

            runtime.stop_container = lambda name: runtime.teardown.append(
                "retry-container"
            )
            runtime.teardown_pending_evaluations()
            runtime.close_private_materializations()

            self.assertFalse(prediction_dir.exists())
            self.assertEqual(runtime._pending_evaluations, {})
            self.assertEqual(
                runtime.teardown[-2:],
                [
                    "retry-container",
                    "retry-container",
                ],
            )

    def test_prediction_cleanup_failure_remains_pending_teardown(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = CapturingDockerRuntime()
            prediction_dir = Path(directory) / "predictions"
            prediction_dir.mkdir()
            key = ("episode", "evaluation")
            runtime._pending_evaluations[key] = (
                "candidate",
                "evaluator",
                prediction_dir,
            )

            with patch.object(
                runtime,
                "_remove_prediction_artifact",
                side_effect=OSError("storage unavailable"),
            ), self.assertRaisesRegex(
                WorkerTeardownFailure, "prediction artifact cleanup failed"
            ):
                runtime._teardown_evaluation(key)

            self.assertIn(key, runtime._pending_evaluations)
            self.assertTrue(prediction_dir.exists())
            runtime._teardown_evaluation(key)
            self.assertNotIn(key, runtime._pending_evaluations)
            self.assertFalse(prediction_dir.exists())

    def test_evaluator_never_accepts_a_stale_result_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            output.mkdir()
            result = output / "analysis.json"
            result.write_text('{"performance_score": 99.0}', encoding="utf-8")
            runtime = CapturingDockerRuntime()

            with self.assertRaisesRegex(
                RuntimeFailure, "valid performance or cost result"
            ):
                runtime.run_evaluator(
                    config=TaskConfig.load(TASK),
                    resources=TrainingResources(),
                    candidate_dir=TASK.parent / "agent",
                    split="analysis",
                    output_dir=output,
                )

            self.assertFalse(result.exists())
            self.assertEqual(list(output.glob("predictions-*")), [])

    def test_candidate_logs_are_private_and_never_enter_runtime_error(self):
        class FailingRuntime(CapturingDockerRuntime):
            def _run(self, args, **kwargs):
                if args[:2] == ["image", "inspect"] and "{{.Id}}" in args:
                    return super()._run(args, **kwargs)
                self.calls.append(list(args))
                self.call_kwargs.append(dict(kwargs))
                if "candidate_offline_runner" in " ".join(args):
                    raise RuntimeFailure("hidden input_ids: [0, 1, 2]")
                return subprocess.CompletedProcess(args, 0, "container-id\n", "")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation"
            runtime = FailingRuntime()
            config = TaskConfig.load(TASK)
            with self.assertRaisesRegex(
                RuntimeFailure, "^isolated evaluation failed$"
            ) as caught:
                runtime.run_evaluator(
                    config=config,
                    resources=TrainingResources(),
                    candidate_dir=TASK.parent / "agent",
                    split="analysis",
                    output_dir=output,
                )
            self.assertNotIn("input_ids", str(caught.exception))
            diagnostic = output / "analysis.failure.log"
            self.assertEqual(diagnostic.stat().st_mode & 0o777, 0o600)
            self.assertIn("hidden input_ids", diagnostic.read_text(encoding="utf-8"))
            self.assertEqual(list(output.glob("predictions-*")), [])

    def test_evaluator_rejects_boolean_and_string_performance_scores(self):
        for raw_score in (True, "72.5"):
            with self.subTest(raw_score=raw_score):
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "evaluation"

                    class ResultRuntime(CapturingDockerRuntime):
                        def _run(self, args, **kwargs):
                            result = super()._run(args, **kwargs)
                            if (
                                "--mode" in args
                                and args[args.index("--mode") + 1] == "score"
                            ):
                                output.mkdir(parents=True, exist_ok=True)
                                (output / "analysis.json").write_text(
                                    json.dumps(
                                        {
                                            "performance_score": raw_score,
                                            "raw_metrics": {"mean_nll": 1.25},
                                            "examples": 3,
                                            "input_tokens": 6,
                                            "prediction_bytes": 64,
                                        }
                                    ),
                                    encoding="utf-8",
                                )
                            return result

                    runtime = ResultRuntime()
                    with self.assertRaisesRegex(
                        RuntimeFailure, "valid performance or cost result"
                    ):
                        runtime.run_evaluator(
                            config=TaskConfig.load(TASK),
                            resources=TrainingResources(),
                            candidate_dir=TASK.parent / "agent",
                            split="analysis",
                            output_dir=output,
                        )

    def test_missing_inference_credential_fails_before_gateway_start(self):
        with tempfile.TemporaryDirectory() as directory:
            inference = InferenceConfig(
                protocol="openai_compatible",
                base_url="https://api.openai.com",
                api_key_env="AUTOTRAIN_BENCH_TEST_MISSING_KEY",
                allowed_models=("researcher-a",),
                allowed_paths=("/v1/responses",),
                timeout_seconds=300,
            )
            config = TaskConfig.load(TASK)
            agent = AgentConfig(
                workspace=Path(directory),
                runner="container",
                command=["agent"],
                initial_workspace="participant_snapshot",
                image="agent:image",
                egress="denied",
                inference=inference,
            )
            runtime = CapturingDockerRuntime()
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("AUTOTRAIN_BENCH_TEST_MISSING_KEY", None)
                with self.assertRaises(RuntimeFailure):
                    runtime.start_inference_gateway(
                        name="gateway",
                        agent_network="agent-net",
                        egress_network="egress-net",
                        config=config,
                        agent=agent,
                        audit_dir=Path(directory) / "audit",
                    )
            self.assertEqual(runtime.calls, [])

    def test_host_accelerators_are_recorded_even_when_none_were_requested(self):
        """A profile declaring no accelerator must still leave a trace.

        A site whose scheduler refuses a zero-accelerator allocation runs a
        `cpu-local` profile on a GPU node. Recording an empty list there would
        assert the host had none, which is indistinguishable from never having
        looked.
        """

        inventory = subprocess.CompletedProcess(
            ["nvidia-smi"],
            0,
            "NVIDIA H200, GPU-abc, 550.54.15, 143771\n",
            "",
        )
        runtime = CapturingDockerRuntime()
        with patch("autotrain_bench.runtime.shutil.which", return_value="nvidia-smi"):
            with patch(
                "autotrain_bench.runtime.subprocess.run", return_value=inventory
            ):
                manifest = runtime.environment_manifest(
                    TaskConfig.load(TASK), resources=TrainingResources()
                )

        self.assertEqual(manifest["gpu_inventory"], "inspected")
        self.assertEqual([gpu["name"] for gpu in manifest["gpus"]], ["NVIDIA H200"])
        self.assertEqual(manifest["training_resources"]["nvidia_gpu_count"], 0)

    def test_an_uninspectable_host_is_not_reported_as_having_no_accelerator(self):
        runtime = CapturingDockerRuntime()
        with patch("autotrain_bench.runtime.shutil.which", return_value=None):
            manifest = runtime.environment_manifest(
                TaskConfig.load(TASK), resources=TrainingResources()
            )

        self.assertEqual(manifest["gpu_inventory"], "unavailable")
        self.assertEqual(manifest["gpus"], [])

    def test_a_failing_inventory_query_only_raises_when_accelerators_were_requested(
        self,
    ):
        failure = subprocess.CompletedProcess(["nvidia-smi"], 9, "", "driver error")
        runtime = CapturingDockerRuntime()

        with patch("autotrain_bench.runtime.shutil.which", return_value="nvidia-smi"):
            with patch("autotrain_bench.runtime.subprocess.run", return_value=failure):
                manifest = runtime.environment_manifest(
                    TaskConfig.load(TASK), resources=TrainingResources()
                )
        self.assertEqual(manifest["gpu_inventory"], "unavailable")

        with patch("autotrain_bench.runtime.shutil.which", return_value="nvidia-smi"):
            with patch("autotrain_bench.runtime.subprocess.run", return_value=failure):
                with self.assertRaisesRegex(RuntimeFailure, "driver error"):
                    runtime.environment_manifest(
                        TaskConfig.load(TASK),
                        resources=TrainingResources(nvidia_gpu_count=1),
                    )


if __name__ == "__main__":
    unittest.main()
