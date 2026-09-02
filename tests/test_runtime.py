import os
import io
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from autotrain_bench.config import (
    AgentConfig,
    InferenceConfig,
    TaskConfig,
    TrainingResources,
)
from autotrain_bench.runtime import DockerRuntime, RuntimeFailure
from autotrain_bench.host_agent import _BoundedLogSink, summarize_host_agent_trace
from autotrain_bench.worker_protocol import (
    DEADLINE_TERMINATION,
    EnforcementLevel,
    ResearchTaskSpec,
    TrustLevel,
    WorkerTeardownFailure,
)


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tests/fixtures/task/task.json"


def publish_agent_api(public_task: Path) -> None:
    package = public_task.parent / "agent_api/autotrain_bench"
    package.mkdir(parents=True)
    (package / "client.py").write_text("class Client: pass\n", encoding="utf-8")
    (package / "candidate_preflight.py").write_text(
        "def main(): pass\n", encoding="utf-8"
    )


class CapturingDockerRuntime(DockerRuntime):
    def __init__(self, *, monotonic=None):
        super().__init__(docker_binary="docker", monotonic=monotonic)
        self.calls = []
        self.call_kwargs = []

    def _run(self, args, **kwargs):
        self.calls.append(list(args))
        self.call_kwargs.append(dict(kwargs))
        return subprocess.CompletedProcess(args, 0, "container-id\n", "")


class RuntimeBoundaryTests(unittest.TestCase):
    def test_local_reference_worker_is_explicitly_development_only(self):
        descriptor = DockerRuntime(docker_binary="docker").describe_capabilities()
        self.assertEqual(descriptor.trust_level, TrustLevel.DEVELOPMENT)
        self.assertIsNone(descriptor.conformance)
        self.assertEqual(
            descriptor.capabilities[DEADLINE_TERMINATION].level,
            EnforcementLevel.BEST_EFFORT,
        )

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
                    )

                self.assertIn(("episode", "lease"), runtime._worker_leases)
                runtime.terminate_lease(episode_id="episode", lease_id="lease")

        self.assertEqual(stop.call_count, 2)
        self.assertNotIn(("episode", "lease"), runtime._worker_leases)

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
                mode="sandboxed",
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
            self.assertNotIn(str(config.val_path), command)
            self.assertNotIn(str(config.test_path), command)

    def test_sandboxed_inference_uses_trusted_sidecar_and_internal_network(self):
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
                mode="sandboxed",
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

    def test_unrestricted_dev_mode_has_bridge_network_and_is_not_training_network(self):
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
                mode="unrestricted-dev",
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
                    split="val",
                    output_dir=output,
                    timeout_seconds=7,
                )
            create = runtime.calls[0]
            run_calls = [call for call in runtime.calls if call and call[0] == "run"]
            candidate = next(
                call for call in run_calls if "candidate_server" in " ".join(call)
            )
            evaluator = next(call for call in run_calls if "/opt/evaluator.py" in call)
            self.assertIn("--internal", create)
            self.assertIn("dst=/candidate,readonly", "\n".join(candidate))
            self.assertIn("dst=/opt/candidate-api,readonly", "\n".join(candidate))
            self.assertNotIn(str(ROOT / "autotrain_bench"), "\n".join(candidate))
            self.assertNotIn(str(config.val_path), "\n".join(candidate))
            self.assertNotIn("dst=/candidate", "\n".join(evaluator))
            self.assertIn(str(config.val_path), "\n".join(evaluator))
            self.assertIn(str(config.evaluator_entrypoint), "\n".join(evaluator))
            self.assertIn("http://candidate:8766", evaluator)
            self.assertIn("--gpus", candidate)
            self.assertNotIn("--gpus", evaluator)
            network = create[-1]
            self.assertEqual(candidate[candidate.index("--network") + 1], network)
            self.assertEqual(evaluator[evaluator.index("--network") + 1], network)
            self.assertIn(["network", "rm", network], runtime.calls)
            evaluator_index = next(
                index
                for index, call in enumerate(runtime.calls)
                if "/opt/evaluator.py" in call
            )
            self.assertEqual(runtime.call_kwargs[evaluator_index]["timeout"], 7)

    def test_evaluator_deadline_includes_network_and_candidate_startup(self):
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
                if args[:2] == ["network", "create"]:
                    clock.advance(2)
                elif "candidate_server" in " ".join(args):
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
                    split="val",
                    output_dir=Path(directory) / "output",
                    timeout_seconds=7,
                )

            network_index = next(
                index
                for index, call in enumerate(runtime.calls)
                if call[:2] == ["network", "create"]
            )
            candidate_index = next(
                index
                for index, call in enumerate(runtime.calls)
                if "candidate_server" in " ".join(call)
            )
            evaluator_index = next(
                index
                for index, call in enumerate(runtime.calls)
                if "/opt/evaluator.py" in call
            )
            self.assertEqual(runtime.call_kwargs[network_index]["timeout"], 7)
            self.assertEqual(runtime.call_kwargs[candidate_index]["timeout"], 5)
            self.assertEqual(runtime.call_kwargs[evaluator_index]["timeout"], 2)

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

            def remove_network(self, name):
                del name
                self.teardown.append("network")

        with tempfile.TemporaryDirectory() as directory:
            runtime = FailingTeardownRuntime()
            with self.assertRaisesRegex(
                WorkerTeardownFailure, "runtime teardown failed"
            ):
                runtime.run_evaluator(
                    config=TaskConfig.load(TASK),
                    resources=TrainingResources(),
                    candidate_dir=TASK.parent / "agent",
                    split="val",
                    output_dir=Path(directory) / "output",
                )

        self.assertEqual(runtime.teardown, ["candidate", "evaluator", "network"])

    def test_candidate_logs_are_private_and_never_enter_runtime_error(self):
        class FailingRuntime(CapturingDockerRuntime):
            def _run(self, args, **kwargs):
                self.calls.append(list(args))
                self.call_kwargs.append(dict(kwargs))
                if args and args[0] == "logs":
                    return subprocess.CompletedProcess(
                        args, 0, "hidden input_ids: [0, 1, 2]\n", ""
                    )
                if "/opt/evaluator.py" in args:
                    raise RuntimeFailure("secret evaluator detail")
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
                    split="val",
                    output_dir=output,
                )
            self.assertNotIn("input_ids", str(caught.exception))
            diagnostic = output / "val.failure.log"
            self.assertEqual(diagnostic.stat().st_mode & 0o777, 0o600)
            self.assertIn("hidden input_ids", diagnostic.read_text(encoding="utf-8"))

    def test_evaluator_rejects_boolean_and_string_nll_values(self):
        for raw_nll in (True, "1.25"):
            with self.subTest(raw_nll=raw_nll):
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "evaluation"

                    class ResultRuntime(CapturingDockerRuntime):
                        def _run(self, args, **kwargs):
                            result = super()._run(args, **kwargs)
                            if "/opt/evaluator.py" in args:
                                output.mkdir(parents=True, exist_ok=True)
                                (output / "val.json").write_text(
                                    json.dumps({"nll": raw_nll}), encoding="utf-8"
                                )
                            return result

                    runtime = ResultRuntime()
                    with self.assertRaisesRegex(RuntimeFailure, "valid NLL result"):
                        runtime.run_evaluator(
                            config=TaskConfig.load(TASK),
                            resources=TrainingResources(),
                            candidate_dir=TASK.parent / "agent",
                            split="val",
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
                mode="sandboxed",
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


if __name__ == "__main__":
    unittest.main()
