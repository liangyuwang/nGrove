import os
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from autotrain_bench.config import InferenceConfig, TaskConfig
from autotrain_bench.runtime import DockerRuntime, RuntimeFailure
from autotrain_bench.host_agent import summarize_host_agent_trace


ROOT = Path(__file__).resolve().parents[1]


class CapturingDockerRuntime(DockerRuntime):
    def __init__(self):
        super().__init__(docker_binary="docker")
        self.calls = []

    def _run(self, args, **kwargs):
        self.calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, "container-id\n", "")


class RuntimeBoundaryTests(unittest.TestCase):
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
            config = replace(
                TaskConfig.load(ROOT / "examples/toy/task.json"),
                agent_workspace=workspace,
                agent_runner="host_cli",
                agent_mode=None,
                agent_command=[sys.executable, "-c", script],
                agent_environment=["VISIBLE"],
                inference=None,
            )
            runtime = DockerRuntime(docker_binary="docker")
            with patch.dict(
                os.environ,
                {"VISIBLE": "yes", "SHOULD_NOT_LEAK": "secret"},
                clear=False,
            ):
                handle = runtime.start_host_agent(
                    config=config,
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

    def test_agent_mount_does_not_expose_repository_or_private_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            control = root / "control"
            workspace.mkdir()
            control.mkdir()
            public_task = root / "task.json"
            public_task.write_text("{}", encoding="utf-8")
            config = replace(
                TaskConfig.load(ROOT / "examples/toy/task.json"),
                agent_workspace=workspace,
            )
            runtime = CapturingDockerRuntime()
            runtime.start_agent(
                name="agent",
                config=config,
                control_dir=control,
                public_task_path=public_task,
            )
            command = "\n".join(runtime.calls[0])
            self.assertIn("dst=/opt/autotrain_bench,readonly", command)
            self.assertIn("/agent-home:rw,nosuid,size=1g,mode=1777", command)
            self.assertIn("HOME=/agent-home", command)
            self.assertNotIn(str(ROOT / "examples/toy/private"), command)
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
            inference = InferenceConfig(
                protocol="openai_compatible",
                base_url="https://api.openai.com",
                api_key_env="OPENAI_API_KEY",
                allowed_models=("researcher-a",),
                allowed_paths=("/v1/responses",),
                timeout_seconds=300,
            )
            config = replace(
                TaskConfig.load(ROOT / "examples/toy/task.json"),
                agent_workspace=workspace,
                agent_command=["node", "/workspace/agent.js"],
                agent_mode="sandboxed",
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
                    audit_dir=audit,
                )
            runtime.start_agent(
                name="agent",
                config=config,
                control_dir=control,
                public_task_path=public_task,
                inference_network="agent-net",
            )
            runtime.start_training(name="training", config=config, lease_dir=lease)

            self.assertIn("--internal", runtime.calls[0])
            run_calls = [call for call in runtime.calls if call and call[0] == "run"]
            gateway_command = next(
                call for call in run_calls if call[call.index("--name") + 1] == "gateway"
            )
            agent_command = next(
                call for call in run_calls if call[call.index("--name") + 1] == "agent"
            )
            training_command = next(
                call for call in run_calls if call[call.index("--name") + 1] == "training"
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
            config = replace(
                TaskConfig.load(ROOT / "examples/toy/task.json"),
                agent_workspace=workspace,
                agent_mode="unrestricted-dev",
            )
            runtime = CapturingDockerRuntime()
            runtime.start_agent(
                name="agent",
                config=config,
                control_dir=control,
                public_task_path=public_task,
            )
            runtime.start_training(name="training", config=config, lease_dir=lease)
            agent_command, training_command = runtime.calls
            self.assertEqual(
                agent_command[agent_command.index("--network") + 1], "bridge"
            )
            self.assertEqual(
                training_command[training_command.index("--network") + 1], "none"
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
            config = replace(
                TaskConfig.load(ROOT / "examples/toy/task.json"),
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
                        audit_dir=Path(directory) / "audit",
                    )
            self.assertEqual(runtime.calls, [])


if __name__ == "__main__":
    unittest.main()
