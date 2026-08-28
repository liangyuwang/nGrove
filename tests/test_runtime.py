import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from autotrain_bench.config import InferenceConfig, TaskConfig
from autotrain_bench.runtime import DockerRuntime, RuntimeFailure


ROOT = Path(__file__).resolve().parents[1]


class CapturingDockerRuntime(DockerRuntime):
    def __init__(self):
        super().__init__(docker_binary="docker")
        self.calls = []

    def _run(self, args, **kwargs):
        self.calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, "container-id\n", "")


class RuntimeBoundaryTests(unittest.TestCase):
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
