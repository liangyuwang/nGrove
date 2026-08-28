import json
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

    def test_sandboxed_inference_uses_local_proxy_and_keeps_host_key_out(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            control = root / "control"
            lease = root / "lease"
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
                agent_mode="sandboxed",
                inference=inference,
            )
            runtime = CapturingDockerRuntime()
            with patch.dict(os.environ, {"OPENAI_API_KEY": "host-secret"}):
                runtime.start_agent(
                    name="agent",
                    config=config,
                    control_dir=control,
                    public_task_path=public_task,
                )
            runtime.start_training(name="training", config=config, lease_dir=lease)

            agent_command = runtime.calls[0]
            training_command = runtime.calls[1]
            agent_network = agent_command.index("--network")
            training_network = training_command.index("--network")
            self.assertEqual(agent_command[agent_network + 1], "none")
            self.assertEqual(training_command[training_network + 1], "none")
            serialized = "\n".join(agent_command)
            self.assertIn("OPENAI_BASE_URL=http://127.0.0.1:8765/v1", serialized)
            self.assertIn("autotrain_bench.agent_entrypoint", serialized)
            self.assertNotIn("host-secret", serialized)
            self.assertNotIn("api.openai.com", serialized)
            self.assertNotIn("OPENAI_API_KEY", "\n".join(training_command))

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
                name="agent", config=config, control_dir=control, public_task_path=public_task
            )
            runtime.start_training(name="training", config=config, lease_dir=lease)
            agent_command, training_command = runtime.calls
            self.assertEqual(agent_command[agent_command.index("--network") + 1], "bridge")
            self.assertEqual(training_command[training_command.index("--network") + 1], "none")

    def test_missing_inference_credential_fails_before_container_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            control = root / "control"
            workspace.mkdir()
            control.mkdir()
            public_task = root / "task.json"
            public_task.write_text("{}", encoding="utf-8")
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
                agent_workspace=workspace,
                inference=inference,
            )
            runtime = CapturingDockerRuntime()
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("AUTOTRAIN_BENCH_TEST_MISSING_KEY", None)
                with self.assertRaises(RuntimeFailure):
                    runtime.start_agent(
                        name="agent",
                        config=config,
                        control_dir=control,
                        public_task_path=public_task,
                    )
            self.assertEqual(runtime.calls, [])


if __name__ == "__main__":
    unittest.main()
