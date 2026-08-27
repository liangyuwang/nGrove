import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from autotrain_bench.config import TaskConfig
from autotrain_bench.runtime import DockerRuntime


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
            self.assertNotIn(str(ROOT / "examples/toy/private"), command)
            self.assertNotIn(str(config.val_path), command)
            self.assertNotIn(str(config.test_path), command)


if __name__ == "__main__":
    unittest.main()

