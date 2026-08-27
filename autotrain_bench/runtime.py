"""Container runtime used by the host-side reference supervisor.

AutoTrain-Bench deliberately does not allocate machines or integrate with a
cluster scheduler. The runtime only consumes devices already visible on the
single host where the supervisor is started.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from .config import TaskConfig


class RuntimeFailure(RuntimeError):
    """A trusted runtime operation failed."""


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class DockerRuntime:
    """Minimal Docker implementation for the local reference prototype."""

    def __init__(self, docker_binary: Optional[str] = None) -> None:
        self.docker = docker_binary or shutil.which("docker") or ""

    def _require_docker(self) -> None:
        if not self.docker:
            raise RuntimeFailure(
                "Docker was not found. Install Docker before running the reference "
                "prototype; AutoTrain-Bench does not allocate remote resources."
            )

    def _run(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        timeout: Optional[float] = None,
    ) -> subprocess.CompletedProcess:
        self._require_docker()
        command = [self.docker, *args]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeFailure(f"runtime command timed out: {' '.join(command)}") from exc
        if check and result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise RuntimeFailure(f"Docker command failed: {detail}")
        return result

    @staticmethod
    def _mount(source: Path, target: str, *, read_only: bool) -> str:
        value = f"type=bind,src={source.resolve()},dst={target}"
        if read_only:
            value += ",readonly"
        return value

    def prepare(self, config: TaskConfig, *, pull_missing: bool = True) -> None:
        self._run(["info"], timeout=20)
        for image in config.images:
            inspected = self._run(["image", "inspect", image], check=False)
            if inspected.returncode == 0:
                continue
            if not pull_missing:
                raise RuntimeFailure(f"required image is missing: {image}")
            self._run(["pull", image])
        self.validate_hardware(config)

    def validate_hardware(self, config: TaskConfig) -> None:
        if config.gpu_count == 0:
            return
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            raise RuntimeFailure(
                f"task requires {config.gpu_count} GPU(s), but nvidia-smi is unavailable"
            )
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeFailure(result.stderr.strip() or "nvidia-smi failed")
        names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(names) != config.gpu_count:
            raise RuntimeFailure(
                f"task requires exactly {config.gpu_count} visible GPU(s); found {len(names)}"
            )
        if config.gpu_model and any(config.gpu_model not in name for name in names):
            raise RuntimeFailure(
                f"task requires GPU model containing {config.gpu_model!r}; found {names}"
            )

    def start_agent(
        self,
        *,
        name: str,
        config: TaskConfig,
        control_dir: Path,
        public_task_path: Path,
    ) -> None:
        package_dir = Path(__file__).resolve().parent
        home = config.agent_workspace / ".autotrain-home"
        home.mkdir(parents=True, exist_ok=True)
        args: List[str] = [
            "run",
            "--detach",
            "--name",
            name,
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            "512",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,size=256m",
            "--mount",
            self._mount(config.agent_workspace, "/workspace", read_only=False),
            "--mount",
            self._mount(config.train_dir, "/benchmark/train", read_only=True),
            "--mount",
            self._mount(public_task_path, "/benchmark/task.json", read_only=True),
            "--mount",
            self._mount(control_dir, "/control", read_only=False),
            "--mount",
            self._mount(package_dir, "/opt/autotrain_bench", read_only=True),
            "--env",
            "PYTHONPATH=/opt",
            "--env",
            "AUTOTRAIN_BENCH_CONTROL_DIR=/control",
            "--env",
            "AUTOTRAIN_BENCH_TASK=/benchmark/task.json",
            "--env",
            "HOME=/workspace/.autotrain-home",
            "--workdir",
            "/workspace",
            config.agent_image,
            *config.agent_command,
        ]
        self._run(args)

    def start_training(
        self,
        *,
        name: str,
        config: TaskConfig,
        lease_dir: Path,
    ) -> None:
        scratch = lease_dir / "scratch"
        output = lease_dir / "output"
        scratch.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        args: List[str] = [
            "run",
            "--detach",
            "--name",
            name,
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            "2048",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,exec,nosuid,size=2g",
            "--mount",
            self._mount(config.agent_workspace, "/workspace", read_only=False),
            "--mount",
            self._mount(config.train_dir, "/benchmark/train", read_only=True),
            "--mount",
            self._mount(scratch, "/scratch", read_only=False),
            "--mount",
            self._mount(output, "/output", read_only=False),
            "--env",
            "HOME=/scratch",
            "--env",
            "AUTOTRAIN_TRAIN_DIR=/benchmark/train",
            "--env",
            "AUTOTRAIN_OUTPUT_DIR=/output",
            "--workdir",
            "/workspace",
        ]
        if config.gpu_count:
            args.extend(["--gpus", str(config.gpu_count)])
        args.extend([config.training_image, "sleep", "infinity"])
        self._run(args)

    def exec_training(
        self,
        *,
        name: str,
        argv: Sequence[str],
        timeout_seconds: float,
    ) -> ExecResult:
        self._require_docker()
        command = [self.docker, "exec", name, *argv]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=max(1, math.ceil(timeout_seconds)),
            )
        except subprocess.TimeoutExpired as exc:
            self.stop_container(name)
            return ExecResult(
                exit_code=124,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "command exceeded the lease deadline",
                timed_out=True,
            )
        return ExecResult(result.returncode, result.stdout, result.stderr)

    def container_running(self, name: str) -> bool:
        result = self._run(
            ["inspect", "--format", "{{.State.Running}}", name], check=False
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def stop_container(self, name: str) -> None:
        if not name:
            return
        self._run(["rm", "--force", name], check=False)

    def run_evaluator(
        self,
        *,
        config: TaskConfig,
        candidate_dir: Path,
        split: str,
        output_dir: Path,
    ) -> float:
        if split not in {"val", "test"}:
            raise RuntimeFailure(f"invalid evaluator split: {split}")
        data_path = config.val_path if split == "val" else config.test_path
        evaluator = Path(__file__).resolve().parent / "toy_evaluator.py"
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / f"{split}.json"
        container_name = f"atb-eval-{uuid.uuid4().hex[:12]}"
        args = [
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            "512",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,size=128m",
            "--mount",
            self._mount(candidate_dir, "/candidate", read_only=True),
            "--mount",
            self._mount(data_path, "/eval-data/split.json", read_only=True),
            "--mount",
            self._mount(evaluator, "/opt/evaluator.py", read_only=True),
            "--mount",
            self._mount(output_dir, "/output", read_only=False),
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            config.evaluator_image,
            "python",
            "/opt/evaluator.py",
            "--candidate",
            "/candidate",
            "--data",
            "/eval-data/split.json",
            "--output",
            f"/output/{split}.json",
            "--vocab-size",
            str(config.vocab_size),
        ]
        try:
            self._run(args, timeout=config.evaluator_timeout_seconds)
        except RuntimeFailure:
            self.stop_container(container_name)
            raise
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            loss = float(payload["nll"])
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeFailure("evaluator did not produce a valid NLL result") from exc
        return loss
