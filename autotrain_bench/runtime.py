"""Container runtime used by the host-side reference supervisor.

AutoTrain-Bench deliberately does not allocate machines or integrate with a
cluster scheduler. The runtime only consumes devices already visible on the
single host where the supervisor is started.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import TaskConfig
from .host_agent import HostAgentProcess, start_host_agent_process


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

    def environment_manifest(self, config: TaskConfig) -> Dict[str, Any]:
        """Record the concrete runtime used for a comparable episode."""

        docker_version = self._run(
            ["version", "--format", "{{.Server.Version}}"], timeout=20
        ).stdout.strip()
        images: Dict[str, Dict[str, str]] = {}
        for role, reference in (
            ("agent", config.agent_image),
            ("training", config.training_image),
            ("candidate", config.candidate_image),
            ("evaluator", config.evaluator_image),
            (
                "inference_gateway",
                config.inference_gateway_image if config.inference else None,
            ),
        ):
            if not reference:
                continue
            inspected = self._run(
                ["image", "inspect", reference, "--format", "{{.Id}}"],
                timeout=20,
            )
            images[role] = {
                "reference": reference,
                "image_id": inspected.stdout.strip(),
            }

        gpus: List[Dict[str, str]] = []
        if config.gpu_count:
            nvidia_smi = shutil.which("nvidia-smi")
            if not nvidia_smi:  # guarded by validate_hardware
                raise RuntimeFailure("nvidia-smi disappeared after hardware validation")
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,uuid,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeFailure(result.stderr.strip() or "nvidia-smi failed")
            for line in result.stdout.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 4:
                    raise RuntimeFailure("nvidia-smi returned malformed GPU metadata")
                gpus.append(
                    {
                        "name": fields[0],
                        "uuid": fields[1],
                        "driver_version": fields[2],
                        "memory_mib": fields[3],
                    }
                )
        return {
            "docker_server_version": docker_version,
            "images": images,
            "gpus": gpus,
        }

    def start_agent(
        self,
        *,
        name: str,
        config: TaskConfig,
        control_dir: Path,
        public_task_path: Path,
        inference_network: Optional[str] = None,
    ) -> None:
        config.validate()
        if config.agent_runner != "container":
            raise RuntimeFailure("start_agent requires the container runner")
        if not config.agent_image:  # guarded by TaskConfig.validate
            raise RuntimeFailure("container runner has no Agent image")
        package_dir = Path(__file__).resolve().parent
        missing_environment = [
            name for name in config.agent_environment if name not in os.environ
        ]
        if missing_environment:
            names = ", ".join(missing_environment)
            raise RuntimeFailure(f"agent environment variable is not set: {names}")
        if config.inference is not None and not inference_network:
            raise RuntimeFailure("inference_network is required when inference is enabled")
        agent_network = (
            inference_network
            if config.inference is not None
            else ("none" if config.agent_mode == "sandboxed" else "bridge")
        )
        args: List[str] = [
            "run",
            "--detach",
            "--init",
            "--name",
            name,
            "--network",
            agent_network,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            "512",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,size=256m",
            "--tmpfs",
            "/agent-home:rw,nosuid,size=1g,mode=1777",
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
            "HOME=/agent-home",
            "--workdir",
            "/workspace",
        ]
        for name in config.agent_environment:
            # Passing only the name makes Docker copy the value from the trusted
            # Supervisor environment without putting the value in argv or traces.
            args.extend(["--env", name])

        if config.inference is not None:
            args.extend(
                [
                    "--env",
                    "AUTOTRAIN_BENCH_INFERENCE_BASE_URL="
                    "http://inference-gateway:8765/v1",
                    "--env",
                    "OPENAI_BASE_URL=http://inference-gateway:8765/v1",
                    "--env",
                    "OPENAI_API_KEY=autotrain-bench-gateway",
                ]
            )
        args.extend([config.agent_image, *config.agent_command])
        self._run(args)

    def start_host_agent(
        self,
        *,
        config: TaskConfig,
        control_dir: Path,
        public_task_path: Path,
        stdout_path: Path,
        stderr_path: Path,
    ) -> HostAgentProcess:
        config.validate()
        if config.agent_runner != "host_cli":
            raise RuntimeFailure("start_host_agent requires the host_cli runner")
        package_root = Path(__file__).resolve().parent.parent
        try:
            return start_host_agent_process(
                command=config.agent_command,
                workspace=config.agent_workspace,
                control_dir=control_dir,
                public_task_path=public_task_path,
                package_root=package_root,
                forwarded_environment=config.agent_environment,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
        except OSError as exc:
            raise RuntimeFailure(f"failed to start host Agent: {exc}") from exc

    def create_network(self, name: str, *, internal: bool) -> None:
        args = ["network", "create", "--driver", "bridge"]
        if internal:
            args.append("--internal")
        args.append(name)
        self._run(args)

    def remove_network(self, name: str) -> None:
        if name:
            self._run(["network", "rm", name], check=False)

    def start_inference_gateway(
        self,
        *,
        name: str,
        agent_network: str,
        egress_network: str,
        config: TaskConfig,
        audit_dir: Path,
        readiness_timeout_seconds: float = 10,
    ) -> None:
        config.validate()
        inference = config.inference
        if inference is None:
            raise RuntimeFailure("cannot start inference gateway without configuration")
        if inference.api_key_env and not os.environ.get(inference.api_key_env):
            raise RuntimeFailure(
                "inference credential is missing from the trusted host: "
                f"{inference.api_key_env}"
            )
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_dir.chmod(0o700)
        package_dir = Path(__file__).resolve().parent
        gateway_config = json.dumps(
            {
                "protocol": inference.protocol,
                "base_url": inference.base_url,
                "api_key_env": inference.api_key_env,
                "allowed_models": list(inference.allowed_models),
                "allowed_paths": list(inference.allowed_paths),
                "timeout_seconds": inference.timeout_seconds,
            },
            separators=(",", ":"),
        )
        args: List[str] = [
            "run",
            "--detach",
            "--init",
            "--name",
            name,
            "--network",
            egress_network,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            "128",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,size=64m",
            "--mount",
            self._mount(package_dir, "/opt/autotrain_bench", read_only=True),
            "--mount",
            self._mount(audit_dir, "/audit", read_only=False),
            "--env",
            "PYTHONPATH=/opt",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            f"AUTOTRAIN_BENCH_INFERENCE_CONFIG_JSON={gateway_config}",
            "--env",
            "AUTOTRAIN_BENCH_INFERENCE_AUDIT_PATH=/audit/inference.jsonl",
        ]
        if inference.api_key_env:
            # Docker copies the value from the host into the trusted Gateway only.
            args.extend(["--env", inference.api_key_env])
        args.extend(
            [
                config.inference_gateway_image,
                "python",
                "-m",
                "autotrain_bench.agent_gateway",
            ]
        )
        try:
            self._run(args)
            self._run(
                [
                    "network",
                    "connect",
                    "--alias",
                    "inference-gateway",
                    agent_network,
                    name,
                ]
            )
            deadline = time.monotonic() + readiness_timeout_seconds
            health_command = (
                "import urllib.request; "
                "urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=1).read()"
            )
            while time.monotonic() < deadline:
                ready = self._run(
                    ["exec", name, "python", "-c", health_command],
                    check=False,
                    timeout=2,
                )
                if ready.returncode == 0:
                    return
                if not self.container_running(name):
                    break
                time.sleep(0.05)
        except RuntimeFailure:
            self.stop_container(name)
            raise
        logs = self.container_logs(name).strip()
        self.stop_container(name)
        detail = f": {logs}" if logs else ""
        raise RuntimeFailure(f"inference gateway did not become ready{detail}")

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
        package_dir = Path(__file__).resolve().parent
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
            "--mount",
            self._mount(package_dir, "/opt/autotrain_bench", read_only=True),
            "--env",
            "HOME=/scratch",
            "--env",
            "PYTHONPATH=/opt",
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

    def container_logs(self, name: str) -> str:
        result = self._run(["logs", name], check=False)
        parts = [value for value in (result.stdout, result.stderr) if value]
        return "".join(parts)

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
        if config.evaluator_backend != "isolated_candidate":
            raise RuntimeFailure(
                f"unsupported evaluator backend: {config.evaluator_backend}"
            )
        data_path = config.val_path if split == "val" else config.test_path
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / f"{split}.json"
        runtime_id = uuid.uuid4().hex[:12]
        candidate_name = f"atb-candidate-{runtime_id}"
        evaluator_name = f"atb-evaluator-{runtime_id}"
        network_name = f"atb-evaluation-{runtime_id}"
        package_dir = Path(__file__).resolve().parent
        candidate_args: List[str] = [
            "run",
            "--detach",
            "--init",
            "--name",
            candidate_name,
            "--network",
            network_name,
            "--network-alias",
            "candidate",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            "512",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,size=1g",
            "--tmpfs",
            "/candidate-home:rw,nosuid,size=1g",
            "--log-opt",
            "max-size=10m",
            "--log-opt",
            "max-file=1",
            "--mount",
            self._mount(candidate_dir, "/candidate", read_only=True),
            "--mount",
            self._mount(package_dir, "/opt/autotrain_bench", read_only=True),
            "--env",
            "HOME=/candidate-home",
            "--env",
            "PYTHONPATH=/opt",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--workdir",
            "/candidate",
        ]
        if config.evaluator_device == "cuda":
            candidate_args.extend(["--gpus", str(config.gpu_count)])
        candidate_args.extend(
            [
                config.candidate_image,
                "python",
                "-m",
                "autotrain_bench.candidate_server",
                "--candidate",
                "/candidate",
                "--device",
                config.evaluator_device,
                "--vocab-size",
                str(config.vocab_size),
                "--context-length",
                str(config.context_length),
                "--maximum-batch-size",
                str(config.evaluator_batch_size),
            ]
        )
        evaluator_args: List[str] = [
            "run",
            "--rm",
            "--name",
            evaluator_name,
            "--network",
            network_name,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            "256",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,size=128m",
            "--mount",
            self._mount(data_path, "/eval-data/split.json", read_only=True),
            "--mount",
            self._mount(
                config.evaluator_entrypoint,
                "/opt/evaluator.py",
                read_only=True,
            ),
            "--mount",
            self._mount(output_dir, "/output", read_only=False),
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            config.evaluator_image,
            "python",
            "/opt/evaluator.py",
            "--candidate-url",
            "http://candidate:8766",
            "--data",
            "/eval-data/split.json",
            "--output",
            f"/output/{split}.json",
            "--vocab-size",
            str(config.vocab_size),
            "--batch-size",
            str(config.evaluator_batch_size),
        ]
        evaluator_started = False
        try:
            self.create_network(network_name, internal=True)
            self._run(candidate_args)
            evaluator_started = True
            self._run(
                evaluator_args,
                timeout=config.evaluator_timeout_seconds,
            )
        except RuntimeFailure as exc:
            candidate_logs = self.container_logs(candidate_name).strip()[-4000:]
            detail = f"; Candidate logs: {candidate_logs}" if candidate_logs else ""
            raise RuntimeFailure(f"isolated evaluation failed: {exc}{detail}") from exc
        finally:
            if evaluator_started:
                self.stop_container(evaluator_name)
            self.stop_container(candidate_name)
            self.remove_network(network_name)
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            loss = float(payload["nll"])
        except (
            FileNotFoundError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeFailure("evaluator did not produce a valid NLL result") from exc
        return loss
