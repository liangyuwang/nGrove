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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .config import AgentConfig, TaskConfig, TrainingResources
from .host_agent import HostAgentProcess, start_host_agent_process
from .worker_protocol import (
    CANDIDATE_EVALUATOR_CHANNEL_ISOLATION,
    DEADLINE_TERMINATION,
    GPU_DEVICE_ISOLATION,
    HIDDEN_DATASET_ISOLATION,
    IMMUTABLE_CANDIDATE_ARTIFACT,
    IMMUTABLE_WORKLOAD_IDENTITY,
    TRAINING_DATA_READ_ONLY,
    TRAINING_NETWORK_ISOLATION,
    Capability,
    CommandResult,
    EnforcementLevel,
    LeaseState,
    TrustLevel,
    WorkerDescriptor,
    WorkerFailure,
)


# Compatibility names for callers of the v0 local reference runtime.  New core
# code depends on the platform-neutral names in worker_protocol instead.
RuntimeFailure = WorkerFailure
ExecResult = CommandResult


def _decode_timeout_output(value: Any) -> str:
    """Normalize TimeoutExpired output to the Worker protocol's text type."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    raise TypeError(f"unexpected subprocess output type: {type(value).__name__}")


class DockerRuntime:
    """Minimal Docker implementation for the local reference prototype."""

    def __init__(
        self,
        docker_binary: Optional[str] = None,
        *,
        monotonic: Optional[Callable[[], float]] = None,
    ) -> None:
        self.docker = docker_binary or shutil.which("docker") or ""
        self.monotonic = monotonic or time.monotonic
        self._worker_leases: Dict[tuple, str] = {}

    def describe_capabilities(self) -> WorkerDescriptor:
        """Describe the bundled development Worker without certifying its host."""

        enforced = EnforcementLevel.ENFORCED
        return WorkerDescriptor(
            worker_id="local-reference-docker",
            trust_level=TrustLevel.DEVELOPMENT,
            roles=("research", "evaluation", "agent"),
            capabilities={
                IMMUTABLE_WORKLOAD_IDENTITY: Capability(
                    enforced, "immutable image reference"
                ),
                GPU_DEVICE_ISOLATION: Capability(
                    enforced, "explicit device grant"
                ),
                TRAINING_DATA_READ_ONLY: Capability(
                    enforced, "read-only task data mount"
                ),
                TRAINING_NETWORK_ISOLATION: Capability(
                    enforced, "training network disabled"
                ),
                HIDDEN_DATASET_ISOLATION: Capability(
                    enforced, "separate candidate and evaluator workloads"
                ),
                CANDIDATE_EVALUATOR_CHANNEL_ISOLATION: Capability(
                    enforced, "episode-private evaluation channel"
                ),
                DEADLINE_TERMINATION: Capability(
                    EnforcementLevel.BEST_EFFORT,
                    "Coordinator-process deadline guard",
                ),
                IMMUTABLE_CANDIDATE_ARTIFACT: Capability(
                    enforced, "content-addressed candidate snapshot"
                ),
            },
        )

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

    def prepare(
        self,
        config: TaskConfig,
        *,
        agent: Optional[AgentConfig] = None,
        resources: Optional[TrainingResources] = None,
        pull_missing: bool = True,
    ) -> None:
        self._run(["info"], timeout=20)
        images = set(config.images)
        if agent is not None:
            agent.validate()
            if agent.image:
                images.add(agent.image)
            if agent.inference is not None:
                images.add(config.inference_gateway_image)
        for image in sorted(images):
            inspected = self._run(["image", "inspect", image], check=False)
            if inspected.returncode == 0:
                continue
            if not pull_missing:
                raise RuntimeFailure(f"required image is missing: {image}")
            self._run(["pull", image])
        if resources is not None:
            resources.validate_for(config)
            self.validate_hardware(resources)

    def validate_hardware(self, resources: TrainingResources) -> None:
        resources.validate()
        if resources.nvidia_gpu_count == 0:
            return
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            raise RuntimeFailure(
                "episode requests NVIDIA GPUs, but nvidia-smi is unavailable"
            )
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeFailure(result.stderr.strip() or "nvidia-smi failed")
        inventory = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            fields = [field.strip() for field in line.rsplit(",", 1)]
            if len(fields) != 2:
                raise RuntimeFailure("nvidia-smi returned malformed GPU metadata")
            try:
                memory_mib = int(fields[1])
            except ValueError as exc:
                raise RuntimeFailure(
                    "nvidia-smi returned invalid GPU memory metadata"
                ) from exc
            inventory.append((fields[0], memory_mib))
        if len(inventory) < resources.nvidia_gpu_count:
            raise RuntimeFailure(
                "episode requests "
                f"{resources.nvidia_gpu_count} NVIDIA GPU(s); found {len(inventory)}"
            )
        matching = [
            (name, memory_mib)
            for name, memory_mib in inventory
            if not resources.required_gpu_model
            or resources.required_gpu_model in name
        ]
        if len(matching) < resources.nvidia_gpu_count:
            raise RuntimeFailure(
                "episode requires GPU model containing "
                f"{resources.required_gpu_model!r}; "
                f"found {[name for name, _memory in inventory]}"
            )
        if resources.minimum_gpu_memory_mib is not None and sum(
            memory_mib >= resources.minimum_gpu_memory_mib
            for _name, memory_mib in matching
        ) < resources.nvidia_gpu_count:
            raise RuntimeFailure(
                "episode requires at least "
                f"{resources.minimum_gpu_memory_mib} MiB on each selected GPU; "
                f"found {matching}"
            )

    def environment_manifest(
        self,
        config: TaskConfig,
        *,
        agent: Optional[AgentConfig] = None,
        resources: Optional[TrainingResources] = None,
    ) -> Dict[str, Any]:
        """Record the concrete runtime used for a comparable episode."""

        docker_version = self._run(
            ["version", "--format", "{{.Server.Version}}"], timeout=20
        ).stdout.strip()
        docker_resources = {
            "cpu_count": self._run(
                ["info", "--format", "{{.NCPU}}"], timeout=20
            ).stdout.strip(),
            "memory_bytes": self._run(
                ["info", "--format", "{{.MemTotal}}"], timeout=20
            ).stdout.strip(),
            "architecture": self._run(
                ["info", "--format", "{{.Architecture}}"], timeout=20
            ).stdout.strip(),
            "operating_system": self._run(
                ["info", "--format", "{{.OperatingSystem}}"], timeout=20
            ).stdout.strip(),
        }
        images: Dict[str, Dict[str, str]] = {}
        for role, reference in (
            ("agent", agent.image if agent is not None else None),
            ("training", config.training_image),
            ("candidate", config.candidate_image),
            ("evaluator", config.evaluator_image),
            (
                "inference_gateway",
                config.inference_gateway_image
                if agent is not None and agent.inference is not None
                else None,
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

        resolved_resources = resources or TrainingResources()
        if resources is None:
            resolved_resources.validate()
        else:
            resolved_resources.validate_for(config)
        gpus: List[Dict[str, str]] = []
        if resolved_resources.nvidia_gpu_count:
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
        manifest = {
            "docker_server_version": docker_version,
            "docker_resources": docker_resources,
            "images": images,
            "gpus": gpus,
        }
        if resources is not None:
            manifest["training_resources"] = resolved_resources.public_manifest()
        return manifest

    def start_agent(
        self,
        *,
        name: str,
        config: TaskConfig,
        agent: AgentConfig,
        control_dir: Path,
        public_task_path: Path,
        inference_network: Optional[str] = None,
    ) -> None:
        config.validate()
        agent.validate()
        if agent.runner != "container":
            raise RuntimeFailure("start_agent requires the container runner")
        if not agent.image:  # guarded by AgentConfig.validate
            raise RuntimeFailure("container runner has no Agent image")
        package_dir = Path(__file__).resolve().parent
        missing_environment = [
            name for name in agent.environment if name not in os.environ
        ]
        if missing_environment:
            names = ", ".join(missing_environment)
            raise RuntimeFailure(f"agent environment variable is not set: {names}")
        if agent.inference is not None and not inference_network:
            raise RuntimeFailure("inference_network is required when inference is enabled")
        agent_network = (
            inference_network
            if agent.inference is not None
            else ("none" if agent.mode == "sandboxed" else "bridge")
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
            self._mount(agent.workspace, "/workspace", read_only=False),
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
        for name in agent.environment:
            # Passing only the name makes Docker copy the value from the trusted
            # Supervisor environment without putting the value in argv or traces.
            args.extend(["--env", name])

        if agent.inference is not None:
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
        args.extend([agent.image, *agent.command])
        self._run(args)

    def start_host_agent(
        self,
        *,
        agent: AgentConfig,
        control_dir: Path,
        public_task_path: Path,
        stdout_path: Path,
        stderr_path: Path,
    ) -> HostAgentProcess:
        agent.validate()
        if agent.runner != "host_cli":
            raise RuntimeFailure("start_host_agent requires the host_cli runner")
        package_root = Path(__file__).resolve().parent.parent
        try:
            return start_host_agent_process(
                command=agent.command,
                workspace=agent.workspace,
                control_dir=control_dir,
                public_task_path=public_task_path,
                package_root=package_root,
                forwarded_environment=agent.environment,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
        except OSError as exc:
            raise RuntimeFailure(f"failed to start host Agent: {exc}") from exc

    def create_network(
        self,
        name: str,
        *,
        internal: bool,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        args = ["network", "create", "--driver", "bridge"]
        if internal:
            args.append("--internal")
        args.append(name)
        self._run(args, timeout=timeout_seconds)

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
        agent: AgentConfig,
        audit_dir: Path,
        readiness_timeout_seconds: float = 10,
    ) -> None:
        config.validate()
        agent.validate()
        inference = agent.inference
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
        workspace: Path,
        resources: TrainingResources,
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
            self._mount(workspace, "/workspace", read_only=False),
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
        resources.validate_for(config)
        if resources.nvidia_gpu_count:
            args.extend(["--gpus", str(resources.nvidia_gpu_count)])
        for device in resources.devices:
            args.extend(["--device", device])
        args.extend([config.training_image, "sleep", "infinity"])
        self._run(args)

    @staticmethod
    def _worker_lease_key(episode_id: str, lease_id: str) -> tuple:
        return (episode_id, lease_id)

    def start_lease(
        self,
        *,
        episode_id: str,
        lease_id: str,
        duration_seconds: float,
        config: TaskConfig,
        workspace: Path,
        resources: TrainingResources,
        lease_dir: Path,
    ) -> None:
        """Map the semantic lease operation to the local reference runtime."""

        del duration_seconds
        key = self._worker_lease_key(episode_id, lease_id)
        if key in self._worker_leases:
            raise RuntimeFailure(f"lease already exists: {lease_id}")
        runtime_id = uuid.uuid5(
            uuid.NAMESPACE_URL, f"{episode_id}:{lease_id}"
        ).hex[:12]
        name = f"atb-lease-{runtime_id}"
        self.start_training(
            name=name,
            config=config,
            workspace=workspace,
            resources=resources,
            lease_dir=lease_dir,
        )
        self._worker_leases[key] = name

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
            stdout = _decode_timeout_output(exc.stdout)
            stderr = _decode_timeout_output(exc.stderr)
            return ExecResult(
                exit_code=124,
                stdout=stdout,
                stderr=stderr or "command exceeded the lease deadline",
                timed_out=True,
            )
        return ExecResult(result.returncode, result.stdout, result.stderr)

    def _worker_lease_name(self, episode_id: str, lease_id: str) -> str:
        try:
            return self._worker_leases[
                self._worker_lease_key(episode_id, lease_id)
            ]
        except KeyError as exc:
            raise RuntimeFailure(f"unknown lease: {lease_id}") from exc

    def exec_lease(
        self,
        *,
        episode_id: str,
        lease_id: str,
        argv: Sequence[str],
        timeout_seconds: float,
    ) -> CommandResult:
        return self.exec_training(
            name=self._worker_lease_name(episode_id, lease_id),
            argv=argv,
            timeout_seconds=timeout_seconds,
        )

    def container_running(self, name: str) -> bool:
        result = self._run(
            ["inspect", "--format", "{{.State.Running}}", name], check=False
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def lease_state(self, *, episode_id: str, lease_id: str) -> LeaseState:
        name = self._worker_leases.get(
            self._worker_lease_key(episode_id, lease_id)
        )
        if name is None:
            return LeaseState.STOPPED
        if self.container_running(name):
            return LeaseState.RUNNING
        return LeaseState.STOPPED

    def container_logs(
        self, name: str, *, timeout_seconds: Optional[float] = None
    ) -> str:
        result = self._run(
            ["logs", name], check=False, timeout=timeout_seconds
        )
        parts = [value for value in (result.stdout, result.stderr) if value]
        return "".join(parts)

    def stop_container(self, name: str) -> None:
        if not name:
            return
        self._run(["rm", "--force", name], check=False)

    def terminate_lease(self, *, episode_id: str, lease_id: str) -> None:
        key = self._worker_lease_key(episode_id, lease_id)
        name = self._worker_leases.pop(key, None)
        if name is not None:
            self.stop_container(name)

    def run_evaluator(
        self,
        *,
        config: TaskConfig,
        resources: TrainingResources,
        candidate_dir: Path,
        split: str,
        output_dir: Path,
        timeout_seconds: Optional[float] = None,
    ) -> float:
        resources.validate_for(config)
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
            candidate_args.extend(["--gpus", str(resources.nvidia_gpu_count)])
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
        effective_timeout = float(config.evaluator_timeout_seconds)
        if timeout_seconds is not None:
            if timeout_seconds <= 0:
                raise RuntimeFailure("evaluation deadline reached")
            effective_timeout = min(effective_timeout, float(timeout_seconds))
        evaluation_deadline = self.monotonic() + effective_timeout

        def remaining_seconds() -> float:
            remaining = evaluation_deadline - self.monotonic()
            if remaining <= 0:
                raise RuntimeFailure("evaluation deadline reached")
            return remaining

        evaluator_started = False
        try:
            self.create_network(
                network_name,
                internal=True,
                timeout_seconds=remaining_seconds(),
            )
            self._run(candidate_args, timeout=remaining_seconds())
            evaluator_started = True
            self._run(
                evaluator_args,
                timeout=remaining_seconds(),
            )
        except RuntimeFailure as exc:
            # Candidate output may contain hidden evaluator inputs.  Preserve it
            # only in the trusted episode directory and never include it in the
            # exception that crosses the Agent-facing Supervisor API.
            try:
                candidate_logs = self.container_logs(
                    candidate_name,
                    timeout_seconds=remaining_seconds(),
                ).strip()[-4000:]
            except RuntimeFailure:
                candidate_logs = "<candidate logs unavailable>"
            diagnostic_path = output_dir / f"{split}.failure.log"
            try:
                diagnostic_path.write_text(
                    f"Runtime failure: {exc}\nCandidate logs:\n{candidate_logs}\n",
                    encoding="utf-8",
                )
                diagnostic_path.chmod(0o600)
            except OSError:
                # Diagnostic persistence must never change the fixed public
                # failure returned across the untrusted boundary.
                pass
            raise RuntimeFailure("isolated evaluation failed") from None
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

    def evaluate_candidate(
        self,
        *,
        episode_id: str,
        config: TaskConfig,
        resources: TrainingResources,
        candidate_dir: Path,
        split: str,
        output_dir: Path,
        timeout_seconds: Optional[float] = None,
    ) -> float:
        """Map the semantic evaluation operation to the reference runtime."""

        del episode_id
        return self.run_evaluator(
            config=config,
            resources=resources,
            candidate_dir=candidate_dir,
            split=split,
            output_dir=output_dir,
            timeout_seconds=timeout_seconds,
        )
