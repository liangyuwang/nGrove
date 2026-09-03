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
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .config import AgentConfig, TaskConfig, TrainingResources
from .host_agent import HostAgentProcess, start_host_agent_process
from .results import validated_nll
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
    EvaluationTaskSpec,
    HiddenSplitRef,
    LeaseState,
    ResearchTaskSpec,
    TrustLevel,
    WorkerDescriptor,
    WorkerFailure,
    WorkerTeardownFailure,
)


# Compatibility names for callers of the v0 local reference runtime.  New core
# code depends on the platform-neutral names in worker_protocol instead.
RuntimeFailure = WorkerFailure
ExecResult = CommandResult
MAX_RUNTIME_STREAM_BYTES = 1024 * 1024
_TRUNCATION_MARKER = b"\n[output truncated by AutoTrain-Bench]\n"


def _decode_timeout_output(value: Any) -> str:
    """Normalize TimeoutExpired output to the Worker protocol's text type."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    raise TypeError(f"unexpected subprocess output type: {type(value).__name__}")


def _run_runtime_teardown_steps(
    steps: Sequence[Tuple[str, Callable[[], Any]]],
) -> None:
    """Attempt every local-runtime teardown step before reporting failures."""

    failures = []
    for resource, cleanup in steps:
        try:
            cleanup()
        except Exception as exc:
            failures.append(f"{resource}: {type(exc).__name__}: {exc}")
    if failures:
        raise WorkerTeardownFailure("runtime teardown failed: " + "; ".join(failures))


class _BoundedCapture:
    """Drain one subprocess pipe while retaining only a bounded prefix."""

    def __init__(self, maximum_bytes: int = MAX_RUNTIME_STREAM_BYTES) -> None:
        self.maximum_bytes = maximum_bytes
        self.buffer = bytearray()
        self.truncated = False

    def drain(self, stream: Any) -> None:
        try:
            while True:
                chunk = stream.read(64 * 1024)
                if not chunk:
                    return
                remaining = self.maximum_bytes - len(self.buffer)
                if remaining > 0:
                    self.buffer.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    self.truncated = True
        finally:
            stream.close()

    def text(self) -> str:
        payload = bytes(self.buffer)
        if self.truncated:
            payload = payload[: max(0, self.maximum_bytes - len(_TRUNCATION_MARKER))]
            payload += _TRUNCATION_MARKER
        return payload.decode("utf-8", errors="replace")


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
        self._pending_evaluations: Dict[tuple, Tuple[str, str, str]] = {}

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
                GPU_DEVICE_ISOLATION: Capability(enforced, "explicit device grant"),
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
                    enforced, "producer stopped before Coordinator capture"
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
            raise RuntimeFailure(
                f"runtime command timed out: {' '.join(command)}"
            ) from exc
        except OSError as exc:
            raise RuntimeFailure(
                f"failed to execute runtime command: {' '.join(command)}: {exc}"
            ) from exc
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
            if not resources.required_gpu_model or resources.required_gpu_model in name
        ]
        if len(matching) < resources.nvidia_gpu_count:
            raise RuntimeFailure(
                "episode requires GPU model containing "
                f"{resources.required_gpu_model!r}; "
                f"found {[name for name, _memory in inventory]}"
            )
        if (
            resources.minimum_gpu_memory_mib is not None
            and sum(
                memory_mib >= resources.minimum_gpu_memory_mib
                for _name, memory_mib in matching
            )
            < resources.nvidia_gpu_count
        ):
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
        agent_api_dir = public_task_path.parent / "agent_api"
        if not (agent_api_dir / "autotrain_bench/client.py").is_file():
            raise RuntimeFailure("published Agent API is missing")
        missing_environment = [
            name for name in agent.environment if name not in os.environ
        ]
        if missing_environment:
            names = ", ".join(missing_environment)
            raise RuntimeFailure(f"agent environment variable is not set: {names}")
        if agent.inference is not None and not inference_network:
            raise RuntimeFailure(
                "inference_network is required when inference is enabled"
            )
        agent_network = (
            inference_network
            if agent.inference is not None
            else ("none" if agent.egress == "denied" else "bridge")
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
            "--log-opt",
            "max-size=10m",
            "--log-opt",
            "max-file=1",
            "--mount",
            self._mount(agent.workspace, "/workspace", read_only=False),
            "--mount",
            self._mount(config.train_dir, "/benchmark/train", read_only=True),
            "--mount",
            self._mount(public_task_path, "/benchmark/task.json", read_only=True),
            "--mount",
            self._mount(control_dir / "inbox", "/control/inbox", read_only=False),
            "--mount",
            self._mount(control_dir / "outbox", "/control/outbox", read_only=False),
            "--mount",
            self._mount(agent_api_dir, "/opt/agent-api", read_only=True),
            "--env",
            "PYTHONPATH=/opt/agent-api",
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
        """Remove a network and fail unless its absence is confirmed."""

        if not name:
            return
        removal = self._run(["network", "rm", name], check=False, timeout=20)
        inventory = self._run(["network", "ls", "--format", "{{.Name}}"], timeout=20)
        if name not in {line.strip() for line in inventory.stdout.splitlines()}:
            return
        detail = removal.stderr.strip() or removal.stdout.strip()
        if not detail:
            detail = "network remains after removal"
        raise RuntimeFailure(f"failed to remove network {name}: {detail}")

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
            "--log-opt",
            "max-size=10m",
            "--log-opt",
            "max-file=1",
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
        task: ResearchTaskSpec,
        workspace: Path,
        resources: TrainingResources,
        lease_dir: Path,
    ) -> None:
        public_helper = (
            task.public_api_dir / "autotrain_bench" / "candidate_preflight.py"
        )
        if not public_helper.is_file():
            raise RuntimeFailure(
                f"published training API is unavailable: {public_helper}"
            )
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
            "--log-opt",
            "max-size=10m",
            "--log-opt",
            "max-file=1",
            "--mount",
            self._mount(workspace, "/workspace", read_only=False),
            "--mount",
            self._mount(task.train_dir, "/benchmark/train", read_only=True),
            "--mount",
            self._mount(scratch, "/scratch", read_only=False),
            "--mount",
            self._mount(output, "/output", read_only=False),
            "--mount",
            self._mount(
                task.public_api_dir,
                "/opt/training-api",
                read_only=True,
            ),
            "--env",
            "HOME=/scratch",
            "--env",
            "PYTHONPATH=/opt/training-api",
            "--env",
            "AUTOTRAIN_TRAIN_DIR=/benchmark/train",
            "--env",
            "AUTOTRAIN_OUTPUT_DIR=/output",
            "--workdir",
            "/workspace",
        ]
        resources.validate()
        if resources.nvidia_gpu_count:
            args.extend(["--gpus", str(resources.nvidia_gpu_count)])
        for device in resources.devices:
            args.extend(["--device", device])
        args.extend([task.training_image, "sleep", "infinity"])
        self._run(args)

    @staticmethod
    def _worker_lease_key(episode_id: str, lease_id: str) -> tuple:
        return (episode_id, lease_id)

    @staticmethod
    def _worker_lease_runtime_name(episode_id: str, lease_id: str) -> str:
        """Derive the local handle from the stable semantic Lease identity."""

        runtime_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{episode_id}:{lease_id}").hex[:12]
        return f"atb-lease-{runtime_id}"

    def start_lease(
        self,
        *,
        episode_id: str,
        lease_id: str,
        duration_seconds: float,
        task: ResearchTaskSpec,
        workspace: Path,
        resources: TrainingResources,
        lease_dir: Path,
    ) -> None:
        """Map the semantic lease operation to the local reference runtime."""

        del duration_seconds
        key = self._worker_lease_key(episode_id, lease_id)
        if key in self._worker_leases:
            raise RuntimeFailure(f"lease already exists: {lease_id}")
        name = self._worker_lease_runtime_name(episode_id, lease_id)
        self._worker_leases[key] = name
        try:
            self.start_training(
                name=name,
                task=task,
                workspace=workspace,
                resources=resources,
                lease_dir=lease_dir,
            )
        except Exception as start_exc:
            # ``docker run`` may create the named container before the client
            # observes a timeout or daemon error.  Register the deterministic
            # handle first, then retain it unless absence is confirmed so a
            # later deployment teardown can retry the semantic Lease stop.
            try:
                self.terminate_lease(episode_id=episode_id, lease_id=lease_id)
            except WorkerTeardownFailure as teardown_exc:
                raise WorkerTeardownFailure(
                    "lease startup failed and cleanup was not confirmed: "
                    f"{teardown_exc}"
                ) from start_exc
            raise

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
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise RuntimeFailure(f"failed to execute training command: {exc}") from exc
        stdout_capture = _BoundedCapture()
        stderr_capture = _BoundedCapture()
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_thread = threading.Thread(
            target=stdout_capture.drain, args=(process.stdout,), daemon=True
        )
        stderr_thread = threading.Thread(
            target=stderr_capture.drain, args=(process.stderr,), daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()
        timed_out = False
        try:
            process.wait(timeout=max(1, math.ceil(timeout_seconds)))
        except subprocess.TimeoutExpired:
            timed_out = True
            timeout_teardown_steps = [
                ("training container", lambda: self.stop_container(name)),
            ]
            if process.poll() is None:
                timeout_teardown_steps.append(("Docker exec client", process.kill))
            timeout_teardown_steps.append(
                ("Docker exec client wait", lambda: process.wait(timeout=5))
            )
            _run_runtime_teardown_steps(timeout_teardown_steps)
        finally:
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
        stdout = stdout_capture.text()
        stderr = stderr_capture.text()
        if timed_out and not stderr:
            stderr = "command exceeded the lease deadline"
        return ExecResult(
            exit_code=124 if timed_out else int(process.returncode),
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            stdout_truncated=stdout_capture.truncated,
            stderr_truncated=stderr_capture.truncated,
        )

    def _worker_lease_name(self, episode_id: str, lease_id: str) -> str:
        key = self._worker_lease_key(episode_id, lease_id)
        return self._worker_leases.get(
            key,
            self._worker_lease_runtime_name(episode_id, lease_id),
        )

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
        # A failed ``docker inspect`` is ambiguous: the container may be absent,
        # or the daemon may be unavailable.  A successful inventory query makes
        # absence/stopped state explicit and never converts infrastructure loss
        # into an Agent-exited classification.
        result = self._run(["container", "ls", "--format", "{{.Names}}"])
        return name in {line.strip() for line in result.stdout.splitlines()}

    def container_exit_code(self, name: str) -> Optional[int]:
        inventory = self._run(["container", "ls", "--all", "--format", "{{.Names}}"])
        if name not in {line.strip() for line in inventory.stdout.splitlines()}:
            return None
        result = self._run(
            ["inspect", "--format", "{{.State.Running}} {{.State.ExitCode}}", name],
            check=False,
        )
        if result.returncode != 0:
            # Tolerate a removal race, but fail closed if Docker still reports
            # the name and cannot provide authoritative state.
            retry = self._run(["container", "ls", "--all", "--format", "{{.Names}}"])
            if name not in {line.strip() for line in retry.stdout.splitlines()}:
                return None
            detail = result.stderr.strip() or result.stdout.strip() or "inspect failed"
            raise RuntimeFailure(f"failed to inspect container {name}: {detail}")
        fields = result.stdout.split()
        if len(fields) != 2 or fields[0] not in {"true", "false"}:
            raise RuntimeFailure(f"invalid state returned for container {name}")
        if fields[0] == "true":
            return None
        try:
            return int(fields[1])
        except ValueError as exc:
            raise RuntimeFailure(
                f"invalid exit code returned for container {name}"
            ) from exc

    def lease_state(self, *, episode_id: str, lease_id: str) -> LeaseState:
        name = self._worker_lease_name(episode_id, lease_id)
        if self.container_running(name):
            return LeaseState.RUNNING
        return LeaseState.STOPPED

    def container_logs(
        self, name: str, *, timeout_seconds: Optional[float] = None
    ) -> str:
        result = self._run(
            ["logs", "--tail", "10000", name], check=False, timeout=timeout_seconds
        )
        parts = [value for value in (result.stdout, result.stderr) if value]
        return "".join(parts)

    def stop_container(self, name: str) -> None:
        """Force-remove a container and fail unless its absence is confirmed."""

        if not name:
            return
        removal = self._run(["rm", "--force", name], check=False, timeout=20)
        inventory = self._run(
            ["container", "ls", "--all", "--format", "{{.Names}}"], timeout=20
        )
        if name not in {line.strip() for line in inventory.stdout.splitlines()}:
            return
        detail = removal.stderr.strip() or removal.stdout.strip()
        if not detail:
            detail = "container remains after force removal"
        raise RuntimeFailure(f"failed to terminate container {name}: {detail}")

    def terminate_lease(self, *, episode_id: str, lease_id: str) -> None:
        key = self._worker_lease_key(episode_id, lease_id)
        name = self._worker_lease_name(episode_id, lease_id)
        try:
            self.stop_container(name)
        except WorkerFailure as exc:
            raise WorkerTeardownFailure(str(exc)) from exc
        self._worker_leases.pop(key, None)

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
        data_path = config.val_path if split == "val" else config.test_path
        return self._run_isolated_evaluator(
            episode_id="local-preview",
            task=EvaluationTaskSpec.from_task(config),
            resources=resources,
            candidate_dir=candidate_dir,
            hidden_split=HiddenSplitRef(split, data_path),
            output_dir=output_dir,
            timeout_seconds=timeout_seconds,
        )

    def _run_isolated_evaluator(
        self,
        *,
        episode_id: str,
        task: EvaluationTaskSpec,
        resources: TrainingResources,
        candidate_dir: Path,
        hidden_split: HiddenSplitRef,
        output_dir: Path,
        timeout_seconds: Optional[float] = None,
    ) -> float:
        resources.validate()
        split = hidden_split.name
        if task.backend != "isolated_candidate":
            raise RuntimeFailure(f"unsupported evaluator backend: {task.backend}")
        data_path = hidden_split.path
        if not data_path.is_file():
            raise RuntimeFailure(f"hidden evaluator split is not a file: {data_path}")
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / f"{split}.json"
        try:
            result_path.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeFailure("failed to clear the prior evaluator result") from exc
        runtime_id = uuid.uuid4().hex[:12]
        candidate_name = f"atb-candidate-{runtime_id}"
        evaluator_name = f"atb-evaluator-{runtime_id}"
        network_name = f"atb-evaluation-{runtime_id}"
        evaluation_key = (episode_id, runtime_id)
        candidate_api_dir = output_dir / ".candidate_api"
        package = candidate_api_dir / "autotrain_bench"
        if not candidate_api_dir.exists():
            package.mkdir(parents=True)
            core_package = Path(__file__).resolve().parent
            for filename in ("candidate_protocol.py", "candidate_server.py"):
                shutil.copy2(core_package / filename, package / filename)
                (package / filename).chmod(0o444)
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "__init__.py").chmod(0o444)
            package.chmod(0o555)
            candidate_api_dir.chmod(0o555)
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
            self._mount(candidate_api_dir, "/opt/candidate-api", read_only=True),
            "--env",
            "HOME=/candidate-home",
            "--env",
            "PYTHONPATH=/opt/candidate-api",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--workdir",
            "/candidate",
        ]
        if task.device == "cuda":
            candidate_args.extend(["--gpus", str(resources.nvidia_gpu_count)])
        candidate_args.extend(
            [
                task.candidate_image,
                "python",
                "-m",
                "autotrain_bench.candidate_server",
                "--candidate",
                "/candidate",
                "--device",
                task.device,
                "--vocab-size",
                str(task.vocab_size),
                "--context-length",
                str(task.context_length),
                "--maximum-batch-size",
                str(task.batch_size),
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
            "--log-opt",
            "max-size=10m",
            "--log-opt",
            "max-file=1",
            "--mount",
            self._mount(data_path, "/eval-data/split.json", read_only=True),
            "--mount",
            self._mount(
                task.evaluator_entrypoint,
                "/opt/evaluator.py",
                read_only=True,
            ),
            "--mount",
            self._mount(output_dir, "/output", read_only=False),
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            task.evaluator_image,
            "python",
            "/opt/evaluator.py",
            "--candidate-url",
            "http://candidate:8766",
            "--data",
            "/eval-data/split.json",
            "--output",
            f"/output/{split}.json",
            "--vocab-size",
            str(task.vocab_size),
            "--batch-size",
            str(task.batch_size),
        ]
        effective_timeout = float(task.timeout_seconds)
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

        self._pending_evaluations[evaluation_key] = (
            candidate_name,
            evaluator_name,
            network_name,
        )
        try:
            self.create_network(
                network_name,
                internal=True,
                timeout_seconds=remaining_seconds(),
            )
            self._run(candidate_args, timeout=remaining_seconds())
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
            self._teardown_evaluation(evaluation_key)
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            loss = validated_nll(payload["nll"])
        except (
            FileNotFoundError,
            KeyError,
            OverflowError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeFailure(
                "evaluator did not produce a valid NLL result"
            ) from exc
        return loss

    def evaluate_candidate(
        self,
        *,
        episode_id: str,
        task: EvaluationTaskSpec,
        resources: TrainingResources,
        candidate_dir: Path,
        hidden_split: HiddenSplitRef,
        output_dir: Path,
        timeout_seconds: Optional[float] = None,
    ) -> float:
        """Map the semantic evaluation operation to the reference runtime."""

        return self._run_isolated_evaluator(
            episode_id=episode_id,
            task=task,
            resources=resources,
            candidate_dir=candidate_dir,
            hidden_split=hidden_split,
            output_dir=output_dir,
            timeout_seconds=timeout_seconds,
        )

    def _teardown_evaluation(self, evaluation_key: tuple) -> None:
        resources = self._pending_evaluations.get(evaluation_key)
        if resources is None:
            return
        candidate_name, evaluator_name, network_name = resources
        _run_runtime_teardown_steps(
            [
                ("candidate", lambda: self.stop_container(candidate_name)),
                ("evaluator", lambda: self.stop_container(evaluator_name)),
                ("evaluation network", lambda: self.remove_network(network_name)),
            ]
        )
        self._pending_evaluations.pop(evaluation_key, None)

    def teardown_pending_evaluations(self) -> None:
        """Retry every local evaluation cleanup whose result was uncertain."""

        failures = []
        for evaluation_key in list(self._pending_evaluations):
            try:
                self._teardown_evaluation(evaluation_key)
            except Exception as exc:
                failures.append(f"{evaluation_key[1]}: {type(exc).__name__}: {exc}")
        if failures:
            raise WorkerTeardownFailure(
                "pending evaluation teardown failed: " + "; ".join(failures)
            )
