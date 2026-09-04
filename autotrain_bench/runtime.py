"""Container runtime used by the host-side reference supervisor.

AutoTrain-Bench deliberately does not allocate machines or integrate with a
cluster scheduler. The runtime only consumes devices already visible on the
single host where the supervisor is started.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .artifact_identity import HIDDEN_BUNDLE_FILENAMES, hidden_bundle_sha256
from .config import AgentConfig, TaskConfig, TrainingResources
from .host_agent import HostAgentProcess, start_host_agent_process
from .worker_protocol import (
    CANDIDATE_EVALUATOR_CHANNEL_ISOLATION,
    CANDIDATE_PREDICTION_ARTIFACT_LIFECYCLE,
    CANDIDATE_PREDICTION_STORAGE_QUOTA,
    DEADLINE_TERMINATION,
    GPU_DEVICE_ISOLATION,
    HIDDEN_DATASET_IDENTITY_VERIFICATION,
    HIDDEN_DATASET_ISOLATION,
    IMMUTABLE_CANDIDATE_ARTIFACT,
    IMMUTABLE_WORKLOAD_IDENTITY,
    TRAINING_DATA_READ_ONLY,
    TRAINING_NETWORK_ISOLATION,
    Capability,
    CandidateEvaluationResult,
    CommandResult,
    EnforcementLevel,
    EvaluationTaskSpec,
    HiddenDatasetIdentityMismatch,
    HiddenDatasetRequest,
    HiddenDatasetResolutionFailure,
    LeaseState,
    ResourceAllocationReceipt,
    ResourceEnvelope,
    ResourceObservation,
    ResourcePropertyEvidence,
    ResolvedHiddenDataset,
    ResearchTaskSpec,
    TrustLevel,
    ResolvedWorkload,
    WorkloadRequest,
    WorkloadResolutionFailure,
    WorkerDescriptor,
    WorkerFailure,
    WorkerTeardownFailure,
    v0_resource_envelope,
)


# Compatibility names for callers of the v0 local reference runtime.  New core
# code depends on the platform-neutral names in worker_protocol instead.
RuntimeFailure = WorkerFailure
ExecResult = CommandResult
MAX_RUNTIME_STREAM_BYTES = 1024 * 1024
_TRUNCATION_MARKER = b"\n[output truncated by AutoTrain-Bench]\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        hidden_datasets: Optional[Mapping[str, Path]] = None,
    ) -> None:
        self.docker = docker_binary or shutil.which("docker") or ""
        self.monotonic = monotonic or time.monotonic
        self._hidden_datasets: Dict[str, Path] = {}
        for name, path in (hidden_datasets or {}).items():
            # Validate the semantic key now, but deliberately do not require
            # the private source to exist before post-research evaluation.
            HiddenDatasetRequest(name)
            self._hidden_datasets[name] = Path(path).expanduser().resolve()
        self._resolved_hidden_datasets: Dict[
            tuple[str, str], tuple[str, ResolvedHiddenDataset, Path]
        ] = {}
        self._hidden_snapshot_root: Optional[Path] = None
        self._resolved_workloads: Dict[
            tuple[str, str], tuple[ResolvedWorkload, str]
        ] = {}
        self._worker_leases: Dict[tuple, str] = {}
        self._pending_evaluations: Dict[tuple, Tuple[str, str, Path]] = {}

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
                HIDDEN_DATASET_IDENTITY_VERIFICATION: Capability(
                    enforced,
                    "hidden-bundle-v1 SHA-256 verified before Candidate startup",
                ),
                CANDIDATE_EVALUATOR_CHANNEL_ISOLATION: Capability(
                    enforced, "target-free read-only input and isolated output mounts"
                ),
                CANDIDATE_PREDICTION_STORAGE_QUOTA: Capability(
                    EnforcementLevel.UNSUPPORTED,
                    "host bind mount has no hard byte or inode quota",
                ),
                CANDIDATE_PREDICTION_ARTIFACT_LIFECYCLE: Capability(
                    enforced,
                    "ephemeral removal after scoring and confirmed workload teardown",
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

    def resolve_workload(
        self,
        *,
        episode_id: str,
        workload: WorkloadRequest,
        timeout_seconds: float,
    ) -> ResolvedWorkload:
        """Bind an OCI reference to an opaque, episode-scoped authorization."""

        if not isinstance(workload, WorkloadRequest):
            raise TypeError("workload must be a WorkloadRequest")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise WorkloadResolutionFailure("workload resolution deadline reached")
        deadline = self.monotonic() + timeout_seconds

        def remaining_seconds() -> float:
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                raise WorkloadResolutionFailure("workload resolution deadline reached")
            return remaining

        marker = "@sha256:"
        declared_sha256 = (
            workload.requested_identity.rsplit(marker, 1)[1]
            if marker in workload.requested_identity
            else None
        )
        if (
            declared_sha256 is not None
            and len(declared_sha256) == 64
            and all(character in "0123456789abcdef" for character in declared_sha256)
        ):
            effective_sha256 = declared_sha256
            effective_format = "oci"
            execution_reference = workload.requested_identity
        else:
            try:
                inspected = self._run(
                    [
                        "image",
                        "inspect",
                        workload.requested_identity,
                        "--format",
                        "{{.Id}}",
                    ],
                    timeout=min(20.0, remaining_seconds()),
                ).stdout.strip()
            except WorkerTeardownFailure:
                raise
            except WorkerFailure as exc:
                raise WorkloadResolutionFailure(str(exc)) from exc
            remaining_seconds()
            if not inspected.startswith("sha256:"):
                raise WorkloadResolutionFailure(
                    "runtime did not return an immutable workload identity"
                )
            effective_sha256 = inspected.removeprefix("sha256:")
            if len(effective_sha256) != 64 or any(
                character not in "0123456789abcdef" for character in effective_sha256
            ):
                raise WorkloadResolutionFailure(
                    "runtime returned an invalid workload digest"
                )
            # Do not launch through the mutable tag after resolving it. Docker
            # accepts its immutable image ID as an execution reference.
            effective_format = "oci-image-config"
            execution_reference = inspected
        resolution_id = (
            "workload:"
            + uuid.uuid5(
                uuid.NAMESPACE_URL,
                ":".join(
                    (
                        episode_id,
                        workload.role,
                        workload.requested_identity,
                        effective_format,
                        effective_sha256,
                    )
                ),
            ).hex
        )
        resolved = ResolvedWorkload(
            role=workload.role,
            requested_identity=workload.requested_identity,
            effective_format=effective_format,
            effective_sha256=effective_sha256,
            transformation_sha256=None,
            resolution_id=resolution_id,
        )
        self._resolved_workloads[(episode_id, resolution_id)] = (
            resolved,
            execution_reference,
        )
        return resolved

    def _workload_reference(
        self,
        *,
        episode_id: str,
        workload: ResolvedWorkload,
        role: str,
        requested_identity: str,
    ) -> str:
        if not isinstance(workload, ResolvedWorkload):
            raise TypeError("workload resolution must be a ResolvedWorkload")
        private = self._resolved_workloads.get((episode_id, workload.resolution_id))
        if private is None:
            raise WorkloadResolutionFailure(
                "workload resolution is unavailable for this episode"
            )
        expected, reference = private
        if (
            workload != expected
            or workload.role != role
            or workload.requested_identity != requested_identity
        ):
            raise WorkloadResolutionFailure(
                "workload resolution evidence does not match the requested role"
            )
        return reference

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
        # `README.md` promises that the discovered runtime inventory is recorded
        # alongside the requested resources, so inspect the host whenever that is
        # possible rather than only when accelerators were requested.  Recording
        # an empty list for an uninspected host would claim the machine had no
        # accelerator, which is how a profile declaring `accelerator: null` can
        # occupy one without leaving a trace in the result.
        requested_gpus = resolved_resources.nvidia_gpu_count
        nvidia_smi = shutil.which("nvidia-smi")
        if requested_gpus and not nvidia_smi:  # guarded by validate_hardware
            raise RuntimeFailure("nvidia-smi disappeared after hardware validation")
        gpus: List[Dict[str, str]] = []
        gpu_inventory = "unavailable"
        if nvidia_smi:
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
                # A failure still matters when accelerators were requested,
                # because validate_hardware already depended on this tool.
                if requested_gpus:
                    raise RuntimeFailure(result.stderr.strip() or "nvidia-smi failed")
            else:
                gpu_inventory = "inspected"
                for line in result.stdout.splitlines():
                    fields = [field.strip() for field in line.split(",")]
                    if len(fields) != 4:
                        raise RuntimeFailure(
                            "nvidia-smi returned malformed GPU metadata"
                        )
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
            # `gpus` is authoritative only when `gpu_inventory` is "inspected".
            # An empty list under "unavailable" means unknown, not none.
            "gpus": gpus,
            "gpu_inventory": gpu_inventory,
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
        workload_image: Optional[str] = None,
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
        args.extend([workload_image or task.training_image, "sleep", "infinity"])
        self._run(args)

    @staticmethod
    def _worker_lease_key(episode_id: str, lease_id: str) -> tuple:
        return (episode_id, lease_id)

    @staticmethod
    def _worker_lease_runtime_name(episode_id: str, lease_id: str) -> str:
        """Derive the local handle from the stable semantic Lease identity."""

        runtime_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{episode_id}:{lease_id}").hex[:12]
        return f"atb-lease-{runtime_id}"

    def _resource_allocation_receipt(
        self,
        *,
        episode_id: str,
        lease_id: str,
        name: str,
        resources: TrainingResources,
        resource_envelope: ResourceEnvelope,
    ) -> ResourceAllocationReceipt:
        """Observe the local workload without making Docker the Core schema."""

        gpu_names: Tuple[str, ...] = ()
        gpu_memory: Tuple[int, ...] = ()
        if resources.nvidia_gpu_count:
            result = self._run(
                [
                    "exec",
                    name,
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                timeout=20,
            )
            names = []
            memory = []
            for line in result.stdout.splitlines():
                if not line.strip():
                    continue
                fields = [field.strip() for field in line.rsplit(",", 1)]
                if len(fields) != 2:
                    raise RuntimeFailure(
                        "training workload returned malformed accelerator evidence"
                    )
                try:
                    memory_mib = int(fields[1])
                except ValueError as exc:
                    raise RuntimeFailure(
                        "training workload returned invalid accelerator memory evidence"
                    ) from exc
                names.append(fields[0])
                memory.append(memory_mib)
            gpu_names = tuple(names)
            gpu_memory = tuple(memory)

        actual = {
            "host.count": 1,
            "accelerator.count": len(gpu_names),
            "accelerator.vendor": "nvidia" if gpu_names else None,
            "accelerator.model": gpu_names,
            "accelerator.memory_mib_per_device": gpu_memory,
        }
        properties = []
        for requirement in resource_envelope.requirements:
            value = actual.get(requirement.name)
            if requirement.name not in actual:
                properties.append(
                    ResourcePropertyEvidence(
                        name=requirement.name,
                        allocator_allocated=ResourceObservation("unavailable"),
                        workload_visible=ResourceObservation("unavailable"),
                        enforcement_level=EnforcementLevel.UNSUPPORTED,
                        enforcement_authority=None,
                    )
                )
                continue
            gpu_property = requirement.name.startswith("accelerator.")
            # The v0 count-based device request observes what became visible,
            # but cannot prove that a preselected physical device set was
            # enforced. Keep that limitation in Worker evidence.
            level = (
                EnforcementLevel.BEST_EFFORT
                if gpu_property and resources.nvidia_gpu_count
                else EnforcementLevel.ENFORCED
            )
            properties.append(
                ResourcePropertyEvidence(
                    name=requirement.name,
                    allocator_allocated=ResourceObservation("not_applicable"),
                    workload_visible=ResourceObservation(
                        "observed",
                        value,
                        "local-reference-worker",
                    ),
                    enforcement_level=level,
                    enforcement_authority="local-reference-runtime",
                )
            )
        receipt_id = (
            "allocation:"
            + uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{episode_id}:{lease_id}:{resource_envelope.sha256}",
            ).hex
        )
        return ResourceAllocationReceipt(
            episode_id=episode_id,
            lease_id=lease_id,
            worker_id=self.describe_capabilities().worker_id,
            resource_request_sha256=resource_envelope.sha256,
            properties=tuple(properties),
            receipt_id=receipt_id,
        )

    def start_lease(
        self,
        *,
        episode_id: str,
        lease_id: str,
        duration_seconds: float,
        task: ResearchTaskSpec,
        workspace: Path,
        resources: TrainingResources,
        resource_envelope: Optional[ResourceEnvelope] = None,
        lease_dir: Path,
        workload: ResolvedWorkload,
    ) -> ResourceAllocationReceipt:
        """Map the semantic lease operation to the local reference runtime."""

        del duration_seconds
        key = self._worker_lease_key(episode_id, lease_id)
        if key in self._worker_leases:
            raise RuntimeFailure(f"lease already exists: {lease_id}")
        name = self._worker_lease_runtime_name(episode_id, lease_id)
        resolved_envelope = resource_envelope or v0_resource_envelope(
            resources,
            minimum_enforcement=EnforcementLevel.BEST_EFFORT,
        )
        workload_image = self._workload_reference(
            episode_id=episode_id,
            workload=workload,
            role="training",
            requested_identity=task.training_image,
        )
        self._worker_leases[key] = name
        try:
            self.start_training(
                name=name,
                task=task,
                workspace=workspace,
                resources=resources,
                lease_dir=lease_dir,
                workload_image=workload_image,
            )
            return self._resource_allocation_receipt(
                episode_id=episode_id,
                lease_id=lease_id,
                name=name,
                resources=resources,
                resource_envelope=resolved_envelope,
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
    ) -> CandidateEvaluationResult:
        resources.validate_for(config)
        if split not in {"analysis", "test"}:
            raise RuntimeFailure(f"invalid evaluator split: {split}")
        data_path = config.analysis_path if split == "analysis" else config.test_path
        task = EvaluationTaskSpec.from_task(config)
        request = HiddenDatasetRequest(split)
        operation_timeout = float(
            task.timeout_seconds if timeout_seconds is None else timeout_seconds
        )
        if not math.isfinite(operation_timeout) or operation_timeout <= 0:
            raise RuntimeFailure("evaluation deadline reached")
        operation_deadline = self.monotonic() + operation_timeout

        def remaining_seconds() -> float:
            remaining = operation_deadline - self.monotonic()
            if remaining <= 0:
                raise RuntimeFailure("evaluation deadline reached")
            return remaining

        resolved = self._resolve_hidden_dataset_path(
            "local-preview",
            task.task_id,
            request,
            data_path,
            remaining_seconds(),
        )
        try:
            candidate_workload = self.resolve_workload(
                episode_id="local-preview",
                workload=WorkloadRequest("candidate", task.candidate_image),
                timeout_seconds=remaining_seconds(),
            )
            evaluator_workload = self.resolve_workload(
                episode_id="local-preview",
                workload=WorkloadRequest("evaluator", task.evaluator_image),
                timeout_seconds=remaining_seconds(),
            )
            result = self.evaluate_candidate(
                episode_id="local-preview",
                task=task,
                resources=resources,
                candidate_dir=candidate_dir,
                hidden_dataset=resolved,
                output_dir=output_dir,
                candidate_workload=candidate_workload,
                evaluator_workload=evaluator_workload,
                timeout_seconds=remaining_seconds(),
            )
        except WorkerTeardownFailure:
            # The original error already reports the unconfirmed evaluation
            # boundary. Keep its private snapshots for operator-level retry;
            # the cleanup guard below would only restate the same condition.
            raise
        except BaseException:
            self.close_private_materializations()
            raise
        self.close_private_materializations()
        return result

    def _resolve_hidden_dataset_path(
        self,
        episode_id: str,
        task_id: str,
        request: HiddenDatasetRequest,
        data_path: Path,
        timeout_seconds: float,
    ) -> ResolvedHiddenDataset:
        """Materialize one two-part private snapshot behind an opaque ID."""

        if not isinstance(request, HiddenDatasetRequest):
            raise TypeError("hidden_dataset must be a HiddenDatasetRequest")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise HiddenDatasetResolutionFailure(
                "hidden dataset resolution deadline reached"
            )
        deadline = self.monotonic() + timeout_seconds
        resolution_id = uuid.uuid4().hex
        if self._hidden_snapshot_root is None:
            try:
                self._hidden_snapshot_root = Path(
                    tempfile.mkdtemp(prefix="autotrain-bench-hidden-")
                ).resolve()
                self._hidden_snapshot_root.chmod(0o700)
            except OSError as exc:
                raise HiddenDatasetResolutionFailure(
                    f"failed to create private hidden dataset storage: {exc}"
                ) from exc
        snapshot = self._hidden_snapshot_root / f"{resolution_id}.snapshot"
        staged = self._hidden_snapshot_root / f".{resolution_id}.tmp"
        try:
            if data_path.is_symlink() or not data_path.is_dir():
                raise OSError("hidden dataset must be a directory")
            source_entries = {entry.name: entry for entry in data_path.iterdir()}
            if set(source_entries) != set(HIDDEN_BUNDLE_FILENAMES):
                raise OSError(
                    "hidden dataset must contain only contexts.json and targets.json"
                )
            staged.mkdir(mode=0o700)
            for name in HIDDEN_BUNDLE_FILENAMES:
                source_path = source_entries[name]
                if source_path.is_symlink() or not source_path.is_file():
                    raise OSError(f"hidden dataset entry is not a regular file: {name}")
                destination_path = staged / name
                with source_path.open("rb") as source, destination_path.open(
                    "xb"
                ) as destination:
                    while True:
                        if self.monotonic() >= deadline:
                            raise HiddenDatasetResolutionFailure(
                                "hidden dataset resolution deadline reached"
                            )
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        destination.write(chunk)
                    destination.flush()
                    os.fsync(destination.fileno())
                destination_path.chmod(0o400)
            digest_hex = hidden_bundle_sha256(staged)
            os.replace(staged, snapshot)
            if self.monotonic() >= deadline:
                raise HiddenDatasetResolutionFailure(
                    "hidden dataset resolution deadline reached"
                )
        except HiddenDatasetResolutionFailure:
            shutil.rmtree(staged, ignore_errors=True)
            shutil.rmtree(snapshot, ignore_errors=True)
            raise
        except OSError as exc:
            shutil.rmtree(staged, ignore_errors=True)
            shutil.rmtree(snapshot, ignore_errors=True)
            raise HiddenDatasetResolutionFailure(
                f"failed to resolve hidden dataset {request.name}: {exc}"
            ) from exc
        resolved = ResolvedHiddenDataset(
            request.name,
            digest_hex,
            resolution_id,
            request.identity_algorithm,
        )
        if (
            request.expected_sha256 is not None
            and digest_hex != request.expected_sha256
        ):
            shutil.rmtree(snapshot, ignore_errors=True)
            raise HiddenDatasetIdentityMismatch(resolved)
        self._resolved_hidden_datasets[(episode_id, resolved.resolution_id)] = (
            task_id,
            resolved,
            snapshot,
        )
        return resolved

    def resolve_hidden_dataset(
        self,
        *,
        episode_id: str,
        task: EvaluationTaskSpec,
        hidden_dataset: HiddenDatasetRequest,
        timeout_seconds: float,
    ) -> ResolvedHiddenDataset:
        """Resolve the local reference source only when Core requests it."""

        if not isinstance(hidden_dataset, HiddenDatasetRequest):
            raise TypeError("hidden_dataset must be a HiddenDatasetRequest")
        data_path = self._hidden_datasets.get(hidden_dataset.name)
        if data_path is None:
            raise HiddenDatasetResolutionFailure(
                "no local hidden dataset source is configured for "
                f"{hidden_dataset.name}"
            )
        return self._resolve_hidden_dataset_path(
            episode_id,
            task.task_id,
            hidden_dataset,
            data_path,
            timeout_seconds,
        )

    def close_private_materializations(self) -> None:
        """Remove Worker-private hidden snapshots after all evaluations stop."""

        root = self._hidden_snapshot_root
        if root is None:
            return
        if self._pending_evaluations:
            raise WorkerTeardownFailure(
                "cannot remove private hidden datasets while evaluation teardown "
                "is unconfirmed"
            )
        try:
            shutil.rmtree(root)
        except OSError as exc:
            raise WorkerTeardownFailure(
                f"private hidden dataset cleanup failed: {exc}"
            ) from exc
        self._hidden_snapshot_root = None
        self._resolved_hidden_datasets.clear()

    def _run_isolated_evaluator(
        self,
        *,
        episode_id: str,
        task: EvaluationTaskSpec,
        resources: TrainingResources,
        candidate_dir: Path,
        split: str,
        data_path: Path,
        output_dir: Path,
        candidate_image: str,
        evaluator_image: str,
        timeout_seconds: Optional[float] = None,
    ) -> CandidateEvaluationResult:
        resources.validate()
        if split not in {"analysis", "test"}:
            raise RuntimeFailure(f"invalid evaluator split: {split}")
        if task.backend != "isolated_candidate":
            raise RuntimeFailure(f"unsupported evaluator backend: {task.backend}")
        if task.candidate_channel.protocol != "offline_next_token_logits_v3":
            raise RuntimeFailure("unsupported Candidate application protocol")
        if not data_path.is_dir():
            raise RuntimeFailure(
                f"hidden evaluator split is not a directory: {data_path}"
            )
        contexts_path = data_path / "contexts.json"
        targets_path = data_path / "targets.json"
        if not contexts_path.is_file() or not targets_path.is_file():
            raise RuntimeFailure("hidden evaluator split is incomplete")
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / f"{split}.json"
        validation_path = output_dir / f"{split}.validation.json"
        try:
            result_path.unlink(missing_ok=True)
            validation_path.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeFailure("failed to clear the prior evaluator result") from exc
        runtime_id = uuid.uuid4().hex[:12]
        candidate_name = f"atb-candidate-{runtime_id}"
        evaluator_name = f"atb-evaluator-{runtime_id}"
        evaluation_key = (episode_id, runtime_id)
        candidate_api_dir = output_dir / ".candidate_api"
        prediction_dir = output_dir / f"predictions-{runtime_id}"
        docker_scorer_adapter = (
            Path(__file__).resolve().parent / "docker_scorer_adapter.py"
        )
        package = candidate_api_dir / "autotrain_bench"
        if not candidate_api_dir.exists():
            package.mkdir(parents=True)
            core_package = Path(__file__).resolve().parent
            for filename in ("candidate_protocol.py", "candidate_offline_runner.py"):
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
            "none",
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
            "--mount",
            self._mount(contexts_path, "/eval-input/contexts.json", read_only=True),
            "--mount",
            self._mount(prediction_dir, "/predictions", read_only=False),
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
                candidate_image,
                "python",
                "-m",
                "autotrain_bench.candidate_offline_runner",
                "--candidate",
                "/candidate",
                "--contexts",
                "/eval-input/contexts.json",
                "--output",
                "/predictions",
                "--device",
                task.device,
            ]
        )

        def scorer_args(mode: str) -> List[str]:
            args = [
                "run",
                "--rm",
                "--name",
                evaluator_name,
                "--network",
                "none",
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
                self._mount(contexts_path, "/eval-data/contexts.json", read_only=True),
                "--mount",
                self._mount(targets_path, "/eval-data/targets.json", read_only=True),
                "--mount",
                self._mount(
                    task.evaluator_entrypoint,
                    "/opt/evaluator.py",
                    read_only=True,
                ),
                "--mount",
                self._mount(
                    docker_scorer_adapter,
                    "/opt/docker-scorer-adapter.py",
                    read_only=True,
                ),
                "--mount",
                self._mount(output_dir, "/output", read_only=False),
            ]
            if mode == "score":
                args.extend(
                    [
                        "--mount",
                        self._mount(prediction_dir, "/predictions", read_only=True),
                    ]
                )
            args.extend(
                [
                    "--env",
                    "PYTHONDONTWRITEBYTECODE=1",
                    evaluator_image,
                    "python",
                    "/opt/docker-scorer-adapter.py",
                    "--mode",
                    mode,
                    "--evaluator",
                    "/opt/evaluator.py",
                    "--contexts",
                    "/eval-data/contexts.json",
                    "--targets",
                    "/eval-data/targets.json",
                    "--output",
                    (
                        f"/output/{split}.validation.json"
                        if mode == "validate"
                        else f"/output/{split}.json"
                    ),
                    "--vocab-size",
                    str(task.vocab_size),
                    "--context-lengths",
                    *(str(length) for length in task.context_lengths),
                ]
            )
            if mode == "score":
                args.extend(["--predictions", "/predictions"])
            return args

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

        try:
            prediction_dir.mkdir()
        except OSError as exc:
            raise RuntimeFailure(
                "failed to create Candidate prediction storage"
            ) from exc
        self._pending_evaluations[evaluation_key] = (
            candidate_name,
            evaluator_name,
            prediction_dir,
        )
        serving_seconds: Optional[float] = None
        try:
            self._run(
                scorer_args("validate"),
                timeout=remaining_seconds(),
            )
            serving_started = self.monotonic()
            self._run(candidate_args, timeout=remaining_seconds())
            wait_result = self._run(
                ["wait", candidate_name],
                timeout=remaining_seconds(),
            )
            try:
                candidate_exit_code = int(wait_result.stdout.strip())
            except ValueError as exc:
                raise RuntimeFailure(
                    "Candidate process returned an invalid exit status"
                ) from exc
            if candidate_exit_code != 0:
                raise RuntimeFailure(
                    f"Candidate process exited with status {candidate_exit_code}"
                )
            serving_seconds = max(0.0, self.monotonic() - serving_started)
            self._run(
                scorer_args("score"),
                timeout=remaining_seconds(),
            )
        except RuntimeFailure as exc:
            diagnostic_path = output_dir / f"{split}.failure.log"
            try:
                diagnostic_path.write_text(
                    f"Runtime failure: {exc}\n",
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
            if set(payload) != {
                "performance_score",
                "raw_metrics",
                "examples",
                "input_tokens",
                "prediction_bytes",
            }:
                raise ValueError("unexpected scorer result fields")
            if serving_seconds is None:
                raise ValueError("serving duration is unavailable")
            result = CandidateEvaluationResult(
                performance_score=payload["performance_score"],
                raw_metrics=payload["raw_metrics"],
                serving_seconds=serving_seconds,
                examples=payload["examples"],
                input_tokens=payload["input_tokens"],
                prediction_bytes=payload["prediction_bytes"],
            )
            result.performance_manifest(task.performance_score)
        except (
            FileNotFoundError,
            KeyError,
            OverflowError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeFailure(
                "evaluator did not produce a valid performance result"
            ) from exc
        return result

    def evaluate_candidate(
        self,
        *,
        episode_id: str,
        task: EvaluationTaskSpec,
        resources: TrainingResources,
        candidate_dir: Path,
        hidden_dataset: ResolvedHiddenDataset,
        output_dir: Path,
        candidate_workload: ResolvedWorkload,
        evaluator_workload: ResolvedWorkload,
        timeout_seconds: Optional[float] = None,
    ) -> CandidateEvaluationResult:
        """Map the semantic evaluation operation to the reference runtime."""

        if not isinstance(hidden_dataset, ResolvedHiddenDataset):
            raise TypeError("hidden_dataset must be a ResolvedHiddenDataset")
        resolution = self._resolved_hidden_datasets.get(
            (episode_id, hidden_dataset.resolution_id)
        )
        if resolution is None:
            raise HiddenDatasetResolutionFailure(
                "hidden dataset resolution is unavailable for this episode"
            )
        expected_task_id, expected_resolution, data_path = resolution
        if task.task_id != expected_task_id or hidden_dataset != expected_resolution:
            raise HiddenDatasetResolutionFailure(
                "hidden dataset resolution evidence does not match the opaque ID"
            )
        try:
            actual_digest = hidden_bundle_sha256(data_path)
        except OSError as exc:
            raise HiddenDatasetResolutionFailure(
                f"resolved hidden dataset became unavailable: {exc}"
            ) from exc
        if actual_digest != hidden_dataset.sha256:
            raise HiddenDatasetIdentityMismatch(
                ResolvedHiddenDataset(
                    hidden_dataset.name,
                    actual_digest,
                    hidden_dataset.resolution_id,
                    hidden_dataset.identity_algorithm,
                )
            )
        candidate_image = self._workload_reference(
            episode_id=episode_id,
            workload=candidate_workload,
            role="candidate",
            requested_identity=task.candidate_image,
        )
        evaluator_image = self._workload_reference(
            episode_id=episode_id,
            workload=evaluator_workload,
            role="evaluator",
            requested_identity=task.evaluator_image,
        )
        return self._run_isolated_evaluator(
            episode_id=episode_id,
            task=task,
            resources=resources,
            candidate_dir=candidate_dir,
            split=hidden_dataset.name,
            data_path=data_path,
            output_dir=output_dir,
            candidate_image=candidate_image,
            evaluator_image=evaluator_image,
            timeout_seconds=timeout_seconds,
        )

    def _teardown_evaluation(self, evaluation_key: tuple) -> None:
        resources = self._pending_evaluations.get(evaluation_key)
        if resources is None:
            return
        candidate_name, evaluator_name, prediction_dir = resources
        _run_runtime_teardown_steps(
            [
                ("candidate", lambda: self.stop_container(candidate_name)),
                ("evaluator", lambda: self.stop_container(evaluator_name)),
            ]
        )
        try:
            self._remove_prediction_artifact(prediction_dir)
        except OSError as exc:
            raise WorkerTeardownFailure(
                f"prediction artifact cleanup failed: {exc}"
            ) from exc
        self._pending_evaluations.pop(evaluation_key, None)

    @staticmethod
    def _remove_prediction_artifact(prediction_dir: Path) -> None:
        if prediction_dir.is_symlink():
            raise OSError("prediction artifact root became a symlink")
        if prediction_dir.exists():
            shutil.rmtree(prediction_dir)

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
