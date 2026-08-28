"""Task configuration for the v0 local reference prototype."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit


class ConfigError(ValueError):
    """Raised when a task configuration is invalid."""


AGENT_RUNNERS = {"container", "host_cli"}
AGENT_MODES = {"sandboxed", "unrestricted-dev"}
RESERVED_AGENT_ENVIRONMENT = {
    "AUTOTRAIN_BENCH_CONTROL_DIR",
    "AUTOTRAIN_BENCH_FORWARDED_ENVIRONMENT",
    "AUTOTRAIN_BENCH_HOST_RUNNER",
    "AUTOTRAIN_BENCH_INFERENCE_BASE_URL",
    "AUTOTRAIN_BENCH_TASK",
    "AUTOTRAIN_BENCH_TRAIN_DIR",
    "AUTOTRAIN_BENCH_TRAIN_WORKSPACE",
    "AUTOTRAIN_BENCH_WORKSPACE",
    "HOME",
    "OPENAI_BASE_URL",
    "PYTHONPATH",
}
RESERVED_INFERENCE_CREDENTIAL_ENVIRONMENT = {
    "AUTOTRAIN_BENCH_GATEWAY_HOST",
    "AUTOTRAIN_BENCH_GATEWAY_PORT",
    "AUTOTRAIN_BENCH_INFERENCE_AUDIT_PATH",
    "AUTOTRAIN_BENCH_INFERENCE_CONFIG_JSON",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONPATH",
}
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _mapping(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be an object")
    return value


def _positive_int(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{field} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        comparator = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"{field} must be {comparator}")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty string")
    return value


def _string_list(value: Any, field: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{field} must be a non-empty string array")
    result: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{field} must contain non-empty strings")
        if item not in result:
            result.append(item)
    return tuple(result)


def agent_environment_names(value: Any, field: str = "agent.environment") -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError(f"{field} must be a string array")
    names: List[str] = []
    for item in value:
        if not isinstance(item, str) or not _ENVIRONMENT_NAME.fullmatch(item):
            raise ConfigError(f"{field} contains an invalid environment variable name")
        if item in RESERVED_AGENT_ENVIRONMENT:
            raise ConfigError(f"{field} may not override reserved variable {item}")
        if item not in names:
            names.append(item)
    return names


def agent_mode(value: Any, field: str = "agent.mode") -> str:
    if not isinstance(value, str) or value not in AGENT_MODES:
        allowed = ", ".join(sorted(AGENT_MODES))
        raise ConfigError(f"{field} must be one of: {allowed}")
    return value


def agent_runner(value: Any, field: str = "agent.runner") -> str:
    if not isinstance(value, str) or value not in AGENT_RUNNERS:
        allowed = ", ".join(sorted(AGENT_RUNNERS))
        raise ConfigError(f"{field} must be one of: {allowed}")
    return value


@dataclass(frozen=True)
class InferenceConfig:
    """Registered external researcher inference channel.

    The Agent Container sees only an OpenAI-compatible trusted sidecar.  The
    upstream origin and credential stay outside the untrusted Agent Container.
    """

    protocol: str
    base_url: str
    api_key_env: Optional[str]
    allowed_models: Tuple[str, ...]
    allowed_paths: Tuple[str, ...]
    timeout_seconds: int

    @classmethod
    def load(cls, value: Any, field: str = "agent.inference") -> Optional["InferenceConfig"]:
        if value is None:
            return None
        raw = _mapping(value, field)
        if raw.get("enabled", True) is False:
            return None
        protocol = _string(raw.get("protocol", "openai_compatible"), f"{field}.protocol")
        if protocol != "openai_compatible":
            raise ConfigError(f"{field}.protocol must be 'openai_compatible' in v0")
        base_url = _string(raw.get("base_url"), f"{field}.base_url")
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ConfigError(f"{field}.base_url must be an http(s) origin")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ConfigError(f"{field}.base_url must not contain a path, query, or fragment")

        api_key_env = raw.get("api_key_env")
        if api_key_env is not None:
            if not isinstance(api_key_env, str) or not _ENVIRONMENT_NAME.fullmatch(api_key_env):
                raise ConfigError(f"{field}.api_key_env must be an environment variable name or null")
            if api_key_env in RESERVED_INFERENCE_CREDENTIAL_ENVIRONMENT:
                raise ConfigError(
                    f"{field}.api_key_env may not use reserved gateway variable "
                    f"{api_key_env}"
                )

        allowed_models = _string_list(raw.get("allowed_models"), f"{field}.allowed_models")
        allowed_paths = _string_list(
            raw.get("allowed_paths", ["/v1/responses", "/v1/chat/completions"]),
            f"{field}.allowed_paths",
        )
        for path in allowed_paths:
            parsed_path = urlsplit(path)
            if (
                not path.startswith("/")
                or parsed_path.scheme
                or parsed_path.netloc
                or parsed_path.query
                or parsed_path.fragment
            ):
                raise ConfigError(f"{field}.allowed_paths must contain exact absolute paths")
        timeout_seconds = _positive_int(
            raw.get("timeout_seconds", 300), f"{field}.timeout_seconds"
        )
        return cls(
            protocol=protocol,
            base_url=base_url.rstrip("/"),
            api_key_env=api_key_env,
            allowed_models=allowed_models,
            allowed_paths=allowed_paths,
            timeout_seconds=timeout_seconds,
        )


@dataclass(frozen=True)
class TaskConfig:
    """Fully resolved benchmark task configuration.

    Private evaluator paths and inference credentials stay in trusted
    host/Gateway components and are never included in :meth:`public_manifest`.
    """

    source: Path
    task_id: str
    train_dir: Path
    val_path: Path
    test_path: Path
    agent_workspace: Path
    agent_image: Optional[str]
    training_image: str
    evaluator_image: str
    inference_gateway_image: str
    agent_command: List[str]
    agent_runner: str
    agent_mode: Optional[str]
    agent_environment: List[str]
    inference: Optional[InferenceConfig]
    gpu_count: int
    gpu_model: Optional[str]
    metric: str
    vocab_size: int
    context_length: int
    track_a_total_seconds: int
    track_b_max_leases: int
    track_b_lease_seconds: int
    evaluator_timeout_seconds: int
    evaluator_batch_size: int
    evaluator_device: str

    @classmethod
    def load(cls, path: Path) -> "TaskConfig":
        source = path.expanduser().resolve()
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(f"task configuration not found: {source}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON in {source}: {exc}") from exc

        root = _mapping(raw, "root")
        paths = _mapping(root.get("paths"), "paths")
        images = _mapping(root.get("images"), "images")
        agent = _mapping(root.get("agent"), "agent")
        hardware = _mapping(root.get("hardware"), "hardware")
        evaluation = _mapping(root.get("evaluation"), "evaluation")
        track_a = _mapping(root.get("track_a"), "track_a")
        track_b = _mapping(root.get("track_b"), "track_b")

        base = source.parent

        def resolved_path(key: str, *, directory: bool) -> Path:
            configured = _string(paths.get(key), f"paths.{key}")
            value = (base / configured).resolve()
            if directory and not value.is_dir():
                raise ConfigError(f"paths.{key} is not a directory: {value}")
            if not directory and not value.is_file():
                raise ConfigError(f"paths.{key} is not a file: {value}")
            return value

        command = agent.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(item, str) and item for item in command
        ):
            raise ConfigError("agent.command must be a non-empty string array")

        resolved_agent_runner = agent_runner(agent.get("runner", "container"))
        if resolved_agent_runner == "container":
            if "mode" in agent and "network" in agent:
                raise ConfigError("agent.mode and legacy agent.network may not both be set")
            if "mode" in agent:
                resolved_agent_mode: Optional[str] = agent_mode(agent.get("mode"))
            else:
                # Backward-compatible migration for the first local prototype.
                legacy_network = agent.get("network", "none")
                if legacy_network == "none":
                    resolved_agent_mode = "sandboxed"
                elif legacy_network == "open":
                    resolved_agent_mode = "unrestricted-dev"
                else:
                    raise ConfigError("legacy agent.network must be 'none' or 'open'")
        else:
            if "mode" in agent or "network" in agent:
                raise ConfigError(
                    "agent.mode and legacy agent.network apply only to the container runner"
                )
            resolved_agent_mode = None

        inference = InferenceConfig.load(agent.get("inference"))
        if resolved_agent_runner == "host_cli" and inference is not None:
            raise ConfigError(
                "agent.inference is not available with the host_cli runner; "
                "the host tool manages its own authenticated inference"
            )
        environment = agent_environment_names(agent.get("environment", []))
        if inference and inference.api_key_env and inference.api_key_env in environment:
            raise ConfigError(
                "agent.environment may not forward the inference API credential into the Agent Container"
            )

        gpu_model_value = hardware.get("gpu_model")
        if gpu_model_value is not None and not isinstance(gpu_model_value, str):
            raise ConfigError("hardware.gpu_model must be a string or null")

        config = cls(
            source=source,
            task_id=_string(root.get("task_id"), "task_id"),
            train_dir=resolved_path("train", directory=True),
            val_path=resolved_path("val", directory=False),
            test_path=resolved_path("test", directory=False),
            agent_workspace=resolved_path("agent_workspace", directory=True),
            agent_image=(
                _string(images.get("agent"), "images.agent")
                if resolved_agent_runner == "container"
                else (
                    _string(images.get("agent"), "images.agent")
                    if images.get("agent") is not None
                    else None
                )
            ),
            training_image=_string(images.get("training"), "images.training"),
            evaluator_image=_string(images.get("evaluator"), "images.evaluator"),
            inference_gateway_image=_string(
                images.get("inference_gateway", "python:3.11-slim"),
                "images.inference_gateway",
            ),
            agent_command=list(command),
            agent_runner=resolved_agent_runner,
            agent_mode=resolved_agent_mode,
            agent_environment=environment,
            inference=inference,
            gpu_count=_positive_int(
                hardware.get("gpu_count"), "hardware.gpu_count", allow_zero=True
            ),
            gpu_model=gpu_model_value,
            metric=_string(evaluation.get("metric"), "evaluation.metric"),
            vocab_size=_positive_int(
                evaluation.get("vocab_size"), "evaluation.vocab_size"
            ),
            context_length=_positive_int(
                evaluation.get("context_length"), "evaluation.context_length"
            ),
            track_a_total_seconds=_positive_int(
                track_a.get("total_seconds"), "track_a.total_seconds"
            ),
            track_b_max_leases=_positive_int(
                track_b.get("max_leases"), "track_b.max_leases"
            ),
            track_b_lease_seconds=_positive_int(
                track_b.get("lease_seconds"), "track_b.lease_seconds"
            ),
            evaluator_timeout_seconds=_positive_int(
                evaluation.get("timeout_seconds", 120),
                "evaluation.timeout_seconds",
            ),
            evaluator_batch_size=_positive_int(
                evaluation.get("batch_size", 1),
                "evaluation.batch_size",
            ),
            evaluator_device=_string(
                evaluation.get("device", "cpu"),
                "evaluation.device",
            ),
        )

        return config.validate()

    def validate(self) -> "TaskConfig":
        """Validate the final effective configuration, including CLI overrides."""

        agent_runner(self.agent_runner)
        if self.agent_runner == "container":
            agent_mode(self.agent_mode)
            if not self.agent_image:
                raise ConfigError("images.agent is required for the container runner")
        else:
            if self.agent_mode is not None:
                raise ConfigError("agent.mode must be null for the host_cli runner")
            if self.inference is not None:
                raise ConfigError(
                    "agent.inference is not available with the host_cli runner"
                )
        normalized_environment = agent_environment_names(self.agent_environment)
        if normalized_environment != self.agent_environment:
            raise ConfigError("agent.environment must be normalized and duplicate-free")
        if not self.agent_command or not all(
            isinstance(item, str) and item for item in self.agent_command
        ):
            raise ConfigError("agent.command must be a non-empty string array")
        if (
            self.inference
            and self.inference.api_key_env
            and self.inference.api_key_env in self.agent_environment
        ):
            raise ConfigError(
                "agent.environment may not forward the inference API credential "
                "into the Agent Container"
            )
        if self.metric != "autoregressive_nll":
            raise ConfigError("v0 only implements evaluation.metric='autoregressive_nll'")
        if self.gpu_count == 0 and self.gpu_model:
            raise ConfigError("hardware.gpu_model requires hardware.gpu_count > 0")
        if self.evaluator_device not in {"cpu", "cuda"}:
            raise ConfigError("evaluation.device must be 'cpu' or 'cuda'")
        if self.evaluator_device == "cuda" and self.gpu_count == 0:
            raise ConfigError("evaluation.device='cuda' requires hardware.gpu_count > 0")
        return self

    @property
    def images(self) -> List[str]:
        images = {self.training_image, self.evaluator_image}
        if self.agent_runner == "container" and self.agent_image:
            images.add(self.agent_image)
        if self.inference is not None:
            images.add(self.inference_gateway_image)
        return sorted(images)

    @property
    def policy_compliant(self) -> bool:
        return self.agent_runner == "container" and self.agent_mode == "sandboxed"

    @property
    def agent_network(self) -> str:
        if self.agent_runner == "host_cli":
            return "host_managed"
        return "disabled" if self.agent_mode == "sandboxed" else "unrestricted_dev"

    def public_manifest(self, track: str) -> Dict[str, Any]:
        normalized = track.upper()
        if normalized not in {"A", "B"}:
            raise ConfigError("track must be A or B")

        budget: Dict[str, Any]
        if normalized == "A":
            budget = {"total_wall_clock_seconds": self.track_a_total_seconds}
        else:
            budget = {
                "max_leases": self.track_b_max_leases,
                "max_seconds_per_lease": self.track_b_lease_seconds,
                "official_val_queries_per_lease": 1,
            }

        inference_manifest: Dict[str, Any] = {"enabled": False}
        if self.inference is not None:
            inference_manifest = {
                "enabled": True,
                "protocol": self.inference.protocol,
                "local_base_url": "http://inference-gateway:8765/v1",
                "gateway_image": self.inference_gateway_image,
                "allowed_models": list(self.inference.allowed_models),
                "allowed_paths": list(self.inference.allowed_paths),
                "provider_hosted_tools": "forbidden_except_function_tools",
            }

        return {
            "protocol_version": 1,
            "task_id": self.task_id,
            "track": normalized,
            "train": {"path": "/benchmark/train", "visibility": "read_only"},
            "validation": {
                "content_visibility": "evaluator_only",
                "feedback": "aggregate_metric_after_commit",
            },
            "test": {
                "content_visibility": "postmortem_evaluator_only",
                "feedback": "after_episode_close",
            },
            "evaluation": {
                "metric": self.metric,
                "vocab_size": self.vocab_size,
                "context_length": self.context_length,
                "batch_size": self.evaluator_batch_size,
                "device": self.evaluator_device,
            },
            "hardware": {
                "gpu_count": self.gpu_count,
                "gpu_model": self.gpu_model,
                "allocation": "provided_by_user_before_episode",
            },
            "budget": budget,
            "agent_environment": {
                "runner": self.agent_runner,
                "mode": self.agent_mode,
                "network": self.agent_network,
                "workspace": "." if self.agent_runner == "host_cli" else "/workspace",
                "training_workspace": "/workspace",
                "workspace_scope": "episode_copy",
                "control": "file_mailbox",
                "forwarded_environment": list(self.agent_environment),
                "inference": inference_manifest,
                "policy_compliant": self.policy_compliant,
            },
            "training_environment": {
                "image": self.training_image,
                "network": "disabled",
                "interactive_within_lease": True,
            },
        }
