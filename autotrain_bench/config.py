"""Task configuration for the v0 local reference prototype."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit


class ConfigError(ValueError):
    """Raised when a task configuration is invalid."""


AGENT_RUNNERS = {"container", "host_cli"}
AGENT_MODES = {"sandboxed", "unrestricted-dev"}
EVALUATOR_BACKENDS = {"isolated_candidate"}
TASK_KINDS = {"benchmark_challenge", "development_smoke"}
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
_IMMUTABLE_IMAGE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")


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


def evaluator_backend(value: Any, field: str = "evaluation.backend") -> str:
    if not isinstance(value, str) or value not in EVALUATOR_BACKENDS:
        allowed = ", ".join(sorted(EVALUATOR_BACKENDS))
        raise ConfigError(f"{field} must be one of: {allowed}")
    return value


def task_kind(value: Any, field: str = "task_kind") -> str:
    if not isinstance(value, str) or value not in TASK_KINDS:
        allowed = ", ".join(sorted(TASK_KINDS))
        raise ConfigError(f"{field} must be one of: {allowed}")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be a boolean")
    return value


def immutable_image_reference(value: Any) -> bool:
    """Return whether a container reference is pinned to a SHA-256 digest."""

    return isinstance(value, str) and bool(_IMMUTABLE_IMAGE.fullmatch(value))


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
    def load(
        cls, value: Any, field: str = "agent.inference"
    ) -> Optional["InferenceConfig"]:
        if value is None:
            return None
        raw = _mapping(value, field)
        if raw.get("enabled", True) is False:
            return None
        protocol = _string(
            raw.get("protocol", "openai_compatible"), f"{field}.protocol"
        )
        if protocol != "openai_compatible":
            raise ConfigError(f"{field}.protocol must be 'openai_compatible' in v0")
        base_url = _string(raw.get("base_url"), f"{field}.base_url")
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ConfigError(f"{field}.base_url must be an http(s) origin")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ConfigError(
                f"{field}.base_url must not contain a path, query, or fragment"
            )

        api_key_env = raw.get("api_key_env")
        if api_key_env is not None:
            if not isinstance(api_key_env, str) or not _ENVIRONMENT_NAME.fullmatch(
                api_key_env
            ):
                raise ConfigError(
                    f"{field}.api_key_env must be an environment variable name or null"
                )
            if api_key_env in RESERVED_INFERENCE_CREDENTIAL_ENVIRONMENT:
                raise ConfigError(
                    f"{field}.api_key_env may not use reserved gateway variable "
                    f"{api_key_env}"
                )

        allowed_models = _string_list(
            raw.get("allowed_models"), f"{field}.allowed_models"
        )
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
                raise ConfigError(
                    f"{field}.allowed_paths must contain exact absolute paths"
                )
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
class AgentConfig:
    """Participant-owned Agent selection for one benchmark episode.

    This is deliberately separate from :class:`TaskConfig`: a task package
    defines data, budgets, evaluation, and the Training Environment, while the
    participant supplies the Agent image/command and any initial workspace.
    """

    workspace: Path
    runner: str
    command: List[str]
    initial_workspace: str
    image: Optional[str] = None
    mode: Optional[str] = None
    environment: List[str] = dataclass_field(default_factory=list)
    inference: Optional[InferenceConfig] = None

    def validate(self) -> "AgentConfig":
        resolved_runner = agent_runner(self.runner, "agent runner")
        if resolved_runner == "container":
            agent_mode(self.mode, "agent mode")
            if not self.image:
                raise ConfigError("--agent-image is required for the container runner")
        else:
            if self.mode is not None:
                raise ConfigError("--agent-mode is not valid with the host_cli runner")
            if self.image is not None:
                raise ConfigError("--agent-image is not valid with the host_cli runner")
            if self.inference is not None:
                raise ConfigError(
                    "registered inference is not available with the host_cli runner; "
                    "the host tool manages its own authenticated inference"
                )
        if not self.workspace.is_dir():
            raise ConfigError(f"agent workspace is not a directory: {self.workspace}")
        if not self.command or not all(
            isinstance(item, str) and item for item in self.command
        ):
            raise ConfigError("--agent-command must be a non-empty command")
        if self.initial_workspace not in {"empty", "participant_snapshot"}:
            raise ConfigError(
                "agent initial_workspace must be 'empty' or 'participant_snapshot'"
            )
        normalized_environment = agent_environment_names(
            self.environment or [], "agent environment"
        )
        if normalized_environment != (self.environment or []):
            raise ConfigError("agent environment must be normalized and duplicate-free")
        if (
            self.inference
            and self.inference.api_key_env
            and self.inference.api_key_env in normalized_environment
        ):
            raise ConfigError(
                "the inference API credential may not be forwarded into the "
                "Agent Container"
            )
        return self

    @property
    def policy_compliant(self) -> bool:
        return self.runner == "container" and self.mode == "sandboxed"

    @property
    def network(self) -> str:
        if self.runner == "host_cli":
            return "host_managed"
        return "disabled" if self.mode == "sandboxed" else "unrestricted_dev"

    def public_manifest(self, gateway_image: str) -> Dict[str, Any]:
        inference_manifest: Dict[str, Any] = {"enabled": False}
        if self.inference is not None:
            inference_manifest = {
                "enabled": True,
                "protocol": self.inference.protocol,
                "local_base_url": "http://inference-gateway:8765/v1",
                "gateway_image": gateway_image,
                "allowed_models": list(self.inference.allowed_models),
                "allowed_paths": list(self.inference.allowed_paths),
                "provider_hosted_tools": "forbidden_except_function_tools",
            }
        return {
            "runner": self.runner,
            "mode": self.mode,
            "network": self.network,
            "workspace": "." if self.runner == "host_cli" else "/workspace",
            "training_workspace": "/workspace",
            "workspace_scope": "episode_snapshot",
            "initial_workspace": self.initial_workspace,
            "control": "file_mailbox",
            "forwarded_environment": list(self.environment or []),
            "inference": inference_manifest,
            "policy_compliant": self.policy_compliant,
        }


@dataclass(frozen=True)
class TrainingResources:
    """Trusted episode-time resource envelope requested from a Worker."""

    nvidia_gpu_count: int = 0
    required_gpu_model: Optional[str] = None
    minimum_gpu_memory_mib: Optional[int] = None
    devices: List[str] = dataclass_field(default_factory=list)

    def validate(self) -> "TrainingResources":
        _positive_int(
            self.nvidia_gpu_count,
            "--training-gpus",
            allow_zero=True,
        )
        if self.required_gpu_model is not None:
            _string(self.required_gpu_model, "--training-gpu-model")
            if self.nvidia_gpu_count == 0:
                raise ConfigError(
                    "--training-gpu-model requires --training-gpus greater than zero"
                )
        if self.minimum_gpu_memory_mib is not None:
            _positive_int(
                self.minimum_gpu_memory_mib,
                "minimum GPU memory",
            )
            if self.nvidia_gpu_count == 0:
                raise ConfigError(
                    "minimum GPU memory requires an NVIDIA GPU count greater than zero"
                )
        normalized: List[str] = []
        for device in self.devices:
            if (
                not isinstance(device, str)
                or not device.startswith("/")
                or ":" in device
            ):
                raise ConfigError(
                    "--training-device must be an absolute device path without ':'"
                )
            if device not in normalized:
                normalized.append(device)
        if normalized != self.devices:
            raise ConfigError("training devices must be normalized and duplicate-free")
        return self

    def validate_for(self, task: "TaskConfig") -> "TrainingResources":
        self.validate()
        if task.evaluator_device == "cuda" and self.nvidia_gpu_count == 0:
            raise ConfigError(
                "evaluation.device='cuda' requires --training-gpus greater than zero"
            )
        return self

    def public_manifest(self) -> Dict[str, Any]:
        return {
            "nvidia_gpu_count": self.nvidia_gpu_count,
            "required_gpu_model": self.required_gpu_model,
            "minimum_gpu_memory_mib": self.minimum_gpu_memory_mib,
            "devices": list(self.devices),
            "selection": "episode_runtime",
        }


@dataclass(frozen=True)
class LeaderboardMetricConfig:
    """Primary score frozen into the immutable task definition."""

    name: str
    field: str
    derived_from: str
    formula: str
    direction: str

    @classmethod
    def load(
        cls,
        value: Any,
        field: str = "evaluation.leaderboard_metric",
    ) -> "LeaderboardMetricConfig":
        raw = _mapping(value, field)
        return cls(
            name=_string(raw.get("name"), f"{field}.name"),
            field=_string(raw.get("field"), f"{field}.field"),
            derived_from=_string(raw.get("derived_from"), f"{field}.derived_from"),
            formula=_string(raw.get("formula"), f"{field}.formula"),
            direction=_string(raw.get("direction"), f"{field}.direction"),
        ).validate()

    def validate(self) -> "LeaderboardMetricConfig":
        expected = {
            "name": "perplexity",
            "field": "test_perplexity",
            "derived_from": "autoregressive_nll",
            "formula": "exp(mean_token_nll)",
            "direction": "minimize",
        }
        if self.public_manifest() != expected:
            raise ConfigError(
                "v0 requires evaluation.leaderboard_metric to declare "
                "test_perplexity = exp(mean_token_nll), minimized"
            )
        return self

    def public_manifest(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "field": self.field,
            "derived_from": self.derived_from,
            "formula": self.formula,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class TaskConfig:
    """Fully resolved benchmark task configuration.

    Participant execution is intentionally absent. Private evaluator paths stay
    in trusted host components and are never included in
    :meth:`public_manifest`.
    """

    source: Path
    task_id: str
    task_kind: str
    leaderboard_eligible: bool
    train_dir: Path
    val_path: Path
    test_path: Path
    training_image: str
    candidate_image: str
    evaluator_image: str
    inference_gateway_image: str
    metric: str
    leaderboard_metric: LeaderboardMetricConfig
    vocab_size: int
    context_length: int
    track_a_total_seconds: int
    track_b_max_leases: int
    track_b_lease_seconds: int
    evaluator_timeout_seconds: int
    evaluator_batch_size: int
    evaluator_device: str
    candidate_max_bytes: int
    candidate_max_files: int
    evaluator_backend: str
    evaluator_entrypoint: Path

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
        if "agent" in root:
            raise ConfigError(
                "agent configuration is participant-owned and may not appear in a task"
            )
        if "agent_workspace" in paths:
            raise ConfigError(
                "paths.agent_workspace is participant-owned and may not appear in a task"
            )
        if "agent" in images:
            raise ConfigError(
                "images.agent is participant-owned and may not appear in a task"
            )
        if "hardware" in root:
            raise ConfigError(
                "hardware allocation is episode-owned and may not appear in a task"
            )
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

        configured_evaluator_entrypoint = _string(
            evaluation.get("entrypoint"), "evaluation.entrypoint"
        )
        evaluator_entrypoint = (base / configured_evaluator_entrypoint).resolve()
        if not evaluator_entrypoint.is_file():
            raise ConfigError(
                "evaluation.entrypoint is not a file: " f"{evaluator_entrypoint}"
            )

        config = cls(
            source=source,
            task_id=_string(root.get("task_id"), "task_id"),
            task_kind=task_kind(root.get("task_kind")),
            leaderboard_eligible=_boolean(
                root.get("leaderboard_eligible"), "leaderboard_eligible"
            ),
            train_dir=resolved_path("train", directory=True),
            val_path=resolved_path("val", directory=False),
            test_path=resolved_path("test", directory=False),
            training_image=_string(images.get("training"), "images.training"),
            candidate_image=_string(images.get("candidate"), "images.candidate"),
            evaluator_image=_string(images.get("evaluator"), "images.evaluator"),
            inference_gateway_image=_string(
                images.get("inference_gateway", "python:3.11-slim"),
                "images.inference_gateway",
            ),
            metric=_string(evaluation.get("metric"), "evaluation.metric"),
            leaderboard_metric=LeaderboardMetricConfig.load(
                evaluation.get("leaderboard_metric")
            ),
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
            candidate_max_bytes=_positive_int(
                evaluation.get("max_candidate_bytes"),
                "evaluation.max_candidate_bytes",
            ),
            candidate_max_files=_positive_int(
                evaluation.get("max_candidate_files"),
                "evaluation.max_candidate_files",
            ),
            evaluator_backend=evaluator_backend(evaluation.get("backend")),
            evaluator_entrypoint=evaluator_entrypoint,
        )

        return config.validate()

    def validate(self) -> "TaskConfig":
        """Validate the immutable task definition and its release classification."""
        task_kind(self.task_kind)
        if self.task_kind == "development_smoke" and self.leaderboard_eligible:
            raise ConfigError("development smoke tasks may not be leaderboard eligible")
        if self.metric != "autoregressive_nll":
            raise ConfigError(
                "v0 only implements evaluation.metric='autoregressive_nll'"
            )
        self.leaderboard_metric.validate()
        if self.leaderboard_metric.derived_from != self.metric:
            raise ConfigError(
                "evaluation.leaderboard_metric.derived_from must match "
                "evaluation.metric"
            )
        if self.evaluator_device not in {"cpu", "cuda"}:
            raise ConfigError("evaluation.device must be 'cpu' or 'cuda'")
        evaluator_backend(self.evaluator_backend)
        for name, hidden_path in (
            ("paths.val", self.val_path),
            ("paths.test", self.test_path),
        ):
            try:
                hidden_path.relative_to(self.train_dir)
            except ValueError:
                pass
            else:
                raise ConfigError(
                    f"{name} must be outside the Research Worker TRAIN directory"
                )
        if self.val_path.samefile(self.test_path):
            raise ConfigError("paths.val and paths.test must be distinct files")
        if self.leaderboard_eligible:
            images = {
                "images.training": self.training_image,
                "images.candidate": self.candidate_image,
                "images.evaluator": self.evaluator_image,
                "images.inference_gateway": self.inference_gateway_image,
            }
            mutable = [
                field
                for field, reference in images.items()
                if not immutable_image_reference(reference)
            ]
            if mutable:
                raise ConfigError(
                    "leaderboard-eligible tasks require digest-pinned images: "
                    + ", ".join(mutable)
                )
        if not self.evaluator_entrypoint.is_file():
            raise ConfigError(
                "evaluation.entrypoint is not a file: " f"{self.evaluator_entrypoint}"
            )
        return self

    @property
    def images(self) -> List[str]:
        return sorted(
            {
                self.training_image,
                self.candidate_image,
                self.evaluator_image,
            }
        )

    @property
    def evaluator_policy_compliant(self) -> bool:
        return self.evaluator_backend == "isolated_candidate"

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

        return {
            "protocol_version": 1,
            "task_id": self.task_id,
            "task_kind": self.task_kind,
            "leaderboard_eligible": self.leaderboard_eligible,
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
                "backend": self.evaluator_backend,
                "candidate_image": self.candidate_image,
                "metric": self.metric,
                "leaderboard_metric": self.leaderboard_metric.public_manifest(),
                "vocab_size": self.vocab_size,
                "context_length": self.context_length,
                "batch_size": self.evaluator_batch_size,
                "device": self.evaluator_device,
                "candidate_artifact_limits": {
                    "max_bytes": self.candidate_max_bytes,
                    "max_files": self.candidate_max_files,
                },
                "policy_compliant": self.evaluator_policy_compliant,
                "candidate_boundary": {
                    "hidden_data_mount": "none",
                    "network": "ephemeral_internal_only",
                    "protocol": "next_token_logits_v1",
                },
            },
            "budget": budget,
            "training_environment": {
                "image": self.training_image,
                "network": "disabled",
                "interactive_within_lease": True,
            },
        }
