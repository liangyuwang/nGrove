"""Task configuration for the v0 local reference prototype."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConfigError(ValueError):
    """Raised when a task configuration is invalid."""


AGENT_NETWORK_POLICIES = {"none", "open"}
RESERVED_AGENT_ENVIRONMENT = {
    "AUTOTRAIN_BENCH_CONTROL_DIR",
    "AUTOTRAIN_BENCH_TASK",
    "HOME",
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


def agent_network_policy(value: Any, field: str = "agent.network") -> str:
    if not isinstance(value, str) or value not in AGENT_NETWORK_POLICIES:
        allowed = ", ".join(sorted(AGENT_NETWORK_POLICIES))
        raise ConfigError(f"{field} must be one of: {allowed}")
    return value


@dataclass(frozen=True)
class TaskConfig:
    """Fully resolved benchmark task configuration.

    Private evaluator paths stay on the trusted host and are never included in
    :meth:`public_manifest`.
    """

    source: Path
    task_id: str
    train_dir: Path
    val_path: Path
    test_path: Path
    agent_workspace: Path
    agent_image: str
    training_image: str
    evaluator_image: str
    agent_command: List[str]
    agent_network: str
    agent_environment: List[str]
    gpu_count: int
    gpu_model: Optional[str]
    metric: str
    vocab_size: int
    context_length: int
    track_a_total_seconds: int
    track_b_max_leases: int
    track_b_lease_seconds: int
    evaluator_timeout_seconds: int

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
            agent_image=_string(images.get("agent"), "images.agent"),
            training_image=_string(images.get("training"), "images.training"),
            evaluator_image=_string(images.get("evaluator"), "images.evaluator"),
            agent_command=list(command),
            agent_network=agent_network_policy(agent.get("network", "none")),
            agent_environment=agent_environment_names(agent.get("environment", [])),
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
        )

        if config.metric != "autoregressive_nll":
            raise ConfigError(
                "v0 only implements evaluation.metric='autoregressive_nll'"
            )
        if config.gpu_count == 0 and config.gpu_model:
            raise ConfigError("hardware.gpu_model requires hardware.gpu_count > 0")
        return config

    @property
    def images(self) -> List[str]:
        return sorted(
            {self.agent_image, self.training_image, self.evaluator_image}
        )

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
            },
            "hardware": {
                "gpu_count": self.gpu_count,
                "gpu_model": self.gpu_model,
                "allocation": "provided_by_user_before_episode",
            },
            "budget": budget,
            "agent_environment": {
                "network": self.agent_network,
                "workspace": "/workspace",
                "workspace_scope": "episode_copy",
                "control": "file_mailbox",
                "forwarded_environment": list(self.agent_environment),
            },
            "training_environment": {
                "image": self.training_image,
                "network": "disabled",
                "interactive_within_lease": True,
            },
        }
