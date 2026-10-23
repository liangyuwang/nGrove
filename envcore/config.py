"""Task configuration for the v0 local reference prototype."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from .candidate_protocol import (
    PREDICTION_ARTIFACT_RETENTION,
    candidate_entrypoint_contract,
    candidate_helper_api_manifest,
    context_manifest_contract,
    prediction_manifest_contract,
    prediction_storage_manifest,
    serving_timing_manifest,
)
from .worker_protocol import CostContract, PerformanceScoreContract


class ConfigError(ValueError):
    """Raised when a task configuration is invalid."""


# Agent-facing control protocol.
#
# Version 2 renamed the ``agent_mode`` status field to ``agent_egress`` and its
# ``"sandboxed"`` value to ``"denied"``.
#
# Version 3 excludes environment-owned preparation from participant timing, so
# ``elapsed_seconds`` and ``remaining_wall_clock_seconds`` no longer track raw
# wall clock since task reveal.  An Agent that measured the budget with its own
# clock instead of the reported one would now disagree with the Coordinator.
#
# Version 4 removes organizer-owned validation feedback. Candidate commit now
# returns artifact acceptance only; hidden ANALYSIS and TEST run after the
# research phase is irreversibly sealed.
#
# Version 5 publishes the exact offline prediction-manifest contract and the
# optional generated helper API in the Agent-visible task manifest.
#
# Version 6 promotes Candidate cost to an independent, release-scoped axis. Its
# primary metric is end-to-end Candidate wall time amortized over the frozen
# number of evaluation tasks.
#
# Version 7 publishes the existing Candidate module/function invocation contract,
# including ordered argument types and meanings, without adding a self-check.
#
# Version 8 publishes the complete context input schema, reader return structure,
# and binary logit layout for the existing offline Candidate ABI.
#
# Version 9 publishes a variable-length policy and max_context_tokens input bound
# instead of the trusted evaluation sampling lengths.
#
# Version 10 limits Agent-visible release metadata to declared identifiers;
# private manifest digests and audit extensions remain in trusted records.
# Version 11 distinguishes operator-stopped episodes from participant failures.
# Operator protection settings themselves are never Agent-visible.
# Version 12 declares bounded prediction runtime storage with temporary files;
# Task schema 7 binds that revised rule into the task release fingerprint.
# Version 13 exposes research execution facts; optional comparison metadata lives
# in an explicitly selected benchmark extension. Task schema 8 follows that split.
AGENT_PROTOCOL_VERSION = 13


def operator_research_limit(value: Any, track: str) -> Optional[float]:
    """Validate a trusted run setting, independent of task/Track budgets."""

    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ConfigError("operator research time limit must be finite and positive")
    if track.upper() != "B":
        raise ConfigError("operator research time limit is only available for Track B")
    return float(value)


def resolve_operator_research_limit(
    task_source: Path,
    track: str,
    *,
    override: Optional[float] = None,
    disabled: bool = False,
) -> Optional[float]:
    """Resolve launcher protection without changing the research task identity.

    The optional sibling file belongs to the trusted operator, not the task
    manifest or its public budget. Only the effective value enters episode audit.
    """

    if disabled:
        if override is not None:
            raise ConfigError("operator research protection cannot be set and disabled")
        if track.upper() != "B":
            raise ConfigError(
                "operator research time limit is only available for Track B"
            )
        return None
    if override is not None:
        return operator_research_limit(override, track)
    if track.upper() != "B":
        return None

    return operator_research_limit(
        _operator_default(task_source, "track_b_max_research_seconds"), "B"
    )


def operator_storage_limit(value: Any) -> Optional[int]:
    """A private deployment protection, never a Track resource envelope."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError("operator storage limit must be a positive integer of bytes")
    return value


def resolve_operator_storage_limit(
    task_source: Path, *, override: Optional[int] = None, disabled: bool = False
) -> Optional[int]:
    if disabled:
        if override is not None:
            raise ConfigError("operator storage protection cannot be set and disabled")
        return None
    if override is not None:
        return operator_storage_limit(override)
    return operator_storage_limit(
        _operator_default(task_source, "max_episode_storage_bytes")
    )


def _operator_default(task_source: Path, field: str) -> Any:
    """Read one effective default; overridden/inapplicable values are irrelevant."""

    source = task_source.expanduser().resolve().with_name("operator-defaults.json")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read operator defaults in {source}: {exc}") from exc
    fields = {"track_b_max_research_seconds", "max_episode_storage_bytes"}
    if not isinstance(raw, dict) or not raw or set(raw) - fields:
        raise ConfigError(f"{source} contains invalid operator defaults")
    if field in raw and raw[field] is None:
        raise ConfigError(f"{source}: {field} must be positive")
    return raw.get(field)


# Phases a release may declare as environment-owned rather than participant time.
# `docs/research-protocol.md` requires this boundary to be release-defined and
# identical across participants, so it belongs in the task manifest digest
# rather than in whichever Core revision happens to execute the episode.
ENVIRONMENT_OWNED_PREPARATION_PHASES = {"worker_lease_startup"}
DEFAULT_ENVIRONMENT_OWNED_PREPARATION = ("worker_lease_startup",)

AGENT_RUNNERS = {"container", "host_cli"}
AGENT_EGRESS_POLICIES = {"denied", "unrestricted-dev"}
EVALUATOR_BACKENDS = {"isolated_candidate"}
ANALYSIS_CANDIDATE_SELECTIONS = {"all_committed"}
ANALYSIS_RESULT_VISIBILITIES = {
    "organizer_only",
    "delayed_public",
    "public_after_finalization",
}
TASK_KINDS = {"research", "development_smoke"}
RESERVED_AGENT_ENVIRONMENT = {
    "ENVCORE_CONTROL_DIR",
    "ENVCORE_FORWARDED_ENVIRONMENT",
    "ENVCORE_HOST_RUNNER",
    "ENVCORE_INFERENCE_BASE_URL",
    "ENVCORE_TASK",
    "ENVCORE_TRAIN_DIR",
    "ENVCORE_TRAIN_WORKSPACE",
    "ENVCORE_WORKSPACE",
    "HOME",
    "OPENAI_BASE_URL",
    "PYTHONPATH",
}
RESERVED_INFERENCE_CREDENTIAL_ENVIRONMENT = {
    "ENVCORE_GATEWAY_HOST",
    "ENVCORE_GATEWAY_PORT",
    "ENVCORE_INFERENCE_AUDIT_PATH",
    "ENVCORE_INFERENCE_CONFIG_JSON",
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


def _positive_int_list(value: Any, field: str) -> Tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{field} must be a non-empty integer array")
    result = tuple(
        _positive_int(item, f"{field}[{index}]") for index, item in enumerate(value)
    )
    if tuple(sorted(set(result))) != result:
        raise ConfigError(f"{field} must be unique and strictly increasing")
    return result


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


def environment_owned_preparation(
    value: Any, field: str = "timing.environment_owned_preparation"
) -> Tuple[str, ...]:
    """Validate the release's participant-timing boundary.

    An absent field keeps the v0 default.  An explicit empty list is a valid and
    different choice: it declares that every second, including Worker Lease
    materialization, is charged to the participant.
    """

    if value is None:
        return DEFAULT_ENVIRONMENT_OWNED_PREPARATION
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ConfigError(f"{field} must be an array of phase names")
    unknown = [
        item for item in value if item not in ENVIRONMENT_OWNED_PREPARATION_PHASES
    ]
    if unknown:
        allowed = ", ".join(sorted(ENVIRONMENT_OWNED_PREPARATION_PHASES))
        raise ConfigError(
            f"{field} contains unknown phases {unknown}; allowed: {allowed}"
        )
    if len(set(value)) != len(value):
        raise ConfigError(f"{field} must not repeat a phase")
    return tuple(value)


def agent_egress(value: Any, field: str = "agent.egress") -> str:
    if not isinstance(value, str) or value not in AGENT_EGRESS_POLICIES:
        allowed = ", ".join(sorted(AGENT_EGRESS_POLICIES))
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


def _performance_score_contract(
    value: Any,
    field: str = "evaluation.performance_score",
    *,
    allow_legacy: bool = False,
) -> PerformanceScoreContract:
    try:
        return PerformanceScoreContract.from_manifest(value, allow_legacy=allow_legacy)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} is invalid: {exc}") from exc


def _cost_contract(
    value: Any, field: str = "evaluation.cost", *, allow_legacy: bool = False
) -> CostContract:
    try:
        return CostContract.from_manifest(value, allow_legacy=allow_legacy)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} is invalid: {exc}") from exc


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
    """Participant-owned Agent selection for one research episode.

    This is deliberately separate from :class:`TaskConfig`: a task package
    defines data, budgets, evaluation, and the Training Environment, while the
    participant supplies the Agent image/command and any initial workspace.
    """

    workspace: Path
    runner: str
    command: List[str]
    initial_workspace: str
    image: Optional[str] = None
    egress: Optional[str] = None
    environment: List[str] = dataclass_field(default_factory=list)
    inference: Optional[InferenceConfig] = None

    def validate(self) -> "AgentConfig":
        resolved_runner = agent_runner(self.runner, "agent runner")
        if resolved_runner == "container":
            agent_egress(self.egress, "agent egress")
            if not self.image:
                raise ConfigError("--agent-image is required for the container runner")
        else:
            if self.egress is not None:
                raise ConfigError(
                    "--agent-egress is not valid with the host_cli runner"
                )
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
    def network(self) -> str:
        if self.runner == "host_cli":
            return "host_managed"
        return "disabled" if self.egress == "denied" else "unrestricted_dev"

    def public_manifest(
        self,
        gateway_image: str,
        *,
        control_transport: Optional[str],
    ) -> Dict[str, Any]:
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
            "egress": self.egress,
            "network": self.network,
            "workspace": "." if self.runner == "host_cli" else "/workspace",
            "training_workspace": "/workspace",
            "workspace_scope": "episode_snapshot",
            "initial_workspace": self.initial_workspace,
            "control": control_transport,
            "forwarded_environment": list(self.environment or []),
            "inference": inference_manifest,
        }


@dataclass(frozen=True)
class TrainingResources:
    """Bundled v0 launch inputs translated into a semantic ResourceEnvelope.

    The fixed NVIDIA and device-path fields belong to the local reference
    composition. They are not the extensibility surface for external Workers.
    """

    nvidia_gpu_count: int = 0
    required_gpu_model: Optional[str] = None
    minimum_gpu_memory_mib: Optional[int] = None
    devices: List[str] = dataclass_field(default_factory=list)
    gpu_model_match_mode: str = "contains"

    def validate(self) -> "TrainingResources":
        _positive_int(
            self.nvidia_gpu_count,
            "--training-gpus",
            allow_zero=True,
        )
        if self.gpu_model_match_mode not in ("contains", "exact"):
            raise ConfigError("GPU model match mode must be 'contains' or 'exact'")
        if self.gpu_model_match_mode == "exact" and self.required_gpu_model is None:
            raise ConfigError("exact GPU model matching requires a GPU model")
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
        manifest = {
            "nvidia_gpu_count": self.nvidia_gpu_count,
            "required_gpu_model": self.required_gpu_model,
            "minimum_gpu_memory_mib": self.minimum_gpu_memory_mib,
            "devices": list(self.devices),
            "selection": "episode_runtime",
        }
        if self.gpu_model_match_mode != "contains":
            manifest["gpu_model_match_mode"] = self.gpu_model_match_mode
        return manifest


@dataclass(frozen=True)
class TaskConfig:
    """Fully resolved research task configuration.

    Participant execution is intentionally absent. Private evaluator paths stay
    in trusted host components and are never included in
    :meth:`public_manifest`.
    """

    source: Path
    schema_version: int
    task_id: str
    task_kind: str
    train_dir: Path
    analysis_path: Path
    test_path: Path
    training_image: str
    candidate_image: str
    evaluator_image: str
    inference_gateway_image: str
    performance_score: PerformanceScoreContract
    cost: CostContract
    vocab_size: int
    context_length: int
    context_lengths: Tuple[int, ...]
    environment_owned_preparation: Tuple[str, ...]
    track_a_total_seconds: int
    track_b_max_leases: int
    track_b_lease_seconds: int
    evaluator_timeout_seconds: int
    evaluator_device: str
    candidate_max_bytes: int
    candidate_max_files: int
    evaluator_backend: str
    evaluator_entrypoint: Path
    candidate_protocol: str
    analysis_candidate_selection: str
    analysis_result_visibility: str
    max_analysis_candidates: int

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
        schema_version = _positive_int(root.get("schema_version"), "schema_version")
        if schema_version not in {7, 8}:
            raise ConfigError("task schema_version must be 7 or 8")
        if schema_version == 8 and "leaderboard_eligible" in root:
            raise ConfigError("task schema 8 does not declare leaderboard eligibility")
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
        analysis = _mapping(root.get("analysis"), "analysis")
        timing = root.get("timing")
        preparation = None
        preparation_field = "timing.environment_owned_preparation"
        if timing is not None:
            timing = _mapping(timing, "timing")
            legacy_key = "benchmark_owned_preparation"
            current_key = "environment_owned_preparation"
            if legacy_key in timing and current_key in timing:
                raise ConfigError("timing may not declare both preparation field names")
            if legacy_key in timing:
                if schema_version != 7:
                    raise ConfigError(
                        "rename timing.benchmark_owned_preparation to "
                        "timing.environment_owned_preparation; preserve its value"
                    )
                preparation = timing[legacy_key]
                preparation_field = "timing.benchmark_owned_preparation"
            else:
                preparation = timing.get(current_key)
        track_a = _mapping(root.get("track_a"), "track_a")
        track_b = _mapping(root.get("track_b"), "track_b")

        base = source.parent

        def resolved_path(
            key: str, *, directory: bool, require_exists: bool = True
        ) -> Path:
            configured = _string(paths.get(key), f"paths.{key}")
            value = (base / configured).resolve()
            if require_exists and directory and not value.is_dir():
                raise ConfigError(f"paths.{key} is not a directory: {value}")
            if require_exists and not directory and not value.is_file():
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
            schema_version=schema_version,
            task_id=_string(root.get("task_id"), "task_id"),
            task_kind=task_kind(root.get("task_kind")),
            train_dir=resolved_path("train", directory=True),
            # Hidden data may be materialized by the Evaluation Worker only
            # after research sealing. Loading a task must not require either
            # file to exist in the research execution environment.
            analysis_path=resolved_path(
                "analysis", directory=True, require_exists=False
            ),
            test_path=resolved_path("test", directory=True, require_exists=False),
            training_image=_string(images.get("training"), "images.training"),
            candidate_image=_string(images.get("candidate"), "images.candidate"),
            evaluator_image=_string(images.get("evaluator"), "images.evaluator"),
            inference_gateway_image=_string(
                images.get("inference_gateway", "python:3.11-slim"),
                "images.inference_gateway",
            ),
            performance_score=_performance_score_contract(
                evaluation.get("performance_score"), allow_legacy=schema_version == 7
            ),
            cost=_cost_contract(
                evaluation.get("cost"), allow_legacy=schema_version == 7
            ),
            vocab_size=_positive_int(
                evaluation.get("vocab_size"), "evaluation.vocab_size"
            ),
            context_length=_positive_int(
                evaluation.get("context_length"), "evaluation.context_length"
            ),
            context_lengths=_positive_int_list(
                evaluation.get("context_lengths"), "evaluation.context_lengths"
            ),
            environment_owned_preparation=environment_owned_preparation(
                preparation, field=preparation_field
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
            candidate_protocol=_string(
                evaluation.get("candidate_protocol", "offline_next_token_logits_v3"),
                "evaluation.candidate_protocol",
            ),
            analysis_candidate_selection=_string(
                analysis.get("candidate_selection"),
                "analysis.candidate_selection",
            ),
            analysis_result_visibility=_string(
                analysis.get("result_visibility"),
                "analysis.result_visibility",
            ),
            max_analysis_candidates=_positive_int(
                analysis.get("max_candidates"),
                "analysis.max_candidates",
            ),
        )

        return config.validate()

    def validate(self) -> "TaskConfig":
        """Validate the immutable task definition and its release classification."""
        if self.schema_version not in {7, 8}:
            raise ConfigError("task schema_version must be 7 or 8")
        task_kind(self.task_kind)
        if not isinstance(self.performance_score, PerformanceScoreContract):
            raise ConfigError("evaluation.performance_score is invalid")
        if not isinstance(self.cost, CostContract):
            raise ConfigError("evaluation.cost is invalid")
        if self.evaluator_device not in {"cpu", "cuda"}:
            raise ConfigError("evaluation.device must be 'cpu' or 'cuda'")
        evaluator_backend(self.evaluator_backend)
        if self.candidate_protocol != "offline_next_token_logits_v3":
            raise ConfigError(
                "v0 only implements evaluation.candidate_protocol="
                "'offline_next_token_logits_v3'"
            )
        if (
            not self.context_lengths
            or any(
                isinstance(length, bool) or not isinstance(length, int) or length <= 0
                for length in self.context_lengths
            )
            or tuple(sorted(set(self.context_lengths))) != self.context_lengths
        ):
            raise ConfigError(
                "evaluation.context_lengths must be positive, unique, and "
                "strictly increasing"
            )
        if self.context_lengths[-1] > self.context_length:
            raise ConfigError(
                "evaluation.context_lengths must not exceed the "
                "evaluation.context_length input limit"
            )
        if self.analysis_candidate_selection not in ANALYSIS_CANDIDATE_SELECTIONS:
            raise ConfigError("analysis.candidate_selection must be 'all_committed'")
        if self.analysis_result_visibility not in ANALYSIS_RESULT_VISIBILITIES:
            allowed = ", ".join(sorted(ANALYSIS_RESULT_VISIBILITIES))
            raise ConfigError(f"analysis.result_visibility must be one of: {allowed}")
        for name, hidden_path in (
            ("paths.analysis", self.analysis_path),
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
        if self.analysis_path == self.test_path:
            raise ConfigError(
                "paths.analysis and paths.test must be distinct locations"
            )
        if self.max_analysis_candidates < self.track_b_max_leases:
            raise ConfigError("analysis.max_candidates must cover every Track B Lease")
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
            }

        return {
            "protocol_version": AGENT_PROTOCOL_VERSION,
            "timing": {
                "environment_owned_preparation": list(
                    self.environment_owned_preparation
                )
            },
            "task_id": self.task_id,
            "task_kind": self.task_kind,
            "track": normalized,
            "train": {"path": "/task/train", "visibility": "read_only"},
            "validation": {
                "organizer_feedback": "none",
                "participant_method": "unprescribed",
            },
            "analysis": {
                "content_visibility": "postmortem_evaluator_only",
                "feedback": "never_agent_visible",
                "candidate_selection": self.analysis_candidate_selection,
                "max_candidates": self.max_analysis_candidates,
                "result_visibility": self.analysis_result_visibility,
            },
            "test": {
                "content_visibility": "postmortem_evaluator_only",
                "feedback": "after_episode_close",
            },
            "evaluation": {
                "backend": self.evaluator_backend,
                "candidate_image": self.candidate_image,
                "candidate_protocol": self.candidate_protocol,
                "performance_score": self.performance_score.public_manifest(),
                "cost": self.cost.public_manifest(),
                "vocab_size": self.vocab_size,
                "max_context_tokens": self.context_length,
                "context_policy": (
                    "Evaluation inputs are variable-length token sequences. "
                    "The exact length distribution is not disclosed during research. "
                    "Participants choose how to use context to optimize next-token "
                    "prediction within the resource budget. max_context_tokens is "
                    "an input bound, not a guarantee that any evaluation example "
                    "reaches it."
                ),
                "device": self.evaluator_device,
                "candidate_artifact_limits": {
                    "max_bytes": self.candidate_max_bytes,
                    "max_files": self.candidate_max_files,
                },
                "candidate_boundary": {
                    "entrypoint": candidate_entrypoint_contract(),
                    "context_manifest": context_manifest_contract(),
                    "target_artifact_mount": "none",
                    "context_artifact_access": "read_only",
                    "prediction_artifact_access": "candidate_read_write_isolated",
                    "prediction_manifest": prediction_manifest_contract(),
                    "prediction_storage": prediction_storage_manifest(),
                    "prediction_artifact_retention": PREDICTION_ARTIFACT_RETENTION,
                    "public_helper_api": candidate_helper_api_manifest(),
                    "channel_mechanism": "worker_materialized_files",
                    "candidate_egress": "denied",
                    "protocol": self.candidate_protocol,
                    "serving_timing": serving_timing_manifest(),
                },
            },
            "budget": budget,
            "training_environment": {
                "image": self.training_image,
                "network": "disabled",
                "interactive_within_lease": True,
            },
        }
