"""Compose an environment with the existing autonomous-training task protocol.

Hardware is run-owned here. Published benchmark lanes remain governed by
ChallengeRelease.resolve and cannot be overridden by this composition.
"""

from dataclasses import dataclass, replace
from typing import Optional

from .challenge import ChallengeRelease
from .config import ConfigError, TaskConfig, TrainingResources
from .environment import EnvironmentConfig
from .worker_protocol import ResourceEnvelope


@dataclass(frozen=True)
class EnvironmentRun:
    task: TaskConfig
    track: str
    environment: EnvironmentConfig
    resources: TrainingResources
    resource_envelope: ResourceEnvelope
    evaluation_resource_envelope: Optional[ResourceEnvelope]


def resolve_environment_run(
    task: TaskConfig,
    *,
    environment: EnvironmentConfig,
    track: Optional[str] = None,
    release: Optional[ChallengeRelease] = None,
    track_config_id: Optional[str] = None,
) -> EnvironmentRun:
    """Reuse a task/budget without claiming membership in a benchmark lane."""

    if release is not None:
        if track is not None or not track_config_id:
            raise ConfigError("--challenge requires --track-config instead of --track")
        if task.task_id != release.task_id or task.source != release.task_path:
            raise ConfigError("environment task must match the selected challenge")
        try:
            budget = release.track_configs[track_config_id]
        except KeyError as exc:
            raise ConfigError(f"unknown track config: {track_config_id}") from exc
        task = budget.apply(task)
        track = budget.track
    elif track_config_id is not None or track is None:
        raise ConfigError("a task environment requires --track, without --track-config")
    track = track.upper()
    if track not in {"A", "B"}:
        raise ConfigError("track must be A or B for the training protocol")
    # Only the run projection changes. Source task/release files remain intact.
    task = replace(task, leaderboard_eligible=False).validate()
    return EnvironmentRun(
        task=task,
        track=track,
        environment=environment,
        resources=environment.profile.training_resources().validate_for(task),
        resource_envelope=environment.resource_envelope(),
        evaluation_resource_envelope=environment.profile.evaluation_resource_contract,
    )
