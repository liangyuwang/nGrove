"""Resolve every research run into one frozen execution environment."""

from dataclasses import dataclass
from typing import Optional

from .config import ConfigError, TaskConfig, TrainingResources
from .environment import EnvironmentConfig, FrozenEnvironment, ResearchBudget
from .worker_protocol import EnforcementLevel, ResourceEnvelope, v0_resource_envelope


@dataclass(frozen=True)
class EnvironmentRun:
    task: TaskConfig
    track: str
    environment: FrozenEnvironment

    @property
    def resources(self) -> TrainingResources:
        return self.environment.resources

    @property
    def resource_envelope(self) -> ResourceEnvelope:
        return self.environment.resource_envelope

    @property
    def evaluation_resource_envelope(self) -> Optional[ResourceEnvelope]:
        return self.environment.evaluation_resource_envelope


def resolve_environment_run(
    task: TaskConfig,
    *,
    track: str,
    environment: Optional[EnvironmentConfig] = None,
    resources: Optional[TrainingResources] = None,
    resource_envelope: Optional[ResourceEnvelope] = None,
    evaluation_resource_envelope: Optional[ResourceEnvelope] = None,
) -> EnvironmentRun:
    """Resolve configuration once; execution does not depend on its source."""
    track = track.upper()
    if track not in {"A", "B"}:
        raise ConfigError("track must be A or B for the training protocol")
    if environment is not None:
        if any(
            value is not None
            for value in (resources, resource_envelope, evaluation_resource_envelope)
        ):
            raise ConfigError(
                "choose an environment configuration or explicit execution resources"
            )
        frozen = FrozenEnvironment.from_config(environment, task, track)
    else:
        resources = resources if resources is not None else TrainingResources()
        frozen = FrozenEnvironment.create(
            budget=ResearchBudget.from_task(task, track),
            resources=resources,
            resource_envelope=resource_envelope
            if resource_envelope is not None
            else v0_resource_envelope(
                resources, minimum_enforcement=EnforcementLevel.BEST_EFFORT
            ),
            evaluation_resource_envelope=evaluation_resource_envelope,
        )
    task = frozen.budget.apply(task)
    frozen.resources.validate_for(task)
    return EnvironmentRun(task=task, track=track, environment=frozen)
