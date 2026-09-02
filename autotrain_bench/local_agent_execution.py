"""Execution evidence emitted by the bundled local Agent launch paths."""

from __future__ import annotations

from .agent_execution import AgentExecutionEvidence
from .config import AgentConfig, immutable_image_reference
from .worker_protocol import EnforcementLevel, TrustLevel


def describe_local_agent_execution(agent: AgentConfig) -> AgentExecutionEvidence:
    """Describe the v0 launcher honestly without making the mechanism proof."""

    agent.validate()
    compliant_boundary = agent.runner == "container" and agent.mode == "sandboxed"
    return AgentExecutionEvidence(
        source=(
            "bundled_local_docker"
            if agent.runner == "container"
            else "bundled_host_process"
        ),
        mechanism=agent.runner,
        enforcement=(
            EnforcementLevel.ENFORCED
            if compliant_boundary
            else EnforcementLevel.UNVERIFIED
        ),
        trust_level=TrustLevel.DEVELOPMENT,
        workload_identity=agent.image if agent.runner == "container" else None,
        immutable_workload_identity=(
            agent.runner == "container" and immutable_image_reference(agent.image)
        ),
        limitations=() if compliant_boundary else ("agent_execution_boundary",),
    )
