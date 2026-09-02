"""Execution evidence emitted by the bundled local Agent launch paths."""

from __future__ import annotations

from .agent_execution import AgentExecutionEvidence
from .config import AgentConfig, immutable_image_reference
from .worker_protocol import EnforcementLevel, TrustLevel


def describe_local_agent_execution(agent: AgentConfig) -> AgentExecutionEvidence:
    """Describe the v0 launcher honestly without making the mechanism proof."""

    agent.validate()
    sandboxed_container = agent.runner == "container" and agent.mode == "sandboxed"
    limitations = []
    if not sandboxed_container:
        limitations.append("agent_execution_boundary")
    # Both bundled launch paths expose writable host bind mounts without a
    # deployment-enforced filesystem quota. Application-level mailbox limits
    # cannot contain bytes written before inspection.
    limitations.append("agent_writable_storage_unbounded")
    return AgentExecutionEvidence(
        source=(
            "bundled_local_docker"
            if agent.runner == "container"
            else "bundled_host_process"
        ),
        mechanism=agent.runner,
        enforcement=(
            EnforcementLevel.BEST_EFFORT
            if sandboxed_container
            else EnforcementLevel.UNVERIFIED
        ),
        trust_level=TrustLevel.DEVELOPMENT,
        workload_identity=agent.image if agent.runner == "container" else None,
        immutable_workload_identity=(
            agent.runner == "container" and immutable_image_reference(agent.image)
        ),
        limitations=tuple(limitations),
    )
