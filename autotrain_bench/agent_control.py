"""Semantic boundary between the Coordinator and an Agent control transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Protocol, Tuple


MAX_AGENT_CONTROL_RESPONSE_BYTES = 3 * 1024 * 1024


class ProtocolError(RuntimeError):
    """An Agent request violates the benchmark protocol."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AgentControlEvent:
    name: str
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentControlStepResult:
    handled: bool
    policy_failure: bool = False
    events: Tuple[AgentControlEvent, ...] = ()


@dataclass(frozen=True)
class AgentControlCleanupResult:
    removed: int
    failures: int

    @property
    def complete(self) -> bool:
        return self.failures == 0


class AgentControlTransport(Protocol):
    """Deployment-supplied transport for the semantic Agent request handler."""

    transport_name: str

    def process_one(
        self,
        handler: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> AgentControlStepResult: ...

    def cleanup(self) -> AgentControlCleanupResult: ...
