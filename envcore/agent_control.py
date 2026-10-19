"""Semantic boundary between the Coordinator and an Agent control transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Tuple


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
class AgentControlExchange:
    """Transport-observed response publication; never proof of Agent consumption."""

    request_id: str
    response: Mapping[str, Any]
    delivery: str

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("control exchange requires a request ID")
        if not isinstance(self.response, dict):
            raise ValueError("control exchange response must be an object")
        if self.delivery not in {"published", "not_published"}:
            raise ValueError("invalid control exchange delivery")


@dataclass(frozen=True)
class AgentControlStepResult:
    handled: bool
    policy_failure: bool = False
    infrastructure_failure_code: Optional[str] = None
    infrastructure_failure: Optional[str] = None
    events: Tuple[AgentControlEvent, ...] = ()
    exchange: Optional[AgentControlExchange] = None

    def __post_init__(self) -> None:
        if self.exchange is not None and not isinstance(
            self.exchange, AgentControlExchange
        ):
            raise TypeError("exchange must be an AgentControlExchange")
        code = self.infrastructure_failure_code
        if code is not None and (
            not isinstance(code, str)
            or not code
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
                for character in code
            )
        ):
            raise ValueError(
                "infrastructure failure code must be a lowercase ASCII identifier"
            )
        if self.infrastructure_failure is not None and not isinstance(
            self.infrastructure_failure, str
        ):
            raise ValueError("infrastructure failure detail must be a string")
        if self.infrastructure_failure is not None and code is None:
            raise ValueError("infrastructure failure detail requires a failure code")
        if self.policy_failure and code is not None:
            raise ValueError(
                "a control step cannot be both policy and infrastructure failure"
            )


@dataclass(frozen=True)
class AgentControlCleanupResult:
    removed: int
    agent_policy_violations: int = 0
    agent_caused_failures: int = 0
    infrastructure_failures: int = 0

    @property
    def failures(self) -> int:
        return self.agent_caused_failures + self.infrastructure_failures

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
