"""A replaceable, non-LLM queue dispatcher for platform-owned handlers.

A handler can call a validated launcher or ask a trusted assistant to prepare a
new mechanism online. Neither the handler nor this dispatcher chooses research
branches or resolves research-level merge conflicts for the target Agent.
"""

from __future__ import annotations

from typing import Protocol

from .session import BranchSession


class OperationHandler(Protocol):
    def apply(self, work: dict, session: BranchSession) -> None:
        """Apply one claimed operation, reporting transitions with its fencing token.

        Prepare: allocate -> session.allocated -> restore all components while gated
        -> session.ready -> release execution ONLY on a new permit.
        Checkpoint: quiesce all research -> session.quiesced -> capture immutable state
        -> session.complete(execution_held=True) -> release continuation on new permit.
        Stop: stop all research -> session.quiesced -> clean allocation
        -> session.released -> session.complete.
        On failure: prove quiescence and cleanup before session.fail_clean. Otherwise
        raise; dispatch preserves uncertain ownership for explicit recovery.
        Platform handlers must independently enforce deadlines and observe revocation.
        """
        ...


class Monitor:
    def __init__(
        self, session: BranchSession, monitor_id: str, handler: OperationHandler
    ):
        self.session, self.monitor_id, self.handler = session, monitor_id, handler

    def run_once(self) -> bool:
        work = self.session.claim(self.monitor_id)
        if work is None:
            return False
        operation = work["operation"]
        try:
            self.handler.apply(work, self.session)
        except BaseException as exc:
            # Cancellation/interruptions can occur after a deployment side effect.
            # Record uncertainty before propagating; never assume absence.
            self.session.uncertain(
                operation["id"],
                operation["token"],
                detail=f"{type(exc).__name__}: {exc}",
            )
            raise
        return True
