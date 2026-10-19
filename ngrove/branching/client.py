"""Small Agent-facing client; no database, filesystem, Worker or Operator access.

The deployment supplies an authenticated transport. Retain request_id when a
transport times out: replaying it is safe; inventing a new ID creates a new intent.
"""

from __future__ import annotations

import uuid
from typing import Callable, Optional


class BranchClientError(ValueError):
    pass


class BranchClient:
    def __init__(self, transport: Callable[[dict], dict]):
        self.transport = transport

    def request(
        self, operation: str, args: dict, *, request_id: Optional[str] = None
    ) -> dict:
        response = self.transport(
            {"id": request_id or uuid.uuid4().hex, "op": operation, "args": args}
        )
        if not response.get("ok"):
            raise BranchClientError(response.get("error", "request_failed"))
        return response["result"]

    def fork(
        self,
        snapshot,
        *,
        resources,
        context_mode,
        prefix_mode="shared",
        request_id=None,
    ):
        return self.request(
            "fork",
            {
                "snapshot": snapshot,
                "resources": resources,
                "context_mode": context_mode,
                "prefix_mode": prefix_mode,
            },
            request_id=request_id,
        )

    def checkout(
        self,
        snapshot,
        *,
        resources,
        context_mode,
        prefix_mode="shared",
        request_id=None,
    ):
        return self.request(
            "checkout",
            {
                "snapshot": snapshot,
                "resources": resources,
                "context_mode": context_mode,
                "prefix_mode": prefix_mode,
            },
            request_id=request_id,
        )

    def checkpoint(self, branch, *, request_id=None):
        return self.request("checkpoint", {"branch": branch}, request_id=request_id)

    def merge(
        self,
        sources,
        selection,
        *,
        resources,
        context_mode,
        prefix_mode="shared",
        request_id=None,
    ):
        return self.request(
            "merge",
            {
                "sources": sources,
                "selection": selection,
                "resources": resources,
                "context_mode": context_mode,
                "prefix_mode": prefix_mode,
            },
            request_id=request_id,
        )

    def stop(self, branch, *, request_id=None):
        return self.request("stop", {"branch": branch}, request_id=request_id)

    def status(self):
        return self.request("status", {})

    def branch(self, branch_id):
        return self.request("branch", {"branch": branch_id})

    def operation(self, operation_id):
        return self.request("operation", {"operation": operation_id})

    def history(self, snapshot):
        return self.request("history", {"snapshot": snapshot})

    def snapshot(self, snapshot):
        return self.request("snapshot", {"snapshot": snapshot})

    def finish(self, snapshot):
        return self.request("finish", {"snapshot": snapshot})
