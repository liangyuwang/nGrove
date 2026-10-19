"""Trusted ingress and allowlisted publication of the thin Agent client."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .contracts import BranchError
from .session import BranchSession


class ControlEndpoint:
    """Bind one authenticated branch at the transport boundary.

    This is not authentication by itself. A deployment binds the instance to a
    sandbox identity, revokes it at teardown, and must never accept actor identity
    or database paths from an Agent request.
    """

    def __init__(self, session: BranchSession, actor_branch: str):
        session.branch_status(actor_branch)
        self.session = session
        self.actor_branch = actor_branch
        self.revoked = False

    def revoke(self):
        self.revoked = True

    def __call__(self, request: dict) -> dict:
        try:
            if self.revoked:
                raise BranchError("control_revoked")
            if not isinstance(request, dict) or set(request) != {"id", "op", "args"}:
                raise BranchError("invalid_request")
            op, args = request["op"], request["args"]
            if op == "status" and args == {}:
                result = self.session.status()
            elif op == "branch" and isinstance(args, dict) and set(args) == {"branch"}:
                result = self.session.branch_status(args["branch"])
            elif (
                op == "operation"
                and isinstance(args, dict)
                and set(args) == {"operation"}
            ):
                result = self.session.operation_status(args["operation"])
            elif (
                op == "history" and isinstance(args, dict) and set(args) == {"snapshot"}
            ):
                result = self.session.history(args["snapshot"])
            elif (
                op == "snapshot"
                and isinstance(args, dict)
                and set(args) == {"snapshot"}
            ):
                result = self.session.snapshot(args["snapshot"])
            elif (
                op == "finish" and isinstance(args, dict) and set(args) == {"snapshot"}
            ):
                self.session.request_seal(
                    self.actor_branch, final_snapshot=args["snapshot"]
                )
                result = {"status": "closing"}
            else:
                result = self.session.submit(self.actor_branch, request)
            return {"ok": True, "result": result}
        except BranchError as exc:
            return {"ok": False, "error": exc.code}
        except Exception:
            # Infrastructure diagnostics stay in trusted service logs. A caller
            # must retry the SAME request ID after an ambiguous transport failure.
            logging.getLogger(__name__).exception("Branch control service failed")
            return {"ok": False, "error": "branch_service_unavailable"}


def publish_client_bundle(destination: Path) -> Path:
    """Publish only the client, without importing trusted package initializers."""
    destination = Path(destination)
    package = destination / "ngrove" / "branching"
    package.mkdir(parents=True, exist_ok=False)
    for directory in (package.parent, package):
        path = directory / "__init__.py"
        path.write_text('"""Public branching client bundle."""\n', encoding="utf-8")
        path.chmod(0o444)
    shutil.copyfile(Path(__file__).with_name("client.py"), package / "client.py")
    (package / "client.py").chmod(0o444)
    return destination
