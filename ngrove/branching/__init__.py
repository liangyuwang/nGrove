"""Optional research branching layer; Core episode protocols stay independent."""

from .client import BranchClient
from .contracts import BranchError, RunnerCapabilities, SessionPolicy
from .session import BranchSession

__all__ = [
    "BranchClient",
    "BranchError",
    "BranchSession",
    "RunnerCapabilities",
    "SessionPolicy",
]
