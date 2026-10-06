"""Publish the bundled Agent-facing API for a local deployment.

The contents are an explicit public allowlist.  This deployment helper must not
copy the trusted Core package wholesale into a Tested Agent's environment.
"""

from __future__ import annotations

import shutil
from pathlib import Path


PUBLIC_AGENT_API_FILES = {
    "client.py": "client.py",
    "agent_cli.py": "cli.py",
    "candidate_protocol.py": "candidate_protocol.py",
}


def publish_agent_api_bundle(
    destination: Path,
    *,
    source_package: Path | None = None,
) -> Path:
    """Create one read-only local bundle and return its resolved directory."""

    destination = destination.resolve()
    source = (
        source_package.resolve()
        if source_package is not None
        else Path(__file__).resolve().parent
    )
    package = destination / "envcore"
    package.mkdir(parents=True, exist_ok=False)
    for source_name, destination_name in PUBLIC_AGENT_API_FILES.items():
        source_path = source / source_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Agent API source is missing: {source_path}")
        shutil.copy2(source_path, package / destination_name)
    (package / "__init__.py").write_text(
        "from .client import Client, ClientError\n"
        "__all__ = ['Client', 'ClientError']\n",
        encoding="utf-8",
    )
    # Publish the same restricted surface for existing participant software.
    # Neither namespace may expose the trusted environment implementation.
    legacy = destination / "benchcore"
    shutil.copytree(package, legacy)
    for path in legacy.iterdir():
        path.chmod(0o444)
    legacy.chmod(0o555)
    for path in package.iterdir():
        path.chmod(0o444)
    package.chmod(0o555)
    destination.chmod(0o555)
    return destination
