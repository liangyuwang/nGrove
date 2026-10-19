"""Minimal CLI surface published into untrusted Agent and Training containers."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

from .client import Client, ClientError


def _print(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ngrove")
    commands = parser.add_subparsers(dest="command", required=True)
    agent = commands.add_parser("agent")
    operations = agent.add_subparsers(dest="operation", required=True)
    operations.add_parser("status")
    lease_start = operations.add_parser("lease-start")
    lease_start.add_argument("--seconds", type=int)
    lease_exec = operations.add_parser("lease-exec")
    lease_exec.add_argument("--timeout", type=float)
    lease_exec.add_argument("argv", nargs=argparse.REMAINDER)
    lease_logs = operations.add_parser("lease-logs")
    lease_logs.add_argument("--limit", type=int, default=50)
    lease_commit = operations.add_parser("lease-commit")
    lease_commit.add_argument("candidate")
    incumbent = operations.add_parser("declare-incumbent")
    incumbent.add_argument("candidate_id")
    operations.add_parser("lease-stop")
    operations.add_parser("finish")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    client = Client()
    if args.operation == "status":
        result = client.status()
    elif args.operation == "lease-start":
        result = client.start_lease(args.seconds)
    elif args.operation == "lease-exec":
        command = list(args.argv)
        if command and command[0] == "--":
            command = command[1:]
        result = client.exec(command, timeout_seconds=args.timeout)
    elif args.operation == "lease-logs":
        result = client.logs(args.limit)
    elif args.operation == "lease-commit":
        result = client.commit(args.candidate)
    elif args.operation == "declare-incumbent":
        result = client.declare_incumbent(args.candidate_id)
    elif args.operation == "lease-stop":
        result = client.stop_lease()
    elif args.operation == "finish":
        result = client.finish()
    else:  # pragma: no cover - argparse prevents this
        raise ClientError(f"unknown operation: {args.operation}")
    _print(result)


if __name__ == "__main__":
    main()
