"""Trusted Operator CLI for the optional branching module.

Do not expose these commands, database paths or monitor tokens to a target Agent.
Publish BranchClient and bind ControlEndpoint through a sandboxed transport instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import BranchError, RunnerCapabilities, SessionPolicy
from .session import BranchSession


def _json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def command(args):
    action = args.branch_action
    if action == "create":
        config = _json(args.config)
        if set(config) != {
            "policy",
            "runner",
            "initial_state",
            "resources",
            "context_mode",
        }:
            raise BranchError("invalid_session_configuration")
        session = BranchSession.create(
            args.database,
            policy=SessionPolicy(**config["policy"]),
            capabilities=RunnerCapabilities(**config["runner"]),
            initial_state=config["initial_state"],
            resources=config["resources"],
            context_mode=config["context_mode"],
        )
        result = session.status()
    else:
        session = BranchSession(args.database)
        if action == "status":
            result = session.status(private=args.private)
        elif action == "history":
            result = session.history(args.snapshot, include_state=args.private)
        elif action == "snapshot":
            result = session.snapshot(args.snapshot, include_state=args.private)
        elif action == "events":
            result = session.events()
        elif action == "request":
            result = session.submit(args.actor, _json(args.request))
        elif action == "claim":
            result = session.claim(args.monitor)
        elif action == "recover":
            result = session.recover(
                args.operation, monitor_id=args.monitor, fencing_evidence=args.evidence
            )
        elif action == "seal":
            session.request_seal(args.actor, final_snapshot=args.snapshot)
            result = session.status()
        elif action == "confirm-sealed":
            session.confirm_sealed(control_revoked=True, evidence_id=args.evidence)
            result = session.status()
        elif action == "receipt":
            body = _json(args.body)
            methods = {
                "allocated": session.allocated,
                "quiesced": session.quiesced,
                "released": session.released,
                "complete": session.complete,
                "uncertain": session.uncertain,
                "fail-clean": session.fail_clean,
            }
            if args.kind == "ready":
                result = session.ready(args.operation, args.token, body)
            else:
                result = methods[args.kind](args.operation, args.token, **body)
        else:
            raise BranchError("unknown_command")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


def configure(parser):
    sub = parser.add_subparsers(dest="branch_action", required=True)
    for action in (
        "create",
        "status",
        "history",
        "snapshot",
        "events",
        "request",
        "claim",
        "recover",
        "receipt",
        "seal",
        "confirm-sealed",
    ):
        p = sub.add_parser(action)
        p.add_argument("--database", required=True, type=Path)
        p.set_defaults(function=command)
        if action == "create":
            p.add_argument("--config", required=True, type=Path)
        if action in {"status", "history", "snapshot"}:
            p.add_argument("--private", action="store_true")
        if action in {"history", "snapshot", "seal"}:
            p.add_argument("--snapshot", required=True)
        if action in {"request", "seal"}:
            p.add_argument("--actor", required=True)
        if action == "request":
            p.add_argument("--request", required=True, type=Path)
        if action in {"claim", "recover"}:
            p.add_argument("--monitor", required=True)
        if action in {"recover", "receipt"}:
            p.add_argument("--operation", required=True)
        if action in {"recover", "confirm-sealed"}:
            p.add_argument("--evidence", required=True)
        if action == "receipt":
            p.add_argument("--token", required=True)
            p.add_argument(
                "--kind",
                required=True,
                choices=[
                    "allocated",
                    "ready",
                    "quiesced",
                    "released",
                    "complete",
                    "uncertain",
                    "fail-clean",
                ],
            )
            p.add_argument("--body", required=True, type=Path)


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "branches", help="manage research snapshots, branches and monitor requests"
    )
    configure(parser)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ngrove.branching")
    configure(parser)
    args = parser.parse_args(argv)
    try:
        return command(args)
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
