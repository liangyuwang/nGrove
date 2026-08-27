"""Command-line interface for users and containerized agents."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .client import Client, ClientError
from .config import ConfigError, TaskConfig
from .runtime import DockerRuntime, RuntimeFailure
from .supervisor import Supervisor


def _print(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _episode_directory(root: Path) -> Path:
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (root / f"{timestamp}-{uuid.uuid4().hex[:8]}").resolve()


def command_prepare(args: argparse.Namespace) -> int:
    config = TaskConfig.load(args.task)
    runtime = DockerRuntime()
    runtime.prepare(config, pull_missing=not args.no_pull)
    _print(
        {
            "status": "ready",
            "task_id": config.task_id,
            "images": config.images,
            "gpu_count": config.gpu_count,
            "gpu_model": config.gpu_model,
        }
    )
    return 0


def command_start(args: argparse.Namespace) -> int:
    config = TaskConfig.load(args.task)
    if args.agent_image or args.agent_command_override:
        command = (
            shlex.split(args.agent_command_override)
            if args.agent_command_override
            else config.agent_command
        )
        if not command:
            raise ConfigError("--agent-command must not be empty")
        config = replace(
            config,
            agent_image=args.agent_image or config.agent_image,
            agent_command=command,
        )
    runtime = DockerRuntime()
    runtime.prepare(config, pull_missing=False)
    episode_dir = (
        args.episode_dir.resolve()
        if args.episode_dir
        else _episode_directory(args.episodes_root)
    )
    episode_dir.mkdir(parents=True, exist_ok=False)
    supervisor = Supervisor(
        config=config,
        track=args.track,
        episode_dir=episode_dir,
        runtime=runtime,
    )
    supervisor.begin()
    agent_name = f"atb-{episode_dir.name[:12].lower()}-agent"
    print(f"Episode: {episode_dir}", flush=True)
    try:
        runtime.start_agent(
            name=agent_name,
            config=config,
            control_dir=supervisor.control_dir,
            public_task_path=supervisor.public_task_path,
        )
    except RuntimeFailure:
        supervisor.agent_exited_without_finish()
        raise

    finalized_at: Optional[float] = None
    try:
        while True:
            processed = supervisor.process_one_request()
            supervisor.enforce_deadlines()
            status = supervisor.public_status()["status"]
            agent_running = runtime.container_running(agent_name)

            if status == "active" and not agent_running:
                supervisor.agent_exited_without_finish()
                status = "dnf"

            if status != "active":
                if finalized_at is None:
                    finalized_at = time.monotonic()
                if not agent_running or time.monotonic() - finalized_at >= 3:
                    break
            if not processed:
                time.sleep(0.05)
    except KeyboardInterrupt:
        supervisor.agent_exited_without_finish()
    finally:
        active = supervisor.state.get("active_lease")
        if active:
            runtime.stop_container(active["container_name"])
        runtime.stop_container(agent_name)

    _print(supervisor.public_status())
    return 0 if supervisor.state["status"] == "finalized" else 1


def command_report(args: argparse.Namespace) -> int:
    state_path = args.episode.resolve() / "state.json"
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"episode state not found: {state_path}") from exc
    _print(payload)
    return 0


def command_agent(args: argparse.Namespace) -> int:
    client = Client()
    if args.agent_command == "status":
        result = client.status()
    elif args.agent_command == "lease-start":
        result = client.start_lease(args.seconds)
    elif args.agent_command == "lease-exec":
        argv = list(args.argv)
        if argv and argv[0] == "--":
            argv = argv[1:]
        result = client.exec(argv, timeout_seconds=args.timeout)
    elif args.agent_command == "lease-logs":
        result = client.logs(args.limit)
    elif args.agent_command == "lease-commit":
        result = client.commit(args.candidate)
    elif args.agent_command == "lease-stop":
        result = client.stop_lease()
    elif args.agent_command == "finish":
        result = client.finish(args.candidate_id)
    else:  # pragma: no cover - argparse prevents this
        raise ClientError(f"unknown agent command: {args.agent_command}")
    _print(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autotrain-bench",
        description="AutoTrain-Bench local reference prototype",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="validate and prepare a task")
    prepare.add_argument("--task", required=True, type=Path)
    prepare.add_argument(
        "--no-pull", action="store_true", help="fail instead of pulling missing images"
    )
    prepare.set_defaults(function=command_prepare)

    start = subparsers.add_parser("start", help="start and supervise an episode")
    start.add_argument("--task", required=True, type=Path)
    start.add_argument("--track", required=True, choices=["A", "B", "a", "b"])
    start.add_argument("--episode-dir", type=Path)
    start.add_argument(
        "--agent-image",
        help="override the registered Agent Container image for this episode",
    )
    start.add_argument(
        "--agent-command",
        dest="agent_command_override",
        help="override the command executed in the Agent Container",
    )
    start.add_argument(
        "--episodes-root", type=Path, default=Path(".autotrain/episodes")
    )
    start.set_defaults(function=command_start)

    report = subparsers.add_parser("report", help="print a completed episode report")
    report.add_argument("--episode", required=True, type=Path)
    report.set_defaults(function=command_report)

    agent = subparsers.add_parser(
        "agent", help="commands used from inside the Agent Container"
    )
    agent_subparsers = agent.add_subparsers(dest="agent_command", required=True)

    agent_status = agent_subparsers.add_parser("status")
    agent_status.set_defaults(function=command_agent)

    lease_start = agent_subparsers.add_parser("lease-start")
    lease_start.add_argument("--seconds", type=int)
    lease_start.set_defaults(function=command_agent)

    lease_exec = agent_subparsers.add_parser("lease-exec")
    lease_exec.add_argument("--timeout", type=float)
    lease_exec.add_argument("argv", nargs=argparse.REMAINDER)
    lease_exec.set_defaults(function=command_agent)

    lease_logs = agent_subparsers.add_parser("lease-logs")
    lease_logs.add_argument("--limit", type=int, default=50)
    lease_logs.set_defaults(function=command_agent)

    lease_commit = agent_subparsers.add_parser("lease-commit")
    lease_commit.add_argument("candidate")
    lease_commit.set_defaults(function=command_agent)

    lease_stop = agent_subparsers.add_parser("lease-stop")
    lease_stop.set_defaults(function=command_agent)

    finish = agent_subparsers.add_parser("finish")
    finish.add_argument("--candidate-id")
    finish.set_defaults(function=command_agent)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "track", None):
        args.track = args.track.upper()
    try:
        return_code = args.function(args)
    except (ClientError, ConfigError, RuntimeFailure, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return_code = 2
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
