"""Codex CLI adapter for the generic host_cli Agent runner."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Mapping, Optional

from ..agent_prompt import HOST_CLI_PROMPT


def build_codex_command(
    *,
    codex_binary: str,
    model: str,
    reasoning_effort: str,
    workspace: Path,
    control_dir: Path,
    shell_environment: Mapping[str, str],
) -> List[str]:
    command = [
        codex_binary,
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--sandbox",
        "workspace-write",
        "--add-dir",
        str(control_dir.resolve()),
        "--model",
        model,
        "-c",
        f"model_reasoning_effort={json.dumps(reasoning_effort)}",
        "-c",
        'web_search="disabled"',
        "-c",
        "sandbox_workspace_write.network_access=false",
        "-c",
        'shell_environment_policy.inherit="none"',
        "-c",
        "shell_environment_policy.experimental_use_profile=false",
        "--cd",
        str(workspace.resolve()),
        "--color",
        "never",
        "--json",
    ]
    controlled_environment = {
        "AUTOTRAIN_BENCH_CONTROL_DIR": shell_environment["AUTOTRAIN_BENCH_CONTROL_DIR"],
        "AUTOTRAIN_BENCH_HOST_RUNNER": "1",
        "AUTOTRAIN_BENCH_TASK": shell_environment["AUTOTRAIN_BENCH_TASK"],
        "AUTOTRAIN_BENCH_TRAIN_DIR": "/benchmark/train",
        "AUTOTRAIN_BENCH_TRAIN_WORKSPACE": "/workspace",
        "AUTOTRAIN_BENCH_WORKSPACE": str(workspace.resolve()),
        "HOME": str(workspace.resolve()),
        "PATH": shell_environment.get("PATH", os.defpath),
        "PYTHONPATH": shell_environment["PYTHONPATH"],
    }
    try:
        forwarded_names = json.loads(
            shell_environment.get("AUTOTRAIN_BENCH_FORWARDED_ENVIRONMENT", "[]")
        )
    except json.JSONDecodeError:
        forwarded_names = []
    if isinstance(forwarded_names, list):
        for name in forwarded_names:
            if isinstance(name, str) and name in shell_environment:
                controlled_environment[name] = shell_environment[name]
    for name, value in controlled_environment.items():
        command.extend(
            [
                "-c",
                f"shell_environment_policy.set.{name}={json.dumps(value)}",
            ]
        )
    command.append(HOST_CLI_PROMPT)
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="run a logged-in Codex CLI as an AutoTrain-Bench host Agent"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--codex-binary", default="codex")
    return parser


def main(
    argv: Optional[List[str]] = None,
    *,
    environment: Optional[Mapping[str, str]] = None,
) -> None:
    args = build_parser().parse_args(argv)
    env = dict(environment or os.environ)
    workspace_value = env.get("AUTOTRAIN_BENCH_WORKSPACE")
    control_value = env.get("AUTOTRAIN_BENCH_CONTROL_DIR")
    task_value = env.get("AUTOTRAIN_BENCH_TASK")
    if not workspace_value or not control_value or not task_value:
        raise SystemExit(
            "Codex CLI adapter must be launched by the AutoTrain-Bench "
            "host_cli runner"
        )

    codex_binary = shutil.which(args.codex_binary, path=env.get("PATH"))
    if not codex_binary:
        raise SystemExit(f"Codex CLI was not found: {args.codex_binary}")
    try:
        version = subprocess.run(
            [codex_binary, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env=env,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        version = "unknown"

    metadata = {
        "type": "benchcore.agent_metadata",
        "tool": "codex",
        "tool_version": version or "unknown",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
    }
    print(json.dumps(metadata, sort_keys=True), flush=True)
    command = build_codex_command(
        codex_binary=codex_binary,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        workspace=Path(workspace_value),
        control_dir=Path(control_value),
        shell_environment=env,
    )
    os.execvpe(codex_binary, command, env)


if __name__ == "__main__":
    main()
