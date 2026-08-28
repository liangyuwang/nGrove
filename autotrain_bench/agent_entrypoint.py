"""PID-1 wrapper that runs the local inference proxy beside the user agent."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from typing import Optional


def main() -> None:
    raw = os.environ.get("AUTOTRAIN_BENCH_AGENT_COMMAND_JSON")
    if not raw:
        raise SystemExit("AUTOTRAIN_BENCH_AGENT_COMMAND_JSON is missing")
    command = json.loads(raw)
    if not isinstance(command, list) or not command or not all(
        isinstance(item, str) and item for item in command
    ):
        raise SystemExit("invalid agent command")

    gateway: Optional[subprocess.Popen] = None
    if os.environ.get("AUTOTRAIN_BENCH_INFERENCE_GATEWAY") == "1":
        gateway = subprocess.Popen(
            [sys.executable, "-m", "autotrain_bench.agent_gateway"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # The gateway is local-only and normally starts in a few milliseconds.
        time.sleep(0.05)
        if gateway.poll() is not None:
            raise SystemExit("failed to start AutoTrain-Bench inference gateway")

    child = subprocess.Popen(command)

    def forward_signal(signum, _frame):
        if child.poll() is None:
            child.send_signal(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, forward_signal)

    try:
        return_code = child.wait()
    finally:
        if gateway is not None and gateway.poll() is None:
            gateway.terminate()
            try:
                gateway.wait(timeout=2)
            except subprocess.TimeoutExpired:
                gateway.kill()
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
