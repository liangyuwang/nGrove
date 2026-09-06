"""Fixed trusted process fixtures; never a Tested Agent."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autotrain_bench import local_watchdog
from autotrain_bench.docker_recovery import DockerRecovery


def main():
    if sys.argv[1] == "output-parent":
        mode = sys.argv[2]
        code = local_watchdog.supervise(
            lambda fd: [sys.executable, __file__, str(fd), "unused", mode]
        )
        raise SystemExit(code)
    fd, directory, mode = sys.argv[1:]
    if mode.startswith("output-child"):
        from autotrain_bench import cli

        def output(argv):
            cli._print({"status": "closed", "TEST": "saved"})
            print("fixture diagnostic", file=sys.stderr, end="")

        with patch.object(cli, "main", output):
            if mode == "output-child-flush-error":

                class BrokenOutput:
                    def write(self, data):
                        return sys.__stdout__.write(data)

                    def flush(self):
                        raise OSError("fixture output flush failed")

                with patch.object(sys, "stdout", BrokenOutput()):
                    local_watchdog.worker_main()
            local_watchdog.worker_main()
        return
    root = Path(directory)
    client = local_watchdog.WatchdogClient(int(fd))
    local_watchdog._client = client
    journal = DockerRecovery(root / "worker", episode=root / "episode")
    client.register(journal, "fixture-docker")
    if mode == "paused":
        os.kill(os.getpid(), signal.SIGSTOP)
    if mode == "killed":
        os.kill(os.getpid(), signal.SIGKILL)
    if mode == "inherited_lock":
        subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            pass_fds=(journal.lock_fd,),
        )
        os.kill(os.getpid(), signal.SIGKILL)
    if mode in {"deadline", "research", "lease"}:
        if mode == "deadline":

            @local_watchdog.watched()
            def stuck(*, timeout_seconds):
                time.sleep(60)  # Heartbeat keeps running while this call is stuck.

            stuck(timeout_seconds=0.05)
        else:
            client.arm(mode, 0.05, mode)
            time.sleep(60)
    if mode == "disarmed":
        client.arm("research", 0.05, "research")
        client.disarm("research")
        time.sleep(0.3)
    if mode == "late_disarm":
        client.arm("research", 0.01, "research")
        time.sleep(0.5)
        client.disarm("research")
    journal.close()
    client._request({"type": "finished", "exit_code": 0 if mode != "failed" else 1})
    while True:
        signal.pause()


if __name__ == "__main__":
    main()
