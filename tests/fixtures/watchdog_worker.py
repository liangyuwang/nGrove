"""Fixed trusted process fixtures; never a Tested Agent."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envcore import local_watchdog
from envcore.docker_recovery import DockerRecovery, _RecoveryDeadline
from envcore.local_recovery import _acquire_journal


def signal_parent(directory, phase, signum):
    """Exercise real parent signals with disposable processes and a fake daemon."""
    from tests.test_docker_recovery import FakeDocker

    root = Path(directory)
    children = []
    popen = subprocess.Popen
    killpg = os.killpg
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def launch(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        if phase == "launch":
            # A signal after creation but before Popen returns must not lose
            # the process handle needed to fence the dedicated group.
            os.kill(os.getpid(), signum)
        return child

    def fence(pid, termination_signal):
        if phase == "fence":
            os.kill(os.getpid(), signum)
        killpg(pid, termination_signal)

    def recover(registration, *, timeout_seconds):
        deadline = _RecoveryDeadline(timeout_seconds)
        with_journal = _acquire_journal(Path(registration["state_dir"]), deadline)
        try:
            if phase == "recovery":
                os.kill(os.getpid(), signum)
                os.kill(os.getpid(), signum)
            report = with_journal._recover_with_deadline(FakeDocker().run, deadline)
            (root / "recovered").write_text(str(report["cleanup_complete"]))
            return report
        finally:
            with_journal.close()

    try:
        with patch.object(subprocess, "Popen", launch), patch.object(
            os, "killpg", fence
        ):
            code = local_watchdog.supervise(
                lambda fd: [
                    sys.executable,
                    __file__,
                    str(fd),
                    str(root),
                    "signal-normal"
                    if phase == "recovery"
                    else "signal-child-" + str(signum),
                ],
                recover=recover,
            )
        # Observe before fixture safety cleanup; cleanup cannot make a broken
        # supervise() implementation appear to have reaped its own child.
        (root / "child-reaped").write_text(str(children[0].returncode is not None))
        (root / "handlers-restored").write_text(
            str(
                all(
                    signal.getsignal(sig) == handler
                    for sig, handler in previous.items()
                )
            )
        )
        return code
    finally:
        for child in children:
            if child.returncode is None:
                # Only this fixture's own unreaped child identity is used.
                try:
                    killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=5)


def main():
    if sys.argv[1] == "signal-parent":
        raise SystemExit(signal_parent(sys.argv[2], sys.argv[3], int(sys.argv[4])))
    if sys.argv[1] == "output-parent":
        mode = sys.argv[2]
        code = local_watchdog.supervise(
            lambda fd: [sys.executable, __file__, str(fd), "unused", mode]
        )
        raise SystemExit(code)
    fd, directory, mode = sys.argv[1:]
    if mode.startswith("output-child"):
        from envcore import cli

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
    if mode == "signal-normal" or mode.startswith("signal-child-"):
        private = journal.private_directory("signal-fixture-")
        (private / "disposable-input").write_text("fixture data, no hidden dataset")
    if mode.startswith("signal-child-"):
        os.kill(os.getppid(), int(mode.removeprefix("signal-child-")))
        time.sleep(60)
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
