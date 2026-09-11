"""Disposable recovery helper faults; no real Docker workloads or hidden data."""

import json
import os
import signal
import sys
import threading
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchcore import local_recovery
from benchcore.docker_recovery import DockerRecovery
from tests.test_docker_recovery import FakeDocker


def main():
    fd, directory, mode = sys.argv[1:]
    root = Path(directory)

    def block(*args, **kwargs):
        (root / "blocked-stage").write_text(mode)
        if mode == "signals":
            os.kill(os.getppid(), signal.SIGTERM)
            os.kill(os.getppid(), signal.SIGINT)
        threading.Event().wait()

    def perform(request):
        if mode == "crash":
            os._exit(7)
        if mode == "bad-result":
            return {"cleanup_complete": True}
        if mode == "huge":
            return {"oversized": "x" * (2 * local_recovery.MAX_FRAME_BYTES)}
        backend = FakeDocker()
        inventory = root / "backend.json"
        if inventory.exists():
            backend.objects = {
                (item["kind"], item["name"]): item["object"]
                for item in json.loads(inventory.read_text())
            }

        def run(args, **kwargs):
            result = backend.run(args, **kwargs)
            inventory.write_text(
                json.dumps(
                    [
                        {"kind": kind, "name": name, "object": obj}
                        for (kind, name), obj in backend.objects.items()
                    ]
                )
            )
            return result

        original_close = DockerRecovery.close

        def close(journal):
            if mode == "close-block" and journal.lock_fd >= 0:
                block()
            original_close(journal)
            if mode == "complete-signals":
                os.kill(os.getppid(), signal.SIGTERM)
                os.kill(os.getppid(), signal.SIGINT)

        from contextlib import ExitStack

        with ExitStack() as stack:
            stack.enter_context(
                patch("benchcore.runtime.DockerRuntime._run_raw", side_effect=run)
            )
            if mode in {"open-block", "signals"}:
                stack.enter_context(patch.object(DockerRecovery, "_read", block))
            if mode == "save-block":
                stack.enter_context(patch.object(DockerRecovery, "_save", block))
            if mode == "delete-block":
                stack.enter_context(
                    patch("benchcore.docker_recovery.shutil.rmtree", block)
                )
            if mode in {"close-block", "complete-signals"}:
                stack.enter_context(patch.object(DockerRecovery, "close", close))
            if mode == "report-block":
                stack.enter_context(
                    patch.object(local_recovery, "_persist_watchdog_report", block)
                )
            return local_recovery._perform(request)

    local_recovery.worker_main(int(fd), perform=perform)


if __name__ == "__main__":
    main()
