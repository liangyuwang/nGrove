"""Local public process families, not a real runtime or accelerator acceptance."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from envcore.config import TrainingResources
from envcore.conformance import (
    CommandTimeoutCase,
    CommandTimeoutObservation,
    probe_worker_leases,
)
from envcore.worker_protocol import CommandResult, ResearchTaskSpec
from tests.test_lease_probe import LeaseWorker, contract


@unittest.skipUnless(os.name == "posix", "fixture uses owned POSIX process groups")
class CommandTimeoutProcessTests(unittest.TestCase):
    def test_command_descendants_stop_before_timeout_result_is_accepted(self):
        self.run_fixture(stop_descendants=True)

    def test_stopping_only_exec_client_cannot_hide_live_command_descendants(self):
        self.run_fixture(stop_descendants=False)

    def run_fixture(self, *, stop_descendants):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("workspace", "output", "train", "api"):
                (root / name).mkdir()
            ready = root / "command-started"
            events = []
            observations = []
            # The parent exits after spawning a child. Its unreaped PID anchors
            # the owned group until cleanup, even while descendants keep working.
            script = """
import subprocess, sys
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
Path(sys.argv[1]).write_text(str(child.pid))
"""
            command = (sys.executable, "-c", script, str(ready))

            def start_process(argv, *, new_session=False):
                return subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=new_session,
                )

            def group_states(process):
                if process is None or process.returncode is not None:
                    return []
                inventory = subprocess.run(
                    ["ps", "-axo", "pid=,pgid=,stat="],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=1,
                )
                values = []
                for line in inventory.stdout.splitlines():
                    pid, group, state = line.split()
                    if int(group) == process.pid:
                        values.append((int(pid), state[0].upper()))
                return values

            def live_group(process):
                return any(state != "Z" for _, state in group_states(process))

            class ProcessWorker(LeaseWorker):
                command_process = None
                client = None

                def cleanup(self):
                    if (
                        self.command_process is not None
                        and self.command_process.returncode is None
                    ):
                        # Never reap the leader before signalling its group: its
                        # PID could otherwise be reused for unrelated work.
                        if live_group(self.command_process):
                            try:
                                os.killpg(self.command_process.pid, signal.SIGKILL)
                            except (ProcessLookupError, PermissionError):
                                # A group containing only an exited leader can
                                # reject signals; actual absence is decisive.
                                if live_group(self.command_process):
                                    raise
                        until = time.monotonic() + 5
                        while live_group(self.command_process):
                            if time.monotonic() >= until:
                                raise RuntimeError("fixture descendants did not stop")
                            time.sleep(0.01)
                        self.command_process.wait(timeout=5)
                    if self.client is not None:
                        if self.client.poll() is None:
                            self.client.kill()
                        self.client.wait(timeout=5)
                    return True

                def exec_lease(self, **kwargs):
                    if tuple(kwargs["argv"]) != command:
                        return super().exec_lease(**kwargs)
                    self.commands.append(kwargs)
                    until = time.monotonic() + kwargs["timeout_seconds"]
                    self.command_process = start_process(command, new_session=True)
                    self.client = start_process(
                        [sys.executable, "-c", "import time; time.sleep(60)"]
                    )
                    while not ready.exists() or not any(
                        pid == self.command_process.pid and state == "Z"
                        for pid, state in group_states(self.command_process)
                    ):
                        if time.monotonic() >= until:
                            raise RuntimeError(
                                "public fixture did not start before timeout"
                            )
                        time.sleep(0.01)
                    binding = {key: kwargs[key] for key in ("episode_id", "lease_id")}
                    binding.update(
                        worker_id=self.describe_capabilities().worker_id,
                        command_sha256=hashlib.sha256(
                            json.dumps(command, separators=(",", ":")).encode()
                        ).hexdigest(),
                    )
                    events.append(("start", binding))
                    try:
                        self.client.wait(timeout=max(0.001, until - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        events.append(("command_timeout", binding))
                        if stop_descendants:
                            os.killpg(self.command_process.pid, signal.SIGKILL)
                            cleanup_until = time.monotonic() + 1
                            while live_group(self.command_process):
                                if time.monotonic() >= cleanup_until:
                                    raise RuntimeError(
                                        "command descendants survived timeout"
                                    )
                                time.sleep(0.01)
                        self.client.kill()
                        self.client.wait(timeout=1)
                        return CommandResult(
                            124, "", "fixture command timed out", timed_out=True
                        )
                    raise AssertionError(
                        "fixture exec client should not finish naturally"
                    )

                def terminate_lease(self, **kwargs):
                    self.cleanup()
                    super().terminate_lease(**kwargs)

            worker = ProcessWorker()

            def observe(**kwargs):
                states = group_states(worker.command_process)
                running = any(state != "Z" for _, state in states)
                observations.append((states, len(worker.stops)))
                binding = {
                    key: kwargs[key]
                    for key in ("episode_id", "lease_id", "worker_id", "command_sha256")
                }
                for _, actual in events:
                    self.assertEqual(actual, binding)
                return CommandTimeoutObservation(
                    **binding,
                    observation_id=kwargs["observation_id"],
                    source="local-process-family-observer",
                    running=running,
                    start_evidence_id="local-start-event"
                    if any(kind == "start" for kind, _ in events)
                    else None,
                    timeout_evidence_id="local-command-timeout-event"
                    if any(kind == "command_timeout" for kind, _ in events)
                    else None,
                )

            try:
                report = probe_worker_leases(
                    worker,
                    task=ResearchTaskSpec(
                        "public-fixture", "fixture:image", root / "train", root / "api"
                    ),
                    workspace=root / "workspace",
                    output_root=root / "output",
                    resources=TrainingResources(),
                    resource_envelope=contract(),
                    rejected_resource_envelope=contract(2),
                    command=("normal-fixture",),
                    expected_stdout="fixture marker\n",
                    command_timeout_case=CommandTimeoutCase(
                        command, observe, timeout_seconds=1, termination_grace_seconds=2
                    ),
                    teardown=worker.cleanup,
                )
                self.assertEqual(
                    report.passed, stop_descendants, report.private_manifest()
                )
                self.assertEqual(
                    report.leases[0]["command_timeout"]["passed"], stop_descendants
                )
                self.assertEqual([count for _, count in observations], [0, 0])
                self.assertTrue(report.leases[0]["termination_confirmed"])
                self.assertIsNotNone(worker.command_process.returncode)
                self.assertIsNotNone(worker.client.returncode)
                if not stop_descendants:
                    self.assertIn(
                        "command_workloads_still_running",
                        {check.code for check in report.failed},
                    )
                    states = observations[-1][0]
                    self.assertIn((worker.command_process.pid, "Z"), states)
                    self.assertTrue(
                        any(
                            pid != worker.command_process.pid and state != "Z"
                            for pid, state in states
                        )
                    )
            finally:
                worker.cleanup()


if __name__ == "__main__":
    unittest.main()
