import io
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from benchcore.docker_recovery import DockerRecovery, _RecoveryDeadline
from benchcore.local_recovery import _acquire_journal
from benchcore.local_watchdog import supervise, watched
from benchcore.worker_protocol import WorkerFailure
from tests.test_docker_recovery import FakeDocker


FIXTURE = Path(__file__).parent / "fixtures/watchdog_worker.py"


@unittest.skipUnless(os.name == "posix", "watchdog uses a private POSIX process group")
class WatchdogProcessTests(unittest.TestCase):
    def test_parent_signals_do_not_interrupt_fencing_or_recovery(self):
        for phase in ("launch", "active", "fence", "recovery"):
            for signum in (signal.SIGTERM, signal.SIGINT):
                with self.subTest(phase=phase, signal=signum):
                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        episode = root / "episode"
                        episode.mkdir()
                        state = episode / "state.json"
                        saved = (
                            '{"research_sealed":true,"TEST":"saved-fixture-decision"}'
                        )
                        state.write_text(saved)
                        result = subprocess.run(
                            [
                                sys.executable,
                                str(FIXTURE),
                                "signal-parent",
                                str(root),
                                phase,
                                str(int(signum)),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=15,
                        )
                        self.assertEqual(result.returncode, 2, result.stderr)
                        self.assertEqual((root / "child-reaped").read_text(), "True")
                        self.assertEqual(
                            (root / "handlers-restored").read_text(), "True"
                        )
                        self.assertEqual(state.read_text(), saved)
                        if phase == "launch":
                            continue  # No journal has been registered yet.
                        self.assertEqual((root / "recovered").read_text(), "True")
                        report = json.loads(
                            (root / "worker/watchdog-report.json").read_text()
                        )
                        self.assertEqual(report["reason"], "operator_interrupted")
                        self.assertEqual(report["operator_signal"], signum)
                        self.assertTrue(report["cleanup"]["cleanup_complete"])
                        self.assertFalse((root / "worker/private").exists())
                        self.assertEqual(report["worker_exit_code"], -signal.SIGKILL)
                        self.assertEqual(
                            report["completed_exit_code"],
                            0 if phase == "recovery" else None,
                        )
                        self.assertEqual(report["failures"], [])
                        self.assertFalse(report["resumed"])

    def test_completed_worker_flushes_final_json_to_a_pipe(self):
        environment = dict(os.environ)
        environment.pop("PYTHONUNBUFFERED", None)
        result = subprocess.run(
            [sys.executable, str(FIXTURE), "output-parent", "output-child"],
            capture_output=True,
            text=True,
            timeout=15,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout), {"status": "closed", "TEST": "saved"}
        )
        self.assertIn("fixture diagnostic", result.stderr)

    def test_output_flush_failure_is_not_reported_as_success(self):
        result = subprocess.run(
            [sys.executable, str(FIXTURE), "output-parent", "output-child-flush-error"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        # Both streams are attempted even when stdout cannot be flushed.
        self.assertIn("fixture diagnostic", result.stderr)

    def run_fixture(
        self, mode, *, failure=None, recovery_seconds=120, recovery_delay=0
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episode"
            episode.mkdir()
            state = episode / "state.json"
            saved = '{"research_sealed":true,"TEST":"saved-fixture-decision"}'
            state.write_text(saved)
            recovered = []

            def recover(registration, *, timeout_seconds):
                # This real lock acquisition also proves that all inherited
                # creation-client ownership descriptors have been released.
                deadline = _RecoveryDeadline(timeout_seconds)
                journal = _acquire_journal(Path(registration["state_dir"]), deadline)
                try:
                    self.assertEqual(journal.state["owner"], registration["owner"])
                    recovered.append(True)
                    if failure is not None:
                        raise WorkerFailure(failure)
                    now = [time.monotonic()]
                    backend = FakeDocker()

                    def run(args, **kwargs):
                        now[0] += recovery_delay
                        return backend.run(args, **kwargs)

                    with patch(
                        "benchcore.docker_recovery.time.monotonic",
                        side_effect=lambda: now[0],
                    ):
                        return journal._recover_with_deadline(run, deadline)
                finally:
                    journal.close()

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = supervise(
                    lambda fd: [sys.executable, str(FIXTURE), str(fd), str(root), mode],
                    heartbeat_timeout=0.8,
                    grace_seconds=0.1,
                    recovery_seconds=recovery_seconds,
                    recover=recover,
                )
            report_path = root / "worker/watchdog-report.json"
            self.assertTrue(report_path.exists(), stderr.getvalue())
            report = json.loads(report_path.read_text())
            diagnostic = f"report={report!r}\nstderr={stderr.getvalue()}"
            self.assertEqual(state.read_text(), saved, diagnostic)
            self.assertEqual(recovered, [True], diagnostic)
            self.assertFalse(report["resumed"], diagnostic)
            return code, report

    def test_normal_completion_and_cooperative_failure_keep_exit_status(self):
        for mode, expected in (("normal", 0), ("failed", 1), ("disarmed", 0)):
            with self.subTest(mode=mode):
                code, report = self.run_fixture(mode)
                self.assertEqual(code, expected)
                self.assertEqual(report["reason"], "worker_completed")
                self.assertEqual(report["completed_exit_code"], expected)
                self.assertTrue(report["cleanup"]["cleanup_complete"])

    def test_paused_or_killed_worker_is_fenced_and_recovered(self):
        for mode, reason in (
            ("paused", "worker_unresponsive"),
            ("killed", "worker_exited"),
            ("inherited_lock", "worker_exited"),
        ):
            with self.subTest(mode=mode):
                code, report = self.run_fixture(mode)
                self.assertEqual(code, 2)
                self.assertEqual(report["reason"], reason)
                self.assertEqual(report["worker_exit_code"], -signal.SIGKILL)
                self.assertTrue(report["cleanup"]["cleanup_complete"])

    def test_operation_lease_and_private_research_deadlines_are_independent(self):
        for mode in ("deadline", "research", "lease", "late_disarm"):
            with self.subTest(mode=mode):
                code, report = self.run_fixture(mode)
                self.assertEqual(code, 2)
                self.assertEqual(report["reason"], "deadline_exceeded")
                self.assertTrue(report["cleanup"]["cleanup_complete"])

    def test_failed_recovery_is_reported_without_changing_saved_result(self):
        code, report = self.run_fixture("killed", failure="daemon unavailable")
        self.assertEqual(code, 2)
        self.assertIsNone(report["cleanup"])
        self.assertTrue(
            any("daemon unavailable" in item for item in report["failures"])
        )

    def test_completed_worker_recovery_timeout_keeps_nonzero_status(self):
        code, report = self.run_fixture("normal", recovery_seconds=2, recovery_delay=3)
        self.assertEqual(code, 2)
        self.assertEqual(report["reason"], "worker_completed")
        self.assertEqual(report["completed_exit_code"], 0)
        self.assertEqual(report["recovery_timeout_seconds"], 2)
        self.assertEqual(report["cleanup"]["timeout_seconds"], 2)
        self.assertTrue(report["cleanup"]["deadline_exceeded"])
        self.assertFalse(report["cleanup"]["cleanup_complete"])

    def test_foreign_child_reaper_is_rejected_before_launch(self):
        with patch(
            "benchcore.local_watchdog.signal.getsignal",
            return_value=signal.SIG_IGN,
        ):
            with patch("benchcore.local_watchdog.subprocess.Popen") as launch:
                with self.assertRaisesRegex(WorkerFailure, "exclusive child reaping"):
                    supervise(lambda fd: ["unused"])
                launch.assert_not_called()


class WatchdogBindingTests(unittest.TestCase):
    def test_recovery_helper_passes_deadline_and_closes_journal(self):
        from benchcore.local_recovery import _perform

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = DockerRecovery(root / "worker", episode=root / "episode")
            registration = {
                "state_dir": str(journal.root),
                "owner": journal.state["owner"],
                "directory_identity": journal.state["directory_identity"],
                "docker_binary": "fixture-docker",
            }
            journal.close()
            now = [0.0]
            timeouts = []

            def late_backend(args, **kwargs):
                timeouts.append(kwargs["timeout"])
                now[0] += 2
                return FakeDocker().run(args, **kwargs)

            with patch(
                "benchcore.runtime.DockerRuntime._run_raw",
                side_effect=late_backend,
            ), patch(
                "benchcore.docker_recovery.time.monotonic",
                side_effect=lambda: now[0],
            ):
                report = _perform(
                    {
                        "operation": "recover",
                        "state_dir": registration["state_dir"],
                        "docker_binary": "fixture-docker",
                        "registration": registration,
                        "timeout_seconds": 1,
                    }
                )
            self.assertEqual(timeouts, [1])
            self.assertFalse(report["cleanup_complete"])
            self.assertTrue(report["deadline_exceeded"])
            # The actual owner lock is released even after a deadline failure.
            reopened = DockerRecovery(root / "worker")
            reopened.close()

    def test_notification_failure_does_not_hide_runtime_cleanup_failure(self):
        from benchcore.local_watchdog import WatchdogClient

        left, right = socket.socketpair()
        client = WatchdogClient(left.detach())
        self.addCleanup(client.close)
        self.addCleanup(right.close)
        client.registered = True

        def request(message):
            if message["type"] == "disarm":
                client.closed.set()
                raise WorkerFailure("watchdog acknowledgement lost")

        client._request = request
        with self.assertRaisesRegex(WorkerFailure, "runtime removal refused"):
            with client.operation("cleanup", 1, cleanup=True):
                raise WorkerFailure("runtime removal refused")

    @unittest.skipUnless(os.name == "posix", "journal uses POSIX ownership locks")
    def test_channel_loss_after_initial_cleanup_info_does_not_block_removal(self):
        from benchcore.local_watchdog import WatchdogClient
        from benchcore.runtime import DockerRuntime

        for kind in ("container", "network", "volume"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                backend = FakeDocker()
                runtime = DockerRuntime(docker_binary="fixture-docker")
                journal = DockerRecovery(
                    Path(directory) / "worker", episode=Path(directory) / "episode"
                )
                self.addCleanup(journal.close)
                runtime._recovery = journal
                journal.execute(
                    ["run", "--detach", "--name", "agent", "image"]
                    if kind == "container"
                    else [kind, "create", "agent"],
                    backend.run,
                )
                backend.calls.clear()
                client = WatchdogClient.__new__(WatchdogClient)
                client.registered = True
                client.control_timeout = 120
                client.closed = threading.Event()
                client._cleanup_context = threading.local()

                def request(message):
                    if message["type"] == "disarm":
                        client.closed.set()
                        raise WorkerFailure("watchdog acknowledgement lost after info")
                    if client.closed.is_set():
                        raise WorkerFailure("watchdog channel is unavailable")

                client._request = request
                with patch("benchcore.local_watchdog._client", client), patch(
                    "benchcore.runtime.subprocess.run",
                    side_effect=lambda command, **kwargs: backend.run(command[1:]),
                ):
                    with self.assertRaisesRegex(WorkerFailure, "watchdog"):
                        if kind == "container":
                            runtime.stop_container("agent")
                        else:
                            runtime._run([kind, "rm", "agent"])
                self.assertEqual(backend.calls[0][0], "info")
                self.assertTrue(any("rm" in command for command in backend.calls))
                self.assertFalse(backend.objects)
                self.assertEqual(
                    journal.state["resources"][kind + ":agent"]["state"], "removed"
                )

    def test_container_cli_routes_to_watchdog_and_validates_private_overrides(self):
        from benchcore.cli import main

        argv = [
            "start",
            "--task",
            "unused-task.json",
            "--track",
            "B",
            "--episode-dir",
            "/unused-episode",
            "--agent-runner",
            "container",
            "--agent-image",
            "fixture-image",
            "--agent-command",
            "fixture-command",
            "--worker-watchdog-heartbeat-seconds",
            "20",
            "--worker-watchdog-grace-seconds",
            "3",
            "--worker-watchdog-recovery-seconds",
            "45",
        ]
        with patch("benchcore.cli.supervise_start", return_value=0) as start:
            with self.assertRaises(SystemExit) as stopped:
                main(argv)
            self.assertEqual(stopped.exception.code, 0)
            start.assert_called_once_with(
                argv, heartbeat_timeout=20, grace_seconds=3, recovery_seconds=45
            )
        for option in ("control", "recovery"):
            for value in ("nan", "inf", "0", "-1"):
                with patch("benchcore.cli.supervise_start") as start, redirect_stderr(
                    io.StringIO()
                ):
                    with self.assertRaises(SystemExit) as stopped:
                        main([*argv, f"--worker-watchdog-{option}-seconds", value])
                    self.assertEqual(stopped.exception.code, 2)
                    start.assert_not_called()

    def test_watchdog_is_not_a_core_or_agent_dependency(self):
        from tests.test_architecture import imported_modules, ROOT

        for name in (
            "supervisor.py",
            "worker_protocol.py",
            "client.py",
            "agent_api.py",
        ):
            path = ROOT / "benchcore" / name
            if path.exists():
                self.assertNotIn("local_watchdog", imported_modules(path))

    def test_operation_uses_actual_allowance_and_disarms_after_failure(self):
        from benchcore.local_watchdog import WatchdogClient

        client = WatchdogClient.__new__(WatchdogClient)
        client.registered = True
        messages = []
        client._request = messages.append

        @watched()
        def operation(*, timeout_seconds):
            raise WorkerFailure("ordinary timeout")

        with patch("benchcore.local_watchdog._client", client):
            with self.assertRaisesRegex(WorkerFailure, "ordinary timeout"):
                operation(timeout_seconds=2.5)
        self.assertEqual(messages[0]["seconds"], 2.5)
        self.assertEqual(messages[0]["type"], "arm")
        self.assertEqual(messages[1], {"type": "disarm", "key": messages[0]["key"]})

    def test_broken_watchdog_channel_allows_stop_but_refuses_new_work(self):
        from benchcore.local_watchdog import WatchdogClient
        from benchcore.runtime import DockerRuntime

        for already_detected in (False, True):
            with self.subTest(already_detected=already_detected):
                left, right = socket.socketpair()
                client = WatchdogClient(left.detach())
                client.registered = True
                self.addCleanup(client.close)
                right.close()
                if already_detected:
                    client.closed.set()
                runtime = DockerRuntime(docker_binary="fixture-docker")
                calls = []

                def run(command, **kwargs):
                    calls.append(command)
                    return subprocess.CompletedProcess(command, 0, "", "")

                with patch("benchcore.local_watchdog._client", client), patch(
                    "benchcore.runtime.subprocess.run", side_effect=run
                ):
                    # Missing watchdog acknowledgement stays an error AFTER
                    # actual removal/absence checks, rather than blocking stop.
                    with self.assertRaisesRegex(WorkerFailure, "watchdog"):
                        runtime.stop_container("owned-container")
                    self.assertIn(
                        ["fixture-docker", "rm", "--force", "owned-container"], calls
                    )
                    self.assertIn("ls", calls[-1])
                    calls.clear()
                    with self.assertRaisesRegex(WorkerFailure, "refusing new"):
                        runtime._run(["run", "--name", "new", "image"])
                    with self.assertRaisesRegex(WorkerFailure, "refusing new"):
                        runtime._run_raw(["start", "existing-id"])
                    self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
