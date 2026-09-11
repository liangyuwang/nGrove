import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from benchcore import local_recovery
from benchcore.docker_recovery import DockerRecovery
from benchcore.local_watchdog import supervise
from benchcore.runtime import DockerRuntime
from benchcore.worker_protocol import WorkerFailure
from tests.test_docker_recovery import FakeDocker


FIXTURE = Path(__file__).parent / "fixtures/recovery_process.py"
WORKER_FIXTURE = Path(__file__).parent / "fixtures/watchdog_worker.py"


@unittest.skipUnless(
    os.name == "posix", "recovery supervision uses POSIX process groups"
)
class LocalRecoveryProcessTests(unittest.TestCase):
    @contextmanager
    def episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episode"
            episode.mkdir()
            (episode / "result.json").write_text('{"TEST":"already_saved"}')
            backend = FakeDocker()
            journal = DockerRecovery(root / "worker", episode=episode)
            try:
                for args in (
                    ["run", "--detach", "--name", "agent", "fixture-image"],
                    [
                        "run",
                        "--detach",
                        "--name",
                        "atb-prediction-holder-1",
                        "fixture-image",
                    ],
                    ["volume", "create", "predictions"],
                ):
                    journal.execute(args, backend.run)
                private = journal.private_directory("hidden-")
                (private / "fixture").write_text("disposable fixture")
                registration = {
                    "state_dir": str(journal.root),
                    "owner": journal.state["owner"],
                    "directory_identity": journal.state["directory_identity"],
                    "docker_binary": "fixture-docker",
                }
            finally:
                journal.close()
            (root / "backend.json").write_text(
                json.dumps(
                    [
                        {"kind": kind, "name": name, "object": obj}
                        for (kind, name), obj in backend.objects.items()
                    ]
                )
            )
            yield root, registration

    def helper(self, root, mode):
        return patch.object(
            local_recovery,
            "_command",
            side_effect=lambda fd: [
                sys.executable,
                str(FIXTURE),
                str(fd),
                str(root),
                mode,
            ],
        )

    def recover(self, root, registration, mode, *, timeout=0.6):
        with self.helper(root, mode):
            return local_recovery.recover_resources(
                root / "worker",
                docker_binary="fixture-docker",
                registration=registration,
                timeout_seconds=timeout,
            )

    def test_filesystem_stalls_are_fenced_and_recovery_can_be_retried(self):
        for mode in ("open-block", "save-block", "delete-block", "close-block"):
            with self.subTest(mode=mode), self.episode() as (root, registration):
                started = time.monotonic()
                report = self.recover(root, registration, mode)
                self.assertLess(time.monotonic() - started, 3)
                self.assertEqual((root / "blocked-stage").read_text(), mode)
                self.assertFalse(report["cleanup_complete"])
                self.assertTrue(report["deadline_exceeded"])
                self.assertTrue(report["process_supervision"]["termination_confirmed"])
                if mode != "close-block":
                    self.assertTrue((root / "worker/private").exists())
                # A fresh helper must be able to acquire the real journal lock
                # and finish from durable progress after the blocked one exits.
                retry = self.recover(root, registration, "normal", timeout=3)
                self.assertTrue(retry["cleanup_complete"], retry)
                self.assertFalse((root / "worker/private").exists())
                self.assertEqual(json.loads((root / "backend.json").read_text()), [])
                self.assertEqual(
                    (root / "episode/result.json").read_text(),
                    '{"TEST":"already_saved"}',
                )

    def test_report_storage_stall_preserves_the_previous_report(self):
        with self.episode() as (root, registration):
            saved = '{"earlier_attempt":true}'
            target = root / "worker/watchdog-report.json"
            target.write_text(saved)
            started = time.monotonic()
            with self.helper(root, "report-block"):
                with self.assertRaisesRegex(WorkerFailure, "deadline exceeded"):
                    local_recovery.write_watchdog_report(
                        registration, {"new": True}, timeout_seconds=0.6
                    )
            self.assertLess(time.monotonic() - started, 3)
            self.assertEqual((root / "blocked-stage").read_text(), "report-block")
            self.assertEqual(target.read_text(), saved)

    def test_manual_recovery_uses_process_boundary_without_parent_journal_io(self):
        from benchcore.cli import build_parser, command_worker_recover

        with self.episode() as (root, registration):
            args = build_parser().parse_args(
                [
                    "worker-recover",
                    "--state-dir",
                    str(root / "worker"),
                    "--timeout-seconds",
                    "0.6",
                ]
            )
            output = io.StringIO()
            with self.helper(root, "open-block"), patch.object(
                DockerRecovery,
                "__init__",
                side_effect=AssertionError("parent must not open storage"),
            ), redirect_stdout(output):
                self.assertEqual(command_worker_recover(args), 2)
            report = json.loads(output.getvalue())
            self.assertTrue(report["deadline_exceeded"])
            self.assertFalse(report["cleanup_complete"])

    def test_default_watchdog_does_not_block_on_report_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "episode").mkdir()
            saved = '{"TEST":"already_saved"}'
            state = root / "episode/state.json"
            state.write_text(saved)
            output = io.StringIO()
            started = time.monotonic()
            with self.helper(root, "report-block"), redirect_stderr(output):
                code = supervise(
                    lambda fd: [
                        sys.executable,
                        str(WORKER_FIXTURE),
                        str(fd),
                        str(root),
                        "normal",
                    ],
                    recovery_seconds=0.8,
                )
            self.assertLess(time.monotonic() - started, 3)
            self.assertEqual(code, 2)
            self.assertEqual((root / "blocked-stage").read_text(), "report-block")
            self.assertIn("current report was not persisted", output.getvalue())
            self.assertFalse((root / "worker/watchdog-report.json").exists())
            self.assertEqual(state.read_text(), saved)

    def test_helper_failures_never_claim_success(self):
        for mode in ("crash", "bad-result", "huge"):
            with self.subTest(mode=mode), self.episode() as (root, registration):
                report = self.recover(root, registration, mode, timeout=3)
                self.assertFalse(report["cleanup_complete"])
                self.assertTrue(report["failures"])
                self.assertIsNotNone(report["process_supervision"]["exit_code"])
                if mode != "crash":
                    self.assertTrue(
                        report["process_supervision"]["termination_confirmed"]
                    )
                # A group-fencing error after abrupt exit stays unconfirmed;
                # reaping its leader alone must not turn that into success.
                self.assertTrue((root / "worker/private").exists())

    def test_cancellation_during_blocked_recovery_restores_handlers(self):
        previous = {
            sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)
        }
        with self.episode() as (root, registration):
            report = self.recover(root, registration, "signals")
            self.assertFalse(report["cleanup_complete"])
            self.assertIn(
                report["process_supervision"]["operator_signal"],
                (signal.SIGTERM, signal.SIGINT),
            )
            self.assertTrue(report["process_supervision"]["termination_confirmed"])
            for sig, handler in previous.items():
                self.assertEqual(signal.getsignal(sig), handler)

    def test_manual_cancellation_returns_failure_even_when_cleanup_completes(self):
        from benchcore.cli import build_parser, command_worker_recover

        with self.episode() as (root, registration):
            args = build_parser().parse_args(
                [
                    "worker-recover",
                    "--state-dir",
                    str(root / "worker"),
                    "--timeout-seconds",
                    "3",
                ]
            )
            output = io.StringIO()
            with self.helper(root, "complete-signals"), redirect_stdout(output):
                self.assertEqual(command_worker_recover(args), 2)
            report = json.loads(output.getvalue())
            self.assertTrue(report["cleanup_complete"], report)
            self.assertIn(
                report["process_supervision"]["operator_signal"],
                (signal.SIGTERM, signal.SIGINT),
            )

    def test_recovery_helper_cancellation_reaches_outer_watchdog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "episode").mkdir()
            with self.helper(root, "complete-signals"), redirect_stderr(io.StringIO()):
                code = supervise(
                    lambda fd: [
                        sys.executable,
                        str(WORKER_FIXTURE),
                        str(fd),
                        str(root),
                        "normal",
                    ],
                    recovery_seconds=3,
                )
            self.assertEqual(code, 2)
            report = json.loads((root / "worker/watchdog-report.json").read_text())
            self.assertTrue(report["cleanup"]["cleanup_complete"], report)
            self.assertEqual(report["reason"], "operator_interrupted")
            self.assertIn(report["operator_signal"], (signal.SIGTERM, signal.SIGINT))

    def test_unconfirmed_helper_termination_is_not_reported_as_success(self):
        with self.episode() as (root, registration):
            children = []
            popen = subprocess.Popen

            def launch(*args, **kwargs):
                child = popen(*args, **kwargs)
                children.append(child)
                return child

            try:
                with patch.object(subprocess, "Popen", launch), patch.object(
                    popen,
                    "wait",
                    side_effect=subprocess.TimeoutExpired("recovery helper", 5),
                ):
                    report = self.recover(root, registration, "normal", timeout=3)
                self.assertFalse(report["cleanup_complete"])
                self.assertFalse(report["process_supervision"]["termination_confirmed"])
            finally:
                for child in children:
                    child.wait(timeout=3)

    def test_source_checkout_helper_runs_from_an_unrelated_working_directory(self):
        with self.episode() as (root, registration):
            popen = subprocess.Popen

            def launch(*args, **kwargs):
                kwargs["cwd"] = str(root)
                return popen(*args, **kwargs)

            with patch.object(subprocess, "Popen", launch):
                local_recovery.write_watchdog_report(
                    registration, {"fixture": True}, timeout_seconds=3
                )
            self.assertEqual(
                json.loads((root / "worker/watchdog-report.json").read_text()),
                {"fixture": True},
            )

    def test_recovery_runtime_clients_retain_ownership_after_helper_loss(self):
        with self.episode() as (root, registration):
            journal = DockerRecovery(root / "worker")
            runtime = DockerRuntime(docker_binary=sys.executable)
            runtime._recovery = journal
            runtime._recovery_clients_hold_lock = True
            clients = []

            def lose_helper(args, **kwargs):
                self.assertEqual(kwargs["pass_fds"], (journal.lock_fd,))
                clients.append(
                    subprocess.Popen(
                        [sys.executable, "-c", "import time; time.sleep(30)"],
                        pass_fds=kwargs["pass_fds"],
                    )
                )
                raise WorkerFailure("simulated helper loss")

            try:
                with patch("benchcore.runtime.subprocess.run", lose_helper):
                    with self.assertRaisesRegex(WorkerFailure, "simulated helper loss"):
                        runtime._run_raw(["info"], timeout=1)
                journal.close()
                with self.assertRaisesRegex(WorkerFailure, "still holds ownership"):
                    DockerRecovery(root / "worker")
            finally:
                journal.close()
                for client in clients:
                    client.kill()
                    client.wait(timeout=3)
            reopened = DockerRecovery(root / "worker")
            reopened.close()


if __name__ == "__main__":
    unittest.main()
