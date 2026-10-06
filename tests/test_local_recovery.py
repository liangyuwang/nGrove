import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from envcore import local_recovery, local_watchdog
from envcore.docker_recovery import DockerRecovery, WorkerOwnershipBusy
from envcore.local_watchdog import supervise
from envcore.runtime import DockerRuntime
from envcore.worker_protocol import WorkerFailure
from tests.test_docker_recovery import FakeDocker


FIXTURE = Path(__file__).parent / "fixtures/recovery_process.py"
WORKER_FIXTURE = Path(__file__).parent / "fixtures/watchdog_worker.py"


@unittest.skipUnless(os.name == "posix", "journal uses POSIX ownership locks")
class RecoveryOwnershipTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.journal = DockerRecovery(
            self.root / "worker", episode=self.root / "episode"
        )
        self.addCleanup(self.journal.close)
        self.backend = FakeDocker()
        self.journal.execute(
            ["run", "--detach", "--name", "agent", "fixture-image"], self.backend.run
        )
        self.private = self.journal.private_directory("fixture-")
        self.path = self.journal.root / "resources.json"
        self.saved = self.path.read_bytes()
        self.request = {
            "operation": "recover",
            "state_dir": str(self.journal.root),
            "docker_binary": "fixture-docker",
            "registration": {
                "owner": self.journal.state["owner"],
                "directory_identity": self.journal.state["directory_identity"],
            },
            "timeout_seconds": 1,
        }

    def test_lock_wait_and_journal_read_spend_the_cleanup_allowance(self):
        now = [0.0]
        sleeps = []
        timeouts = []
        read = DockerRecovery._read

        def release(seconds):
            sleeps.append(seconds)
            now[0] += seconds
            if now[0] >= 0.2:
                self.journal.close()

        def slow_read(journal):
            state = read(journal)
            now[0] += 0.1
            return state

        def slow_run(args, **kwargs):
            timeouts.append(kwargs["timeout"])
            now[0] += 0.8
            return self.backend.run(args, **kwargs)

        with patch(
            "envcore.local_recovery.time.monotonic", side_effect=lambda: now[0]
        ), patch("envcore.local_recovery.time.sleep", side_effect=release), patch(
            "envcore.runtime.DockerRuntime._run_raw", side_effect=slow_run
        ), patch.object(DockerRecovery, "_read", slow_read):
            report = local_recovery._perform(self.request)
        self.assertEqual(len(sleeps), 4)
        self.assertEqual(len(timeouts), 1)
        self.assertAlmostEqual(timeouts[0], 0.7)
        self.assertEqual(report["timeout_seconds"], 1)
        self.assertTrue(report["deadline_exceeded"], report)
        self.assertFalse(report["cleanup_complete"], report)
        self.assertFalse(report["resumed"])
        self.assertEqual(self.path.read_bytes(), self.saved)
        self.assertTrue(self.private.exists())
        self.assertIn(("container", "agent"), self.backend.objects)
        reopened = DockerRecovery(self.journal.root)
        reopened.close()

    def test_busy_lock_exhausts_one_budget_without_modifying_storage(self):
        now = [0.0]
        sleeps = []
        lock_identity = (self.journal.root / "owner.lock").stat().st_ino

        def wait(seconds):
            sleeps.append(seconds)
            now[0] += seconds

        with patch(
            "envcore.local_recovery.time.monotonic", side_effect=lambda: now[0]
        ), patch("envcore.local_recovery.time.sleep", side_effect=wait), patch(
            "envcore.runtime.DockerRuntime._run_raw"
        ) as run:
            report = local_recovery._perform({**self.request, "timeout_seconds": 0.12})
        run.assert_not_called()
        self.assertEqual(len(sleeps), 3)
        self.assertAlmostEqual(sum(sleeps), 0.12)
        self.assertAlmostEqual(sleeps[-1], 0.02)
        self.assertTrue(report["deadline_exceeded"], report)
        self.assertFalse(report["cleanup_complete"], report)
        self.assertFalse(report["resumed"])
        self.assertIsNone(report["episode"])
        self.assertEqual(report["failures"][0]["name"], "ownership")
        self.assertEqual(self.path.read_bytes(), self.saved)
        self.assertTrue(self.private.exists())
        self.assertEqual(
            (self.journal.root / "owner.lock").stat().st_ino, lock_identity
        )
        with self.assertRaises(WorkerOwnershipBusy):
            DockerRecovery(self.journal.root)

    def test_registration_mismatch_after_wait_is_not_retried(self):
        self.journal.close()
        for field, value in (("owner", "0" * 32), ("directory_identity", [0, 0])):
            with self.subTest(field=field):
                owner = DockerRecovery(self.journal.root)
                self.addCleanup(owner.close)
                with patch(
                    "envcore.local_recovery.time.sleep",
                    side_effect=lambda seconds: owner.close(),
                ) as wait, patch("envcore.runtime.DockerRuntime._run_raw") as run:
                    request = {
                        **self.request,
                        "registration": {**self.request["registration"], field: value},
                    }
                    with self.assertRaisesRegex(
                        WorkerFailure, "differs from registration"
                    ):
                        local_recovery._perform(request)
                wait.assert_called_once()
                run.assert_not_called()
                self.assertEqual(self.path.read_bytes(), self.saved)
                self.assertTrue(self.private.exists())
        reopened = DockerRecovery(self.journal.root)
        reopened.close()

    def test_unsafe_or_damaged_storage_is_not_retried(self):
        self.journal.close()
        copied = self.root / "copied"
        shutil.copytree(self.journal.root, copied)
        for mode in ("copied", "damaged", "permissions"):
            with self.subTest(mode=mode):
                self.path.write_bytes(
                    b"{truncated" if mode == "damaged" else self.saved
                )
                self.journal.root.chmod(0o755 if mode == "permissions" else 0o700)
                with patch("envcore.local_recovery.time.sleep") as wait, patch(
                    "envcore.runtime.DockerRuntime._run_raw"
                ) as run:
                    with self.assertRaises(WorkerFailure) as failure:
                        local_recovery._perform(
                            {
                                **self.request,
                                "state_dir": str(copied)
                                if mode == "copied"
                                else self.request["state_dir"],
                            }
                        )
                self.assertNotIsInstance(failure.exception, WorkerOwnershipBusy)
                wait.assert_not_called()
                run.assert_not_called()
                self.assertTrue(self.private.exists())

    def test_lock_permission_error_is_not_classified_as_busy(self):
        self.journal.close()
        with patch(
            "fcntl.flock", side_effect=PermissionError("denied")
        ) as flock, patch("envcore.local_recovery.time.sleep") as wait, patch(
            "envcore.runtime.DockerRuntime._run_raw"
        ) as run:
            with self.assertRaisesRegex(WorkerFailure, "cannot open") as failure:
                local_recovery._perform(self.request)
        self.assertNotIsInstance(failure.exception, WorkerOwnershipBusy)
        flock.assert_called_once()
        wait.assert_not_called()
        run.assert_not_called()
        self.assertEqual(self.path.read_bytes(), self.saved)


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

    def test_supervised_recovery_waits_for_an_inherited_lock_to_be_released(self):
        with self.episode() as (root, registration):
            journal = DockerRecovery(root / "worker")
            self.addCleanup(journal.close)
            client = subprocess.Popen(
                [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
                pass_fds=(journal.lock_fd,),
                stdin=subprocess.PIPE,
            )
            journal.close()
            stop = threading.Event()
            released = threading.Event()

            def release_after_contention():
                while not stop.wait(0.01):
                    if (root / "ownership-busy").exists():
                        client.stdin.write(b"!")
                        client.stdin.flush()
                        released.set()
                        return

            release = threading.Thread(target=release_after_contention)
            release.start()
            try:
                report = self.recover(root, registration, "ownership-wait", timeout=3)
                self.assertTrue(released.is_set(), report)
                self.assertTrue(report["cleanup_complete"], report)
                self.assertFalse(report["deadline_exceeded"], report)
                self.assertTrue(report["process_supervision"]["termination_confirmed"])
                self.assertFalse(report["resumed"])
                self.assertFalse((root / "worker/private").exists())
                self.assertEqual(json.loads((root / "backend.json").read_text()), [])
                self.assertEqual(
                    (root / "episode/result.json").read_text(),
                    '{"TEST":"already_saved"}',
                )
            finally:
                stop.set()
                release.join(timeout=3)
                client.stdin.close()
                if client.poll() is None:
                    client.kill()
                client.wait(timeout=3)

    def test_supervised_busy_lock_times_out_and_preserves_unconfirmed_resources(self):
        with self.episode() as (root, registration):
            owner = DockerRecovery(root / "worker")
            self.addCleanup(owner.close)
            saved = (root / "worker/resources.json").read_bytes()
            inventory = (root / "backend.json").read_bytes()
            started = time.monotonic()
            report = self.recover(root, registration, "ownership-wait")
            self.assertLess(time.monotonic() - started, 3)
            self.assertTrue((root / "ownership-busy").exists(), report)
            self.assertFalse(report["cleanup_complete"], report)
            self.assertTrue(report["deadline_exceeded"], report)
            self.assertTrue(report["process_supervision"]["termination_confirmed"])
            self.assertFalse(report["resumed"])
            self.assertTrue(report["failures"], report)
            self.assertEqual((root / "worker/resources.json").read_bytes(), saved)
            self.assertEqual((root / "backend.json").read_bytes(), inventory)
            self.assertTrue((root / "worker/private").exists())
            with self.assertRaises(WorkerOwnershipBusy):
                DockerRecovery(root / "worker")
            owner.close()
            retry = self.recover(root, registration, "normal", timeout=3)
            self.assertTrue(retry["cleanup_complete"], retry)
            self.assertEqual(
                (root / "episode/result.json").read_text(), '{"TEST":"already_saved"}'
            )

    def test_default_watchdog_recovers_after_inherited_lock_worker_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "episode").mkdir()
            saved = '{"TEST":"already_saved"}'
            state = root / "episode/state.json"
            state.write_text(saved)
            stderr = io.StringIO()
            with self.helper(root, "normal"), redirect_stderr(stderr):
                code = supervise(
                    lambda fd: [
                        sys.executable,
                        str(WORKER_FIXTURE),
                        str(fd),
                        str(root),
                        "inherited_lock",
                    ],
                    recovery_seconds=3,
                )
            path = root / "worker/watchdog-report.json"
            self.assertTrue(path.exists(), stderr.getvalue())
            report = json.loads(path.read_text())
            diagnostic = f"report={report!r}\nstderr={stderr.getvalue()}"
            self.assertEqual(code, 2, diagnostic)
            self.assertEqual(report["reason"], "worker_exited", diagnostic)
            self.assertEqual(report["worker_exit_code"], -signal.SIGKILL, diagnostic)
            self.assertTrue(report["cleanup"]["cleanup_complete"], diagnostic)
            self.assertTrue(
                report["cleanup"]["process_supervision"]["termination_confirmed"],
                diagnostic,
            )
            self.assertEqual(report["failures"], [], diagnostic)
            self.assertFalse(report["resumed"], diagnostic)
            self.assertEqual(state.read_text(), saved, diagnostic)

    def test_watchdog_report_only_receives_budget_remaining_after_lock_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "episode").mkdir()
            elapsed = [0.0]
            monotonic = time.monotonic
            write_report = local_watchdog._write_report
            report_timeouts = []

            def recover(registration, *, timeout_seconds):
                owner = DockerRecovery(Path(registration["state_dir"]))

                def release(seconds):
                    elapsed[0] += seconds
                    if elapsed[0] >= 1:
                        owner.close()

                try:
                    with patch(
                        "envcore.local_recovery.time.sleep", side_effect=release
                    ), patch(
                        "envcore.runtime.DockerRuntime._run_raw",
                        side_effect=FakeDocker().run,
                    ):
                        return local_recovery._perform(
                            {
                                "operation": "recover",
                                "state_dir": registration["state_dir"],
                                "docker_binary": "fixture-docker",
                                "registration": registration,
                                "timeout_seconds": timeout_seconds,
                            }
                        )
                finally:
                    owner.close()

            def persist(registration, report, *, timeout_seconds):
                report_timeouts.append(timeout_seconds)
                return write_report(
                    registration, report, timeout_seconds=timeout_seconds
                )

            stderr = io.StringIO()
            with patch(
                "envcore.local_watchdog.time.monotonic",
                side_effect=lambda: monotonic() + elapsed[0],
            ), patch.object(local_watchdog, "_write_report", persist), redirect_stderr(
                stderr
            ):
                code = supervise(
                    lambda fd: [
                        sys.executable,
                        str(WORKER_FIXTURE),
                        str(fd),
                        str(root),
                        "normal",
                    ],
                    recovery_seconds=3,
                    recover=recover,
                )
            self.assertEqual(code, 0, stderr.getvalue())
            self.assertEqual(len(report_timeouts), 1)
            self.assertGreater(report_timeouts[0], 0)
            self.assertLessEqual(report_timeouts[0], 2)
            report = json.loads((root / "worker/watchdog-report.json").read_text())
            self.assertTrue(report["report_persisted"], report)
            self.assertEqual(report["recovery_timeout_seconds"], 3)
            self.assertEqual(report["cleanup"]["timeout_seconds"], 3)

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
        from envcore.cli import build_parser, command_worker_recover

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
        from envcore.cli import build_parser, command_worker_recover

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
                with patch("envcore.runtime.subprocess.run", lose_helper):
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
