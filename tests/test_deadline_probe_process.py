"""A local process fixture; it does not validate a Docker or accelerator Worker."""

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
from envcore.deadline_probe import (
    LeaseDeadlineObservation,
    WorkerFaultEvidence,
    WorkerFaultKind,
    probe_worker_lease_deadline,
)
from envcore.worker_protocol import LeaseState, ResearchTaskSpec
from tests.test_lease_probe import LeaseWorker, contract


@unittest.skipUnless(os.name == "posix", "local fixture uses POSIX deadline signals")
class DeadlineProcessProbeTests(unittest.TestCase):
    def test_separate_controller_stops_real_idle_process_before_probe_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("workspace", "output", "train", "api"):
                (root / name).mkdir()
            event_path = root / "controller-event.json"
            processes = []
            cleanup_was_early = []

            def cleanup():
                # Fence the controller before reaping the workload and allowing
                # its PID to be reused by an unrelated process.
                for process in reversed(processes):
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)
                return True

            class ProcessWorker(LeaseWorker):
                workload = None

                def start_lease(self, **kwargs):
                    receipt = super().start_lease(**kwargs)
                    self.workload = subprocess.Popen(
                        [sys.executable, "-c", "import time; time.sleep(60)"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    processes.append(self.workload)
                    event = {
                        "episode_id": kwargs["episode_id"],
                        "lease_id": kwargs["lease_id"],
                        "worker_id": receipt.worker_id,
                    }
                    controller = subprocess.Popen(
                        [
                            sys.executable,
                            "-c",
                            "import os, signal, sys, time; from pathlib import Path; "
                            "time.sleep(float(sys.argv[2])); "
                            "p = Path(sys.argv[3]); "
                            "p.with_suffix('.tmp').write_text(sys.argv[4]); "
                            "p.with_suffix('.tmp').replace(p); "
                            "os.kill(int(sys.argv[1]), signal.SIGTERM)",
                            str(self.workload.pid),
                            str(kwargs["duration_seconds"]),
                            str(event_path),
                            json.dumps(event),
                        ],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    processes.append(controller)
                    return receipt

                def lease_state(self, **kwargs):
                    raise AssertionError("Worker polling must not cause expiry")

                def exec_lease(self, **kwargs):
                    raise AssertionError("fixture must stay idle")

                def terminate_lease(self, **kwargs):
                    cleanup_was_early.append(workload_running())
                    cleanup()
                    super().terminate_lease(**kwargs)

            worker = ProcessWorker()

            def workload_running():
                if worker.workload.returncode is not None:
                    return False
                # Read without reaping: the controller may still signal this
                # child, whose PID must remain owned until controller shutdown.
                result = subprocess.run(
                    ["ps", "-o", "stat=", "-p", str(worker.workload.pid)],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=1,
                )
                state = result.stdout.strip()
                if not state:
                    raise RuntimeError("process state is unavailable")
                return not state.startswith("Z")

            def observe(**kwargs):
                running = workload_running()
                event = (
                    json.loads(event_path.read_text()) if event_path.exists() else None
                )
                if event is not None:
                    for key in ("episode_id", "lease_id", "worker_id"):
                        self.assertEqual(event[key], kwargs[key])
                return LeaseDeadlineObservation(
                    episode_id=kwargs["episode_id"],
                    lease_id=kwargs["lease_id"],
                    worker_id=kwargs["worker_id"],
                    observation_id=kwargs["observation_id"],
                    source="local-process-controller",
                    state=LeaseState.RUNNING if running else LeaseState.STOPPED,
                    deadline_triggered=event is not None,
                    deadline_evidence_id="local-controller-event"
                    if event is not None
                    else None,
                )

            try:
                report = probe_worker_lease_deadline(
                    worker,
                    task=ResearchTaskSpec(
                        "process-fixture", "fixture:image", root / "train", root / "api"
                    ),
                    workspace=root / "workspace",
                    output_root=root / "output",
                    resources=TrainingResources(),
                    resource_envelope=contract(),
                    observe=observe,
                    teardown=cleanup,
                    duration_seconds=1,
                    termination_grace_seconds=2,
                    poll_interval_seconds=0.025,
                )
                self.assertTrue(report.passed, report.private_manifest())
                self.assertEqual(cleanup_was_early, [False])
                self.assertTrue(report.lease["deadline_passed"])
                self.assertTrue(report.lease["cleanup_confirmed"])
                self.assertTrue(
                    all(process.poll() is not None for process in processes)
                )
            finally:
                cleanup()


@unittest.skipUnless(os.name == "posix", "local fixture uses POSIX fault signals")
class WorkerFaultProcessProbeTests(unittest.TestCase):
    def test_paused_worker_does_not_disable_separate_deadline_controller(self):
        self.run_fixture(WorkerFaultKind.PAUSED, independent=True)

    def test_killed_worker_does_not_disable_separate_deadline_controller(self):
        self.run_fixture(WorkerFaultKind.TERMINATED, independent=True)

    def test_pausing_worker_that_owns_deadline_enforcement_fails_the_diagnostic(self):
        self.run_fixture(WorkerFaultKind.PAUSED, independent=False)

    def run_fixture(self, kind, *, independent):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("workspace", "output", "train", "api"):
                (root / name).mkdir()
            event_path = root / "deadline-event.json"
            ready_path = root / "worker-admission.json"
            processes = []
            injected = {}
            cleanup_was_early = []
            # The Worker control process acknowledges this scope's admission.
            # In the negative case it also owns deadline enforcement; pausing
            # that process must leave the real workload running past its grant.
            guard = """
import json, os, signal, sys, time
from pathlib import Path
spec = json.loads(sys.argv[1])
if spec['admission']:
    ready = Path(spec['admission'])
    ready.with_suffix('.tmp').write_text(json.dumps(spec['binding']))
    ready.with_suffix('.tmp').replace(ready)
if spec['enforce']:
    # Start this controller's own relative clock. Python 3.9 monotonic origins
    # need not be shared across processes on every supported host.
    time.sleep(spec['duration_seconds'])
    event = Path(spec['event'])
    event.with_suffix('.tmp').write_text(json.dumps(spec['binding']))
    event.with_suffix('.tmp').replace(event)
    os.kill(spec['workload_pid'], signal.SIGTERM)
else:
    time.sleep(60)
"""

            def launch(argv):
                process = subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                processes.append(process)
                return process

            def process_state(process):
                if process.returncode is not None:
                    return "Z"
                result = subprocess.run(
                    ["ps", "-o", "stat=", "-p", str(process.pid)],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=1,
                )
                state = result.stdout.strip()
                if not state:
                    raise RuntimeError("owned process state is unavailable")
                # Do not reap any child that a still-owned controller could
                # signal. Cleanup fences controllers before workload PID reuse.
                return state[0].upper()

            def cleanup():
                for process in reversed(processes):
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)
                return True

            class ProcessWorker(LeaseWorker):
                def start_lease(self, **kwargs):
                    receipt = super().start_lease(**kwargs)
                    self.workload = launch(
                        [sys.executable, "-c", "import time; time.sleep(60)"]
                    )
                    self.binding = {
                        key: kwargs[key] for key in ("episode_id", "lease_id")
                    }
                    self.binding["worker_id"] = receipt.worker_id
                    spec = dict(
                        workload_pid=self.workload.pid,
                        duration_seconds=kwargs["duration_seconds"],
                        event=str(event_path),
                        admission=str(ready_path),
                        binding=self.binding,
                        enforce=not independent,
                    )
                    self.control = launch(
                        [sys.executable, "-c", guard, json.dumps(spec)]
                    )
                    readiness_deadline = time.monotonic() + 5
                    while not ready_path.exists():
                        if time.monotonic() >= readiness_deadline:
                            raise RuntimeError(
                                "fixture Worker did not acknowledge admission"
                            )
                        time.sleep(0.01)
                    if json.loads(ready_path.read_text()) != self.binding:
                        raise RuntimeError("fixture admission binding mismatch")
                    if independent:
                        spec.update(admission=None, enforce=True)
                        launch([sys.executable, "-c", guard, json.dumps(spec)])
                    return receipt

                def lease_state(self, **kwargs):
                    raise AssertionError("Worker polling cannot enforce expiry")

                def exec_lease(self, **kwargs):
                    raise AssertionError("the Lease must remain idle")

                def terminate_lease(self, **kwargs):
                    if injected:
                        raise AssertionError("must not call the faulted Worker")
                    cleanup()
                    super().terminate_lease(**kwargs)

            worker = ProcessWorker()

            def inject(**kwargs):
                self.assertEqual(
                    {key: kwargs[key] for key in worker.binding}, worker.binding
                )
                injected.update(kwargs)
                worker.control.send_signal(
                    signal.SIGSTOP if kind is WorkerFaultKind.PAUSED else signal.SIGKILL
                )
                deadline = time.monotonic() + kwargs["timeout_seconds"]
                expected = "T" if kind is WorkerFaultKind.PAUSED else "Z"
                while process_state(worker.control) != expected:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("fixture fault was not observed in time")
                    time.sleep(0.01)

            def observe(**kwargs):
                running = process_state(worker.workload) != "Z"
                control_state = process_state(worker.control)
                event = (
                    json.loads(event_path.read_text()) if event_path.exists() else None
                )
                if event is not None:
                    self.assertEqual(event, worker.binding)
                fault = None
                if injected and control_state == (
                    "T" if kind is WorkerFaultKind.PAUSED else "Z"
                ):
                    fault = WorkerFaultEvidence(
                        injected["fault_id"], kind, "local-control-process-state"
                    )
                return LeaseDeadlineObservation(
                    episode_id=kwargs["episode_id"],
                    lease_id=kwargs["lease_id"],
                    worker_id=kwargs["worker_id"],
                    observation_id=kwargs["observation_id"],
                    source="local-independent-process-observer",
                    state=LeaseState.RUNNING if running else LeaseState.STOPPED,
                    deadline_triggered=event is not None,
                    deadline_evidence_id="local-deadline-event"
                    if event is not None
                    else None,
                    worker_fault=fault,
                )

            def teardown():
                cleanup_was_early.append(process_state(worker.workload) != "Z")
                return cleanup()

            try:
                report = probe_worker_lease_deadline(
                    worker,
                    task=ResearchTaskSpec(
                        "process-fixture", "fixture:image", root / "train", root / "api"
                    ),
                    workspace=root / "workspace",
                    output_root=root / "output",
                    resources=TrainingResources(),
                    resource_envelope=contract(),
                    observe=observe,
                    teardown=teardown,
                    worker_fault=kind,
                    inject_fault=inject,
                    duration_seconds=1,
                    termination_grace_seconds=1,
                    poll_interval_seconds=0.025,
                )
                self.assertEqual(report.passed, independent, report.private_manifest())
                self.assertEqual(report.lease["deadline_passed"], independent)
                self.assertTrue(report.lease["worker_fault"]["accepted"])
                self.assertTrue(report.lease["cleanup_confirmed"])
                self.assertEqual(cleanup_was_early, [not independent])
                self.assertEqual(worker.stops, [])
                self.assertTrue(
                    all(process.poll() is not None for process in processes)
                )
                if not independent:
                    self.assertEqual(
                        report.failed[0].name, "deadline.expiry_during_worker_fault"
                    )
                    self.assertIn(
                        report.failed[0].code,
                        {"idle_lease_deadline_exceeded", "operation_deadline_exceeded"},
                    )
            finally:
                cleanup()


if __name__ == "__main__":
    unittest.main()
