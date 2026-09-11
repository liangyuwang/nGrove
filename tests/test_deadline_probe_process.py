"""A local process fixture; it does not validate a Docker or accelerator Worker."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from autotrain_bench.config import TrainingResources
from autotrain_bench.deadline_probe import (
    LeaseDeadlineObservation,
    probe_worker_lease_deadline,
)
from autotrain_bench.worker_protocol import LeaseState, ResearchTaskSpec
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
                    deadline = time.monotonic() + kwargs["duration_seconds"]
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
                            "time.sleep(max(0, float(sys.argv[2]) - time.monotonic())); "
                            "p = Path(sys.argv[3]); "
                            "p.with_suffix('.tmp').write_text(sys.argv[4]); "
                            "p.with_suffix('.tmp').replace(p); "
                            "os.kill(int(sys.argv[1]), signal.SIGTERM)",
                            str(self.workload.pid),
                            str(deadline),
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


if __name__ == "__main__":
    unittest.main()
