import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

from autotrain_bench.config import TaskConfig
from autotrain_bench.client import Client
from autotrain_bench.runtime import ExecResult, RuntimeFailure
from autotrain_bench.supervisor import ProtocolError, Supervisor


ROOT = Path(__file__).resolve().parents[1]


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class FakeRuntime:
    def __init__(self):
        self.running = set()
        self.fail_start = False
        self.evaluations = []

    def start_training(self, *, name, config, lease_dir):
        del config, lease_dir
        if self.fail_start:
            self.fail_start = False
            raise RuntimeFailure("simulated infrastructure failure")
        self.running.add(name)

    def exec_training(self, *, name, argv, timeout_seconds):
        self.assert_running(name)
        return ExecResult(0, "executed " + " ".join(argv), "")

    def assert_running(self, name):
        if name not in self.running:
            raise AssertionError(f"container is not running: {name}")

    def container_running(self, name):
        return name in self.running

    def stop_container(self, name):
        self.running.discard(name)

    def run_evaluator(self, *, config, candidate_dir, split, output_dir):
        del config, candidate_dir, output_dir
        self.evaluations.append(split)
        return 1.25 if split == "val" else 1.5


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        workspace = root / "workspace"
        workspace.mkdir()
        candidate = workspace / "candidate"
        candidate.mkdir()
        (candidate / "candidate.py").write_text(
            "def load(candidate_dir, device): return None\n"
            "def next_token_logits(model, input_ids): return []\n",
            encoding="utf-8",
        )
        base = TaskConfig.load(ROOT / "examples/toy/task.json")
        self.config = replace(base, agent_workspace=workspace)
        self.workspace = workspace
        self.runtime = FakeRuntime()
        self.clock = FakeClock()
        self.episode = root / "episode"

    def tearDown(self):
        self.temporary.cleanup()

    def supervisor(self, track="B", config=None):
        supervisor = Supervisor(
            config=config or self.config,
            track=track,
            episode_dir=self.episode,
            runtime=self.runtime,
            monotonic=self.clock,
            wall_time=lambda: 1_700_000_000.0,
        )
        supervisor.begin()
        return supervisor

    def test_track_b_interactive_lease_commit_and_finish(self):
        supervisor = self.supervisor()
        self.assertEqual(supervisor.state["agent_system"]["network"], "none")
        self.assertEqual(
            len(supervisor.state["agent_system"]["initial_workspace_sha256"]), 64
        )
        lease = supervisor.start_lease()
        self.assertEqual(lease["lease_id"], "lease-0001")
        command = supervisor.exec_lease(["python", "train.py"])
        self.assertEqual(command["exit_code"], 0)
        committed = supervisor.commit_lease("candidate")
        self.assertEqual(committed["val_nll"], 1.25)
        self.assertIsNone(supervisor.state["active_lease"])

        with self.assertRaises(ProtocolError):
            supervisor.commit_lease("candidate")

        supervisor.start_lease()
        supervisor.stop_lease()
        with self.assertRaises(ProtocolError):
            supervisor.start_lease()

        final = supervisor.finish(committed["candidate_id"])
        self.assertEqual(final["status"], "finalized")
        self.assertEqual(final["test_nll"], 1.5)
        self.assertEqual(self.runtime.evaluations, ["val", "test"])
        with self.assertRaises(ProtocolError):
            supervisor.start_lease()

        events = [
            json.loads(line)
            for line in supervisor.events_path.read_text(encoding="utf-8").splitlines()
        ]
        types = [event["type"] for event in events]
        self.assertLess(types.index("EPISODE_CLOSED"), types.index("TEST_EVALUATED"))

    def test_infrastructure_failure_does_not_consume_lease(self):
        supervisor = self.supervisor()
        self.runtime.fail_start = True
        with self.assertRaises(ProtocolError) as caught:
            supervisor.start_lease()
        self.assertEqual(caught.exception.code, "infrastructure_failure")
        self.assertEqual(supervisor.state["leases_used"], 0)
        supervisor.start_lease()
        self.assertEqual(supervisor.state["leases_used"], 1)

    def test_track_a_deadline_revokes_lease_and_closes_episode(self):
        config = replace(self.config, track_a_total_seconds=5)
        supervisor = self.supervisor(track="A", config=config)
        supervisor.start_lease()
        self.clock.advance(6)
        self.assertTrue(supervisor.enforce_deadlines())
        self.assertEqual(supervisor.state["status"], "finalized")
        self.assertIsNone(supervisor.state["active_lease"])
        self.assertFalse(self.runtime.running)

    def test_agent_mailbox_exposes_only_public_status(self):
        supervisor = self.supervisor()
        result = {}

        def request_status():
            result.update(Client(supervisor.control_dir, poll_seconds=0.001).status())

        thread = threading.Thread(target=request_status)
        thread.start()
        deadline = time.monotonic() + 2
        while thread.is_alive() and time.monotonic() < deadline:
            supervisor.process_one_request()
            time.sleep(0.001)
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["status"], "active")
        self.assertEqual(result["agent_network"], "none")
        serialized = json.dumps(result)
        self.assertNotIn(str(self.config.val_path), serialized)
        self.assertNotIn(str(self.config.test_path), serialized)


if __name__ == "__main__":
    unittest.main()
