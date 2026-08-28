import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

from autotrain_bench.config import InferenceConfig, TaskConfig
from autotrain_bench.client import Client
from autotrain_bench.runtime import ExecResult, RuntimeFailure
from autotrain_bench.supervisor import ProtocolError, Supervisor


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tests/fixtures/task/task.json"


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
        self.val_nlls = []

    def start_training(self, *, name, config, lease_dir):
        if self.fail_start:
            self.fail_start = False
            raise RuntimeFailure("simulated infrastructure failure")
        output = lease_dir / "output" / "candidate"
        output.mkdir(parents=True)
        shutil.copy2(
            config.agent_workspace / "candidate/candidate.py",
            output / "candidate.py",
        )
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
        if split == "val" and self.val_nlls:
            return self.val_nlls.pop(0)
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
        base = TaskConfig.load(TASK)
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
        self.assertEqual(supervisor.state["agent_system"]["network"], "disabled")
        self.assertTrue(supervisor.state["agent_system"]["policy_compliant"])
        self.assertTrue(supervisor.state["policy_compliant"])
        self.assertEqual(supervisor.state["policy_violations"], [])
        self.assertEqual(
            supervisor.state["evaluation_system"]["backend"],
            "isolated_candidate",
        )
        self.assertTrue(
            supervisor.state["evaluation_system"]["policy_compliant"]
        )
        self.assertEqual(
            len(supervisor.state["agent_system"]["initial_workspace_sha256"]), 64
        )
        lease = supervisor.start_lease()
        self.assertEqual(lease["lease_id"], "lease-0001")
        command = supervisor.exec_lease(["python", "train.py"])
        self.assertEqual(command["exit_code"], 0)
        committed = supervisor.commit_lease("/output/candidate")
        self.assertEqual(committed["val_nll"], 1.25)
        self.assertIsNone(supervisor.state["active_lease"])
        self.assertIsNone(supervisor.state["incumbent_id"])
        declared = supervisor.declare_incumbent(committed["candidate_id"])
        self.assertEqual(declared["incumbent_id"], committed["candidate_id"])

        with self.assertRaises(ProtocolError):
            supervisor.commit_lease("/output/candidate")

        supervisor.start_lease()
        supervisor.stop_lease()
        with self.assertRaises(ProtocolError):
            supervisor.start_lease()

        final = supervisor.finish()
        self.assertEqual(final["status"], "finalized")
        self.assertEqual(final["test_nll"], 1.5)
        self.assertTrue(final["agent_finish_observed"])
        self.assertEqual(self.runtime.evaluations, ["val", "test"])
        with self.assertRaises(ProtocolError):
            supervisor.start_lease()

        events = [
            json.loads(line)
            for line in supervisor.events_path.read_text(encoding="utf-8").splitlines()
        ]
        types = [event["type"] for event in events]
        self.assertLess(types.index("VAL_EVALUATED"), types.index("INCUMBENT_DECLARED"))
        self.assertLess(types.index("EPISODE_CLOSED"), types.index("TEST_EVALUATED"))

    def test_official_elapsed_time_freezes_before_agent_drain(self):
        supervisor = self.supervisor()
        self.clock.advance(4)
        supervisor.finish()
        self.assertEqual(supervisor.public_status()["elapsed_seconds"], 4.0)
        self.clock.advance(30)
        self.assertEqual(supervisor.public_status()["elapsed_seconds"], 4.0)
        self.assertEqual(supervisor.state["elapsed_seconds"], 4.0)

    def test_infrastructure_failure_does_not_consume_lease(self):
        supervisor = self.supervisor()
        self.runtime.fail_start = True
        with self.assertRaises(ProtocolError) as caught:
            supervisor.start_lease()
        self.assertEqual(caught.exception.code, "infrastructure_failure")
        self.assertEqual(supervisor.state["leases_used"], 0)
        supervisor.start_lease()
        self.assertEqual(supervisor.state["leases_used"], 1)

    def test_commit_accepts_only_stopped_lease_output(self):
        supervisor = self.supervisor()
        supervisor.start_lease()
        with self.assertRaises(ProtocolError) as caught:
            supervisor.commit_lease("/workspace/candidate")
        self.assertEqual(caught.exception.code, "invalid_candidate")
        self.assertIsNone(supervisor.state["active_lease"])
        self.assertEqual(supervisor.state["candidates"], {})

    def test_candidate_output_symlink_is_rejected_after_lease_stop(self):
        supervisor = self.supervisor()
        lease = supervisor.start_lease()
        candidate = (
            supervisor.runs_dir / lease["lease_id"] / "output" / "candidate"
        )
        (candidate / "candidate.py").unlink()
        (candidate / "candidate.py").symlink_to(self.config.val_path)
        with self.assertRaises(ProtocolError) as caught:
            supervisor.commit_lease("/output/candidate")
        self.assertEqual(caught.exception.code, "invalid_candidate")
        self.assertIsNone(supervisor.state["active_lease"])

    def test_candidate_output_hard_link_is_rejected_after_lease_stop(self):
        supervisor = self.supervisor()
        lease = supervisor.start_lease()
        candidate_file = (
            supervisor.runs_dir
            / lease["lease_id"]
            / "output/candidate/candidate.py"
        )
        candidate_file.unlink()
        os.link(self.workspace / "candidate/candidate.py", candidate_file)
        with self.assertRaises(ProtocolError) as caught:
            supervisor.commit_lease("/output/candidate")
        self.assertEqual(caught.exception.code, "invalid_candidate")
        self.assertIsNone(supervisor.state["active_lease"])

    def test_timing_report_separates_lease_use_and_unused_track_a_budget(self):
        supervisor = self.supervisor(
            track="A", config=replace(self.config, track_a_total_seconds=100)
        )
        supervisor.start_lease(10)
        self.clock.advance(3)
        candidate = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(candidate["candidate_id"])
        final = supervisor.finish()
        timing = final["timing"]
        self.assertEqual(timing["official_elapsed_seconds"], 3.0)
        self.assertEqual(timing["training_lease_seconds"], 3.0)
        self.assertEqual(timing["validation_evaluation_seconds"], 0.0)
        self.assertEqual(timing["agent_and_runtime_seconds"], 0.0)
        self.assertEqual(timing["unused_budget_seconds"], 97.0)

    def test_track_a_deadline_revokes_lease_and_closes_episode(self):
        config = replace(self.config, track_a_total_seconds=5)
        supervisor = self.supervisor(track="A", config=config)
        supervisor.start_lease()
        self.clock.advance(6)
        self.assertTrue(supervisor.enforce_deadlines())
        self.assertEqual(supervisor.state["status"], "finalized")
        self.assertIsNone(supervisor.state["active_lease"])
        self.assertFalse(self.runtime.running)

    def test_track_a_deadline_uses_only_agent_declared_incumbent(self):
        config = replace(self.config, track_a_total_seconds=100)
        supervisor = self.supervisor(track="A", config=config)
        self.runtime.val_nlls = [1.0, 2.0]

        with self.assertRaises(ProtocolError) as unknown:
            supervisor.declare_incumbent("sha256:missing")
        self.assertEqual(unknown.exception.code, "unknown_candidate")

        supervisor.start_lease(10)
        first = supervisor.commit_lease("/output/candidate")
        self.assertIsNone(supervisor.state["incumbent_id"])
        supervisor.declare_incumbent(first["candidate_id"])

        (self.workspace / "candidate/candidate.py").write_text(
            "def load(candidate_dir, device): return None\n"
            "def next_token_logits(model, input_ids): return []\n"
            "# a distinct later candidate\n",
            encoding="utf-8",
        )
        supervisor.start_lease(10)
        later = supervisor.commit_lease("/output/candidate")
        self.assertNotEqual(first["candidate_id"], later["candidate_id"])
        self.assertEqual(supervisor.state["incumbent_id"], first["candidate_id"])

        self.clock.advance(101)
        self.assertTrue(supervisor.enforce_deadlines())
        self.assertEqual(supervisor.state["final_candidate_id"], first["candidate_id"])
        self.assertEqual(supervisor.state["test_nll"], 1.5)

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
        self.assertEqual(result["agent_network"], "disabled")
        self.assertEqual(result["agent_mode"], "sandboxed")
        serialized = json.dumps(result)
        self.assertNotIn(str(self.config.val_path), serialized)
        self.assertNotIn(str(self.config.test_path), serialized)

    def test_mailbox_rejects_agent_created_request_symlinks(self):
        supervisor = self.supervisor()
        request = supervisor.inbox / "attack.json"
        request.symlink_to(supervisor.state_path)
        self.assertTrue(supervisor.process_one_request())
        response = json.loads(
            (supervisor.outbox / "attack.json").read_text(encoding="utf-8")
        )
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "invalid_request")
        self.assertTrue(supervisor.state_path.is_file())

    def test_host_cli_runner_is_recorded_as_non_compliant(self):
        config = replace(
            self.config,
            agent_runner="host_cli",
            agent_mode=None,
            inference=None,
        )
        supervisor = self.supervisor(config=config)
        self.assertEqual(supervisor.state["agent_system"]["runner"], "host_cli")
        self.assertIsNone(supervisor.state["agent_system"]["image"])
        self.assertEqual(supervisor.state["agent_system"]["network"], "host_managed")
        self.assertFalse(supervisor.state["agent_system"]["policy_compliant"])
        usage = {"tool": "codex", "total_tokens": 42}
        supervisor.finish()
        supervisor.record_agent_execution(
            exit_code=0,
            usage=usage,
            termination_reason="exited",
            drain_seconds=2.5,
        )
        self.assertEqual(supervisor.state["agent_system"]["exit_code"], 0)
        self.assertEqual(
            supervisor.state["agent_system"]["termination_reason"], "exited"
        )
        self.assertTrue(supervisor.state["agent_system"]["finish_observed"])
        self.assertEqual(supervisor.state["agent_system"]["drain_seconds"], 2.5)
        self.assertEqual(supervisor.state["agent_usage"], usage)
        status = supervisor.public_status()
        self.assertEqual(status["agent_runner"], "host_cli")
        self.assertIsNone(status["agent_mode"])
        self.assertEqual(status["agent_usage"], usage)
        self.assertEqual(status["agent_exit_code"], 0)
        self.assertEqual(status["agent_termination_reason"], "exited")
        self.assertTrue(status["agent_finish_observed"])
        self.assertEqual(status["agent_drain_seconds"], 2.5)
        self.assertFalse(status["policy_compliant"])
        self.assertIn("agent_execution_boundary", status["policy_violations"])
        self.assertEqual(status["policy_violations"], ["agent_execution_boundary"])

    def test_registered_inference_usage_is_summarized_from_trusted_trace(self):
        inference = InferenceConfig(
            protocol="openai_compatible",
            base_url="https://api.example.com",
            api_key_env=None,
            allowed_models=("researcher-a",),
            allowed_paths=("/v1/responses",),
            timeout_seconds=30,
        )
        config = replace(self.config, inference=inference)
        supervisor = self.supervisor(config=config)

        records = [
            {
                "type": "INFERENCE_STARTED",
                "request_id": "request-1",
                "model": "researcher-a",
                "path": "/v1/responses",
            },
            {
                "type": "INFERENCE_COMPLETED",
                "request_id": "request-1",
                "model": "researcher-a",
                "path": "/v1/responses",
                "status": 200,
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "total_tokens": 10,
                },
            },
        ]
        supervisor.inference_trace_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        summary = supervisor.refresh_inference_usage()
        self.assertEqual(summary["model_calls"], 1)
        self.assertEqual(summary["total_tokens"], 10)
        self.assertTrue(summary["usage_complete"])
        state = json.loads(supervisor.state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["inference_usage"], summary)



if __name__ == "__main__":
    unittest.main()
