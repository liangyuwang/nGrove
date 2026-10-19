import ast
import hashlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ngrove.branching import (
    BranchError,
    BranchSession,
    RunnerCapabilities,
    SessionPolicy,
)
from ngrove.branching.artifacts import ArtifactStore
from ngrove.branching.client import BranchClient, BranchClientError
from ngrove.branching.control import ControlEndpoint, publish_client_bundle
from ngrove.branching.monitor import Monitor
from ngrove.branching.recording import episode_segment
from ngrove.branching.contracts import identity
from envcore.cli import build_parser


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def reference(value, format="fixture-context-v1"):
    return {
        "format": format,
        "sha256": hashlib.sha256(value.encode()).hexdigest(),
        "locator": "private-adapter:" + value,
    }


class BranchingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clock = Clock()
        self.state = {
            "agent_context": reference("context"),
            "workspace": reference("workspace", "directory-v1"),
        }
        self.capabilities = RunnerCapabilities(
            "fixture-runner",
            "fixture-probe",
            {"agent_context": ("fixture-context-v1",), "workspace": ("directory-v1",)},
            context_modes=("exact", "replay"),
            prefix_modes=("shared", "copy"),
        )
        self.session = self.create()
        self.root_branch = self.session.status()["root_branch"]
        self.endpoint = ControlEndpoint(self.session, self.root_branch)
        self.client = BranchClient(self.endpoint)
        self.counter = 0

    def create(self, **policy):
        values = {
            "capacity": {"gpu": 4},
            "wall_seconds": 10000,
            "research_seconds": 10000,
            "branch_seconds": 1000,
            **policy,
        }
        return BranchSession.create(
            self.root / (str(len(list(self.root.glob("*.db")))) + ".db"),
            policy=SessionPolicy(**values),
            capabilities=self.capabilities,
            initial_state=self.state,
            resources={"gpu": 1},
            clock=self.clock,
        )

    def branch(self, branch=None, session=None):
        session = session or self.session
        return session.status(private=True)["branches"][branch or self.root_branch]

    def ready(self, work=None, session=None):
        session = session or self.session
        work = work or session.claim("monitor")
        op = work["operation"]
        bid = work["branch"]["id"]
        allocation = "allocation-" + bid
        session.allocated(
            op["id"],
            op["token"],
            allocation_id=allocation,
            resources=work["branch"]["resources"],
        )
        receipt = {
            "snapshot": work["branch"]["head"],
            "allocation_id": allocation,
            "episode_id": "episode-" + bid,
            "execution_held": True,
            "isolated_writes": True,
            "context_mode": work["branch"]["context_mode"],
            "prefix_mode": work["branch"]["prefix_mode"],
            "evidence_id": "restore-proof",
        }
        result = session.ready(op["id"], op["token"], receipt)
        return work, receipt, result

    def checkpoint(self, branch=None, state=None, records=()):
        branch = branch or self.root_branch
        self.counter += 1
        self.client.checkpoint(branch, request_id=f"checkpoint-{self.counter}")
        work = self.session.claim("monitor")
        op = work["operation"]
        self.session.quiesced(op["id"], op["token"], evidence_id="quiescence-proof")
        result = self.session.complete(
            op["id"],
            op["token"],
            state_components=state or self.state,
            records=records,
            execution_held=True,
        )
        return result["head"]

    def stop_work(self, work, session=None):
        session = session or self.session
        op = work["operation"]
        session.quiesced(op["id"], op["token"], evidence_id="quiescent")
        session.released(op["id"], op["token"], evidence_id="cleaned")
        return session.complete(op["id"], op["token"])

    def test_queue_prepare_ready_and_active_allocation_clocks_are_distinct(self):
        self.clock.advance(10)
        work = self.session.claim("monitor")
        op = work["operation"]
        self.clock.advance(20)
        self.session.allocated(
            op["id"],
            op["token"],
            allocation_id="allocation-" + self.root_branch,
            resources={"gpu": 1},
        )
        self.clock.advance(30)
        self.assertEqual(self.session.status()["research_seconds"], 0)
        self.assertEqual(self.session.status()["resource_seconds"]["gpu"], 30)
        _, receipt, permit = self.ready(work)
        self.assertFalse(permit["already_applied"])
        self.clock.advance(15)
        self.assertEqual(self.branch()["active_seconds"], 15)
        self.assertEqual(self.session.status()["resource_seconds"]["gpu"], 45)
        self.assertTrue(
            self.session.ready(op["id"], op["token"], receipt)["already_applied"]
        )

    def test_child_wait_does_not_pause_parent_or_reset_prefix_budget(self):
        self.ready()
        self.clock.advance(100)
        snapshot = self.checkpoint(records=[{"observation": "public experiment"}])
        request = self.client.fork(snapshot, resources={"gpu": 1}, context_mode="exact")
        self.clock.advance(40)
        work = self.session.claim("monitor")
        self.assertEqual(self.branch()["active_seconds"], 140)
        self.ready(work)
        self.clock.advance(30)
        child = self.branch(request["branch_id"])
        self.assertEqual(child["active_seconds"], 30)
        self.assertEqual(self.session._path_seconds(child), 130)
        self.assertEqual(self.session.status()["research_seconds"], 200)
        self.assertEqual(
            self.session.history(snapshot)["nodes"][-1]["records"],
            [{"observation": "public experiment"}],
        )

    def test_checkpoint_capture_excludes_active_time_but_preserves_allocation_charge(
        self,
    ):
        self.ready()
        self.clock.advance(10)
        self.client.checkpoint(self.root_branch)
        work = self.session.claim("monitor")
        op = work["operation"]
        self.clock.advance(5)  # Requested pause is not proof of quiescence.
        self.session.quiesced(op["id"], op["token"], evidence_id="stopped")
        self.clock.advance(20)
        result = self.session.complete(
            op["id"], op["token"], state_components=self.state, execution_held=True
        )
        self.assertEqual(result["active_seconds"], 15)
        self.assertEqual(self.session.status()["resource_seconds"]["gpu"], 35)
        self.clock.advance(4)
        self.assertEqual(self.branch()["active_seconds"], 19)
        retry = self.session.complete(
            op["id"], op["token"], state_components=self.state, execution_held=True
        )
        self.assertTrue(retry["already_applied"])

    def test_checkout_restores_old_node_without_mutating_original_branch(self):
        self.ready()
        self.clock.advance(5)
        first = self.checkpoint()
        changed = {**self.state, "agent_context": reference("later")}
        self.clock.advance(10)
        second = self.checkpoint(state=changed)
        req = self.client.checkout(
            first, resources={}, context_mode="exact", prefix_mode="copy"
        )
        work, _, _ = self.ready()
        self.assertEqual(work["snapshot"]["components"], self.state)
        self.assertEqual(self.branch()["head"], second)
        self.assertEqual(self.branch(req["branch_id"])["head"], first)
        self.assertEqual(self.branch(req["branch_id"])["prefix_mode"], "copy")

    def test_dag_merge_keeps_both_histories_and_deduplicates_shared_ancestry(self):
        self.ready()
        self.clock.advance(10)
        common = self.checkpoint()
        left = self.client.fork(common, resources={}, context_mode="exact")["branch_id"]
        self.ready()
        right = self.client.fork(common, resources={}, context_mode="exact")[
            "branch_id"
        ]
        self.ready()
        self.clock.advance(5)
        left_state = {**self.state, "agent_context": reference("left")}
        right_state = {**self.state, "workspace": reference("right", "directory-v1")}
        left_node = self.checkpoint(left, left_state, records=[{"side": "left"}])
        right_node = self.checkpoint(right, right_state, records=[{"side": "right"}])
        with self.assertRaisesRegex(BranchClientError, "explicit_component_selection"):
            self.client.merge(
                [left_node, right_node], {}, resources={}, context_mode="exact"
            )
        req = self.client.merge(
            [left_node, right_node],
            {"agent_context": left_node, "workspace": right_node},
            resources={},
            context_mode="exact",
        )
        work, _, _ = self.ready()
        self.assertEqual(
            work["snapshot"]["components"]["agent_context"], left_state["agent_context"]
        )
        self.assertEqual(
            work["snapshot"]["components"]["workspace"], right_state["workspace"]
        )
        self.assertEqual(self.session._path_seconds(self.branch(req["branch_id"])), 20)
        history = self.session.history(self.branch(req["branch_id"])["head"])
        self.assertEqual(sum(n["id"] == common for n in history["nodes"]), 1)
        self.assertEqual(history["nodes"][-1]["parents"], [left_node, right_node])

    def test_idempotent_requests_survive_restart_and_reject_different_intent(self):
        self.ready()
        head = self.branch()["head"]
        first = self.client.fork(
            head, resources={}, context_mode="exact", request_id="stable"
        )
        reopened = BranchSession(self.session.path, clock=self.clock)
        client = BranchClient(ControlEndpoint(reopened, self.root_branch))
        self.assertEqual(
            first,
            client.fork(head, resources={}, context_mode="exact", request_id="stable"),
        )
        with self.assertRaisesRegex(BranchClientError, "request_id_conflict"):
            client.fork(
                head, resources={"gpu": 1}, context_mode="exact", request_id="stable"
            )
        self.assertEqual(len(reopened.status()["branches"]), 2)

    def test_parallel_monitors_claim_once_and_never_overbook_capacity(self):
        self.ready()
        head = self.branch()["head"]
        for _ in range(8):
            self.client.fork(head, resources={"gpu": 1}, context_mode="exact")

        def claim(i):
            return BranchSession(self.session.path, clock=self.clock).claim(
                f"monitor-{i}"
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            claimed = [x for x in executor.map(claim, range(8)) if x]
        self.assertEqual(len(claimed), 3)
        self.assertEqual(len({x["operation"]["id"] for x in claimed}), 3)
        self.assertIsNone(self.session.claim("other"))
        self.client.stop(claimed[0]["branch"]["id"])
        # Cancellation alone never releases a claimed allocation reservation.
        self.assertIsNone(self.session.claim("other"))
        op = claimed[0]["operation"]
        self.session.quiesced(op["id"], op["token"], evidence_id="no-workloads")
        self.session.released(op["id"], op["token"], evidence_id="no-resources")
        self.session.fail_clean(op["id"], op["token"])
        self.assertIsNotNone(self.session.claim("other"))

    def test_queued_stop_is_idempotent_and_cancelling_checkpoint_still_stops(self):
        self.ready()
        self.client.checkpoint(self.root_branch)
        stop = self.client.stop(self.root_branch)
        again = self.client.stop(self.root_branch)
        self.assertEqual(stop["id"], again["id"])
        self.assertEqual(stop["kind"], "stop")
        self.stop_work(self.session.claim("monitor"))
        self.assertEqual(self.branch()["status"], "stopped")

    def test_queued_child_cancellation_has_no_platform_side_effect(self):
        self.ready()
        child = self.client.fork(
            self.branch()["head"], resources={}, context_mode="exact"
        )
        result = self.client.stop(child["branch_id"])
        self.assertEqual(result["status"], "cancelled")
        self.assertIsNone(self.session.claim("monitor"))
        self.assertEqual(self.branch(child["branch_id"])["status"], "stopped")

    def test_unsupported_context_fails_before_request_or_allocation(self):
        self.ready()
        before = self.session.status()
        with self.assertRaisesRegex(BranchClientError, "unsupported_restore_mode"):
            self.client.fork(
                self.branch()["head"], resources={}, context_mode="provider-magic"
            )
        self.assertEqual(
            len(self.session.status()["branches"]), len(before["branches"])
        )
        with self.assertRaisesRegex(BranchError, "unsupported_state_format"):
            BranchSession.create(
                self.root / "bad.db",
                policy=SessionPolicy({}, 10, 10, 10),
                capabilities=self.capabilities,
                initial_state={
                    **self.state,
                    "agent_context": reference("x", "unsupported"),
                },
                resources={},
            )
        self.assertFalse((self.root / "bad.db").exists())

    def test_explicit_replay_cannot_be_relabelled_exact(self):
        self.ready()
        replay = self.client.fork(
            self.branch()["head"], resources={}, context_mode="replay"
        )
        self.ready()
        node = self.checkpoint(replay["branch_id"])
        with self.assertRaisesRegex(BranchClientError, "cannot_upgrade"):
            self.client.fork(node, resources={}, context_mode="exact")

    def test_uncertain_calls_require_fenced_cleanup_and_old_tokens_cannot_commit(self):
        class Broken:
            def apply(inner, work, session):
                raise RuntimeError("private host and credential detail")

        with self.assertRaisesRegex(RuntimeError, "private host"):
            Monitor(self.session, "monitor", Broken()).run_once()
        old = self.session.status(private=True)["operations"]
        op = next(iter(old.values()))
        self.assertEqual(op["status"], "needs_recovery")
        self.assertNotIn("private host", json.dumps(self.session.status()))
        self.assertIn("private host", json.dumps(self.session.events()))
        recovered = self.session.recover(
            op["id"], monitor_id="replacement", fencing_evidence="executor-fenced"
        )
        with self.assertRaisesRegex(BranchError, "stale_operation_token"):
            self.session.quiesced(op["id"], op["token"], evidence_id="stale")
        new = recovered["operation"]
        with self.assertRaisesRegex(BranchError, "failure_requires_cleanup"):
            self.session.fail_clean(new["id"], new["token"])
        self.session.quiesced(new["id"], new["token"], evidence_id="stopped")
        self.session.released(new["id"], new["token"], evidence_id="cleaned")
        self.session.fail_clean(new["id"], new["token"])
        self.assertFalse(self.branch()["reserved"])

    def test_late_ready_is_rejected_and_resources_retained_for_cleanup(self):
        short = self.create(wall_seconds=5)
        work = short.claim("monitor")
        op = work["operation"]
        short.allocated(
            op["id"],
            op["token"],
            allocation_id="allocation-" + work["branch"]["id"],
            resources={"gpu": 1},
        )
        self.clock.advance(6)
        with self.assertRaises(BranchError):
            self.ready(work, short)
        status = short.status(private=True)
        self.assertEqual(status["status"], "closing")
        branch = status["branches"][work["branch"]["id"]]
        self.assertTrue(branch["reserved"])
        self.assertFalse(branch["clock_running"])
        self.assertEqual(status["resource_seconds"]["gpu"], 6)

    def test_pool_exhaustion_requests_stop_without_fabricating_termination(self):
        short = self.create(research_seconds=3)
        self.ready(session=short)
        self.clock.advance(4)
        status = short.status(private=True)
        self.assertEqual(status["status"], "closing")
        self.assertTrue(next(iter(status["branches"].values()))["clock_running"])
        self.assertEqual(short.claim("cleanup")["operation"]["kind"], "stop")

    def test_sealing_requires_every_branch_and_control_teardown(self):
        self.ready()
        self.client.fork(self.branch()["head"], resources={}, context_mode="exact")
        self.ready()
        self.client.finish(self.branch()["head"])
        with self.assertRaisesRegex(BranchError, "teardown_incomplete"):
            self.session.confirm_sealed(control_revoked=True, evidence_id="too-early")
        while True:
            work = self.session.claim("cleanup")
            if work is None:
                break
            self.stop_work(work)
        self.session.confirm_sealed(
            control_revoked=True, evidence_id="all-controls-revoked"
        )
        self.assertEqual(self.session.status()["status"], "sealed")
        self.endpoint.revoke()
        with self.assertRaisesRegex(BranchClientError, "control_revoked"):
            self.client.status()

    def test_public_views_exclude_locators_tokens_and_private_receipts(self):
        self.ready()
        public = json.dumps(self.client.history(self.branch()["head"])) + json.dumps(
            self.client.status()
        )
        self.assertNotIn("private-adapter", public)
        self.assertNotIn("allocation-", public)
        self.assertNotIn('"token"', public)
        self.assertIn(
            "private-adapter",
            json.dumps(self.session.history(self.branch()["head"], include_state=True)),
        )

    def test_large_history_is_bounded_and_can_be_traversed_one_node_at_a_time(self):
        self.ready()
        original = self.branch()["head"]
        head = self.checkpoint(records=[{"public": "experiment"}])
        with self.assertRaisesRegex(BranchError, "history_limit_use_snapshot"):
            self.session.history(head, max_nodes=1)
        node = self.client.snapshot(head)
        self.assertEqual(node["parents"], [original])
        self.assertEqual(node["records"], [{"public": "experiment"}])
        self.assertNotIn("private-adapter", json.dumps(node))
        self.assertEqual(self.client.snapshot(original)["parents"], [])
        with self.assertRaisesRegex(BranchError, "snapshot_metadata_limit"):
            self.create(max_payload_bytes=1)

    def test_snapshot_corruption_and_clock_regression_are_rejected(self):
        self.ready()
        head = self.branch()["head"]
        with sqlite3.connect(str(self.session.path)) as db:
            db.execute(
                "UPDATE snapshots SET body=? WHERE id=?",
                (json.dumps({"parents": []}), head),
            )
        with self.assertRaisesRegex(BranchError, "identity_mismatch"):
            self.session.history(head)
        self.clock.advance(-1)
        with self.assertRaisesRegex(BranchError, "clock_regressed"):
            self.session.status()

    def test_failed_transaction_does_not_leave_orphan_branch_or_event(self):
        self.ready()
        before = self.session.status(private=True)
        with patch.object(self.session, "_event", side_effect=OSError("disk full")):
            with self.assertLogs(
                "ngrove.branching.control", level="ERROR"
            ), self.assertRaisesRegex(BranchClientError, "branch_service_unavailable"):
                self.client.fork(
                    self.branch()["head"], resources={}, context_mode="exact"
                )
        after = self.session.status(private=True)
        self.assertEqual(before["branches"], after["branches"])
        self.assertEqual(before["requests"], after["requests"])
        self.assertEqual(before["sequence"], after["sequence"])

    def test_cli_surface_and_read_only_client_publication(self):
        args = build_parser().parse_args(
            ["branches", "status", "--database", str(self.session.path)]
        )
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(args.function(args), 0)
        self.assertEqual(json.loads(output.getvalue())["root_branch"], self.root_branch)
        bundle = publish_client_bundle(self.root / "public")
        files = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*.py")}
        self.assertEqual(
            files,
            {
                "ngrove/__init__.py",
                "ngrove/branching/__init__.py",
                "ngrove/branching/client.py",
            },
        )
        code = "import sys; sys.path.insert(0, sys.argv[1]); from ngrove.branching.client import BranchClient; print(BranchClient.__name__)"
        result = subprocess.run(
            [sys.executable, "-c", code, str(bundle)],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "BranchClient")

    def test_malformed_agent_requests_return_stable_errors(self):
        self.ready()
        head = self.branch()["head"]
        for request in (
            {"id": "bad", "op": [], "args": {}},
            {"id": "bad", "op": "stop", "args": {"branch": []}},
            {
                "id": "bad",
                "op": "fork",
                "args": {
                    "snapshot": head,
                    "resources": {"gpu": 10**1000},
                    "context_mode": "exact",
                    "prefix_mode": "shared",
                },
            },
            {
                "id": "bad",
                "op": "merge",
                "args": {
                    "sources": [[], []],
                    "selection": {},
                    "resources": {},
                    "context_mode": "exact",
                    "prefix_mode": "shared",
                },
            },
            {"id": "bad", "op": "fork", "args": {"snapshot": head}},
        ):
            with self.subTest(request=request):
                result = self.endpoint(request)
                self.assertFalse(result["ok"])
                self.assertIsInstance(result["error"], str)

    def test_combined_bundle_supports_both_clients_without_trusted_modules(self):
        bundle = publish_client_bundle(
            self.root / "combined-api", include_episode_api=True
        )
        code = """import importlib.util
from envcore.client import Client
from ngrove.branching.client import BranchClient
for name in ('envcore.supervisor', 'envcore.operator_status',
             'ngrove.branching.session', 'ngrove.branching.control',
             'ngrove.branching.monitor', 'ngrove.branching.artifacts',
             'ngrove.branching.cli'):
    assert importlib.util.find_spec(name) is None, name
print(Client.__name__, BranchClient.__name__)
"""
        environment = {
            **os.environ,
            # Keep the trusted source later on the path to verify that neither
            # public package extends its namespace into it.
            "PYTHONPATH": os.pathsep.join(
                [str(bundle), str(Path(__file__).resolve().parents[1])]
            ),
        }
        for arguments, expected_code, expected_text in (
            (["-c", code], 0, "Client BranchClient"),
            (["-m", "ngrove", "agent", "--help"], 0, "lease-start"),
            (["-m", "ngrove", "start", "--help"], 2, "invalid choice"),
            (["-m", "ngrove", "report", "--help"], 2, "invalid choice"),
            (["-m", "ngrove", "branches", "--help"], 2, "invalid choice"),
        ):
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    [sys.executable, "-B", "-S", *arguments],
                    cwd=self.root,
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, expected_code, result.stderr)
                self.assertIn(expected_text, result.stdout + result.stderr)
        if os.name == "posix":
            for path in (bundle, *bundle.rglob("*")):
                self.assertEqual(
                    path.stat().st_mode & 0o777, 0o555 if path.is_dir() else 0o444
                )

    def test_ready_cannot_bind_an_allocation_or_episode_twice(self):
        work, receipt, _ = self.ready()
        self.client.fork(
            self.branch()["head"], resources={"gpu": 1}, context_mode="exact"
        )
        child = self.session.claim("monitor")
        op = child["operation"]
        with self.assertRaisesRegex(BranchError, "allocation_already_bound"):
            self.session.allocated(
                op["id"],
                op["token"],
                allocation_id=receipt["allocation_id"],
                resources={"gpu": 1},
            )
        self.session.allocated(
            op["id"], op["token"], allocation_id="child", resources={"gpu": 1}
        )
        duplicate = {
            **receipt,
            "snapshot": child["branch"]["head"],
            "allocation_id": "child",
        }
        with self.assertRaisesRegex(BranchError, "episode_already_bound"):
            self.session.ready(op["id"], op["token"], duplicate)
        self.assertFalse(self.branch(child["branch"]["id"])["clock_running"])

    def test_ready_response_loss_does_not_authorize_a_second_launch(self):
        work, receipt, _ = self.ready()
        op = work["operation"]
        self.session.uncertain(
            op["id"], op["token"], detail="release acknowledgement lost"
        )
        with self.assertRaisesRegex(BranchError, "ready_not_admissible"):
            self.session.ready(op["id"], op["token"], receipt)
        self.assertTrue(self.branch()["reserved"])
        self.clock.advance(5)
        self.assertEqual(self.branch()["active_seconds"], 5)

    def test_expiry_during_checkpoint_capture_never_rearms_execution(self):
        session = self.create(wall_seconds=5)
        self.ready(session=session)
        root = session.status()["root_branch"]
        client = BranchClient(ControlEndpoint(session, root))
        client.checkpoint(root)
        work = session.claim("monitor")
        op = work["operation"]
        session.quiesced(op["id"], op["token"], evidence_id="paused")
        self.clock.advance(6)
        with self.assertRaisesRegex(BranchError, "completion_not_admissible"):
            session.complete(
                op["id"], op["token"], state_components=self.state, execution_held=True
            )
        self.assertFalse(session.branch_status(root)["research_active"])
        self.assertEqual(session.status()["status"], "closing")

    def test_seal_retry_preserves_intent_and_final_wall_clock(self):
        self.ready()
        snapshot = self.branch()["head"]
        self.client.finish(snapshot)
        self.client.finish(snapshot)
        self.stop_work(self.session.claim("monitor"))
        self.clock.advance(5)
        self.session.confirm_sealed(control_revoked=True, evidence_id="revoked")
        elapsed = self.session.status()["wall_seconds"]
        self.clock.advance(100)
        self.assertEqual(self.session.status()["wall_seconds"], elapsed)

    def test_target_can_poll_one_operation_without_loading_all_histories(self):
        self.ready()
        request = self.client.fork(
            self.branch()["head"], resources={}, context_mode="exact"
        )
        self.assertEqual(self.client.operation(request["id"])["status"], "queued")
        self.assertEqual(self.client.branch(request["branch_id"])["status"], "queued")
        self.assertNotIn("token", self.client.operation(request["id"]))

    def test_core_semantics_do_not_depend_on_branching(self):
        root = Path(__file__).resolve().parents[1]
        for name in (
            "supervisor.py",
            "worker_protocol.py",
            "recording.py",
            "run_config.py",
        ):
            tree = ast.parse((root / "envcore" / name).read_text())
            imports = [
                n.module
                for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.module
            ]
            self.assertFalse(any("branching" in module for module in imports))


class ArtifactAndProjectionTests(unittest.TestCase):
    def test_real_workspace_and_context_restore_with_shared_or_copied_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "original"
            workspace.mkdir()
            (workspace / "train.py").write_text("step = 12\n")
            (workspace / "empty").mkdir()
            store = ArtifactStore(root / "objects")
            tree = store.capture_directory(workspace)
            context = store.put_json(
                {"messages": [{"role": "user", "content": "continue"}], "step": 12},
                format="fixture-context-v1",
            )
            before = len(list(store.root.iterdir()))
            self.assertEqual(store.capture_directory(workspace), tree)
            self.assertEqual(len(list(store.root.iterdir())), before)
            store.restore_directory(tree, root / "branch-a")
            copied = ArtifactStore(root / "copied-objects")
            copied_tree = store.copy_component(tree, copied)
            copied_context = store.copy_component(context, copied)
            copied.restore_directory(copied_tree, root / "branch-b")
            self.assertEqual(json.loads(copied.read(copied_context))["step"], 12)
            (root / "branch-a" / "train.py").write_text("step = 13\n")
            self.assertEqual(
                (root / "branch-b" / "train.py").read_text(), "step = 12\n"
            )
            self.assertEqual((workspace / "train.py").read_text(), "step = 12\n")
            self.assertTrue((root / "branch-b" / "empty").is_dir())

    def test_snapshot_paths_and_corrupt_objects_cannot_escape_or_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ArtifactStore(root / "objects")
            bad = store.put_json(
                {"files": {"../outside": {}}, "directories": []}, format="directory-v1"
            )
            with self.assertRaisesRegex(BranchError, "invalid_snapshot_path"):
                store.restore_directory(bad, root / "destination")
            self.assertFalse((root / "outside").exists())
            ref = store.put(b"original", format="bytes")
            path = store.root / ref["sha256"]
            path.chmod(0o600)
            path.write_bytes(b"corrupted")
            with self.assertRaisesRegex(BranchError, "identity_mismatch"):
                store.read(ref)

    def test_projection_consumes_real_core_recording_without_hidden_evidence(self):
        from tests.test_recording import RecordingIntegrationTests

        fixture = RecordingIntegrationTests()
        fixture.setUp()
        try:
            supervisor = fixture.fixture.supervisor()
            fixture.exchange(supervisor, "status")
            supervisor.record_observation(
                source="operator",
                source_event_id="private",
                kind="diagnostic",
                data={"secret": "operator-only"},
            )
            segment = episode_segment(
                supervisor.episode_dir, through_seq=supervisor.sequence
            )
            self.assertTrue(segment["events"])
            self.assertTrue(
                any(e["type"] == "CONTROL_RESPONSE" for e in segment["events"])
            )
            self.assertNotIn("operator-only", json.dumps(segment))
            self.assertFalse(segment["complete_agent_context"])
        finally:
            fixture.doCleanups()

    def test_projection_rejects_missing_or_corrupt_events_in_its_interval(self):
        from tests.test_recording import RecordingIntegrationTests

        fixture = RecordingIntegrationTests()
        fixture.setUp()
        try:
            supervisor = fixture.fixture.supervisor()
            fixture.exchange(supervisor, "status")
            fixture.exchange(supervisor, "status")
            journal = supervisor.events_path
            lines = journal.read_text().splitlines()
            index = next(
                i
                for i, line in enumerate(lines)
                if json.loads(line)["type"] == "ACTION_RETURNED"
            )
            for replacement in ([], ["invalid JSON"]):
                with self.subTest(replacement=replacement):
                    journal.write_text(
                        "\n".join(lines[:index] + replacement + lines[index + 1 :])
                        + "\n"
                    )
                    with self.assertRaisesRegex(
                        BranchError, "recording_interval_incomplete"
                    ):
                        episode_segment(
                            supervisor.episode_dir, through_seq=supervisor.sequence
                        )
        finally:
            fixture.doCleanups()

    def test_projection_completion_is_scoped_to_the_selected_boundary(self):
        from tests.test_recording import RecordingIntegrationTests

        fixture = RecordingIntegrationTests()
        fixture.setUp()
        try:
            supervisor = fixture.fixture.supervisor()
            fixture.exchange(supervisor, "status")
            events = [
                json.loads(line)
                for line in supervisor.events_path.read_text().splitlines()
            ]
            start = next(e for e in events if e["type"] == "ACTION_STARTED")
            end = next(e for e in events if e["type"] == "ACTION_RETURNED")
            expected_issue = {
                "code": "action_completion_unknown",
                "event_id": start["event_id"],
            }
            prefix = episode_segment(supervisor.episode_dir, through_seq=start["seq"])
            self.assertEqual(prefix["issues"], [expected_issue])
            complete = episode_segment(supervisor.episode_dir, through_seq=end["seq"])
            self.assertEqual(complete["issues"], [])

            # A complete journal can still record an interrupted action. Record
            # a public start without a return and keep the uncertainty explicit.
            with patch.object(
                supervisor, "_handle_request", side_effect=KeyboardInterrupt
            ):
                with self.assertRaises(KeyboardInterrupt):
                    supervisor.handle({"op": "status"})
            pending = episode_segment(
                supervisor.episode_dir, through_seq=supervisor.sequence
            )
            self.assertEqual(len(pending["issues"]), 1)
            self.assertEqual(pending["issues"][0]["code"], "action_completion_unknown")
        finally:
            fixture.doCleanups()

    def test_projection_ignores_gaps_and_private_issues_outside_its_interval(self):
        events = [
            {
                "seq": seq,
                "event_id": f"ep:{seq}",
                "type": "CONTROL_RESPONSE",
                "phase": "research",
                "visibility": "agent_exchange",
            }
            for seq in (1, 3, 4, 6)
        ]
        recording = {
            "episode_id": "ep",
            "events": events,
            "payloads": {},
            "issues": [
                {"code": "event_gap", "event_id": "ep:3"},
                {"code": "event_gap", "event_id": "ep:6"},
                {
                    "code": "action_completion_unknown",
                    "action_id": "private-action",
                },
                {"code": "invalid_event", "detail": "private-diagnostic"},
            ],
        }
        with patch("ngrove.branching.recording.read_recording", return_value=recording):
            segment = episode_segment(Path("unused"), after_seq=2, through_seq=4)
        self.assertEqual(segment["issues"], [])
        self.assertEqual([e["seq"] for e in segment["events"]], [3, 4])
        self.assertNotIn("private-action", json.dumps(segment))
        self.assertNotIn("private-diagnostic", json.dumps(segment))

    def test_demo_exercises_full_shared_copy_merge_and_cleanup_lifecycle(self):
        root = Path(__file__).resolve().parents[1]
        import os

        result = subprocess.run(
            [sys.executable, str(root / "examples/branching/demo.py")],
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root)},
            text=True,
            capture_output=True,
            check=True,
            timeout=20,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "sealed")
        self.assertEqual(payload["branches"], 4)
        self.assertEqual(payload["shared_prefix_occurrences"], 1)

    def test_projection_never_exports_postmortem_or_private_events(self):
        public = {"ok": True, "result": {"lease_id": "lease-1"}}
        digest = identity(public)
        event = {
            "event_id": "ep:1",
            "seq": 1,
            "phase": "research",
            "type": "CONTROL_RESPONSE",
            "visibility": "agent_exchange",
            "payload": {"status": "saved", "sha256": digest},
        }
        hidden = {**event, "seq": 2, "event_id": "ep:2", "phase": "after_seal"}
        private = {**event, "seq": 3, "event_id": "ep:3", "visibility": "operator_only"}
        value = {
            "episode_id": "ep",
            "events": [event, hidden, private],
            "payloads": {digest: public},
            "postmortem": {"secret_score": 99},
            "usage": {"private_limit": 5},
        }
        with patch("ngrove.branching.recording.read_recording", return_value=value):
            segment = episode_segment(Path("unused"), through_seq=3)
        self.assertEqual(len(segment["events"]), 1)
        self.assertNotIn("secret_score", json.dumps(segment))
        self.assertNotIn("private_limit", json.dumps(segment))
        self.assertEqual(segment["events"][0]["consumption"], "unknown")


if __name__ == "__main__":
    unittest.main()
