import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from envcore.cli import build_parser, command_report
from envcore.operator_status import format_operator_status, read_operator_status
from tests.test_results import _finalized


NOW = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)


def active_state():
    return _finalized(
        episode_id="episode-1",
        status="active",
        outcome=None,
        final_candidate_id=None,
        test_status=None,
        test_cost=None,
        test_performance_score=None,
        test_raw_metrics=None,
        research_sealed_at=None,
        closure_reason=None,
        analysis_complete=None,
        analysis_status=None,
        analysis_identity_verified=None,
        analysis_trajectory=[],
        candidate_commits=[],
        incumbent_declarations=[],
        agent_system={"runner": "host_cli", "termination_reason": None},
        active_lease={
            "lease_id": "lease-0001",
            "resource_allocation_receipt_id": "receipt:lease-0001",
        },
    )


class OperatorStatusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.state_path = self.root / "state.json"
        self.observation_path = self.root / "observation.json"

    def state(self, payload=None, age=5):
        payload = active_state() if payload is None else payload
        self.state_path.write_text(json.dumps(payload), encoding="utf-8")
        stamp = (NOW - timedelta(seconds=age)).timestamp()
        os.utime(self.state_path, (stamp, stamp))

    def observation(self, **overrides):
        observation = {
            "schema_version": 1,
            "episode_id": "episode-1",
            "source": "test deployment monitor",
            "observed_at": (NOW - timedelta(seconds=5)).isoformat(),
            "execution_state": "running",
        }
        observation.update(overrides)
        self.observation_path.write_text(json.dumps(observation), encoding="utf-8")

    def read(self):
        return read_operator_status(
            self.root, observation_path=self.observation_path, now=NOW
        )

    def test_fresh_core_snapshot_alone_does_not_prove_liveness(self):
        self.state()
        view = read_operator_status(self.root, now=NOW)
        self.assertEqual(view["core"]["freshness"], "fresh")
        self.assertEqual(view["core"]["phase"], "research")
        self.assertEqual(view["summary"], "unknown")
        self.assertEqual(view["execution_state"], "unknown")
        self.assertIn("cannot be confirmed", format_operator_status(view))

    def test_long_running_command_can_have_old_core_snapshot_and_fresh_execution(self):
        self.state(age=3600)
        self.observation()
        view = self.read()
        self.assertEqual(view["summary"], "running")
        self.assertEqual(view["core"]["freshness"], "stale")
        self.assertEqual(view["core"]["active_lease_id"], "lease-0001")
        self.assertEqual(view["workload_cleanup"], "unknown")

    def test_terminated_execution_does_not_rewrite_active_core_or_create_result(self):
        self.state(age=3600)
        self.observation(execution_state="terminated", storage_cleanup="confirmed")
        before = self.state_path.read_bytes()
        view = self.read()
        self.assertEqual(view["summary"], "interrupted")
        self.assertIsNone(view["core"]["outcome"])
        self.assertEqual(view["core"]["status"], "active")
        self.assertIsNone(view["core"]["test_performance_score"])
        self.assertEqual(view["workload_cleanup"], "unknown")
        self.assertEqual(view["storage_cleanup"], "confirmed")
        self.assertEqual(self.state_path.read_bytes(), before)

    def test_stale_or_future_observation_cannot_prove_execution_or_cleanup(self):
        self.state()
        for offset, freshness in ((-61, "stale"), (10, "future")):
            with self.subTest(offset=offset):
                self.observation(
                    observed_at=(NOW + timedelta(seconds=offset)).isoformat(),
                    workload_cleanup="confirmed",
                    storage_cleanup="confirmed",
                )
                view = self.read()
                self.assertEqual(view["deployment"]["freshness"], freshness)
                self.assertEqual(view["summary"], "unknown")
                self.assertEqual(view["workload_cleanup"], "unknown")
                self.assertEqual(view["storage_cleanup"], "unknown")

    def test_explicit_freshness_limit_is_honored(self):
        self.state()
        self.observation(observed_at=(NOW - timedelta(seconds=90)).isoformat())
        view = read_operator_status(
            self.root,
            observation_path=self.observation_path,
            now=NOW,
            max_age_seconds=120,
        )
        self.assertEqual(view["summary"], "running")
        for invalid in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                read_operator_status(self.root, max_age_seconds=invalid, now=NOW)

    def test_final_state_wins_over_old_progress_and_stale_running_observation(self):
        self.state(
            _finalized(
                episode_id="episode-1",
                closure_reason="agent_finished",
                agent_system={
                    "runner": "host_cli",
                    "termination_reason": "research_closed",
                    "exit_code": -9,
                },
                agent_control={"cleanup_complete": True},
            ),
            age=86400,
        )
        (self.root / "progress.json").write_text(
            json.dumps(
                {"status": "closed", "outcome": None, "analysis_status": "pending"}
            )
        )
        self.observation(observed_at=(NOW - timedelta(days=1)).isoformat())
        view = self.read()
        self.assertEqual(view["summary"], "finalized")
        self.assertEqual(view["core"]["outcome"], "success")
        self.assertEqual(view["core"]["analysis_status"], "complete")
        self.assertEqual(view["core"]["closure_reason"], "agent_finished")
        self.assertEqual(view["core"]["agent_termination_reason"], "research_closed")
        self.assertEqual(view["execution_state"], "unknown")
        self.assertEqual(view["workload_cleanup"], "unknown")
        self.assertIn("TEST score: 100.0", format_operator_status(view))

    def test_bad_deployment_input_cannot_hide_a_final_result(self):
        self.state(_finalized(episode_id="episode-1"))
        for content in ("{", '{"schema_version": 999}', '["invalid"]'):
            with self.subTest(content=content):
                self.observation_path.write_text(content)
                view = self.read()
                self.assertEqual(view["summary"], "finalized")
                self.assertEqual(view["deployment"]["availability"], "invalid")
        self.observation_path.unlink()
        self.assertEqual(self.read()["summary"], "finalized")

    def test_prior_result_with_null_optional_metadata_retains_terminal_outcome(self):
        self.state(
            _finalized(
                episode_id="episode-1",
                result_schema_version=12,
                agent_system=None,
                agent_control=None,
                policy_compliant=False,
                leaderboard_eligible=False,
            )
        )
        view = self.read()
        self.assertEqual(view["summary"], "finalized")
        self.assertIsNone(view["core"]["agent_termination_reason"])
        self.assertNotIn("compliant", format_operator_status(view))

    def test_wrong_episode_observation_is_not_used(self):
        self.state()
        self.observation(episode_id="another-episode", execution_state="terminated")
        view = self.read()
        self.assertEqual(view["summary"], "unknown")
        self.assertEqual(view["deployment"]["availability"], "mismatched")

    def test_missing_or_bad_core_does_not_assert_failure_or_trust_unbound_observation(
        self,
    ):
        self.observation(execution_state="terminated")
        for content, availability in (
            (None, "missing"),
            ("{", "invalid"),
            ('{"result_schema_version": 999}', "invalid"),
        ):
            with self.subTest(content=content):
                if content is not None:
                    self.state_path.write_text(content)
                view = self.read()
                self.assertEqual(view["summary"], "unknown")
                self.assertEqual(view["core"]["availability"], availability)
                self.assertEqual(view["deployment"]["availability"], "unbound")
                self.assertEqual(
                    view["deployment"]["source"], "test deployment monitor"
                )
        self.state()
        self.assertEqual(self.read()["summary"], "interrupted")

    def test_naive_timestamps_and_unknown_fields_are_rejected(self):
        self.state()
        for override in (
            {"observed_at": "2026-09-10T12:00:00"},
            {"schema_version": True},
            {"research_closure": "wrong-field"},
            {"execution_state": "COMPLETED"},
            {"workload_cleanup": True},
        ):
            with self.subTest(override=override):
                self.observation(**override)
                self.assertEqual(self.read()["deployment"]["availability"], "invalid")

    def test_closed_research_and_postmortem_phases_are_snapshot_facts(self):
        state = _finalized(
            episode_id="episode-1",
            status="closed",
            outcome=None,
            test_status="pending",
            test_performance_score=None,
            test_raw_metrics=None,
            test_cost=None,
            analysis_status="pending",
            analysis_complete=False,
            analysis_trajectory=[],
            agent_system={"termination_reason": None},
            agent_control={"cleanup_complete": False},
        )
        self.state(state)
        self.observation()
        self.assertEqual(self.read()["core"]["phase"], "research_teardown")
        state["agent_system"]["termination_reason"] = "research_closed"
        state["agent_control"]["cleanup_complete"] = True
        self.state(state)
        self.assertEqual(self.read()["core"]["phase"], "test_pending")
        state["test_status"] = "evaluated"
        self.state(state)
        self.assertEqual(self.read()["core"]["phase"], "analysis_pending")

    def test_cli_text_and_json_status_do_not_change_existing_report(self):
        self.state(_finalized(episode_id="episode-1"))
        parser = build_parser()
        for flags in ([], ["--json"]):
            args = parser.parse_args(["status", "--episode", str(self.root), *flags])
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(args.function(args), 0)
            if flags:
                self.assertEqual(json.loads(output.getvalue())["summary"], "finalized")
            else:
                self.assertIn("Summary: finalized", output.getvalue())
        args = parser.parse_args(["report", "--episode", str(self.root)])
        with redirect_stdout(io.StringIO()) as output:
            command_report(args)
        report = json.loads(output.getvalue())
        self.assertEqual(report["result_schema_version"], 18)
        self.assertEqual(report["outcome"], "success")
        self.assertNotIn("deployment", report)

    def test_one_unreadable_source_does_not_block_reading_the_other(self):
        self.state(_finalized(episode_id="episode-1"))
        self.observation()
        open_path = Path.open

        def deny_observation(path, *args, **kwargs):
            if path == self.observation_path:
                raise PermissionError("observation access denied")
            return open_path(path, *args, **kwargs)

        with patch.object(Path, "open", deny_observation):
            view = self.read()
        self.assertEqual(view["summary"], "finalized")
        self.assertEqual(view["deployment"]["availability"], "unreadable")

    def test_cli_accepts_deployment_observation_and_marks_interruption(self):
        self.state()
        self.observation(execution_state="terminated")
        args = build_parser().parse_args(
            [
                "status",
                "--episode",
                str(self.root),
                "--observation",
                str(self.observation_path),
                "--max-age-seconds",
                "120",
                "--json",
            ]
        )

        def at_fixed_time(*args, **kwargs):
            return read_operator_status(*args, now=NOW, **kwargs)

        with patch("envcore.cli.read_operator_status", side_effect=at_fixed_time):
            with redirect_stdout(io.StringIO()) as output:
                args.function(args)
        view = json.loads(output.getvalue())
        self.assertEqual(view["summary"], "interrupted")
        self.assertEqual(view["core"]["status"], "active")


if __name__ == "__main__":
    unittest.main()
