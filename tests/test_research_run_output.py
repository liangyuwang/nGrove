"""Research runs retain execution evidence without comparison qualification."""

import copy
import json
import unittest
from unittest.mock import patch

from envcore.operator_status import format_operator_status, read_operator_status
from envcore.recording import read_recording
from envcore.results import ResultSchemaError, normalize_result_report
from envcore.worker_protocol import BASELINE_REQUIREMENTS, assess_worker
from tests.test_results import _finalized
from tests import test_supervisor


COMPARISON_FIELDS = {
    "benchmark",
    "benchmark_release",
    "benchmark_manifest_sha256",
    "comparison_key",
    "leaderboard_eligible",
    "leaderboard_eligibility_violations",
    "formal_release_verified",
    "formal_release_verification",
    "formal_conformance",
    "policy_compliant",
    "evaluator_policy_compliant",
    "conformance",
    "conformance_satisfied",
    "trust_satisfied",
}


class ResearchRunOutputTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_supervisor.SupervisorTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def assert_research_only(self, value):
        if isinstance(value, dict):
            self.assertFalse(COMPARISON_FIELDS.intersection(value), value.keys())
            for nested in value.values():
                self.assert_research_only(nested)
        elif isinstance(value, list):
            for nested in value:
                self.assert_research_only(nested)
        elif isinstance(value, str):
            self.assertNotEqual(value, "benchmark_release")

    def complete(self, supervisor):
        supervisor.handle({"op": "status"})
        supervisor.start_lease()
        candidate = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(candidate["candidate_id"])
        supervisor.finish()
        self.fixture.finalize_after_agent_stop(supervisor)
        return normalize_result_report(supervisor.state)

    def test_research_lifecycle_never_assesses_or_exports_comparison_qualification(
        self,
    ):
        with patch(
            "envcore.benchmark.benchmark_eligibility_violations",
            side_effect=AssertionError("qualification must be explicitly selected"),
        ), patch(
            "envcore.benchmark.assess_benchmark_worker",
            side_effect=AssertionError("certification must be explicitly selected"),
        ):
            supervisor = self.fixture.supervisor()
            self.assert_research_only(
                json.loads(supervisor.public_task_path.read_text())
            )
            self.assert_research_only(supervisor.public_status())
            self.assert_research_only(self.complete(supervisor))
            report = normalize_result_report(supervisor.state)
            self.assert_research_only(report)
            self.assert_research_only(json.loads(supervisor.state_path.read_text()))
            for line in supervisor.events_path.read_text().splitlines():
                self.assert_research_only(json.loads(line))
            self.assert_research_only(read_recording(self.fixture.episode))
            view = read_operator_status(self.fixture.episode)
            self.assert_research_only(view)
            output = format_operator_status(view).lower()
            for word in ("benchmark", "leaderboard", "compliant", "comparison key"):
                self.assertNotIn(word, output)
        self.assertEqual(
            report["cost_contract"]["comparison_requires"],
            ["hidden_dataset_identity", "effective_resources"],
        )
        self.assertEqual(report["outcome"], "success")
        self.assertEqual(self.fixture.worker.evaluations, ["test", "analysis"])
        self.assertIn("agent_writable_storage_unbounded", report["execution_issues"])
        self.assertEqual(
            report["performance_score_contract"]["comparison_scope"], "task_release"
        )

    def test_execution_failure_keeps_concrete_evidence_without_qualification(self):
        supervisor = self.fixture.supervisor()
        supervisor.deployment_teardown_failed("research_worker_teardown_failure")
        self.assertIn(
            "deployment_teardown_incomplete", supervisor.state["execution_issues"]
        )
        self.assertIn(
            "research_worker_teardown_failure",
            supervisor.state["deployment_teardown_failures"],
        )
        self.assert_research_only(supervisor.public_status())
        self.assert_research_only(supervisor.state)

    def test_capability_admission_has_no_certification_requirement(self):
        assessment = assess_worker(
            self.fixture.worker.describe_capabilities(), BASELINE_REQUIREMENTS
        )
        self.assertTrue(assessment.satisfied)
        self.assert_research_only(assessment.public_manifest())

    def test_explicit_benchmark_preserves_qualification_and_operator_output(self):
        supervisor, _ = self.fixture.formal_supervisor()
        report = self.complete(supervisor)
        self.assertTrue(report["benchmark"]["leaderboard_eligible"])
        historical = copy.deepcopy(report)
        historical["result_schema_version"] = 17
        original = copy.deepcopy(historical)
        self.assertEqual(normalize_result_report(historical), original)
        self.assertEqual(historical, original)
        self.assertNotIn("leaderboard_eligible", report)
        self.assert_research_only(report["agent_system"])
        self.assert_research_only(report["workers"])
        self.assertIn("benchmark", read_recording(self.fixture.episode))
        text = format_operator_status(read_operator_status(self.fixture.episode))
        self.assertIn("Benchmark comparison key:", text)
        self.assertIn("leaderboard eligible: true", text)
        for field in ("leaderboard_eligible", "policy_compliant", "comparison_key"):
            changed = copy.deepcopy(report)
            changed[field] = False
            with self.subTest(field=field), self.assertRaises(ResultSchemaError):
                normalize_result_report(changed)
        changed = copy.deepcopy(report)
        changed["benchmark"]["policy_violations"].append("injected")
        with self.assertRaises(ResultSchemaError):
            normalize_result_report(changed)

    def test_benchmark_certification_cannot_disagree_with_execution_evidence(self):
        supervisor, _ = self.fixture.formal_supervisor()
        report = self.complete(supervisor)
        for mutation in (
            "missing_workers",
            "missing_receipt",
            "different_worker",
            "trust",
        ):
            changed = copy.deepcopy(report)
            if mutation == "missing_workers":
                changed["benchmark"]["workers"] = {}
            elif mutation == "missing_receipt":
                changed["benchmark"]["workers"]["research"]["conformance"] = None
            elif mutation == "different_worker":
                changed["benchmark"]["workers"]["research"]["conformance"][
                    "worker_id"
                ] = "other"
            else:
                changed["workers"]["research"]["trust_level"] = "development"
            with self.subTest(mutation=mutation), self.assertRaises(ResultSchemaError):
                normalize_result_report(changed)

    def test_benchmark_stays_bound_after_failed_result_publication(self):
        supervisor, _ = self.fixture.formal_supervisor()
        supervisor.start_lease()
        candidate = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(candidate["candidate_id"])
        supervisor.finish()
        supervisor.record_agent_execution(exit_code=0, termination_reason="exited")
        self.assertTrue(supervisor.cleanup_agent_control())
        supervisor.evaluate_postmortem()
        original_save = supervisor._save_state

        def fail_final_save():
            if supervisor.state["status"] == "finalized":
                raise OSError("publication failed")
            original_save()

        with patch.object(supervisor, "_save_state", side_effect=fail_final_save):
            with self.assertRaisesRegex(OSError, "publication failed"):
                supervisor.publish_postmortem_result()
        self.assertIs(supervisor.benchmark.record, supervisor.state["benchmark"])
        supervisor.deployment_teardown_failed("agent_control_teardown_failure")
        report = normalize_result_report(supervisor.state)
        self.assertFalse(report["benchmark"]["leaderboard_eligible"])
        self.assertIn(
            "deployment_teardown_incomplete", report["benchmark"]["policy_violations"]
        )

    def test_historical_result_is_read_without_rewriting_or_reclassifying_it(self):
        historical = _finalized(
            result_schema_version=16,
            leaderboard_eligible=False,
            policy_compliant=False,
            policy_violations=["agent_execution_boundary"],
        )
        path = self.fixture.episode.parent / "historical.json"
        path.write_text(json.dumps(historical))
        original = path.read_bytes()
        report = normalize_result_report(json.loads(path.read_text()))
        self.assertEqual(report["result_schema_version"], 16)
        self.assertFalse(report["leaderboard_eligible"])
        self.assertEqual(path.read_bytes(), original)
        self.assertNotIn("benchmark", report)

    def test_empty_extension_cannot_silently_enable_comparison(self):
        supervisor = self.fixture.supervisor()
        for value in (None, {}, False):
            with self.subTest(value=value), self.assertRaises(ResultSchemaError):
                normalize_result_report({**supervisor.state, "benchmark": value})


if __name__ == "__main__":
    unittest.main()
