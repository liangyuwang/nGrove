import math
import unittest

from autotrain_bench.results import (
    AGENT_MODE_RESULT_SCHEMA_VERSION,
    PRE_ANALYSIS_RESULT_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    ResultSchemaError,
    normalize_result_report,
    perplexity_from_nll,
    validated_nll,
)


def _analysis_point(candidate_id="sha256:candidate"):
    return {
        "commit_index": 1,
        "event_seq": 1,
        "candidate_id": candidate_id,
        "source_lease": "lease-0001",
        "elapsed_seconds": 1.0,
        "declared_incumbent": True,
        "analysis_status": "evaluated",
        "analysis_nll": 0.0,
        "analysis_perplexity": 1.0,
        "analysis_error": None,
    }


def _finalized(*, schema_version=RESULT_SCHEMA_VERSION, **overrides):
    """A minimal valid finalized result, so each test varies only one thing."""

    payload = {
        "result_schema_version": schema_version,
        "status": "finalized",
        "outcome": "success",
        "final_candidate_id": "sha256:candidate",
        "test_nll": 0.0,
        "test_perplexity": 1.0,
        "failure": None,
        "research_sealed_at": "2026-01-01T00:00:00+00:00",
        "analysis_complete": True,
        "analysis_trajectory": [_analysis_point()],
    }
    payload.update(overrides)
    return payload


class ResultMetricTests(unittest.TestCase):
    def test_nll_validation_does_not_coerce_non_numeric_json_values(self):
        invalid = (True, "1.25", None, float("nan"), float("inf"), -0.1, 10**1000)
        for value in invalid:
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(ValueError):
                    validated_nll(value)

    def test_perplexity_is_derived_from_mean_nll(self):
        self.assertEqual(validated_nll(0), 0.0)
        self.assertAlmostEqual(perplexity_from_nll(1.5), math.exp(1.5))

    def test_current_result_schema_rejects_inconsistent_states(self):
        invalid = {
            "terminal_without_outcome": {
                "status": "finalized",
                "outcome": None,
                "test_nll": None,
                "test_perplexity": None,
                "failure": None,
                "research_sealed_at": "2026-01-01T00:00:00+00:00",
                "analysis_complete": False,
                "analysis_trajectory": [_analysis_point()],
            },
            "success_without_candidate": {
                "status": "finalized",
                "outcome": "success",
                "final_candidate_id": None,
                "test_nll": 0.0,
                "test_perplexity": 1.0,
                "failure": None,
                "research_sealed_at": "2026-01-01T00:00:00+00:00",
                "analysis_complete": True,
                "analysis_trajectory": [_analysis_point()],
            },
            "active_failure": {
                "status": "active",
                "outcome": "failed",
                "test_nll": None,
                "test_perplexity": None,
                "failure": "unexpected",
            },
            "pending_with_metrics": {
                "status": "active",
                "outcome": None,
                "test_nll": 0.0,
                "test_perplexity": 1.0,
                "failure": None,
                "research_sealed_at": "2026-01-01T00:00:00+00:00",
                "analysis_complete": False,
                "analysis_trajectory": [],
            },
            "success_with_active_lease": {
                "status": "finalized",
                "outcome": "success",
                "final_candidate_id": "sha256:candidate",
                "active_lease": {"lease_id": "lease-0001"},
                "test_nll": 0.0,
                "test_perplexity": 1.0,
                "failure": None,
            },
            "success_with_pending_teardown": {
                "status": "finalized",
                "outcome": "success",
                "final_candidate_id": "sha256:candidate",
                "pending_lease_teardowns": ["lease-0001"],
                "test_nll": 0.0,
                "test_perplexity": 1.0,
                "failure": None,
            },
            "finalized_with_empty_publication": {
                "status": "finalized",
                "outcome": "success",
                "final_candidate_id": "sha256:candidate",
                "test_nll": 0.0,
                "test_perplexity": 1.0,
                "failure": None,
                "result_publication": None,
            },
            "active_with_publication": {
                "status": "active",
                "outcome": None,
                "final_candidate_id": None,
                "test_nll": None,
                "test_perplexity": None,
                "failure": None,
                "result_publication": {
                    "publication_id": "publication",
                    "prepared_event_seq": 1,
                },
            },
        }
        for name, payload in invalid.items():
            payload["result_schema_version"] = RESULT_SCHEMA_VERSION
            with self.subTest(name=name):
                with self.assertRaises(ResultSchemaError):
                    normalize_result_report(payload)

    def test_current_result_schema_accepts_test_in_progress(self):
        report = normalize_result_report(
            {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "status": "closed",
                "outcome": None,
                "final_candidate_id": "sha256:candidate",
                "test_nll": None,
                "test_perplexity": None,
                "failure": None,
                "research_sealed_at": "2026-01-01T00:00:00+00:00",
                "analysis_complete": False,
                "analysis_trajectory": [],
            }
        )
        self.assertIsNone(report["outcome"])

    def test_current_result_accepts_a_prepared_publication_reference(self):
        report = normalize_result_report(
            {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "status": "finalized",
                "outcome": "success",
                "final_candidate_id": "sha256:candidate",
                "test_nll": 0.0,
                "test_perplexity": 1.0,
                "failure": None,
                "research_sealed_at": "2026-01-01T00:00:00+00:00",
                "analysis_complete": True,
                "analysis_trajectory": [_analysis_point()],
                "result_publication": {
                    "publication_id": "publication",
                    "prepared_event_seq": 7,
                },
            }
        )

        self.assertEqual(report["result_publication"]["prepared_event_seq"], 7)

    def test_current_result_rejects_a_malformed_analysis_trajectory(self):
        malformed = _analysis_point()
        malformed["analysis_perplexity"] = 2.0

        with self.assertRaisesRegex(ResultSchemaError, "ANALYSIS perplexity"):
            normalize_result_report(_finalized(analysis_trajectory=[malformed]))

    def test_final_candidate_must_be_in_the_analysis_trajectory(self):
        with self.assertRaisesRegex(ResultSchemaError, "final Candidate"):
            normalize_result_report(
                _finalized(analysis_trajectory=[_analysis_point("sha256:other")])
            )

    def test_older_v2_finalized_result_without_publication_remains_readable(self):
        report = normalize_result_report(
            {
                "result_schema_version": 2,
                "status": "finalized",
                "outcome": "success",
                "final_candidate_id": "sha256:candidate",
                "test_nll": 0.0,
                "test_perplexity": 1.0,
                "failure": None,
            }
        )

        self.assertNotIn("result_publication", report)

    def test_v3_validation_evidence_remains_explicitly_legacy(self):
        report = normalize_result_report(
            {
                "result_schema_version": PRE_ANALYSIS_RESULT_SCHEMA_VERSION,
                "status": "finalized",
                "outcome": "success",
                "final_candidate_id": "sha256:candidate",
                "test_nll": 0.0,
                "test_perplexity": 1.0,
                "failure": None,
                "candidates": {
                    "sha256:candidate": {
                        "val_nll": 1.25,
                        "val_perplexity": math.exp(1.25),
                        "evaluation_error": None,
                    }
                },
                "timing": {"validation_evaluation_seconds": 2.0},
            }
        )

        candidate = report["candidates"]["sha256:candidate"]
        self.assertEqual(candidate["legacy_validation"]["nll"], 1.25)
        self.assertNotIn("val_nll", candidate)
        self.assertEqual(candidate["analysis_status"], "legacy_not_run")
        self.assertFalse(report["analysis_complete"])
        self.assertEqual(report["analysis_trajectory"], [])
        self.assertTrue(report["legacy_semantics"]["organizer_validation_feedback"])
        self.assertEqual(report["timing"]["legacy_validation_evaluation_seconds"], 2.0)

    def test_private_staged_result_is_not_part_of_a_report(self):
        report = normalize_result_report(
            {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "status": "closed",
                "outcome": None,
                "final_candidate_id": "sha256:candidate",
                "test_nll": None,
                "test_perplexity": None,
                "failure": None,
                "research_sealed_at": "2026-01-01T00:00:00+00:00",
                "analysis_complete": False,
                "analysis_trajectory": [],
                "pending_result": {
                    "outcome": "success",
                    "test_nll": 1.0,
                    "test_perplexity": math.e,
                    "failure": None,
                },
            }
        )

        self.assertNotIn("pending_result", report)
        self.assertIsNone(report["outcome"])

    def test_closed_result_cannot_claim_failure(self):
        with self.assertRaisesRegex(ResultSchemaError, "closed"):
            normalize_result_report(
                {
                    "result_schema_version": RESULT_SCHEMA_VERSION,
                    "status": "closed",
                    "outcome": "failed",
                    "final_candidate_id": "sha256:candidate",
                    "test_nll": None,
                    "test_perplexity": None,
                    "failure": "legacy_unscored_result",
                    "research_sealed_at": "2026-01-01T00:00:00+00:00",
                    "analysis_complete": False,
                    "analysis_trajectory": [],
                }
            )

    def test_legacy_closed_result_remains_pending(self):
        report = normalize_result_report(
            {
                "status": "closed",
                "final_candidate_id": "sha256:candidate",
                "test_nll": None,
                "failure": None,
            }
        )
        self.assertIsNone(report["outcome"])
        self.assertIsNone(report["failure"])

    def test_legacy_migration_rejects_an_unknown_status(self):
        with self.assertRaisesRegex(ResultSchemaError, "invalid result status"):
            normalize_result_report(
                {
                    "status": "unknown",
                    "final_candidate_id": None,
                    "test_nll": None,
                    "failure": None,
                }
            )

    def test_v2_agent_mode_is_migrated_to_the_v3_egress_field(self):
        report = normalize_result_report(
            {
                "result_schema_version": AGENT_MODE_RESULT_SCHEMA_VERSION,
                "status": "finalized",
                "outcome": "success",
                "final_candidate_id": "sha256:candidate",
                "test_nll": 0.0,
                "test_perplexity": 1.0,
                "failure": None,
                "agent_system": {"runner": "container", "mode": "sandboxed"},
            }
        )

        self.assertEqual(report["result_schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(
            report["migrated_from_result_schema_version"],
            AGENT_MODE_RESULT_SCHEMA_VERSION,
        )
        self.assertEqual(report["agent_system"]["egress"], "denied")
        self.assertNotIn("mode", report["agent_system"])

    def test_v2_host_runner_agent_mode_migrates_without_inventing_a_policy(self):
        report = normalize_result_report(
            {
                "result_schema_version": AGENT_MODE_RESULT_SCHEMA_VERSION,
                "status": "dnf",
                "outcome": "failed",
                "final_candidate_id": None,
                "test_nll": None,
                "test_perplexity": None,
                "failure": "agent exited without finishing the episode",
                "agent_system": {"runner": "host_cli", "mode": None},
            }
        )

        self.assertIsNone(report["agent_system"]["egress"])
        self.assertNotIn("mode", report["agent_system"])

    def test_current_schema_report_is_not_marked_as_migrated(self):
        report = normalize_result_report(
            {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "status": "finalized",
                "outcome": "success",
                "final_candidate_id": "sha256:candidate",
                "test_nll": 0.0,
                "test_perplexity": 1.0,
                "failure": None,
                "agent_system": {"runner": "container", "egress": "denied"},
                "research_sealed_at": "2026-01-01T00:00:00+00:00",
                "analysis_complete": True,
                "analysis_trajectory": [_analysis_point()],
            }
        )

        self.assertNotIn("migrated_from_result_schema_version", report)
        self.assertEqual(report["agent_system"]["egress"], "denied")

    def test_v3_rejects_a_result_that_still_carries_the_v2_agent_mode(self):
        with self.assertRaisesRegex(ResultSchemaError, "agent_system.mode"):
            normalize_result_report(
                _finalized(agent_system={"runner": "container", "mode": "sandboxed"})
            )

    def test_v2_migration_rejects_an_unknown_agent_mode(self):
        with self.assertRaisesRegex(ResultSchemaError, "unknown Agent mode"):
            normalize_result_report(
                _finalized(
                    schema_version=AGENT_MODE_RESULT_SCHEMA_VERSION,
                    agent_system={"runner": "container", "mode": "unknown-policy"},
                )
            )

    def test_an_unknown_agent_egress_policy_is_rejected(self):
        with self.assertRaisesRegex(ResultSchemaError, "agent_system.egress"):
            normalize_result_report(
                _finalized(
                    agent_system={"runner": "container", "egress": "unknown-policy"}
                )
            )

    def test_a_host_runner_result_may_not_declare_an_egress_policy(self):
        with self.assertRaisesRegex(ResultSchemaError, "host_cli"):
            normalize_result_report(
                _finalized(agent_system={"runner": "host_cli", "egress": "denied"})
            )

    def test_a_container_runner_result_must_declare_an_egress_policy(self):
        with self.assertRaisesRegex(ResultSchemaError, "must declare"):
            normalize_result_report(
                _finalized(agent_system={"runner": "container", "egress": None})
            )

    def test_an_unknown_agent_runner_is_rejected(self):
        with self.assertRaisesRegex(ResultSchemaError, "agent_system.runner"):
            normalize_result_report(
                _finalized(agent_system={"runner": "microvm", "egress": "denied"})
            )

    def test_a_valid_host_runner_result_is_accepted(self):
        report = normalize_result_report(
            _finalized(agent_system={"runner": "host_cli", "egress": None})
        )

        self.assertIsNone(report["agent_system"]["egress"])

    def test_a_future_result_schema_version_is_rejected(self):
        with self.assertRaisesRegex(ResultSchemaError, "unsupported result schema"):
            normalize_result_report(
                {
                    "result_schema_version": RESULT_SCHEMA_VERSION + 1,
                    "status": "finalized",
                    "outcome": "success",
                    "final_candidate_id": "sha256:candidate",
                    "test_nll": 0.0,
                    "test_perplexity": 1.0,
                    "failure": None,
                }
            )


if __name__ == "__main__":
    unittest.main()
