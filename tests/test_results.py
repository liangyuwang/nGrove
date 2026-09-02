import math
import unittest

from autotrain_bench.results import (
    ResultSchemaError,
    normalize_result_report,
    perplexity_from_nll,
    validated_nll,
)


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
            },
            "success_without_candidate": {
                "status": "finalized",
                "outcome": "success",
                "final_candidate_id": None,
                "test_nll": 0.0,
                "test_perplexity": 1.0,
                "failure": None,
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
            payload["result_schema_version"] = 2
            with self.subTest(name=name):
                with self.assertRaises(ResultSchemaError):
                    normalize_result_report(payload)

    def test_current_result_schema_accepts_test_in_progress(self):
        report = normalize_result_report(
            {
                "result_schema_version": 2,
                "status": "closed",
                "outcome": None,
                "final_candidate_id": "sha256:candidate",
                "test_nll": None,
                "test_perplexity": None,
                "failure": None,
            }
        )
        self.assertIsNone(report["outcome"])

    def test_current_result_accepts_a_prepared_publication_reference(self):
        report = normalize_result_report(
            {
                "result_schema_version": 2,
                "status": "finalized",
                "outcome": "success",
                "final_candidate_id": "sha256:candidate",
                "test_nll": 0.0,
                "test_perplexity": 1.0,
                "failure": None,
                "result_publication": {
                    "publication_id": "publication",
                    "prepared_event_seq": 7,
                },
            }
        )

        self.assertEqual(report["result_publication"]["prepared_event_seq"], 7)

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

    def test_private_staged_result_is_not_part_of_a_report(self):
        report = normalize_result_report(
            {
                "result_schema_version": 2,
                "status": "closed",
                "outcome": None,
                "final_candidate_id": "sha256:candidate",
                "test_nll": None,
                "test_perplexity": None,
                "failure": None,
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
                    "result_schema_version": 2,
                    "status": "closed",
                    "outcome": "failed",
                    "final_candidate_id": "sha256:candidate",
                    "test_nll": None,
                    "test_perplexity": None,
                    "failure": "legacy_unscored_result",
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


if __name__ == "__main__":
    unittest.main()
