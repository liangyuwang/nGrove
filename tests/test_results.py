import math
import unittest

from autotrain_bench.results import (
    UNCOUNTED_ANALYSIS_RESULT_SCHEMA_VERSION,
    AGENT_MODE_RESULT_SCHEMA_VERSION,
    POSTMORTEM_RESULT_SCHEMA_VERSION,
    PRE_RESOURCE_ALLOCATION_RESULT_SCHEMA_VERSION,
    PRE_SERVING_EFFICIENCY_RESULT_SCHEMA_VERSION,
    PRE_WORKLOAD_RESOLUTION_RESULT_SCHEMA_VERSION,
    PRE_ANALYSIS_RESULT_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    ResultSchemaError,
    normalize_result_report,
    perplexity_from_nll,
    validated_nll,
)


def _candidate_commit(candidate_id="sha256:candidate"):
    return {
        "commit_index": 1,
        "event_seq": 1,
        "candidate_id": candidate_id,
        "source_lease": "lease-0001",
        "timestamp": "2026-01-01T00:00:01+00:00",
        "elapsed_seconds": 1.0,
    }


def _incumbent_declaration(candidate_id="sha256:candidate"):
    return {
        "declaration_index": 1,
        "event_seq": 2,
        "candidate_id": candidate_id,
        "previous_candidate_id": None,
        "source_leases": ["lease-0001"],
        "leases_used": 1,
        "timestamp": "2026-01-01T00:00:01+00:00",
        "elapsed_seconds": 1.0,
    }


def _analysis_point(candidate_id="sha256:candidate"):
    return {
        **_candidate_commit(candidate_id),
        "declared_incumbent": True,
        "declaration_event_seqs": [2],
        "analysis_status": "evaluated",
        "analysis_nll": 0.0,
        "analysis_perplexity": 1.0,
        "analysis_serving_metrics": _serving_metrics(),
        "analysis_error": None,
    }


def _serving_metrics():
    return {
        "wall_seconds": 2.0,
        "examples": 4,
        "input_tokens": 20,
        "prediction_bytes": 128,
        "examples_per_second": 2.0,
        "input_tokens_per_second": 10.0,
    }


def _empty_workload_resolutions():
    return {"training": None, "candidate": None, "evaluator": None}


def _workload_resolution(role, acceptance="canonical_identity"):
    return {
        "role": role,
        "requested_identity": f"example/{role}@sha256:" + "a" * 64,
        "effective_format": "oci",
        "effective_sha256": "a" * 64,
        "transformation_sha256": None,
        "worker_id": "certified-worker",
        "acceptance": acceptance,
    }


def _resource_fields():
    envelope = {
        "schema_version": 1,
        "normative": [
            {
                "name": "host.count",
                "relation": "exact",
                "value": 1,
                "minimum_enforcement": "best_effort",
            }
        ],
        "observational": [],
    }
    receipt = {
        "receipt_version": 1,
        "receipt_id": "receipt:lease-0001",
        "episode_id": "episode-1",
        "lease_id": "lease-0001",
        "worker_id": "research-worker",
        "resource_request_sha256": (
            "4faff5339f77c121b322f6c2017097cc11f91720eacbe8cc47844274da5b9dda"
        ),
        "properties": [
            {
                "name": "host.count",
                "allocator_allocated": {
                    "status": "observed",
                    "value": 1,
                    "authority": "test-allocator",
                },
                "workload_visible": {
                    "status": "not_applicable",
                    "value": None,
                    "authority": None,
                },
                "enforcement_level": "enforced",
                "enforcement_authority": "test-worker",
            }
        ],
    }
    return {
        "episode_id": "episode-1",
        "workers": {"research": {"worker_id": "research-worker"}},
        "leases_used": 1,
        "resource_envelope": envelope,
        "resource_allocations": [
            {
                "receipt": receipt,
                "assessment": {
                    "satisfied": True,
                    "missing_properties": [],
                    "mismatched_properties": [],
                    "insufficient_enforcement": [],
                    "unexpected_properties": [],
                },
                "accepted": True,
            }
        ],
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
        "test_serving_metrics": _serving_metrics(),
        "failure": None,
        "research_sealed_at": "2026-01-01T00:00:00+00:00",
        "analysis_complete": True,
        "analysis_status": "complete",
        "analysis_failure": None,
        "analysis_identity_verified": True,
        "analysis_trajectory": [_analysis_point()],
        "candidate_commits": [_candidate_commit()],
        "incumbent_declarations": [_incumbent_declaration()],
        "test_status": "evaluated",
        "workload_resolutions": _empty_workload_resolutions(),
        **_resource_fields(),
    }
    payload.update(overrides)
    # Derive the summary counts from whatever trajectory the test ended up with,
    # so a test that varies the trajectory does not also have to restate them.
    # A test that wants an inconsistent summary overrides them explicitly.
    trajectory = payload.get("analysis_trajectory")
    if isinstance(trajectory, list):
        payload.setdefault("analysis_selected_points", len(trajectory))
        payload.setdefault(
            "analysis_evaluated_points",
            sum(
                isinstance(point, dict) and point.get("analysis_status") == "evaluated"
                for point in trajectory
            ),
        )
    return payload


class ResultMetricTests(unittest.TestCase):
    def test_v8_migration_marks_serving_evidence_as_unavailable(self):
        legacy = _finalized(schema_version=PRE_SERVING_EFFICIENCY_RESULT_SCHEMA_VERSION)
        legacy.pop("test_serving_metrics")
        legacy["analysis_trajectory"][0].pop("analysis_serving_metrics")

        report = normalize_result_report(legacy)

        self.assertEqual(report["result_schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(
            report["migrated_from_result_schema_version"],
            PRE_SERVING_EFFICIENCY_RESULT_SCHEMA_VERSION,
        )
        self.assertIsNone(report["test_serving_metrics"])
        self.assertIsNone(report["analysis_trajectory"][0]["analysis_serving_metrics"])

    def test_v7_migration_marks_resource_evidence_as_unavailable(self):
        legacy = _finalized(
            schema_version=PRE_RESOURCE_ALLOCATION_RESULT_SCHEMA_VERSION
        )
        legacy.pop("resource_envelope")
        legacy.pop("resource_allocations")

        report = normalize_result_report(legacy)

        self.assertEqual(report["result_schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(
            report["migrated_from_result_schema_version"],
            PRE_RESOURCE_ALLOCATION_RESULT_SCHEMA_VERSION,
        )
        self.assertIsNone(report["resource_envelope"])
        self.assertEqual(report["resource_allocations"], [])

    def test_current_resource_receipt_cannot_be_replayed_for_another_request(self):
        payload = _finalized()
        payload["resource_allocations"][0]["receipt"]["resource_request_sha256"] = (
            "a" * 64
        )

        with self.assertRaisesRegex(ResultSchemaError, "different request"):
            normalize_result_report(payload)

    def test_current_resource_views_cannot_disagree_with_a_stored_assessment(self):
        payload = _finalized()
        payload["resource_allocations"][0]["receipt"]["properties"][0][
            "workload_visible"
        ] = {
            "status": "observed",
            "value": 2,
            "authority": "test-workload",
        }

        with self.assertRaisesRegex(ResultSchemaError, "assessment"):
            normalize_result_report(payload)

    def test_nll_validation_does_not_coerce_non_numeric_json_values(self):
        invalid = (True, "1.25", None, float("nan"), float("inf"), -0.1, 10**1000)
        for value in invalid:
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(ValueError):
                    validated_nll(value)

    def test_perplexity_is_derived_from_mean_nll(self):
        self.assertEqual(validated_nll(0), 0.0)
        self.assertAlmostEqual(perplexity_from_nll(1.5), math.exp(1.5))

    def test_serving_rates_are_bound_to_counts_and_wall_time(self):
        payload = _finalized()
        payload["test_serving_metrics"]["examples_per_second"] = 999.0

        with self.assertRaisesRegex(ResultSchemaError, "does not match"):
            normalize_result_report(payload)

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
                "test_serving_metrics": _serving_metrics(),
                "failure": None,
                "research_sealed_at": "2026-01-01T00:00:00+00:00",
                "analysis_complete": False,
                "analysis_status": "pending",
                "analysis_failure": None,
                "analysis_identity_verified": None,
                "analysis_trajectory": [],
                "analysis_selected_points": 0,
                "analysis_evaluated_points": 0,
                "candidate_commits": [_candidate_commit()],
                "incumbent_declarations": [_incumbent_declaration()],
                "test_status": "pending",
                "workload_resolutions": _empty_workload_resolutions(),
            },
            "success_with_active_lease": {
                "status": "finalized",
                "outcome": "success",
                "final_candidate_id": "sha256:candidate",
                "active_lease": {"lease_id": "lease-0001"},
                "test_nll": 0.0,
                "test_perplexity": 1.0,
                "test_serving_metrics": _serving_metrics(),
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
                **_resource_fields(),
                "status": "closed",
                "outcome": None,
                "final_candidate_id": "sha256:candidate",
                "test_nll": None,
                "test_perplexity": None,
                "failure": None,
                "research_sealed_at": "2026-01-01T00:00:00+00:00",
                "analysis_complete": False,
                "analysis_status": "pending",
                "analysis_failure": None,
                "analysis_identity_verified": None,
                "analysis_trajectory": [],
                "analysis_selected_points": 0,
                "analysis_evaluated_points": 0,
                "candidate_commits": [_candidate_commit()],
                "incumbent_declarations": [_incumbent_declaration()],
                "test_status": "pending",
                "workload_resolutions": _empty_workload_resolutions(),
            }
        )
        self.assertIsNone(report["outcome"])

    def test_current_result_accepts_a_prepared_publication_reference(self):
        report = normalize_result_report(
            {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                **_resource_fields(),
                "status": "finalized",
                "outcome": "success",
                "final_candidate_id": "sha256:candidate",
                "test_nll": 0.0,
                "test_perplexity": 1.0,
                "test_serving_metrics": _serving_metrics(),
                "failure": None,
                "research_sealed_at": "2026-01-01T00:00:00+00:00",
                "analysis_complete": True,
                "analysis_status": "complete",
                "analysis_failure": None,
                "analysis_identity_verified": True,
                "analysis_trajectory": [_analysis_point()],
                "analysis_selected_points": 1,
                "analysis_evaluated_points": 1,
                "candidate_commits": [_candidate_commit()],
                "incumbent_declarations": [_incumbent_declaration()],
                "test_status": "evaluated",
                "workload_resolutions": _empty_workload_resolutions(),
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

    def test_successful_test_may_retain_incomplete_analysis_diagnostics(self):
        point = _analysis_point()
        point.update(
            analysis_status="infrastructure_failure",
            analysis_nll=None,
            analysis_perplexity=None,
            analysis_serving_metrics=None,
            analysis_error="worker_teardown_failure",
        )

        report = normalize_result_report(
            _finalized(
                analysis_complete=False,
                analysis_status="incomplete",
                analysis_failure="analysis_worker_teardown_failure",
                analysis_identity_verified=True,
                analysis_trajectory=[point],
                deployment_teardown_failures=["evaluation_worker_teardown_failure"],
            )
        )

        self.assertEqual(report["outcome"], "success")
        self.assertEqual(report["analysis_status"], "incomplete")

    def test_analysis_trajectory_is_bound_to_candidate_commit_inventory(self):
        tampered = _analysis_point()
        tampered["elapsed_seconds"] = 2.0

        with self.assertRaisesRegex(ResultSchemaError, "bound to candidate commit"):
            normalize_result_report(_finalized(analysis_trajectory=[tampered]))

    def test_analysis_trajectory_is_bound_to_incumbent_declarations(self):
        tampered = _analysis_point()
        tampered["declaration_event_seqs"] = []

        with self.assertRaisesRegex(ResultSchemaError, "declaration evidence"):
            normalize_result_report(_finalized(analysis_trajectory=[tampered]))

    def test_final_candidate_must_be_in_the_analysis_trajectory(self):
        with self.assertRaisesRegex(ResultSchemaError, "final Candidate"):
            normalize_result_report(
                _finalized(
                    analysis_trajectory=[_analysis_point("sha256:other")],
                    candidate_commits=[_candidate_commit("sha256:other")],
                    incumbent_declarations=[_incumbent_declaration("sha256:other")],
                )
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

    def test_v4_postmortem_result_remains_readable_without_invented_history(self):
        payload = _finalized(schema_version=POSTMORTEM_RESULT_SCHEMA_VERSION)
        for field in (
            "analysis_status",
            "analysis_failure",
            "analysis_identity_verified",
            "incumbent_declarations",
            "test_status",
        ):
            payload.pop(field, None)
        for point in payload["analysis_trajectory"]:
            point.pop("declaration_event_seqs", None)

        report = normalize_result_report(payload)

        self.assertEqual(report["result_schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(
            report["migrated_from_result_schema_version"],
            POSTMORTEM_RESULT_SCHEMA_VERSION,
        )
        self.assertEqual(report["analysis_status"], "complete")
        self.assertEqual(report["test_status"], "evaluated")
        self.assertEqual(report["incumbent_declarations"], [])
        self.assertEqual(
            report["migration_limitations"],
            ["incumbent_declaration_history_unavailable"],
        )

    def test_private_staged_result_is_not_part_of_a_report(self):
        report = normalize_result_report(
            {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                **_resource_fields(),
                "status": "closed",
                "outcome": None,
                "final_candidate_id": "sha256:candidate",
                "test_nll": None,
                "test_perplexity": None,
                "failure": None,
                "research_sealed_at": "2026-01-01T00:00:00+00:00",
                "analysis_complete": False,
                "analysis_status": "pending",
                "analysis_failure": None,
                "analysis_identity_verified": None,
                "analysis_trajectory": [],
                "analysis_selected_points": 0,
                "analysis_evaluated_points": 0,
                "candidate_commits": [_candidate_commit()],
                "incumbent_declarations": [_incumbent_declaration()],
                "test_status": "evaluated",
                "workload_resolutions": _empty_workload_resolutions(),
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
                    **_resource_fields(),
                    "status": "closed",
                    "outcome": "failed",
                    "final_candidate_id": "sha256:candidate",
                    "test_nll": None,
                    "test_perplexity": None,
                    "failure": "legacy_unscored_result",
                    "research_sealed_at": "2026-01-01T00:00:00+00:00",
                    "analysis_complete": False,
                    "analysis_status": "pending",
                    "analysis_failure": None,
                    "analysis_identity_verified": None,
                    "analysis_trajectory": [],
                    "analysis_selected_points": 0,
                    "analysis_evaluated_points": 0,
                    "candidate_commits": [_candidate_commit()],
                    "incumbent_declarations": [_incumbent_declaration()],
                    "test_status": "failed",
                    "workload_resolutions": _empty_workload_resolutions(),
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
                **_resource_fields(),
                "status": "finalized",
                "outcome": "success",
                "final_candidate_id": "sha256:candidate",
                "test_nll": 0.0,
                "test_perplexity": 1.0,
                "test_serving_metrics": _serving_metrics(),
                "failure": None,
                "agent_system": {"runner": "container", "egress": "denied"},
                "research_sealed_at": "2026-01-01T00:00:00+00:00",
                "analysis_complete": True,
                "analysis_status": "complete",
                "analysis_failure": None,
                "analysis_identity_verified": True,
                "analysis_trajectory": [_analysis_point()],
                "analysis_selected_points": 1,
                "analysis_evaluated_points": 1,
                "candidate_commits": [_candidate_commit()],
                "incumbent_declarations": [_incumbent_declaration()],
                "test_status": "evaluated",
                "workload_resolutions": _empty_workload_resolutions(),
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

    def test_analysis_counts_must_agree_with_the_trajectory_they_summarize(self):
        """A summary that disagrees with its detail would defeat the counts."""

        with self.assertRaisesRegex(ResultSchemaError, "analysis_selected_points"):
            normalize_result_report(_finalized(analysis_selected_points=7))
        with self.assertRaisesRegex(ResultSchemaError, "analysis_evaluated_points"):
            normalize_result_report(_finalized(analysis_evaluated_points=0))

    def test_a_complete_pass_with_a_failed_point_is_visible_in_the_summary(self):
        """`complete` means the pass finished, not that every point measured."""

        failed = _analysis_point()
        failed.update(
            analysis_status="failed",
            analysis_nll=None,
            analysis_perplexity=None,
            analysis_serving_metrics=None,
            analysis_error="evaluation_failed",
        )
        report = normalize_result_report(
            _finalized(analysis_trajectory=[failed], test_status="evaluated")
        )

        self.assertEqual(report["analysis_status"], "complete")
        self.assertEqual(report["analysis_selected_points"], 1)
        self.assertEqual(report["analysis_evaluated_points"], 0)

    def test_a_v5_result_gains_counts_derived_from_its_own_trajectory(self):
        legacy = _finalized(schema_version=UNCOUNTED_ANALYSIS_RESULT_SCHEMA_VERSION)
        legacy.pop("analysis_selected_points")
        legacy.pop("analysis_evaluated_points")

        report = normalize_result_report(legacy)

        self.assertEqual(report["result_schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(
            report["migrated_from_result_schema_version"],
            UNCOUNTED_ANALYSIS_RESULT_SCHEMA_VERSION,
        )
        self.assertEqual(report["analysis_selected_points"], 1)
        self.assertEqual(report["analysis_evaluated_points"], 1)

    def test_v5_migration_overwrites_colliding_analysis_count_extensions(self):
        legacy = _finalized(schema_version=UNCOUNTED_ANALYSIS_RESULT_SCHEMA_VERSION)
        legacy["analysis_selected_points"] = 999
        legacy["analysis_evaluated_points"] = 998

        report = normalize_result_report(legacy)

        self.assertEqual(report["analysis_selected_points"], 1)
        self.assertEqual(report["analysis_evaluated_points"], 1)

    def test_v6_result_records_missing_workload_evidence_without_inventing_it(self):
        legacy = _finalized(
            schema_version=PRE_WORKLOAD_RESOLUTION_RESULT_SCHEMA_VERSION,
            leaderboard_eligible=True,
        )
        legacy.pop("workload_resolutions")

        report = normalize_result_report(legacy)

        self.assertEqual(report["result_schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(report["workload_resolutions"], _empty_workload_resolutions())

    def test_leaderboard_result_rejects_unverified_workload_resolution(self):
        resolutions = {
            role: _workload_resolution(role)
            for role in ("training", "candidate", "evaluator")
        }
        resolutions["candidate"] = _workload_resolution(
            "candidate", acceptance="unverified"
        )

        with self.assertRaisesRegex(ResultSchemaError, "unverified workload"):
            normalize_result_report(
                _finalized(
                    leaderboard_eligible=True,
                    workload_resolutions=resolutions,
                )
            )

    def test_result_cannot_self_declare_workload_acceptance(self):
        forged_identity = _workload_resolution("training")
        forged_identity["effective_sha256"] = "b" * 64
        with self.assertRaisesRegex(ResultSchemaError, "canonical identity"):
            normalize_result_report(
                _finalized(
                    workload_resolutions={
                        "training": forged_identity,
                        "candidate": None,
                        "evaluator": None,
                    }
                )
            )

        derived = _workload_resolution("training", "organizer_receipt")
        derived.update(
            effective_format="sif",
            effective_sha256="b" * 64,
            transformation_sha256="c" * 64,
        )
        with self.assertRaisesRegex(ResultSchemaError, "organizer acceptance"):
            normalize_result_report(
                _finalized(
                    workload_resolutions={
                        "training": derived,
                        "candidate": None,
                        "evaluator": None,
                    }
                )
            )

    def test_migration_marker_cannot_bypass_v6_validation(self):
        payload = _finalized(
            analysis_status="invalid",
            migrated_from_result_schema_version=(
                UNCOUNTED_ANALYSIS_RESULT_SCHEMA_VERSION
            ),
        )

        with self.assertRaisesRegex(ResultSchemaError, "ANALYSIS lifecycle status"):
            normalize_result_report(payload)


if __name__ == "__main__":
    unittest.main()
