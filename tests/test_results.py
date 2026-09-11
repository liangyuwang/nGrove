import unittest

from autotrain_bench.artifact_identity import (
    CANDIDATE_TREE_IDENTITY_ALGORITHM,
    FILE_IDENTITY_ALGORITHM,
    HIDDEN_BUNDLE_IDENTITY_ALGORITHM,
)
from autotrain_bench.results import (
    RESULT_SCHEMA_VERSION,
    ResultSchemaError,
    normalize_result_report,
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
        "analysis_performance_score": 100.0,
        "analysis_raw_metrics": {"mean_nll": 0.0},
        "analysis_cost": _cost(),
        "analysis_error": None,
    }


def _cost():
    return {
        "time_per_task": 0.5,
        "candidate_wall_seconds": 2.0,
        "task_count": 4,
        "input_tokens": 20,
        "prediction_bytes": 128,
        "tasks_per_second": 2.0,
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
        "gateway_execution": None,
        "evaluation_resource_envelope": None,
        "evaluation_resource_allocations": [],
        "performance_score_contract": {
            "definition_id": "uniform_normalized_log_score_v1",
            "range": {"minimum": 0.0, "maximum": 100.0},
            "direction": "maximize",
            "comparison_scope": "benchmark_release",
            "raw_metrics": ["mean_nll"],
        },
        "cost_contract": {
            "definition_id": "candidate_end_to_end_time_per_task_v1",
            "task_unit": "next_token_prediction",
            "primary_metric": "time_per_task",
            "unit": "seconds_per_task",
            "direction": "minimize",
            "aggregation": "candidate_wall_seconds_divided_by_task_count",
            "comparison_scope": "benchmark_release",
            "comparison_requires": [
                "hidden_dataset_identity",
                "runtime_profile",
            ],
            "submetrics": [
                "candidate_wall_seconds",
                "task_count",
                "input_tokens",
                "prediction_bytes",
                "tasks_per_second",
                "input_tokens_per_second",
            ],
        },
        "artifact_identity_algorithms": {
            "candidate_tree": CANDIDATE_TREE_IDENTITY_ALGORITHM,
            "file": FILE_IDENTITY_ALGORITHM,
            "hidden_evaluation": HIDDEN_BUNDLE_IDENTITY_ALGORITHM,
        },
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


def _finalized(**overrides):
    """A minimal valid finalized result, so each test varies only one thing."""

    payload = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "status": "finalized",
        "outcome": "success",
        "final_candidate_id": "sha256:candidate",
        "test_cost": _cost(),
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
        "test_performance_score": 100.0,
        "test_raw_metrics": {"mean_nll": 0.0},
    }
    payload.update(overrides)
    if payload["result_schema_version"] < 15:
        payload.pop("gateway_execution", None)
    if payload["result_schema_version"] >= 14:
        payload.setdefault("evaluation_resource_envelope", None)
        if "evaluation_resource_allocations" not in overrides:
            measurements = (
                [("test", payload.get("final_candidate_id"))]
                if payload.get("test_cost") is not None
                else []
            )
            trajectory = payload.get("analysis_trajectory")
            if isinstance(trajectory, list):
                measurements.extend(
                    ("analysis", point.get("candidate_id"))
                    for point in trajectory
                    if isinstance(point, dict)
                    and point.get("analysis_cost") is not None
                )
            payload["evaluation_resource_allocations"] = [
                {
                    "evaluation_id": "evaluation:" + str(index),
                    "candidate_id": candidate,
                    "split": split,
                    "receipt": None,
                    "assessment": None,
                    "accepted": False,
                }
                for index, (split, candidate) in enumerate(measurements)
            ]
    else:
        payload.pop("evaluation_resource_envelope", None)
        payload.pop("evaluation_resource_allocations", None)
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
    def test_prior_v12_reports_remain_readable_without_operator_semantics(self):
        payload = _finalized(result_schema_version=12)
        self.assertEqual(normalize_result_report(payload)["outcome"], "success")
        payload["operator_limits"] = {"max_research_seconds": 60}
        with self.assertRaisesRegex(ResultSchemaError, "version 13"):
            normalize_result_report(payload)

    def test_operator_abort_is_distinct_and_cannot_carry_benchmark_measurements(self):
        payload = _finalized(
            track="B",
            status="operator_stopped",
            outcome="aborted",
            failure="operator_time_limit_reached",
            closure_reason="operator_time_limit_reached",
            leaderboard_eligible=False,
            final_candidate_id=None,
            test_status=None,
            test_performance_score=None,
            test_raw_metrics=None,
            test_cost=None,
            analysis_status=None,
            analysis_complete=None,
            analysis_failure=None,
            analysis_identity_verified=None,
            analysis_trajectory=[],
            operator_limits={
                "max_research_seconds": 60,
                "clock": "monotonic_since_task_reveal_without_refunds",
            },
        )
        self.assertEqual(normalize_result_report(payload)["outcome"], "aborted")
        for override in (
            {"outcome": "failed"},
            {"outcome": "success"},
            {"test_performance_score": 10},
            {"leaderboard_eligible": True},
            {"operator_limits": None},
            {"track": "A"},
        ):
            with self.subTest(override=override), self.assertRaises(ResultSchemaError):
                normalize_result_report({**payload, **override})

    def test_current_result_rejects_a_different_hidden_bundle_algorithm(self):
        payload = _finalized()
        payload["artifact_identity_algorithms"]["hidden_evaluation"] = "sha256"

        with self.assertRaisesRegex(ResultSchemaError, "active protocol"):
            normalize_result_report(payload)

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

    def test_cost_is_bound_to_counts_and_wall_time(self):
        payload = _finalized()
        payload["test_cost"]["tasks_per_second"] = 999.0

        with self.assertRaisesRegex(ResultSchemaError, "does not match"):
            normalize_result_report(payload)

        payload = _finalized()
        payload["test_cost"]["time_per_task"] = 0.01

        with self.assertRaisesRegex(ResultSchemaError, "does not match"):
            normalize_result_report(payload)

    def test_unrepresentable_cost_counts_are_schema_errors(self):
        payload = _finalized()
        payload["test_cost"]["task_count"] = 10**10000

        with self.assertRaisesRegex(ResultSchemaError, "cannot be represented"):
            normalize_result_report(payload)

    def test_current_result_schema_rejects_inconsistent_states(self):
        invalid = {
            "terminal_without_outcome": {
                "status": "finalized",
                "outcome": None,
                "test_performance_score": None,
                "test_raw_metrics": None,
                "failure": None,
                "research_sealed_at": "2026-01-01T00:00:00+00:00",
                "analysis_complete": False,
                "analysis_trajectory": [_analysis_point()],
            },
            "success_without_candidate": {
                "status": "finalized",
                "outcome": "success",
                "final_candidate_id": None,
                "test_performance_score": 100.0,
                "test_raw_metrics": {"mean_nll": 0.0},
                "failure": None,
                "research_sealed_at": "2026-01-01T00:00:00+00:00",
                "analysis_complete": True,
                "analysis_trajectory": [_analysis_point()],
            },
            "active_failure": {
                "status": "active",
                "outcome": "failed",
                "test_performance_score": None,
                "test_raw_metrics": None,
                "failure": "unexpected",
            },
            "pending_with_metrics": {
                "status": "active",
                "outcome": None,
                "test_performance_score": 100.0,
                "test_raw_metrics": {"mean_nll": 0.0},
                "test_cost": _cost(),
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
                "test_performance_score": 100.0,
                "test_raw_metrics": {"mean_nll": 0.0},
                "test_cost": _cost(),
                "failure": None,
            },
            "success_with_pending_teardown": {
                "status": "finalized",
                "outcome": "success",
                "final_candidate_id": "sha256:candidate",
                "pending_lease_teardowns": ["lease-0001"],
                "test_performance_score": 100.0,
                "test_raw_metrics": {"mean_nll": 0.0},
                "failure": None,
            },
            "finalized_with_empty_publication": {
                "status": "finalized",
                "outcome": "success",
                "final_candidate_id": "sha256:candidate",
                "test_performance_score": 100.0,
                "test_raw_metrics": {"mean_nll": 0.0},
                "failure": None,
                "result_publication": None,
            },
            "active_with_publication": {
                "status": "active",
                "outcome": None,
                "final_candidate_id": None,
                "test_performance_score": None,
                "test_raw_metrics": None,
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
                "test_performance_score": None,
                "test_raw_metrics": None,
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
                "evaluation_resource_allocations": _finalized()[
                    "evaluation_resource_allocations"
                ],
                "status": "finalized",
                "outcome": "success",
                "final_candidate_id": "sha256:candidate",
                "test_performance_score": 100.0,
                "test_raw_metrics": {"mean_nll": 0.0},
                "test_cost": _cost(),
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
        malformed["analysis_raw_metrics"] = {"unexpected": 2.0}

        with self.assertRaisesRegex(ResultSchemaError, "ANALYSIS performance"):
            normalize_result_report(_finalized(analysis_trajectory=[malformed]))

    def test_current_result_rejects_legacy_metric_fields(self):
        with self.assertRaisesRegex(ResultSchemaError, "legacy TEST metric"):
            normalize_result_report(_finalized(test_nll=0.0))

        point = _analysis_point()
        point["analysis_nll"] = 0.0
        with self.assertRaisesRegex(ResultSchemaError, "legacy metric"):
            normalize_result_report(_finalized(analysis_trajectory=[point]))

    def test_successful_test_may_retain_incomplete_analysis_diagnostics(self):
        point = _analysis_point()
        point.update(
            analysis_status="infrastructure_failure",
            analysis_performance_score=None,
            analysis_raw_metrics=None,
            analysis_cost=None,
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

    def test_private_staged_result_is_not_part_of_a_report(self):
        report = normalize_result_report(
            {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                **_resource_fields(),
                "status": "closed",
                "outcome": None,
                "final_candidate_id": "sha256:candidate",
                "test_performance_score": None,
                "test_raw_metrics": None,
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
                    "test_performance_score": 50.0,
                    "test_raw_metrics": {"mean_nll": 1.0},
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
                    "test_performance_score": None,
                    "test_raw_metrics": None,
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

    def test_current_schema_normalization_is_idempotent(self):
        report = normalize_result_report(
            {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                **_resource_fields(),
                "evaluation_resource_allocations": _finalized()[
                    "evaluation_resource_allocations"
                ],
                "status": "finalized",
                "outcome": "success",
                "final_candidate_id": "sha256:candidate",
                "test_performance_score": 100.0,
                "test_raw_metrics": {"mean_nll": 0.0},
                "test_cost": _cost(),
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

        self.assertEqual(normalize_result_report(report), report)
        self.assertEqual(report["agent_system"]["egress"], "denied")

    def test_current_schema_rejects_the_removed_agent_mode(self):
        with self.assertRaisesRegex(ResultSchemaError, "agent_system.mode"):
            normalize_result_report(
                _finalized(agent_system={"runner": "container", "mode": "sandboxed"})
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
                }
            )

    def test_unversioned_and_pre_preview_result_schemas_are_rejected(self):
        for payload in ({}, {"result_schema_version": RESULT_SCHEMA_VERSION - 1}):
            with self.subTest(payload=payload):
                with self.assertRaises(ResultSchemaError):
                    normalize_result_report(payload)

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
            analysis_performance_score=None,
            analysis_raw_metrics=None,
            analysis_cost=None,
            analysis_error="evaluation_failed",
        )
        report = normalize_result_report(
            _finalized(analysis_trajectory=[failed], test_status="evaluated")
        )

        self.assertEqual(report["analysis_status"], "complete")
        self.assertEqual(report["analysis_selected_points"], 1)
        self.assertEqual(report["analysis_evaluated_points"], 0)

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


if __name__ == "__main__":
    unittest.main()
