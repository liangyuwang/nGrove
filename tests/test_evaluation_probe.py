"""Black-box evaluation probe diagnostics; no Docker, GPU or real hidden data."""

import json
import tempfile
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from envcore.config import TrainingResources
from envcore.conformance import (
    EVALUATION_PROBE_VERSION,
    probe_worker_evaluation,
)
from envcore.worker_protocol import (
    CandidateEvaluationResult,
    EnforcementLevel,
    EvaluationResourceFailure,
    EvaluationResourceReceipt,
    ResourceEnvelope,
    ResourceObservation,
    ResourcePropertyEvidence,
    ResourceRequirement,
    WorkerFailure,
    WorkerTeardownFailure,
)
from tests.test_conformance import TASK, ResolutionWorker


def contract(value=1):
    # This vocabulary belongs to the fixture deployment, not to Core.
    return ResourceEnvelope(
        requirements=(
            ResourceRequirement(
                "fixture.capacity", "exact", value, EnforcementLevel.BEST_EFFORT
            ),
        ),
        observational_properties=("fixture.observation",),
    )


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class EvaluationWorker(ResolutionWorker):
    def __init__(self):
        super().__init__()
        self.evaluations = []

    def evaluate_candidate(self, **kwargs):
        self.evaluations.append(kwargs)
        self.episode_ids.append(kwargs["episode_id"])
        self.timeouts.append(kwargs["timeout_seconds"])
        envelope = kwargs["resource_envelope"]
        if envelope.sha256 == contract(2).sha256:
            raise EvaluationResourceFailure("private fixture capacity unavailable")
        return CandidateEvaluationResult(
            performance_score=50.0,
            raw_metrics={"mean_nll": 1.0},
            candidate_wall_seconds=1.0,
            task_count=2,
            input_tokens=4,
            prediction_bytes=24,
            resource_receipt=EvaluationResourceReceipt(
                episode_id=kwargs["episode_id"],
                evaluation_id=kwargs["evaluation_id"],
                worker_id=self.describe_capabilities().worker_id,
                resource_request_sha256=envelope.sha256,
                receipt_id="fixture:" + uuid.uuid4().hex,
                properties=(
                    ResourcePropertyEvidence(
                        name="fixture.capacity",
                        allocator_allocated=ResourceObservation("unavailable"),
                        workload_visible=ResourceObservation("observed", 1, "fixture"),
                        enforcement_level=EnforcementLevel.BEST_EFFORT,
                        enforcement_authority="fixture",
                    ),
                ),
            ),
        )


class EvaluationProbeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.candidate = self.root / "candidate"
        self.candidate.mkdir()
        self.output = self.root / "diagnostics"
        self.output.mkdir()
        self.worker = EvaluationWorker()
        self.cleanup_calls = []

    def teardown(self):
        self.cleanup_calls.append(True)
        return True

    def probe(self, **overrides):
        kwargs = {
            "task": TASK,
            "candidate_dir": self.candidate,
            "output_root": self.output,
            "resources": TrainingResources(),
            "hidden_dataset_identities": self.worker.sources,
            "resource_envelope": contract(),
            "rejected_resource_envelope": contract(2),
            "teardown": self.teardown,
        }
        kwargs.update(overrides)
        return probe_worker_evaluation(self.worker, **kwargs)

    def assert_failed(self, report, code):
        self.assertFalse(report.passed, report.private_manifest())
        self.assertIn(code, {check.code for check in report.failed})

    def test_all_cases_use_bound_calls_and_remain_diagnostic_only(self):
        report = self.probe()

        self.assertTrue(report.passed, report.private_manifest())
        self.assertEqual(report.probe_version, EVALUATION_PROBE_VERSION)
        self.assertEqual(self.cleanup_calls, [True])
        self.assertEqual(set(self.worker.episode_ids), {report.run_id})
        calls = self.worker.evaluations
        self.assertEqual(
            [call["hidden_dataset"].name for call in calls],
            ["test", "analysis", "test", "analysis"],
        )
        self.assertEqual(len({call["evaluation_id"] for call in calls}), 4)
        self.assertEqual(len({call["output_dir"] for call in calls}), 4)
        self.assertTrue(
            all(call["candidate_dir"] == self.candidate.resolve() for call in calls)
        )
        self.assertTrue(
            all(call["candidate_workload"].role == "candidate" for call in calls)
        )
        self.assertTrue(
            all(call["evaluator_workload"].role == "evaluator" for call in calls)
        )
        self.assertNotIn("formal_conformance", report.public_manifest())
        self.assertFalse(hasattr(report, "receipt"))
        self.assertFalse(hasattr(self.worker, "start_lease"))
        self.assertEqual(list(self.candidate.iterdir()), [])
        self.assertEqual(len(report.private_manifest()["evaluations"]), 4)
        json.dumps(report.private_manifest())

    def test_fixture_measurements_and_rejection_details_stay_private(self):
        report = self.probe()
        public = json.dumps(report.public_manifest())
        private = json.dumps(report.private_manifest())

        for value in (
            "fixture.capacity",
            "private fixture capacity unavailable",
            self.worker.sources["test"],
            "performance_score",
            TASK.candidate_image,
        ):
            self.assertNotIn(value, public)
            self.assertIn(value, private)

    def test_old_worker_interface_fails_and_cleans_up(self):
        self.worker = ResolutionWorker()
        report = self.probe()

        self.assert_failed(report, "unexpected_worker_exception")
        self.assertEqual(self.cleanup_calls, [True])
        self.assertEqual(len(report.evaluations), 1)
        self.assertTrue(report.skipped)

    def test_missing_receipt_and_each_wrong_binding_fail_before_analysis(self):
        original = self.worker.evaluate_candidate
        for changes in (
            None,
            {"episode_id": "other"},
            {"evaluation_id": "other"},
            {"worker_id": "other"},
            {"resource_request_sha256": "0" * 64},
        ):

            def evaluate(**kwargs):
                result = original(**kwargs)
                return replace(
                    result,
                    resource_receipt=None
                    if changes is None
                    else replace(result.resource_receipt, **changes),
                )

            with self.subTest(changes=changes), patch.object(
                self.worker, "evaluate_candidate", side_effect=evaluate
            ):
                report = self.probe()
            self.assert_failed(report, "invalid_worker_evidence")
            self.assertEqual(len(report.evaluations), 1)
        self.assertEqual(len(self.cleanup_calls), 5)

    def test_resource_mismatch_and_weak_enforcement_are_not_passing_receipts(self):
        original = self.worker.evaluate_candidate
        for mismatch in (True, False):

            def evaluate(**kwargs):
                result = original(**kwargs)
                receipt = result.resource_receipt
                evidence = receipt.properties[0]
                changed = (
                    replace(
                        evidence,
                        workload_visible=ResourceObservation("observed", 0, "fixture"),
                    )
                    if mismatch
                    else replace(
                        evidence, enforcement_level=EnforcementLevel.UNVERIFIED
                    )
                )
                return replace(
                    result, resource_receipt=replace(receipt, properties=(changed,))
                )

            with self.subTest(mismatch=mismatch), patch.object(
                self.worker, "evaluate_candidate", side_effect=evaluate
            ):
                report = self.probe()
            self.assert_failed(report, "invalid_worker_evidence")
            self.assertFalse(report.evaluations[0]["assessment"]["satisfied"])

    def test_receipt_id_replay_is_rejected_even_with_fresh_operation_binding(self):
        original = self.worker.evaluate_candidate

        def evaluate(**kwargs):
            result = original(**kwargs)
            return replace(
                result,
                resource_receipt=replace(result.resource_receipt, receipt_id="reused"),
            )

        with patch.object(self.worker, "evaluate_candidate", side_effect=evaluate):
            report = self.probe()
        self.assert_failed(report, "invalid_worker_evidence")
        self.assertEqual(len(report.evaluations), 2)
        self.assertIn("receipt_id was reused", str(report.private_manifest()))

    def test_resource_failure_for_supported_contract_is_not_expected_rejection(self):
        with patch.object(
            self.worker,
            "evaluate_candidate",
            side_effect=EvaluationResourceFailure("GPU probe failed"),
        ):
            report = self.probe()
        self.assert_failed(report, "unexpected_resource_rejection")
        self.assertEqual(len(report.evaluations), 1)

    def test_ordinary_failure_and_unconfirmed_teardown_do_not_pass_rejection_case(self):
        original = self.worker.evaluate_candidate
        for error, code in (
            (WorkerFailure("resource query failed"), "worker_failure"),
            (
                WorkerTeardownFailure("resource boundary unknown"),
                "worker_teardown_unconfirmed",
            ),
        ):

            def evaluate(**kwargs):
                if kwargs["resource_envelope"].sha256 == contract(2).sha256:
                    raise error
                return original(**kwargs)

            with self.subTest(code=code), patch.object(
                self.worker, "evaluate_candidate", side_effect=evaluate
            ):
                report = self.probe()
            self.assert_failed(report, code)
            self.assertEqual(len(report.evaluations), 3)
            self.assertTrue(report.skipped)

    def test_successful_return_for_unsupported_contract_fails_probe(self):
        original = self.worker.evaluate_candidate

        def evaluate(**kwargs):
            return original(**{**kwargs, "resource_envelope": contract()})

        with patch.object(self.worker, "evaluate_candidate", side_effect=evaluate):
            report = self.probe()
        self.assert_failed(report, "invalid_worker_evidence")
        self.assertEqual(len(report.evaluations), 3)

    def test_hidden_identity_mismatch_never_reaches_candidate(self):
        original = self.worker.resolve_hidden_dataset

        def resolve(**kwargs):
            return replace(original(**kwargs), sha256="0" * 64)

        with patch.object(self.worker, "resolve_hidden_dataset", side_effect=resolve):
            report = self.probe()
        self.assert_failed(report, "invalid_worker_evidence")
        self.assertEqual(self.worker.evaluations, [])
        self.assertEqual(self.cleanup_calls, [True])

    def test_workload_identity_drift_stops_later_evaluation(self):
        original = self.worker.resolve_workload

        def resolve(**kwargs):
            result = original(**kwargs)
            return (
                replace(result, effective_sha256="0" * 64)
                if self.worker.evaluations
                else result
            )

        with patch.object(self.worker, "resolve_workload", side_effect=resolve):
            report = self.probe()
        self.assert_failed(report, "invalid_worker_evidence")
        self.assertEqual(len(self.worker.evaluations), 1)

    def test_resolution_and_invocation_share_each_case_deadline(self):
        clock = Clock()

        def tick(method):
            def invoke(**kwargs):
                try:
                    return method(**kwargs)
                finally:
                    clock.now += 1

            return invoke

        with (
            patch("envcore.conformance.time.monotonic", clock),
            patch.object(
                self.worker,
                "resolve_workload",
                side_effect=tick(self.worker.resolve_workload),
            ),
            patch.object(
                self.worker,
                "resolve_hidden_dataset",
                side_effect=tick(self.worker.resolve_hidden_dataset),
            ),
            patch.object(
                self.worker,
                "evaluate_candidate",
                side_effect=tick(self.worker.evaluate_candidate),
            ),
        ):
            report = self.probe(timeout_seconds=5)
        self.assertTrue(report.passed, report.private_manifest())
        self.assertEqual(self.worker.timeouts, [5, 4, 3, 2] * 4)

    def test_late_resolution_does_not_receive_another_budget(self):
        clock = Clock()
        original = self.worker.resolve_workload

        def resolve(**kwargs):
            result = original(**kwargs)
            clock.now += 2
            return result

        with patch("envcore.conformance.time.monotonic", clock), patch.object(
            self.worker, "resolve_workload", side_effect=resolve
        ):
            report = self.probe(timeout_seconds=3)
        self.assert_failed(report, "operation_deadline_exceeded")
        self.assertEqual(self.worker.timeouts, [3, 1])
        self.assertEqual(self.worker.evaluations, [])
        self.assertEqual(self.cleanup_calls, [True])

    def test_late_success_and_late_resource_rejection_both_fail(self):
        original = self.worker.evaluate_candidate
        for late_rejection in (False, True):
            clock = Clock()

            def evaluate(**kwargs):
                try:
                    return original(**kwargs)
                finally:
                    if (
                        not late_rejection
                        or kwargs["resource_envelope"].sha256 == contract(2).sha256
                    ):
                        clock.now += 10

            with self.subTest(late_rejection=late_rejection), patch(
                "envcore.conformance.time.monotonic", clock
            ), patch.object(self.worker, "evaluate_candidate", side_effect=evaluate):
                report = self.probe(timeout_seconds=3)
            self.assert_failed(report, "operation_deadline_exceeded")
            self.assertEqual(len(report.evaluations), 3 if late_rejection else 1)

    def test_cleanup_requires_explicit_confirmation(self):
        for result in (None, False, 1, "confirmed"):
            with self.subTest(result=result):
                report = self.probe(teardown=lambda: result)
            self.assert_failed(report, "deployment_teardown_unconfirmed")

        def teardown():
            raise OSError("/private/cleanup/detail")

        report = self.probe(teardown=teardown)
        self.assert_failed(report, "deployment_teardown_failed")
        self.assertNotIn("/private/cleanup/detail", str(report.public_manifest()))
        self.assertIn("/private/cleanup/detail", str(report.private_manifest()))

    def test_late_teardown_failure_preserves_the_unconfirmed_boundary(self):
        clock = Clock()

        def evaluate(**kwargs):
            clock.now += 10
            raise WorkerTeardownFailure("workload still present")

        with (
            patch("envcore.conformance.time.monotonic", clock),
            patch.object(self.worker, "evaluate_candidate", side_effect=evaluate),
        ):
            report = self.probe(timeout_seconds=3)
        self.assert_failed(report, "worker_teardown_unconfirmed")
        self.assertEqual(self.cleanup_calls, [True])
        self.assertFalse(report.evaluations[0]["passed"])

    def test_descriptor_failure_and_interrupt_still_call_cleanup(self):
        with patch.object(
            self.worker,
            "describe_capabilities",
            side_effect=OSError("descriptor unavailable"),
        ):
            report = self.probe()
        self.assert_failed(report, "unexpected_worker_exception")
        self.assertEqual(self.worker.evaluations, [])
        with patch.object(
            self.worker, "evaluate_candidate", side_effect=KeyboardInterrupt
        ), self.assertRaises(KeyboardInterrupt):
            self.probe()
        self.assertEqual(self.cleanup_calls, [True, True])

    def test_each_probe_uses_an_independent_output_and_episode_scope(self):
        first = self.probe()
        second = self.probe()
        self.assertNotEqual(first.run_id, second.run_id)
        paths = [call["output_dir"] for call in self.worker.evaluations]
        self.assertEqual(len(set(paths)), 8)

    def test_invalid_inputs_fail_before_worker_or_output_creation(self):
        for overrides in (
            {"timeout_seconds": value}
            for value in (0, -1, float("nan"), float("inf"), True)
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.probe(**overrides)
        for overrides in (
            {"resource_envelope": ResourceEnvelope(requirements=())},
            {"rejected_resource_envelope": contract()},
            {"hidden_dataset_identities": {"test": "0" * 64}},
            {"hidden_dataset_identities": {"test": None, "analysis": "0" * 64}},
            {"output_root": self.candidate},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.probe(**overrides)
        self.assertEqual(self.cleanup_calls, [])
        self.assertEqual(self.worker.episode_ids, [])
        self.assertEqual(list(self.output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
