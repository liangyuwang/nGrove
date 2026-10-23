"""Worker-fault evidence and recovery checks with deterministic fixture clocks."""

import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from envcore.deadline_probe import (
    WORKER_FAULT_DEADLINE_PROBE_VERSION,
    WorkerFaultEvidence,
    WorkerFaultKind,
)
from envcore.worker_protocol import LeaseState, WorkerFailure
from tests import test_deadline_probe as idle


class WorkerFaultDeadlineTests(unittest.TestCase):
    def setUp(self):
        idle.DeadlineProbeTests.setUp(self)
        self.injections = []
        self.active_fault = None

    assert_failed = idle.DeadlineProbeTests.assert_failed

    def inject(self, **kwargs):
        self.injections.append(kwargs)
        self.worker.history.append(("inject", self.clock.now))
        self.active_fault = WorkerFaultEvidence(
            fault_id=kwargs["fault_id"],
            kind=kwargs["kind"],
            evidence_id="private-fault-event",
        )

    def observe(self, **kwargs):
        return replace(
            idle.DeadlineProbeTests.observe(self, **kwargs),
            worker_fault=self.active_fault,
        )

    def teardown(self):
        self.worker.history.append(("external_cleanup", self.clock.now))
        self.active_fault = None
        return idle.DeadlineProbeTests.teardown(self)

    def probe(self, **overrides):
        inputs = dict(worker_fault=WorkerFaultKind.PAUSED, inject_fault=self.inject)
        inputs.update(overrides)
        return idle.DeadlineProbeTests.probe(self, **inputs)

    def test_pause_and_termination_require_live_work_under_fault_through_expiry(self):
        for kind in WorkerFaultKind:
            with self.subTest(kind=kind):
                report = self.probe(worker_fault=kind)
                self.assertTrue(report.passed, report.private_manifest())
                self.assertEqual(
                    report.probe_version, WORKER_FAULT_DEADLINE_PROBE_VERSION
                )
                self.assertNotIn("formal_conformance", report.public_manifest())
                self.assertTrue(report.lease["deadline_passed"])
                self.assertTrue(report.lease["worker_fault"]["accepted"])
                self.assertTrue(report.lease["cleanup_confirmed"])
                self.assertEqual(self.worker.stops, [])
                self.assertEqual(self.worker.states, [])
                self.assertEqual(self.worker.commands, [])
                self.assertEqual(report.observations[0]["phase"], "initial")
                self.assertNotIn("worker_fault", report.observations[0]["observation"])
                post_fault = report.observations[1]["observation"]
                self.assertEqual(post_fault["state"], "running")
                self.assertFalse(post_fault["deadline_triggered"])
                self.assertEqual(post_fault["worker_fault"]["kind"], kind.value)
                for item in report.observations[1:-1]:
                    self.assertEqual(
                        item["observation"]["worker_fault"]["kind"], kind.value
                    )
                self.assertNotIn("worker_fault", report.observations[-1]["observation"])
                request = self.injections[-1]
                self.assertEqual(request["episode_id"], report.run_id)
                self.assertEqual(request["lease_id"], report.lease["lease_id"])
                self.assertEqual(request["worker_id"], report.worker_id)
                self.assertEqual(
                    request["fault_id"], report.lease["worker_fault"]["fault_id"]
                )
                self.assertEqual(self.worker.history[-2][0], "external_cleanup")
                self.assertEqual(self.worker.history[-1][0], "observe")

    def test_stalled_controller_does_not_pass_because_external_cleanup_later_stops_work(
        self,
    ):
        self.worker.enforce = False
        report = self.probe()
        self.assert_failed(report, "idle_lease_deadline_exceeded")
        self.assertFalse(report.lease["deadline_passed"])
        self.assertTrue(report.lease["cleanup_confirmed"])
        self.assertEqual(self.worker.stops, [])

    def test_injector_acknowledgement_alone_cannot_establish_the_fault(self):
        report = self.probe(inject_fault=lambda **kwargs: None)
        self.assert_failed(report, "worker_fault_unconfirmed")
        self.assertFalse(report.lease["worker_fault"]["accepted"])
        self.assertTrue(report.skipped)

    def test_fault_must_bind_the_fresh_request_and_selected_kind(self):
        for changes in (
            {"fault_id": "old-fault"},
            {"kind": WorkerFaultKind.TERMINATED},
        ):
            with self.subTest(changes=changes):

                def inject(**kwargs):
                    self.inject(**kwargs)
                    self.active_fault = replace(self.active_fault, **changes)

                report = self.probe(inject_fault=inject)
                self.assert_failed(report, "worker_fault_binding_mismatch")
                self.assertEqual(self.worker.stops, [])

    def test_worker_revival_or_unknown_fault_state_before_expiry_fails(self):
        self.clock.schedule(1, lambda: setattr(self, "active_fault", None))
        report = self.probe()
        self.assert_failed(report, "worker_fault_unconfirmed")
        self.assertFalse(report.lease["deadline_passed"])

    def test_final_stopped_observation_must_still_confirm_the_fault(self):
        def observe(**kwargs):
            value = self.observe(**kwargs)
            return (
                replace(value, worker_fault=None)
                if value.state is LeaseState.STOPPED
                else value
            )

        report = self.probe(observe=observe)
        self.assert_failed(report, "worker_fault_unconfirmed")
        self.assertFalse(report.lease["deadline_passed"])
        self.assertTrue(report.lease["cleanup_confirmed"])

    def test_fault_that_immediately_kills_workload_is_not_deadline_evidence(self):
        def inject(**kwargs):
            self.inject(**kwargs)
            self.worker.running.clear()

        report = self.probe(inject_fault=inject)
        self.assert_failed(report, "running_after_fault_not_observed")
        self.assertFalse(report.lease["deadline_passed"])

    def test_partial_failed_injection_never_calls_worker_even_for_cleanup(self):
        def inject(**kwargs):
            self.inject(**kwargs)
            raise WorkerFailure("fault partially applied")

        with patch.object(
            self.worker, "terminate_lease", side_effect=AssertionError("unavailable")
        ) as stop:
            report = self.probe(inject_fault=inject)
        self.assert_failed(report, "worker_failure")
        stop.assert_not_called()
        self.assertEqual(self.cleanup_calls, [True])
        self.assertTrue(report.lease["cleanup_confirmed"])

    def test_late_injection_uses_original_expiry_and_external_recovery(self):
        def inject(**kwargs):
            self.inject(**kwargs)
            self.clock.advance(2)

        report = self.probe(inject_fault=inject)
        self.assert_failed(report, "operation_deadline_exceeded")
        self.assertEqual(report.lease["worker_fault"]["finished_monotonic"], 2)
        self.assertEqual(report.lease["latest_stop_monotonic"], 3)
        self.assertFalse(report.lease["worker_fault"]["accepted"])
        self.assertTrue(report.lease["cleanup_confirmed"])
        self.assertEqual(self.worker.stops, [])

    def test_fault_confirmation_shares_injection_allowance(self):
        def inject(**kwargs):
            self.inject(**kwargs)
            self.clock.advance(0.75)

        def observe(**kwargs):
            if self.active_fault is not None and len(self.observer_calls) == 1:
                self.assertEqual(kwargs["timeout_seconds"], 0.25)
                self.clock.advance(0.5)
            return self.observe(**kwargs)

        report = self.probe(inject_fault=inject, observe=observe)
        self.assert_failed(report, "operation_deadline_exceeded")
        self.assertFalse(report.observations[1]["accepted"])
        self.assertIn("worker_fault", report.observations[1]["observation"])

    def test_injection_allowance_cannot_extend_past_earliest_possible_expiry(self):
        original = self.worker.start_lease

        def start(**kwargs):
            receipt = original(**kwargs)
            self.clock.advance(1.5)
            return receipt

        with patch.object(self.worker, "start_lease", side_effect=start):
            report = self.probe(fault_timeout_seconds=10)
        self.assertTrue(report.passed, report.private_manifest())
        self.assertEqual(self.injections[0]["timeout_seconds"], 0.5)
        self.assertEqual(report.lease["latest_stop_monotonic"], 4.5)

    def test_failed_initial_observation_never_injects_and_uses_normal_compensation(
        self,
    ):
        self.active_fault = WorkerFaultEvidence(
            "prior-fault", WorkerFaultKind.PAUSED, "old-evidence"
        )
        report = self.probe()
        self.assert_failed(report, "initial_running_not_observed")
        self.assertEqual(self.injections, [])
        self.assertEqual(len(self.worker.stops), 1)

    def test_injection_interrupt_still_runs_external_cleanup_then_propagates(self):
        def inject(**kwargs):
            self.inject(**kwargs)
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            self.probe(inject_fault=inject)
        self.assertEqual(self.cleanup_calls, [True])
        self.assertEqual(self.worker.stops, [])
        self.assertEqual(
            self.observer_calls[-1]["episode_id"], self.injections[0]["episode_id"]
        )

    def test_interrupt_during_fault_observation_still_uses_external_recovery(self):
        def observe(**kwargs):
            if len(self.observer_calls) == 1 and self.active_fault is not None:
                raise KeyboardInterrupt()
            return self.observe(**kwargs)

        with self.assertRaises(KeyboardInterrupt):
            self.probe(observe=observe)
        self.assertEqual(self.cleanup_calls, [True])
        self.assertEqual(self.worker.stops, [])
        self.assertEqual(len(self.observer_calls), 2)

    def test_interrupt_in_external_teardown_still_attempts_independent_absence(self):
        def teardown():
            self.teardown()
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            self.probe(teardown=teardown)
        self.assertEqual(self.cleanup_calls, [True])
        self.assertEqual(self.worker.stops, [])
        self.assertEqual(self.worker.history[-1][0], "observe")

    def test_external_teardown_and_absence_observation_share_cleanup_budget(self):
        def teardown():
            self.clock.advance(0.75)
            return self.teardown()

        report = self.probe(teardown=teardown, cleanup_timeout_seconds=1)
        self.assertTrue(report.passed, report.private_manifest())
        self.assertEqual(self.observer_calls[-1]["timeout_seconds"], 0.25)

    def test_late_cleanup_cannot_claim_confirmation_or_erase_deadline_success(self):
        def teardown():
            self.clock.advance(2)
            return self.teardown()

        report = self.probe(teardown=teardown, cleanup_timeout_seconds=1)
        self.assert_failed(report, "deployment_teardown_failed")
        self.assert_failed(report, "lease_cleanup_unconfirmed")
        self.assertTrue(report.lease["deadline_passed"])
        self.assertFalse(report.lease["cleanup_confirmed"])

    def test_successful_teardown_reply_cannot_replace_independent_absence(self):
        self.worker.enforce = False
        report = self.probe(teardown=lambda: True)
        self.assert_failed(report, "lease_cleanup_unconfirmed")
        self.assertFalse(report.lease["cleanup_confirmed"])

    def test_failed_or_unconfirmed_teardown_retains_successful_expiry(self):
        for value in (False, None, 1):
            self.active_fault = None
            report = self.probe(teardown=lambda: value)
            self.assert_failed(report, "deployment_teardown_unconfirmed")
            self.assertTrue(report.lease["deadline_passed"])

        def fail():
            raise OSError("external recovery unavailable")

        self.active_fault = None
        report = self.probe(teardown=fail)
        self.assert_failed(report, "deployment_teardown_failed")
        self.assertTrue(report.lease["deadline_passed"])

    def test_fault_evidence_stays_private_and_requests_are_fresh(self):
        first, second = self.probe(), self.probe()
        self.assertNotEqual(
            first.lease["worker_fault"]["fault_id"],
            second.lease["worker_fault"]["fault_id"],
        )
        for report in (first, second):
            public = json.dumps(report.public_manifest())
            private = json.dumps(report.private_manifest())
            for evidence in (
                "private-fault-event",
                report.lease["worker_fault"]["fault_id"],
            ):
                self.assertNotIn(evidence, public)
                self.assertIn(evidence, private)

    def test_fault_inputs_are_validated_before_worker_calls_and_writes(self):
        for overrides in (
            {"worker_fault": "paused"},
            {"worker_fault": None},
            {"inject_fault": None},
            *(
                {"fault_timeout_seconds": value}
                for value in (0, -1, True, float("inf"), float("nan"))
            ),
        ):
            with self.subTest(overrides=overrides), self.assertRaises(
                (TypeError, ValueError)
            ):
                self.probe(**overrides)
        self.assertEqual(self.worker.starts, [])
        self.assertEqual(self.injections, [])
        self.assertEqual(self.cleanup_calls, [])
        self.assertEqual(list((self.root / "output").iterdir()), [])

    def test_fault_evidence_requires_a_typed_kind_and_nonempty_binding(self):
        for values in (
            ("fault", "paused", "evidence"),
            ("", WorkerFaultKind.PAUSED, "evidence"),
            ("fault", WorkerFaultKind.TERMINATED, " "),
        ):
            with self.assertRaises((TypeError, ValueError)):
                WorkerFaultEvidence(*values)


if __name__ == "__main__":
    unittest.main()
