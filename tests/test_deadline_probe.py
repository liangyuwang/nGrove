"""Independent controller observations with a deterministic simulated clock."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from autotrain_bench.config import TrainingResources
from autotrain_bench.deadline_probe import (
    LEASE_DEADLINE_PROBE_VERSION,
    LeaseDeadlineObservation,
    probe_worker_lease_deadline,
)
from autotrain_bench.worker_protocol import (
    LeaseState,
    ResearchTaskSpec,
    ResourceEnvelope,
    WorkerFailure,
    WorkerTeardownFailure,
)
from tests.test_lease_probe import LeaseWorker, contract


class Clock:
    def __init__(self):
        self.now = 0.0
        self.timers = []

    def __call__(self):
        return self.now

    def schedule(self, seconds, action):
        self.timers.append((self.now + seconds, action))

    def advance(self, seconds):
        target = self.now + seconds
        while self.timers and min(item[0] for item in self.timers) <= target:
            earliest = min(self.timers, key=lambda item: item[0])
            self.timers.remove(earliest)
            self.now = earliest[0]
            earliest[1]()
        self.now = target


class IndependentlyTimedWorker(LeaseWorker):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock
        self.fired = {}
        self.enforce = True
        self.arm = True
        self.trigger_after = None
        self.history = []

    def start_lease(self, **kwargs):
        receipt = super().start_lease(**kwargs)
        key = (kwargs["episode_id"], kwargs["lease_id"])
        self.history.append(("start", self.clock.now))
        if self.arm:

            def expire():
                self.fired[key] = "private-controller-event:" + kwargs["lease_id"]
                if self.enforce:
                    self.running.discard(key)
                self.history.append(("deadline", self.clock.now))

            self.clock.schedule(
                self.trigger_after
                if self.trigger_after is not None
                else kwargs["duration_seconds"],
                expire,
            )
        return receipt

    def lease_state(self, **kwargs):
        raise AssertionError("Worker state must not be polled to enforce expiry")

    def exec_lease(self, **kwargs):
        raise AssertionError("the Lease must stay idle")

    def terminate_lease(self, **kwargs):
        self.history.append(("terminate", self.clock.now))
        super().terminate_lease(**kwargs)


class DeadlineProbeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ("workspace", "output", "train", "api"):
            (self.root / name).mkdir()
        self.task = ResearchTaskSpec(
            task_id="deadline-fixture",
            training_image="fixture/training@sha256:" + "a" * 64,
            train_dir=self.root / "train",
            public_api_dir=self.root / "api",
        )
        self.clock = Clock()
        self.worker = IndependentlyTimedWorker(self.clock)
        self.observer_calls = []
        self.cleanup_calls = []

    def observe(self, **kwargs):
        self.observer_calls.append(kwargs)
        self.worker.history.append(("observe", self.clock.now))
        key = (kwargs["episode_id"], kwargs["lease_id"])
        event = self.worker.fired.get(key)
        return LeaseDeadlineObservation(
            episode_id=kwargs["episode_id"],
            lease_id=kwargs["lease_id"],
            worker_id=kwargs["worker_id"],
            observation_id=kwargs["observation_id"],
            source="independent-fixture-controller",
            state=LeaseState.RUNNING
            if key in self.worker.running
            else LeaseState.STOPPED,
            deadline_triggered=event is not None,
            deadline_evidence_id=event,
        )

    def teardown(self):
        self.cleanup_calls.append(True)
        self.worker.running.clear()
        return True

    def probe(self, **overrides):
        inputs = dict(
            task=self.task,
            workspace=self.root / "workspace",
            output_root=self.root / "output",
            resources=TrainingResources(),
            resource_envelope=contract(),
            observe=self.observe,
            teardown=self.teardown,
            duration_seconds=2,
            termination_grace_seconds=1,
            poll_interval_seconds=0.25,
        )
        inputs.update(overrides)
        with patch("autotrain_bench.deadline_probe.time.monotonic", self.clock), patch(
            "autotrain_bench.deadline_probe.time.sleep", self.clock.advance
        ):
            return probe_worker_lease_deadline(self.worker, **inputs)

    def assert_failed(self, report, code):
        self.assertFalse(report.passed, report.private_manifest())
        self.assertIn(code, {check.code for check in report.failed})

    def test_independent_idle_expiry_is_observed_before_any_cleanup(self):
        report = self.probe()
        self.assertTrue(report.passed, report.private_manifest())
        self.assertEqual(report.probe_version, LEASE_DEADLINE_PROBE_VERSION)
        self.assertFalse(report.public_manifest()["formal_conformance"])
        self.assertFalse(hasattr(report, "receipt"))
        self.assertEqual(len(self.worker.starts), 1)
        self.assertEqual(len(self.worker.stops), 1)
        self.assertTrue(report.lease["deadline_passed"])
        self.assertTrue(report.lease["cleanup_confirmed"])
        self.assertEqual(self.cleanup_calls, [True])
        events = [event for event, _ in self.worker.history]
        self.assertLess(events.index("deadline"), events.index("terminate"))
        self.assertTrue(all(item["accepted"] for item in report.observations))
        ids = [call["observation_id"] for call in self.observer_calls]
        self.assertEqual(len(ids), len(set(ids)))
        for call in self.observer_calls:
            self.assertEqual(call["episode_id"], report.run_id)
            self.assertEqual(call["lease_id"], report.lease["lease_id"])
            self.assertEqual(call["worker_id"], report.worker_id)
        self.assertEqual(self.worker.states, [])
        self.assertEqual(self.worker.commands, [])

    def test_lazy_worker_state_cleanup_cannot_pass_as_independent_enforcement(self):
        self.worker.arm = False

        def lazy_state(**kwargs):
            self.worker.running.clear()
            return LeaseState.STOPPED

        with patch.object(self.worker, "lease_state", side_effect=lazy_state) as poll:
            report = self.probe()
        self.assertEqual(poll.call_count, 0)
        self.assert_failed(report, "idle_lease_deadline_exceeded")
        self.assertFalse(report.lease["deadline_passed"])
        self.assertTrue(report.lease["cleanup_confirmed"])
        self.assertEqual(
            self.worker.stops[0], (report.run_id, report.lease["lease_id"])
        )
        self.assertGreaterEqual(self.worker.history[-2][1], 3)

    def test_deadline_event_without_workload_absence_does_not_pass(self):
        self.worker.enforce = False
        report = self.probe()
        self.assert_failed(report, "idle_lease_deadline_exceeded")
        self.assertTrue(
            any(
                item.get("observation", {}).get("deadline_triggered")
                for item in report.observations
            )
        )
        self.assertTrue(report.lease["cleanup_confirmed"])

    def test_unrelated_exit_does_not_count_as_deadline_termination(self):
        self.worker.arm = False
        self.clock.schedule(1, self.worker.running.clear)
        report = self.probe()
        self.assert_failed(report, "deadline_cause_unconfirmed")
        self.assertFalse(report.lease["deadline_passed"])

    def test_premature_deadline_is_rejected(self):
        self.worker.trigger_after = 1
        report = self.probe()
        self.assert_failed(report, "deadline_triggered_early")

    def test_already_stopped_fixture_does_not_establish_a_live_idle_lease(self):
        original = self.worker.start_lease

        def start(**kwargs):
            receipt = original(**kwargs)
            self.clock.advance(3)
            return receipt

        with patch.object(self.worker, "start_lease", side_effect=start):
            report = self.probe()
        self.assert_failed(report, "initial_running_not_observed")
        self.assertTrue(report.skipped)

    def test_observer_scope_and_fresh_request_binding_are_required(self):
        for field in ("episode_id", "lease_id", "worker_id", "observation_id"):
            with self.subTest(field=field):

                def observe(**kwargs):
                    return replace(self.observe(**kwargs), **{field: "wrong-binding"})

                report = self.probe(observe=observe)
                self.assert_failed(report, "observation_binding_mismatch")
                self.assertFalse(report.observations[0]["accepted"])

    def test_replayed_snapshot_is_rejected_even_if_the_source_and_lease_match(self):
        cached = []

        def observe(**kwargs):
            fresh = self.observe(**kwargs)
            if not cached:
                cached.append(fresh)
                return fresh
            return cached[0]

        report = self.probe(observe=observe)
        self.assert_failed(report, "observation_binding_mismatch")

    def test_observer_identity_cannot_change_mid_run(self):
        def observe(**kwargs):
            result = self.observe(**kwargs)
            return (
                replace(result, source="different-controller")
                if self.clock.now > 0
                else result
            )

        report = self.probe(observe=observe)
        self.assert_failed(report, "observation_source_changed")

    def test_unknown_idle_state_cannot_prove_workload_absence(self):
        def observe(**kwargs):
            result = self.observe(**kwargs)
            return (
                replace(result, state=LeaseState.FAILED)
                if self.clock.now > 0
                else result
            )

        self.assert_failed(self.probe(observe=observe), "workload_state_unconfirmed")

    def test_late_valid_observation_does_not_pass_and_is_retained_as_unaccepted(self):
        def observe(**kwargs):
            if len(self.observer_calls) == 1:
                self.clock.advance(3)
            return self.observe(**kwargs)

        report = self.probe(observe=observe)
        self.assert_failed(report, "operation_deadline_exceeded")
        self.assertFalse(report.lease["deadline_passed"])
        self.assertFalse(report.observations[1]["accepted"])
        self.assertEqual(report.observations[1]["observation"]["state"], "stopped")
        self.assertTrue(report.observations[1]["observation"]["deadline_triggered"])
        self.assertTrue(report.lease["cleanup_confirmed"])

    def test_observation_timeout_never_exceeds_remaining_termination_allowance(self):
        self.worker.enforce = False
        self.probe(observation_timeout_seconds=30)
        idle = self.observer_calls[:-1]
        self.assertEqual(idle[0]["timeout_seconds"], 3)
        self.assertEqual(idle[-1]["timeout_seconds"], 0.25)

    def test_workload_or_receipt_mismatch_prevents_idle_observation(self):
        original_resolve = self.worker.resolve_workload
        with patch.object(
            self.worker,
            "resolve_workload",
            side_effect=lambda **kw: replace(
                original_resolve(**kw), requested_identity="different-workload"
            ),
        ):
            report = self.probe()
        self.assert_failed(report, "invalid_worker_evidence")
        self.assertEqual(self.worker.starts, [])
        self.assertEqual(self.observer_calls, [])
        original_start = self.worker.start_lease
        with patch.object(
            self.worker,
            "start_lease",
            side_effect=lambda **kw: replace(
                original_start(**kw), lease_id="wrong-lease"
            ),
        ):
            report = self.probe()
        self.assert_failed(report, "invalid_worker_evidence")
        self.assertEqual([item["phase"] for item in report.observations], ["cleanup"])

    def test_late_start_uses_separate_cleanup_budget(self):
        original = self.worker.start_lease

        def start(**kwargs):
            receipt = original(**kwargs)
            self.clock.advance(6)
            return receipt

        with patch.object(self.worker, "start_lease", side_effect=start):
            report = self.probe(startup_timeout_seconds=5)
        self.assert_failed(report, "operation_deadline_exceeded")
        self.assertTrue(report.lease["start_attempted"])
        self.assertTrue(report.lease["cleanup_confirmed"])
        self.assertFalse(report.lease["deadline_passed"])

    def test_partial_start_failure_and_teardown_failure_remain_distinct(self):
        original = self.worker.start_lease
        for error in (WorkerFailure, WorkerTeardownFailure):

            def start(**kwargs):
                original(**kwargs)
                raise error("private startup fault")

            with patch.object(self.worker, "start_lease", side_effect=start):
                report = self.probe()
            self.assert_failed(
                report,
                "worker_failure"
                if error is WorkerFailure
                else "worker_teardown_unconfirmed",
            )
            self.assertTrue(report.lease["cleanup_confirmed"])

    def test_failed_compensation_does_not_erase_observed_deadline_or_skip_outer_cleanup(
        self,
    ):
        with patch.object(
            self.worker,
            "terminate_lease",
            side_effect=WorkerTeardownFailure("cleanup failed"),
        ):
            report = self.probe()
        self.assert_failed(report, "lease_cleanup_unconfirmed")
        self.assertTrue(report.lease["deadline_passed"])
        self.assertFalse(report.lease["cleanup_confirmed"])
        self.assertEqual(self.cleanup_calls, [True])

    def test_interrupt_in_observation_or_compensation_still_runs_outer_teardown(self):
        def interrupt(**kwargs):
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            self.probe(observe=interrupt)
        self.assertEqual(self.cleanup_calls, [True])
        with patch.object(
            self.worker, "terminate_lease", side_effect=KeyboardInterrupt()
        ):
            with self.assertRaises(KeyboardInterrupt):
                self.probe()
        self.assertEqual(self.cleanup_calls, [True, True])

    def test_outer_cleanup_requires_exact_confirmation(self):
        for value in (None, False, 1, "confirmed"):
            self.assert_failed(
                self.probe(teardown=lambda: value), "deployment_teardown_unconfirmed"
            )

        def fail():
            raise OSError("private outer teardown error")

        self.assert_failed(self.probe(teardown=fail), "deployment_teardown_failed")

    def test_reports_keep_controller_and_resource_evidence_private(self):
        report = self.probe()
        public = json.dumps(report.public_manifest())
        private = json.dumps(report.private_manifest())
        for value in (
            "independent-fixture-controller",
            "private-controller-event:",
            "fixture.capacity",
            self.task.training_image,
            "observation:",
        ):
            self.assertNotIn(value, public)
            self.assertIn(value, private)

    def test_fresh_scopes_are_used_for_each_diagnostic(self):
        first, second = self.probe(), self.probe()
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertNotEqual(first.lease["lease_id"], second.lease["lease_id"])
        self.assertNotEqual(
            self.worker.starts[0]["lease_dir"], self.worker.starts[1]["lease_dir"]
        )

    def test_observation_requires_typed_state_and_real_trigger_evidence(self):
        valid = dict(
            episode_id="episode-1",
            lease_id="lease-1",
            worker_id="worker-1",
            observation_id="observation-1",
            source="controller",
            state=LeaseState.RUNNING,
            deadline_triggered=False,
        )
        for changes in (
            {"state": "stopped"},
            {"deadline_triggered": 1},
            {"deadline_triggered": True},
            {"deadline_evidence_id": "untriggered"},
            {"source": ""},
            {"worker_fault": {}},
        ):
            with self.subTest(changes=changes), self.assertRaises(
                (ValueError, TypeError)
            ):
                LeaseDeadlineObservation(**{**valid, **changes})

    def test_invalid_inputs_are_rejected_before_worker_calls_or_writes(self):
        invalid = [
            {name: value}
            for name in (
                "duration_seconds",
                "startup_timeout_seconds",
                "termination_grace_seconds",
                "observation_timeout_seconds",
                "poll_interval_seconds",
                "cleanup_timeout_seconds",
            )
            for value in (0, -1, True, float("inf"), float("nan"))
        ] + [
            {"resource_envelope": ResourceEnvelope(requirements=())},
            {"observe": None},
            {"teardown": None},
            {"duration_seconds": 10000},
            {"poll_interval_seconds": 2},
            {"output_root": self.root / "workspace"},
            {"output_root": self.root / "train"},
            {"output_root": self.root / "api"},
            {"output_root": self.root},
            {"workspace": self.root / "missing"},
        ]
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(
                (ValueError, TypeError)
            ):
                self.probe(**overrides)
        self.assertEqual(self.worker.starts, [])
        self.assertEqual(self.worker.episode_ids, [])
        self.assertEqual(self.observer_calls, [])
        self.assertEqual(self.cleanup_calls, [])
        self.assertEqual(list((self.root / "output").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
