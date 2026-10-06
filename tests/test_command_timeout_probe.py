"""Independent observation contracts for the optional Lease command timeout case."""

import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from envcore.conformance import (
    LEASE_COMMAND_TIMEOUT_PROBE_VERSION,
    CommandTimeoutCase,
    CommandTimeoutObservation,
)
from envcore.worker_protocol import (
    CommandResult,
    WorkerFailure,
    WorkerTeardownFailure,
)
from tests import test_lease_probe as leases


TIMEOUT_COMMAND = ("public-fixture", "keep-descendants-running")


class CommandWorker(leases.LeaseWorker):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock
        self.command_scopes = {}
        self.leave_running = False
        self.event_start = True
        self.event_timeout = True
        self.elapsed = None
        self.command_result = CommandResult(
            124, "partial fixture output", "", timed_out=True
        )
        self.command_error = None

    def exec_lease(self, **kwargs):
        if tuple(kwargs["argv"]) != TIMEOUT_COMMAND:
            return super().exec_lease(**kwargs)
        self.commands.append(kwargs)
        key = (kwargs["episode_id"], kwargs["lease_id"])
        self.command_scopes[key] = dict(
            running=self.leave_running,
            start_evidence_id="private-command-start" if self.event_start else None,
            timeout_evidence_id="private-command-timeout"
            if self.event_timeout
            else None,
        )
        self.clock.now += (
            kwargs["timeout_seconds"] if self.elapsed is None else self.elapsed
        )
        if self.command_error is not None:
            raise self.command_error
        return self.command_result

    def terminate_lease(self, **kwargs):
        super().terminate_lease(**kwargs)
        key = (kwargs["episode_id"], kwargs["lease_id"])
        if key in self.command_scopes:
            self.command_scopes[key]["running"] = False


class CommandTimeoutProbeTests(unittest.TestCase):
    def setUp(self):
        leases.LeaseProbeTests.setUp(self)
        self.clock = leases.Clock()
        self.worker = CommandWorker(self.clock)
        self.observed = []

    assert_failed = leases.LeaseProbeTests.assert_failed
    teardown = leases.LeaseProbeTests.teardown

    def observe(self, **kwargs):
        self.observed.append((kwargs, len(self.worker.stops)))
        state = self.worker.command_scopes.get(
            (kwargs["episode_id"], kwargs["lease_id"]), dict(running=False)
        )
        return CommandTimeoutObservation(
            **{
                key: kwargs[key]
                for key in (
                    "episode_id",
                    "lease_id",
                    "worker_id",
                    "command_sha256",
                    "observation_id",
                )
            },
            source="private-independent-command-observer",
            **state,
        )

    def probe(self, *, observe=None, **overrides):
        fixture = CommandTimeoutCase(TIMEOUT_COMMAND, observe or self.observe)
        inputs = dict(command_timeout_case=fixture)
        inputs.update(overrides)
        with patch("envcore.conformance.time.monotonic", self.clock):
            return leases.LeaseProbeTests.probe(self, **inputs)

    def test_timeout_is_observed_before_any_worker_cleanup_and_without_extra_leases(
        self,
    ):
        report = self.probe()
        self.assertTrue(report.passed, report.private_manifest())
        self.assertEqual(report.probe_version, LEASE_COMMAND_TIMEOUT_PROBE_VERSION)
        self.assertFalse(report.public_manifest()["formal_conformance"])
        self.assertEqual(len(self.worker.starts), 3)
        self.assertEqual(len(self.worker.commands), 5)
        evidence = report.leases[0]["command_timeout"]
        self.assertTrue(evidence["attempted"])
        self.assertTrue(evidence["passed"])
        self.assertEqual([stop_count for _, stop_count in self.observed], [0, 0])
        self.assertEqual(self.observed[0][0]["argv"], TIMEOUT_COMMAND)
        self.assertEqual(self.observed[0][0]["lease_id"], report.leases[0]["lease_id"])
        self.assertEqual(self.observed[0][0]["worker_id"], report.worker_id)
        self.assertNotEqual(
            self.observed[0][0]["observation_id"], self.observed[1][0]["observation_id"]
        )
        self.assertEqual(
            evidence["returned_monotonic"] - evidence["started_monotonic"], 2
        )
        self.assertEqual(
            evidence["deadline_monotonic"] - evidence["started_monotonic"], 3
        )
        self.assertEqual(self.cleanups, [True])
        self.assertTrue(all(record["passed"] for record in report.leases))

    def test_reported_timeout_with_surviving_descendants_fails_before_compensation(
        self,
    ):
        self.worker.leave_running = True
        report = self.probe()
        self.assert_failed(report, "command_workloads_still_running")
        self.assertEqual(len(self.worker.starts), 1)
        evidence = report.leases[0]["command_timeout"]
        self.assertTrue(evidence["observations"][-1]["observation"]["running"])
        self.assertFalse(evidence["passed"])
        self.assertTrue(report.leases[0]["termination_confirmed"])
        self.assertFalse(next(iter(self.worker.command_scopes.values()))["running"])

    def test_actual_start_and_command_timeout_cause_are_both_required(self):
        for missing in ("event_start", "event_timeout"):
            self.worker = CommandWorker(self.clock)
            setattr(self.worker, missing, False)
            self.assert_failed(self.probe(), "command_timeout_cause_unconfirmed")

    def test_preexisting_command_or_event_is_rejected_before_dispatch(self):
        for changes in (
            {"running": True},
            {"start_evidence_id": "old-start"},
            {"timeout_evidence_id": "old-timeout"},
        ):
            with self.subTest(changes=changes):
                report = self.probe(
                    observe=lambda **kw: replace(self.observe(**kw), **changes)
                )
                self.assert_failed(report, "command_scope_not_clean")
                self.assertFalse(report.leases[0]["command_timeout"]["attempted"])
        self.assertFalse(
            any(tuple(call["argv"]) == TIMEOUT_COMMAND for call in self.worker.commands)
        )

    def test_all_identity_bindings_and_fresh_observations_are_checked(self):
        for name in (
            "episode_id",
            "lease_id",
            "worker_id",
            "command_sha256",
            "observation_id",
        ):
            with self.subTest(name=name):

                def observe(**kwargs):
                    value = self.observe(**kwargs)
                    if value.start_evidence_id is not None:
                        return replace(
                            value,
                            **{name: "a" * 64 if name == "command_sha256" else "wrong"},
                        )
                    return value

                report = self.probe(observe=observe)
                self.assert_failed(report, "command_observation_binding_mismatch")
                self.assertFalse(
                    report.leases[0]["command_timeout"]["observations"][-1]["accepted"]
                )

    def test_observer_source_change_or_replayed_snapshot_fails(self):
        cached = []

        def replay(**kwargs):
            fresh = self.observe(**kwargs)
            if not cached:
                cached.append(fresh)
            return cached[0]

        self.assert_failed(
            self.probe(observe=replay), "command_observation_binding_mismatch"
        )

        def changed(**kwargs):
            fresh = self.observe(**kwargs)
            return (
                replace(fresh, source="different-source")
                if fresh.start_evidence_id
                else fresh
            )

        self.assert_failed(
            self.probe(observe=changed), "command_observation_source_changed"
        )

    def test_normal_exit_and_failure_are_not_timeout_enforcement(self):
        for result in (
            CommandResult(0, "", ""),
            CommandResult(1, "", ""),
            CommandResult(0, "", "", timed_out=True),
        ):
            self.worker.command_result = result
            self.assert_failed(self.probe(), "command_timeout_not_reported")

    def test_premature_timeout_reply_is_not_accepted(self):
        self.worker.elapsed = 0.5
        self.assert_failed(self.probe(), "command_timeout_reported_early")

    def test_late_reply_is_retained_but_cannot_pass(self):
        self.worker.elapsed = 3
        report = self.probe()
        self.assert_failed(report, "operation_deadline_exceeded")
        evidence = report.leases[0]["command_timeout"]
        self.assertTrue(evidence["result"]["timed_out"])
        self.assertFalse(evidence["passed"])
        self.assertEqual(evidence["returned_monotonic"], evidence["deadline_monotonic"])
        self.assertTrue(report.leases[0]["termination_confirmed"])

    def test_observation_uses_remaining_return_grace_and_late_evidence_fails(self):
        self.worker.elapsed = 2.75

        def observe(**kwargs):
            if self.worker.command_scopes:
                self.assertEqual(kwargs["timeout_seconds"], 0.25)
                self.clock.now += 0.5
            return self.observe(**kwargs)

        report = self.probe(observe=observe)
        self.assert_failed(report, "operation_deadline_exceeded")
        last = report.leases[0]["command_timeout"]["observations"][-1]
        self.assertFalse(last["accepted"])
        self.assertFalse(last["observation"]["running"])

    def test_case_or_lease_expiry_cannot_supply_the_command_timeout(self):
        for limits in ({"duration_seconds": 3}, {"timeout_seconds": 3}):
            report = self.probe(**limits)
            self.assert_failed(report, "command_timeout_allowance_insufficient")
            self.assertFalse(report.leases[0]["command_timeout"]["attempted"])
        self.assertEqual(self.worker.command_scopes, {})

    def test_startup_and_baseline_observation_consume_the_available_lease_window(self):
        original = self.worker.start_lease

        def start(**kwargs):
            receipt = original(**kwargs)
            self.clock.now += 2
            return receipt

        def observe(**kwargs):
            self.clock.now += 0.75
            return self.observe(**kwargs)

        with patch.object(self.worker, "start_lease", side_effect=start):
            report = self.probe(observe=observe, duration_seconds=5)
        self.assert_failed(report, "command_timeout_allowance_insufficient")
        self.assertFalse(report.leases[0]["command_timeout"]["attempted"])

    def test_worker_failure_and_unconfirmed_teardown_remain_distinct(self):
        for error, code in (
            (WorkerFailure, "worker_failure"),
            (WorkerTeardownFailure, "worker_teardown_unconfirmed"),
        ):
            self.worker.command_error = error("private timeout failure")
            report = self.probe()
            self.assert_failed(report, code)
            self.assertTrue(report.leases[0]["termination_confirmed"])
            self.assertFalse(report.leases[0]["command_timeout"]["passed"])

    def test_compensation_failure_does_not_skip_outer_teardown_or_hide_live_command(
        self,
    ):
        self.worker.leave_running = True
        with patch.object(
            self.worker,
            "terminate_lease",
            side_effect=WorkerTeardownFailure("failed stop"),
        ):
            report = self.probe()
        self.assert_failed(report, "command_workloads_still_running")
        self.assert_failed(report, "lease_cleanup_unconfirmed")
        self.assertEqual(self.cleanups, [True])

    def test_interrupt_in_command_or_observer_compensates_before_propagating(self):
        self.worker.command_error = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.probe()
        self.assertEqual(self.cleanups, [True])
        self.assertEqual(len(self.worker.stops), 1)
        self.worker.command_error = None

        def observe(**kwargs):
            value = self.observe(**kwargs)
            if value.start_evidence_id:
                raise KeyboardInterrupt()
            return value

        with self.assertRaises(KeyboardInterrupt):
            self.probe(observe=observe)
        self.assertEqual(self.cleanups, [True, True])
        self.assertEqual(len(self.worker.stops), 2)

    def test_successful_timeout_is_preserved_when_later_cleanup_fails(self):
        report = self.probe(teardown=lambda: False)
        self.assert_failed(report, "deployment_teardown_unconfirmed")
        self.assertTrue(report.leases[0]["command_timeout"]["passed"])

    def test_command_streams_are_bounded_and_evidence_remains_private(self):
        self.worker.command_result = CommandResult(
            124, "x" * 5000, "y" * 5000, timed_out=True
        )
        report = self.probe()
        self.assertTrue(report.passed, report.private_manifest())
        result = report.leases[0]["command_timeout"]["result"]
        self.assertEqual(len(result["stdout"]), 4096)
        self.assertEqual(len(result["stderr"]), 4096)
        self.assertTrue(result["stdout_truncated"])
        self.assertTrue(result["stderr_truncated"])
        for item in (
            *TIMEOUT_COMMAND,
            "private-command-timeout",
            "private-command-start",
            "private-independent-command-observer",
        ):
            self.assertNotIn(item, json.dumps(report.public_manifest()))
            self.assertIn(item, json.dumps(report.private_manifest()))

    def test_invalid_inputs_are_rejected_before_worker_calls(self):
        for changes in (
            {"command": "shell string"},
            {"command": ()},
            {"command": ("",)},
            {"observe": None},
            *(
                {name: value}
                for name in (
                    "timeout_seconds",
                    "termination_grace_seconds",
                    "observation_timeout_seconds",
                )
                for value in (0, -1, True, float("nan"), float("inf"))
            ),
        ):
            with self.subTest(changes=changes), self.assertRaises(
                (TypeError, ValueError)
            ):
                CommandTimeoutCase(
                    **{**dict(command=TIMEOUT_COMMAND, observe=self.observe), **changes}
                )
        with self.assertRaises(TypeError):
            self.probe(command_timeout_case={})
        with self.assertRaises(ValueError):
            self.probe(command=TIMEOUT_COMMAND)
        self.assertEqual(self.worker.starts, [])
        self.assertEqual(self.cleanups, [])
        self.assertEqual(list((self.root / "output").iterdir()), [])

    def test_observation_requires_typed_state_binding_and_nonempty_evidence(self):
        valid = dict(
            episode_id="episode",
            lease_id="lease",
            worker_id="worker",
            command_sha256="a" * 64,
            observation_id="observation",
            source="observer",
            running=False,
        )
        for changes in (
            {"running": None},
            {"running": 1},
            {"source": ""},
            {"command_sha256": "invalid"},
            {"start_evidence_id": ""},
            {"timeout_evidence_id": 1},
        ):
            with self.subTest(changes=changes), self.assertRaises(
                (TypeError, ValueError)
            ):
                CommandTimeoutObservation(**{**valid, **changes})


if __name__ == "__main__":
    unittest.main()
