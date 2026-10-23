"""Exercise the diagnostic against fake Workers, with no accelerator allocation."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from envcore.config import TrainingResources
from envcore.conformance import LEASE_PROBE_VERSION, probe_worker_leases
from envcore.worker_protocol import (
    CommandResult,
    EnforcementLevel,
    LeaseState,
    ResearchTaskSpec,
    ResourceAllocationReceipt,
    ResourceEnvelope,
    ResourceObservation,
    ResourcePropertyEvidence,
    ResourceRequirement,
    WorkerFailure,
    WorkerTeardownFailure,
)
from tests.test_conformance import ResolutionWorker


def contract(value=1):
    return ResourceEnvelope(
        requirements=(
            ResourceRequirement(
                "fixture.capacity", "exact", value, EnforcementLevel.ENFORCED
            ),
        )
    )


class LeaseWorker(ResolutionWorker):
    def __init__(self):
        super().__init__()
        self.starts = []
        self.commands = []
        self.stops = []
        self.states = []
        self.running = set()

    def start_lease(self, **kwargs):
        self.starts.append(kwargs)
        if kwargs["resource_envelope"].sha256 == contract(2).sha256:
            raise WorkerFailure("private fixture capacity unavailable")
        self.running.add((kwargs["episode_id"], kwargs["lease_id"]))
        return ResourceAllocationReceipt(
            receipt_id="receipt:" + kwargs["lease_id"],
            episode_id=kwargs["episode_id"],
            lease_id=kwargs["lease_id"],
            worker_id=self.describe_capabilities().worker_id,
            resource_request_sha256=kwargs["resource_envelope"].sha256,
            properties=(
                ResourcePropertyEvidence(
                    name="fixture.capacity",
                    allocator_allocated=ResourceObservation(
                        "observed", 1, "fixture-allocator"
                    ),
                    workload_visible=ResourceObservation("not_applicable"),
                    enforcement_level=EnforcementLevel.ENFORCED,
                    enforcement_authority="fixture-worker",
                ),
            ),
        )

    def exec_lease(self, **kwargs):
        self.commands.append(kwargs)
        if (kwargs["episode_id"], kwargs["lease_id"]) in self.running:
            return CommandResult(0, "fixture marker\n", "")
        return CommandResult(1, "", "private stopped Lease detail")

    def terminate_lease(self, *, episode_id, lease_id):
        self.stops.append((episode_id, lease_id))
        self.running.discard((episode_id, lease_id))

    def lease_state(self, *, episode_id, lease_id):
        self.states.append((episode_id, lease_id))
        return (
            LeaseState.RUNNING
            if (episode_id, lease_id) in self.running
            else LeaseState.STOPPED
        )

    def resolve_hidden_dataset(self, **kwargs):
        raise AssertionError("Research diagnostic must not resolve hidden data")


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class LeaseProbeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for directory in ("workspace", "output", "train", "api"):
            (self.root / directory).mkdir()
        self.task = ResearchTaskSpec(
            task_id="fixture-task",
            training_image="fixture/training@sha256:" + "a" * 64,
            train_dir=self.root / "train",
            public_api_dir=self.root / "api",
        )
        self.worker = LeaseWorker()
        self.cleanups = []

    def teardown(self):
        self.cleanups.append(True)
        return True

    def probe(self, **overrides):
        inputs = dict(
            task=self.task,
            workspace=self.root / "workspace",
            output_root=self.root / "output",
            resources=TrainingResources(),
            resource_envelope=contract(),
            rejected_resource_envelope=contract(2),
            command=("fixture-command",),
            expected_stdout="fixture marker\n",
            teardown=self.teardown,
        )
        inputs.update(overrides)
        return probe_worker_leases(self.worker, **inputs)

    def assert_failed(self, report, code="invalid_worker_evidence"):
        self.assertFalse(report.passed, report.private_manifest())
        self.assertIn(code, {check.code for check in report.failed})

    def test_lifecycle_resources_and_cleanup_remain_diagnostic_only(self):
        report = self.probe()
        self.assertTrue(report.passed, report.private_manifest())
        self.assertEqual(report.probe_version, LEASE_PROBE_VERSION)
        self.assertNotIn("formal_conformance", report.public_manifest())
        self.assertFalse(hasattr(report, "receipt"))
        self.assertEqual(self.cleanups, [True])
        self.assertEqual(len(self.worker.starts), 3)
        self.assertEqual(len(self.worker.commands), 4)
        self.assertEqual(len(self.worker.stops), 6)
        self.assertEqual(len({call["lease_id"] for call in self.worker.starts}), 3)
        self.assertEqual(len({call["lease_dir"] for call in self.worker.starts}), 3)
        self.assertEqual(set(self.worker.episode_ids), {report.run_id})
        self.assertEqual(self.worker.running, set())
        for call in self.worker.starts:
            self.assertEqual(call["workspace"], (self.root / "workspace").resolve())
            self.assertEqual(call["workload"].role, "training")
            self.assertEqual(call["duration_seconds"], 30)
            self.assertFalse(hasattr(call["task"], "test_path"))
        self.assertEqual(list((self.root / "workspace").iterdir()), [])
        self.assertTrue(
            all(
                record["termination_confirmed"] and record["passed"]
                for record in report.leases
            )
        )

    def test_reports_keep_workloads_resource_and_command_details_private(self):
        report = self.probe()
        public = json.dumps(report.public_manifest())
        private = json.dumps(report.private_manifest())
        for value in (
            "fixture.capacity",
            self.task.training_image,
            "fixture marker",
            "private stopped Lease detail",
            "private fixture capacity unavailable",
        ):
            self.assertNotIn(value, public)
            self.assertIn(value, private)

    def test_wrong_or_missing_receipts_stop_before_commands_and_compensate(self):
        for change in (
            lambda receipt: None,
            lambda receipt: replace(receipt, episode_id="other-episode"),
            lambda receipt: replace(receipt, lease_id="other-lease"),
            lambda receipt: replace(receipt, worker_id="other-worker"),
            lambda receipt: replace(receipt, resource_request_sha256="e" * 64),
            lambda receipt: replace(receipt, properties=()),
            lambda receipt: replace(
                receipt,
                properties=(
                    replace(
                        receipt.properties[0],
                        enforcement_level=EnforcementLevel.BEST_EFFORT,
                    ),
                ),
            ),
            lambda receipt: replace(
                receipt,
                properties=(
                    replace(
                        receipt.properties[0],
                        allocator_allocated=ResourceObservation(
                            "observed", 2, "fixture-allocator"
                        ),
                    ),
                ),
            ),
        ):
            self.worker = LeaseWorker()
            original = self.worker.start_lease
            with patch.object(
                self.worker,
                "start_lease",
                side_effect=lambda **kw: change(original(**kw)),
            ):
                report = self.probe()
            self.assert_failed(report)
            self.assertEqual(len(self.worker.starts), 1)
            self.assertEqual(self.worker.commands, [])
            self.assertEqual(len(self.worker.stops), 1)
            self.assertTrue(report.leases[0]["termination_confirmed"])
            self.assertFalse(report.leases[0]["passed"])

    def test_receipt_id_reuse_is_rejected_on_second_lease(self):
        original = self.worker.start_lease
        with patch.object(
            self.worker,
            "start_lease",
            side_effect=lambda **kw: replace(
                original(**kw), receipt_id="reused-receipt"
            ),
        ):
            report = self.probe()
        self.assert_failed(report)
        self.assertEqual(len(self.worker.starts), 2)
        self.assertEqual(len(self.worker.commands), 2)
        self.assertTrue(report.leases[0]["passed"])
        self.assertFalse(report.leases[1]["passed"])

    def test_resolution_mismatch_or_change_prevents_admission(self):
        original = self.worker.resolve_workload
        for field, value in (
            ("role", "candidate"),
            ("requested_identity", "other-image"),
        ):
            with patch.object(
                self.worker,
                "resolve_workload",
                side_effect=lambda **kw: replace(original(**kw), **{field: value}),
            ):
                self.assert_failed(self.probe())
        self.assertEqual(self.worker.starts, [])
        calls = []

        def resolve(**kwargs):
            calls.append(True)
            value = original(**kwargs)
            return (
                replace(value, effective_sha256="b" * 64) if len(calls) > 1 else value
            )

        with patch.object(self.worker, "resolve_workload", side_effect=resolve):
            report = self.probe()
        self.assert_failed(report)
        self.assertEqual(len(self.worker.starts), 1)

    def test_descriptor_failure_or_wrong_role_cleans_without_admission(self):
        original = self.worker.describe_capabilities()
        for descriptor in (
            None,
            replace(original, roles=("evaluation",)),
        ):
            with patch.object(
                self.worker, "describe_capabilities", return_value=descriptor
            ):
                self.assert_failed(self.probe())
        with patch.object(
            self.worker,
            "describe_capabilities",
            side_effect=ValueError("unsupported Worker protocol version: 15"),
        ):
            self.assert_failed(self.probe())
        self.assertEqual(self.worker.starts, [])
        self.assertEqual(self.cleanups, [True] * 3)

    def test_partial_start_failure_is_registered_before_the_worker_call(self):
        for error in (WorkerFailure, WorkerTeardownFailure, RuntimeError):
            self.worker = LeaseWorker()
            original = self.worker.start_lease

            def start(**kwargs):
                original(**kwargs)
                raise error("created before failing")

            with patch.object(self.worker, "start_lease", side_effect=start):
                report = self.probe()
            self.assert_failed(
                report,
                "worker_teardown_unconfirmed"
                if error is WorkerTeardownFailure
                else "worker_failure"
                if error is WorkerFailure
                else "unexpected_worker_exception",
            )
            self.assertEqual(self.worker.running, set())
            self.assertEqual(len(self.worker.starts), 1)
            self.assertTrue(report.leases[0]["termination_confirmed"])

    def test_bad_command_results_stop_and_preserve_failed_attempt(self):
        for value in (
            None,
            CommandResult(0, "wrong", ""),
            CommandResult(1, "", "failed"),
            CommandResult(124, "", "", timed_out=True),
            CommandResult(0, "fixture marker\n", "", stdout_truncated=True),
        ):
            self.worker = LeaseWorker()
            with patch.object(self.worker, "exec_lease", return_value=value):
                report = self.probe()
            self.assert_failed(report)
            self.assertEqual(len(self.worker.starts), 1)
            self.assertEqual(self.worker.running, set())
            self.assertFalse(report.leases[0]["passed"])

    def test_post_stop_success_timeout_or_output_is_not_accepted_as_rejection(self):
        for result in (
            CommandResult(0, "", ""),
            CommandResult(0, "fixture marker\n", ""),
            CommandResult(1, "fixture marker\n", ""),
            CommandResult(124, "", "", timed_out=True),
        ):
            self.worker = LeaseWorker()
            original = self.worker.exec_lease

            def execute(**kwargs):
                if (
                    kwargs["episode_id"],
                    kwargs["lease_id"],
                ) not in self.worker.running:
                    return result
                return original(**kwargs)

            with patch.object(self.worker, "exec_lease", side_effect=execute):
                report = self.probe()
            self.assert_failed(report)
            self.assertEqual(len(self.worker.starts), 1)

    def test_post_stop_worker_failure_is_valid_but_teardown_failure_is_not(self):
        for error in (WorkerFailure, WorkerTeardownFailure):
            self.worker = LeaseWorker()
            original = self.worker.exec_lease

            def execute(**kwargs):
                if (
                    kwargs["episode_id"],
                    kwargs["lease_id"],
                ) not in self.worker.running:
                    raise error("stopped")
                return original(**kwargs)

            with patch.object(self.worker, "exec_lease", side_effect=execute):
                report = self.probe()
            if error is WorkerFailure:
                self.assertTrue(report.passed)
            else:
                self.assert_failed(report, "worker_teardown_unconfirmed")

    def test_rejected_admission_cannot_hide_a_live_workload(self):
        original = self.worker.start_lease

        def start(**kwargs):
            try:
                return original(**kwargs)
            except WorkerFailure:
                self.worker.running.add((kwargs["episode_id"], kwargs["lease_id"]))
                raise

        with patch.object(self.worker, "start_lease", side_effect=start):
            report = self.probe()
        self.assert_failed(report)
        self.assertEqual(len(self.worker.starts), 3)
        self.assertTrue(report.leases[-1]["termination_confirmed"])
        self.assertFalse(report.leases[-1]["passed"])

    def test_unexpected_admission_of_rejected_contract_is_compensated(self):
        original = self.worker.start_lease

        def start(**kwargs):
            return original(**{**kwargs, "resource_envelope": contract()})

        with patch.object(self.worker, "start_lease", side_effect=start):
            report = self.probe()
        self.assert_failed(report)
        self.assertIn("resource_receipt", report.leases[-1])
        self.assertEqual(self.worker.running, set())

    def test_termination_is_checked_and_successful_cleanup_does_not_erase_failure(self):
        original = self.worker.terminate_lease
        attempts = []

        def stop(**kwargs):
            attempts.append(True)
            if len(attempts) > 1:
                original(**kwargs)

        with patch.object(self.worker, "terminate_lease", side_effect=stop):
            report = self.probe()
        self.assert_failed(report)
        self.assertTrue(report.leases[0]["termination_confirmed"])
        self.assertEqual(len(self.worker.starts), 1)

    def test_repeated_termination_failure_is_visible(self):
        original = self.worker.terminate_lease
        attempts = []

        def stop(**kwargs):
            attempts.append(True)
            if len(attempts) == 2:
                raise WorkerTeardownFailure("duplicate stop failed")
            original(**kwargs)

        with patch.object(self.worker, "terminate_lease", side_effect=stop):
            report = self.probe()
        self.assert_failed(report, "worker_teardown_unconfirmed")
        self.assertTrue(report.leases[0]["termination_confirmed"])
        self.assertEqual(len(attempts), 3)

    def test_operation_timeout_is_shared_with_resolution_and_startup(self):
        clock = Clock()
        original_resolve = self.worker.resolve_workload
        original_start = self.worker.start_lease

        def resolve(**kwargs):
            clock.now += 2
            return original_resolve(**kwargs)

        def start(**kwargs):
            clock.now += 3
            return original_start(**kwargs)

        with patch("envcore.conformance.time.monotonic", clock), patch.object(
            self.worker, "resolve_workload", side_effect=resolve
        ), patch.object(self.worker, "start_lease", side_effect=start):
            report = self.probe(timeout_seconds=10)
        self.assertTrue(report.passed, report.private_manifest())
        self.assertEqual(self.worker.commands[0]["timeout_seconds"], 5)
        self.assertEqual(self.worker.starts[0]["duration_seconds"], 30)

    def test_late_start_response_gets_fresh_compensation_allowance(self):
        clock = Clock()
        original = self.worker.start_lease

        def start(**kwargs):
            result = original(**kwargs)
            clock.now += 10
            return result

        with patch("envcore.conformance.time.monotonic", clock), patch.object(
            self.worker, "start_lease", side_effect=start
        ):
            report = self.probe(timeout_seconds=10)
        self.assert_failed(report, "operation_deadline_exceeded")
        self.assertEqual(self.worker.commands, [])
        self.assertEqual(self.worker.running, set())
        self.assertTrue(report.leases[0]["termination_confirmed"])

    def test_late_rejection_is_not_a_pass_and_late_teardown_stays_explicit(self):
        for error in (WorkerFailure, WorkerTeardownFailure):
            clock = Clock()
            self.worker = LeaseWorker()
            original = self.worker.start_lease

            def start(**kwargs):
                try:
                    return original(**kwargs)
                except WorkerFailure:
                    clock.now += 11
                    raise error("late rejection")

            with patch("envcore.conformance.time.monotonic", clock), patch.object(
                self.worker, "start_lease", side_effect=start
            ):
                report = self.probe(timeout_seconds=10)
            self.assert_failed(
                report,
                "operation_deadline_exceeded"
                if error is WorkerFailure
                else "worker_teardown_unconfirmed",
            )
            self.assertFalse(report.leases[-1]["passed"])

    def test_cleanup_failure_or_late_cleanup_still_invokes_deployment_teardown(self):
        for late in (False, True):
            self.worker = LeaseWorker()
            clock = Clock()

            def stop(**kwargs):
                if late:
                    clock.now += 31
                    return None
                raise WorkerTeardownFailure("still running")

            with patch("envcore.conformance.time.monotonic", clock), patch.object(
                self.worker, "exec_lease", side_effect=WorkerFailure("failed")
            ), patch.object(self.worker, "terminate_lease", side_effect=stop):
                report = self.probe(cleanup_timeout_seconds=30)
            self.assert_failed(report, "lease_cleanup_unconfirmed")
            self.assertFalse(report.leases[0]["termination_confirmed"])
        self.assertEqual(self.cleanups, [True, True])

    def test_interrupt_after_partial_start_compensates_and_propagates(self):
        original = self.worker.start_lease

        def start(**kwargs):
            original(**kwargs)
            raise KeyboardInterrupt()

        with patch.object(self.worker, "start_lease", side_effect=start):
            with self.assertRaises(KeyboardInterrupt):
                self.probe()
        self.assertEqual(self.worker.running, set())
        self.assertEqual(self.cleanups, [True])

    def test_deployment_teardown_requires_exact_confirmation(self):
        for value in (None, False, 1, "yes"):
            self.worker = LeaseWorker()
            self.assert_failed(
                self.probe(teardown=lambda: value), "deployment_teardown_unconfirmed"
            )

        def fail():
            raise OSError("private cleanup failure")

        self.assert_failed(self.probe(teardown=fail), "deployment_teardown_failed")

    def test_interrupt_in_compensation_still_runs_deployment_teardown(self):
        with patch.object(
            self.worker, "exec_lease", side_effect=WorkerFailure("failed")
        ), patch.object(
            self.worker, "terminate_lease", side_effect=KeyboardInterrupt()
        ):
            with self.assertRaises(KeyboardInterrupt):
                self.probe()
        self.assertEqual(self.cleanups, [True])

    def test_late_command_success_is_rejected_and_compensated(self):
        clock = Clock()
        original = self.worker.exec_lease

        def execute(**kwargs):
            result = original(**kwargs)
            clock.now += 11
            return result

        with patch("envcore.conformance.time.monotonic", clock), patch.object(
            self.worker, "exec_lease", side_effect=execute
        ):
            report = self.probe(timeout_seconds=10)
        self.assert_failed(report, "operation_deadline_exceeded")
        self.assertEqual(len(self.worker.starts), 1)
        self.assertEqual(self.worker.running, set())

    def test_untyped_state_is_not_a_termination_confirmation(self):
        with patch.object(self.worker, "lease_state", return_value="stopped"):
            report = self.probe()
        self.assert_failed(report)
        self.assert_failed(report, "lease_cleanup_unconfirmed")
        self.assertFalse(report.leases[0]["termination_confirmed"])
        self.assertEqual(self.cleanups, [True])

    def test_repeated_probe_uses_fresh_scopes(self):
        first, second = self.probe(), self.probe()
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(len({value["lease_dir"] for value in self.worker.starts}), 6)

    def test_invalid_inputs_fail_before_worker_or_writes(self):
        invalid = [
            {name: value}
            for name in (
                "duration_seconds",
                "timeout_seconds",
                "cleanup_timeout_seconds",
            )
            for value in (0, -1, True, float("inf"), float("nan"))
        ] + [
            {"resource_envelope": ResourceEnvelope(requirements=())},
            {"rejected_resource_envelope": contract()},
            {"command": "shell command"},
            {"command": []},
            {"command": [None]},
            {"expected_stdout": ""},
            {"output_root": self.root / "workspace"},
            {"output_root": self.root},
            {"output_root": self.root / "train"},
            {"output_root": self.root / "api"},
            {"workspace": self.root / "missing"},
            {"teardown": None},
        ]
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(
                (ValueError, TypeError)
            ):
                self.probe(**overrides)
        self.assertEqual(self.cleanups, [])
        self.assertEqual(self.worker.episode_ids, [])
        self.assertEqual(list((self.root / "output").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
