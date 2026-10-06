import hashlib
import unittest
import uuid
from pathlib import Path

from envcore.conformance import (
    RESOLUTION_PROBE_VERSION,
    probe_worker_resolution,
)
from envcore.config import TaskConfig
from envcore.worker_protocol import (
    BASELINE_REQUIREMENTS,
    Capability,
    EnforcementLevel,
    EvaluationTaskSpec,
    HiddenDatasetIdentityMismatch,
    ResolvedHiddenDataset,
    ResolvedWorkload,
    TrustLevel,
    WorkerDescriptor,
    WorkloadRequest,
)


ROOT = Path(__file__).resolve().parents[1]
TASK = EvaluationTaskSpec.from_task(
    TaskConfig.load(ROOT / "tests/fixtures/task/task.json")
)


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ResolutionWorker:
    """Enough Worker surface for the deliberately limited resolution probe."""

    def __init__(self):
        self.sources = {"analysis": _digest("analysis"), "test": _digest("test")}
        self.episode_ids = []
        self.timeouts = []

    def describe_capabilities(self):
        return WorkerDescriptor(
            worker_id="resolution-worker",
            trust_level=TrustLevel.DEVELOPMENT,
            roles=("research", "evaluation"),
            capabilities={
                requirement.name: Capability(EnforcementLevel.ENFORCED)
                for requirement in BASELINE_REQUIREMENTS
            },
        )

    def resolve_hidden_dataset(
        self, *, episode_id, task, hidden_dataset, timeout_seconds
    ):
        del task
        self.episode_ids.append(episode_id)
        self.timeouts.append(timeout_seconds)
        digest = self.sources[hidden_dataset.name]
        resolved = ResolvedHiddenDataset(hidden_dataset.name, digest, "shared-handle")
        if (
            hidden_dataset.expected_sha256 is not None
            and hidden_dataset.expected_sha256 != digest
        ):
            raise HiddenDatasetIdentityMismatch(resolved)
        return resolved

    def resolve_workload(self, *, episode_id, workload, timeout_seconds):
        self.episode_ids.append(episode_id)
        self.timeouts.append(timeout_seconds)
        return ResolvedWorkload(
            role=workload.role,
            requested_identity=workload.requested_identity,
            effective_format="oci",
            effective_sha256=_digest(workload.requested_identity),
            transformation_sha256=None,
            resolution_id="shared-handle",
        )


class ResolutionProbeTests(unittest.TestCase):
    def report(self, worker, *, teardown=lambda: None, identities=None, **kwargs):
        return probe_worker_resolution(
            worker,
            task=TASK,
            workloads=(WorkloadRequest("training", "example@sha256:" + "a" * 64),),
            hidden_dataset_identities=identities or dict(worker.sources),
            teardown=teardown,
            **kwargs,
        )

    def failed_names(self, report):
        return {item.name for item in report.failed}

    def test_passing_probe_is_explicitly_not_formal_conformance(self):
        worker = ResolutionWorker()

        report = self.report(worker)

        self.assertTrue(report.passed, report.private_manifest())
        self.assertEqual(report.probe_version, RESOLUTION_PROBE_VERSION)
        self.assertFalse(report.public_manifest()["formal_conformance"])
        self.assertFalse(hasattr(report, "receipt"))
        # This object intentionally lacks Lease and evaluation methods. Passing
        # the narrow probe must never imply full Worker Protocol conformance.
        self.assertFalse(hasattr(worker, "start_lease"))
        self.assertFalse(hasattr(worker, "evaluate_candidate"))

    def test_release_owned_hidden_identity_is_required_and_verified(self):
        worker = ResolutionWorker()
        expected = dict(worker.sources)
        expected["test"] = _digest("organizer-test")

        report = self.report(worker, identities=expected)

        self.assertFalse(report.passed)
        self.assertIn(
            "hidden_dataset[test].resolves_expected_identity",
            self.failed_names(report),
        )

    def test_worker_that_ignores_expected_identity_is_caught(self):
        class IgnoresIdentity(ResolutionWorker):
            def resolve_hidden_dataset(
                self, *, episode_id, task, hidden_dataset, timeout_seconds
            ):
                del task, timeout_seconds
                self.episode_ids.append(episode_id)
                return ResolvedHiddenDataset(
                    hidden_dataset.name,
                    self.sources[hidden_dataset.name],
                    uuid.uuid4().hex,
                )

        report = self.report(IgnoresIdentity())

        self.assertFalse(report.passed)
        self.assertIn(
            "hidden_dataset[analysis].rejects_wrong_identity",
            self.failed_names(report),
        )

    def test_release_may_deliberately_use_the_same_content_for_both_roles(self):
        worker = ResolutionWorker()
        shared = _digest("shared")
        worker.sources = {"analysis": shared, "test": shared}

        report = self.report(worker)

        self.assertTrue(report.passed, report.private_manifest())

    def test_opaque_resolution_handles_may_be_reused(self):
        report = self.report(ResolutionWorker())

        self.assertTrue(report.passed, report.private_manifest())

    def test_workload_request_drift_is_caught(self):
        class DriftsIdentity(ResolutionWorker):
            def resolve_workload(self, *, episode_id, workload, timeout_seconds):
                del episode_id, timeout_seconds
                return ResolvedWorkload(
                    role=workload.role,
                    requested_identity="substituted@sha256:" + "b" * 64,
                    effective_format="oci",
                    effective_sha256=_digest("substituted"),
                    transformation_sha256=None,
                    resolution_id=uuid.uuid4().hex,
                )

        report = self.report(DriftsIdentity())

        self.assertFalse(report.passed)
        self.assertIn(
            "workload[training].binds_release_request",
            self.failed_names(report),
        )

    def test_unexpected_worker_exception_still_runs_teardown(self):
        private_path = "/organizer/private/hidden.json"

        class Crashes(ResolutionWorker):
            def resolve_workload(self, *, episode_id, workload, timeout_seconds):
                raise OSError(private_path)

        teardown_calls = []
        report = self.report(Crashes(), teardown=lambda: teardown_calls.append(True))

        self.assertEqual(teardown_calls, [True])
        self.assertFalse(report.passed)
        self.assertNotIn(private_path, str(report.public_manifest()))
        self.assertIn(private_path, str(report.private_manifest()))

    def test_unconfirmed_deployment_teardown_fails_the_probe(self):
        def teardown():
            raise RuntimeError("private scheduler teardown failed")

        report = self.report(ResolutionWorker(), teardown=teardown)

        self.assertFalse(report.passed)
        self.assertIn("deployment.teardown_confirmed", self.failed_names(report))

        incomplete = self.report(ResolutionWorker(), teardown=lambda: False)
        self.assertFalse(incomplete.passed)
        self.assertIn("deployment.teardown_confirmed", self.failed_names(incomplete))

    def test_each_probe_uses_a_unique_operation_scope(self):
        first = ResolutionWorker()
        second = ResolutionWorker()

        first_report = self.report(first)
        second_report = self.report(second)

        self.assertNotEqual(first_report.run_id, second_report.run_id)
        self.assertEqual(set(first.episode_ids), {first_report.run_id})
        self.assertEqual(set(second.episode_ids), {second_report.run_id})

    def test_probe_passes_only_positive_deadlines_to_worker(self):
        worker = ResolutionWorker()

        report = self.report(worker, timeout_seconds=0.25)

        self.assertTrue(report.passed, report.private_manifest())
        self.assertTrue(worker.timeouts)
        self.assertTrue(all(value == 0.25 for value in worker.timeouts))
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            self.report(ResolutionWorker(), timeout_seconds=0.0)

    def test_required_probe_inputs_cannot_be_skipped(self):
        worker = ResolutionWorker()
        with self.assertRaisesRegex(ValueError, "workload"):
            probe_worker_resolution(
                worker,
                task=TASK,
                workloads=(),
                hidden_dataset_identities=worker.sources,
                teardown=lambda: None,
            )
        with self.assertRaisesRegex(ValueError, "hidden dataset"):
            probe_worker_resolution(
                worker,
                task=TASK,
                workloads=(WorkloadRequest("training", "example@sha256:" + "a" * 64),),
                hidden_dataset_identities={},
                teardown=lambda: None,
            )


if __name__ == "__main__":
    unittest.main()
