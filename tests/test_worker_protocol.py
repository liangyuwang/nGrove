import json
import unittest

from benchcore.artifact_identity import HIDDEN_BUNDLE_IDENTITY_ALGORITHM
from benchcore.candidate_protocol import (
    PREDICTION_ARTIFACT_RETENTION,
    PREDICTION_MANIFEST_MAX_BYTES,
    prediction_storage_limit,
)
from benchcore.worker_protocol import (
    BASELINE_REQUIREMENTS,
    CONFORMANCE_RECEIPT_VERSION,
    IMMUTABLE_WORKLOAD_IDENTITY,
    Capability,
    CapabilityRequirement,
    CandidateChannelRequirements,
    CandidateEvaluationResult,
    CommandResult,
    ConformanceReceipt,
    CostContract,
    EnforcementLevel,
    HiddenDatasetIdentityMismatch,
    HiddenDatasetRequest,
    PerformanceScoreContract,
    ResourceAllocationReceipt,
    ResourceEnvelope,
    ResourceObservation,
    ResourcePropertyEvidence,
    ResourceRequirement,
    ResolvedHiddenDataset,
    ResolvedWorkload,
    TrustLevel,
    WorkloadRequest,
    WorkloadResolutionAcceptance,
    WORKER_CONFORMANCE_SUITE_VERSION,
    WorkerDescriptor,
    assess_resource_allocation,
    assess_worker,
)


def descriptor(*, trust=TrustLevel.DEVELOPMENT, overrides=None):
    capabilities = {
        requirement.name: Capability(EnforcementLevel.ENFORCED)
        for requirement in BASELINE_REQUIREMENTS
    }
    capabilities.update(overrides or {})
    return WorkerDescriptor(
        worker_id="test-deployment",
        trust_level=trust,
        roles=("research", "evaluation"),
        capabilities=capabilities,
    )


def conformance_receipt(
    *,
    worker_id="test-deployment",
    suite_version=WORKER_CONFORMANCE_SUITE_VERSION,
    passed=True,
):
    return ConformanceReceipt(
        receipt_version=CONFORMANCE_RECEIPT_VERSION,
        suite_version=suite_version,
        worker_id=worker_id,
        authority="test-organizer",
        report_sha256="c" * 64,
        passed=passed,
    )


class WorkerProtocolTests(unittest.TestCase):
    def test_resource_receipt_is_property_extensible_and_operation_bound(self):
        envelope = ResourceEnvelope(
            requirements=(
                ResourceRequirement(
                    "site.scratch.capacity_bytes",
                    "at_least",
                    1024,
                    EnforcementLevel.ENFORCED,
                ),
            ),
            observational_properties=("site.fabric.class",),
        )
        receipt = ResourceAllocationReceipt(
            episode_id="episode-1",
            lease_id="lease-0001",
            worker_id="site-worker",
            resource_request_sha256=envelope.sha256,
            properties=(
                ResourcePropertyEvidence(
                    name="site.scratch.capacity_bytes",
                    allocator_allocated=ResourceObservation(
                        "observed", 2048, "site-allocator"
                    ),
                    workload_visible=ResourceObservation("unavailable"),
                    enforcement_level=EnforcementLevel.ENFORCED,
                    enforcement_authority="site-storage-policy",
                ),
            ),
            receipt_id="allocation-1",
        )

        assessment = assess_resource_allocation(
            envelope,
            receipt,
            episode_id="episode-1",
            lease_id="lease-0001",
            worker_id="site-worker",
        )

        self.assertTrue(assessment.satisfied)
        serialized = json.dumps(receipt.public_manifest()).lower()
        for forbidden in ("path", "container_id", "job_id", "cgroup"):
            self.assertNotIn(forbidden, serialized)
        with self.assertRaisesRegex(ValueError, "different episode"):
            assess_resource_allocation(
                envelope,
                receipt,
                episode_id="episode-2",
                lease_id="lease-0001",
                worker_id="site-worker",
            )

    def test_resource_assessment_detects_view_disagreement_and_weak_enforcement(self):
        envelope = ResourceEnvelope(
            requirements=(
                ResourceRequirement(
                    "accelerator.count",
                    "exact",
                    4,
                    EnforcementLevel.ENFORCED,
                ),
            )
        )
        receipt = ResourceAllocationReceipt(
            episode_id="episode-1",
            lease_id="lease-0001",
            worker_id="site-worker",
            resource_request_sha256=envelope.sha256,
            properties=(
                ResourcePropertyEvidence(
                    name="accelerator.count",
                    allocator_allocated=ResourceObservation("observed", 4, "allocator"),
                    workload_visible=ResourceObservation("observed", 3, "sandbox"),
                    enforcement_level=EnforcementLevel.BEST_EFFORT,
                    enforcement_authority="allocator",
                ),
            ),
            receipt_id="allocation-1",
        )

        assessment = assess_resource_allocation(
            envelope,
            receipt,
            episode_id="episode-1",
            lease_id="lease-0001",
            worker_id="site-worker",
        )

        self.assertFalse(assessment.satisfied)
        self.assertEqual(assessment.mismatched, ("accelerator.count",))
        self.assertEqual(
            assessment.insufficient_enforcement,
            ("accelerator.count",),
        )

    def test_effective_workload_view_may_be_smaller_than_allocator_grant(self):
        envelope = ResourceEnvelope(
            requirements=(
                ResourceRequirement(
                    "accelerator.count",
                    "exact",
                    4,
                    EnforcementLevel.ENFORCED,
                ),
            )
        )
        receipt = ResourceAllocationReceipt(
            episode_id="episode-1",
            lease_id="lease-0001",
            worker_id="site-worker",
            resource_request_sha256=envelope.sha256,
            properties=(
                ResourcePropertyEvidence(
                    name="accelerator.count",
                    allocator_allocated=ResourceObservation(
                        "observed", 8, "whole-node-allocator"
                    ),
                    workload_visible=ResourceObservation(
                        "observed", 4, "workload-sandbox"
                    ),
                    enforcement_level=EnforcementLevel.ENFORCED,
                    enforcement_authority="workload-sandbox",
                ),
            ),
            receipt_id="allocation-1",
        )

        assessment = assess_resource_allocation(
            envelope,
            receipt,
            episode_id="episode-1",
            lease_id="lease-0001",
            worker_id="site-worker",
        )

        self.assertTrue(assessment.satisfied)
        self.assertEqual(assessment.mismatched, ())

    def test_resource_ceiling_uses_at_most_relation(self):
        envelope = ResourceEnvelope(
            requirements=(
                ResourceRequirement(
                    "memory.bytes",
                    "at_most",
                    1024,
                    EnforcementLevel.ENFORCED,
                ),
            )
        )

        def assessment_for(value):
            receipt = ResourceAllocationReceipt(
                episode_id="episode-1",
                lease_id="lease-0001",
                worker_id="site-worker",
                resource_request_sha256=envelope.sha256,
                properties=(
                    ResourcePropertyEvidence(
                        name="memory.bytes",
                        allocator_allocated=ResourceObservation("not_applicable"),
                        workload_visible=ResourceObservation(
                            "observed", value, "sandbox"
                        ),
                        enforcement_level=EnforcementLevel.ENFORCED,
                        enforcement_authority="sandbox",
                    ),
                ),
                receipt_id="allocation-1",
            )
            return assess_resource_allocation(
                envelope,
                receipt,
                episode_id="episode-1",
                lease_id="lease-0001",
                worker_id="site-worker",
            )

        self.assertTrue(assessment_for(512).satisfied)
        self.assertEqual(assessment_for(2048).mismatched, ("memory.bytes",))

    def test_resource_schema_versions_and_exact_types_are_strict(self):
        with self.assertRaisesRegex(ValueError, "schema version"):
            ResourceEnvelope(requirements=(), schema_version=True)
        with self.assertRaisesRegex(ValueError, "receipt version"):
            ResourceAllocationReceipt(
                episode_id="episode-1",
                lease_id="lease-0001",
                worker_id="site-worker",
                resource_request_sha256="a" * 64,
                properties=(),
                receipt_id="allocation-1",
                receipt_version=True,
            )

        envelope = ResourceEnvelope(
            requirements=(
                ResourceRequirement(
                    "host.count", "exact", 1, EnforcementLevel.BEST_EFFORT
                ),
            )
        )
        receipt = ResourceAllocationReceipt(
            episode_id="episode-1",
            lease_id="lease-0001",
            worker_id="site-worker",
            resource_request_sha256=envelope.sha256,
            properties=(
                ResourcePropertyEvidence(
                    name="host.count",
                    allocator_allocated=ResourceObservation(
                        "observed", True, "allocator"
                    ),
                    workload_visible=ResourceObservation("not_applicable"),
                    enforcement_level=EnforcementLevel.ENFORCED,
                    enforcement_authority="allocator",
                ),
            ),
            receipt_id="allocation-1",
        )

        assessment = assess_resource_allocation(
            envelope,
            receipt,
            episode_id="episode-1",
            lease_id="lease-0001",
            worker_id="site-worker",
        )
        self.assertEqual(assessment.mismatched, ("host.count",))

    def test_candidate_channel_denies_target_feedback_and_reexposure(self):
        manifest = CandidateChannelRequirements(
            protocol="offline_next_token_logits_v3"
        ).public_manifest()

        self.assertEqual(manifest["mechanism"], "worker_materialized_files")
        self.assertEqual(manifest["request_content"], "target_free_context_artifact")
        self.assertEqual(manifest["predictions_per_context"], 1)
        self.assertEqual(manifest["context_artifact_access"], "read_only")
        self.assertEqual(
            manifest["prediction_artifact_access"],
            "candidate_read_write_isolated",
        )
        self.assertEqual(
            manifest["prediction_manifest"]["exact_fields"],
            [
                "schema_version",
                "candidate_protocol",
                "dataset_id",
                "examples",
                "vocab_size",
                "dtype",
                "logits_file",
            ],
        )
        self.assertFalse(manifest["prediction_manifest"]["additional_fields_allowed"])
        self.assertEqual(
            manifest["public_helper_api"]["prediction_writer"],
            "write_prediction_artifact(output_dir, *, dataset_id, logits, "
            "examples, vocab_size)",
        )
        self.assertEqual(
            manifest["prediction_storage"]["hard_limit"],
            "required_before_candidate_start",
        )
        self.assertEqual(manifest["prediction_storage"]["artifact_max_entries"], 2)
        self.assertEqual(manifest["prediction_storage"]["max_inodes"], 128)
        self.assertEqual(
            manifest["prediction_storage"]["manifest_max_bytes"],
            PREDICTION_MANIFEST_MAX_BYTES,
        )
        self.assertEqual(
            manifest["prediction_artifact_retention"],
            PREDICTION_ARTIFACT_RETENTION,
        )
        self.assertEqual(manifest["candidate_scheduling"], "participant_controlled")
        self.assertEqual(
            manifest["serving_timing"]["start"],
            "immediately_before_candidate_process_invocation",
        )
        self.assertIn("model_loading", manifest["serving_timing"]["includes"])
        self.assertIn("metric_computation", manifest["serving_timing"]["excludes"])
        self.assertEqual(manifest["target_artifact_visibility"], "denied")
        self.assertEqual(manifest["current_target_visibility"], "denied")
        self.assertEqual(manifest["future_token_visibility"], "denied")
        self.assertEqual(manifest["target_feedback_visibility"], "denied")
        self.assertEqual(manifest["cross_context_target_overlap"], "denied")
        self.assertEqual(manifest["candidate_egress"], "denied")
        with self.assertRaisesRegex(ValueError, "hard limits"):
            CandidateChannelRequirements(
                protocol="offline_next_token_logits_v3",
                prediction_storage_hard_limit_required=False,
            )
        with self.assertRaisesRegex(ValueError, "retention"):
            CandidateChannelRequirements(
                protocol="offline_next_token_logits_v3",
                prediction_artifact_retention="retain",
            )

    def test_candidate_evaluation_result_exposes_average_cost_and_evidence(self):
        result = CandidateEvaluationResult(
            performance_score=72.5,
            raw_metrics={"loss": 1.25},
            candidate_wall_seconds=2.0,
            task_count=4,
            input_tokens=20,
            prediction_bytes=128,
        )

        self.assertEqual(result.performance_score, 72.5)
        self.assertEqual(
            result.performance_manifest(
                PerformanceScoreContract("example_score_v1", ("loss",))
            ),
            {
                "performance_score": 72.5,
                "raw_metrics": {"loss": 1.25},
            },
        )
        self.assertEqual(
            result.cost_manifest(
                CostContract(
                    "candidate_end_to_end_time_per_task_v1",
                    "next_token_prediction",
                )
            ),
            {
                "time_per_task": 0.5,
                "candidate_wall_seconds": 2.0,
                "task_count": 4,
                "input_tokens": 20,
                "prediction_bytes": 128,
                "tasks_per_second": 2.0,
                "input_tokens_per_second": 10.0,
            },
        )
        with self.assertRaisesRegex(ValueError, "candidate_wall_seconds"):
            CandidateEvaluationResult(50.0, {"loss": 1.0}, 0.0, 1, 1, 1)
        with self.assertRaisesRegex(ValueError, "cost"):
            CandidateEvaluationResult(50.0, {"loss": 1.0}, 5e-324, 1, 1, 1)

    def test_cost_contract_fixes_average_time_semantics(self):
        contract = CostContract(
            "candidate_end_to_end_time_per_task_v1",
            "next_token_prediction",
        )

        self.assertEqual(
            CostContract.from_manifest(contract.public_manifest()),
            contract,
        )
        with self.assertRaisesRegex(ValueError, "primary_metric"):
            CostContract(
                "candidate_end_to_end_time_per_task_v1",
                "next_token_prediction",
                primary_metric="candidate_wall_seconds",
            )

    def test_cost_contract_rejects_unrepresentable_counts(self):
        contract = CostContract(
            "candidate_end_to_end_time_per_task_v1",
            "next_token_prediction",
        )
        cost = {
            "time_per_task": 1.0,
            "candidate_wall_seconds": 1.0,
            "task_count": 10**10000,
            "input_tokens": 1,
            "prediction_bytes": 1,
            "tasks_per_second": 1.0,
            "input_tokens_per_second": 1.0,
        }

        with self.assertRaisesRegex(ValueError, "cannot be represented"):
            contract.validate_measurement(cost)

    def test_performance_score_contract_is_generic_and_release_scoped(self):
        contract = PerformanceScoreContract("accuracy_v2", ("accuracy",))

        self.assertEqual(
            contract.validate_measurement(84.5, {"accuracy": 0.845}),
            (84.5, {"accuracy": 0.845}),
        )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            contract.validate_measurement(84.5, {"loss": 0.2})
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            contract.validate_measurement(101.0, {"accuracy": 0.845})
        with self.assertRaisesRegex(ValueError, "raw metric name"):
            PerformanceScoreContract("accuracy_v2", (1,))
        with self.assertRaisesRegex(ValueError, "range"):
            PerformanceScoreContract(
                "accuracy_v2",
                ("accuracy",),
                minimum=10**1000,
            )

    def test_candidate_evaluation_rejects_unrepresentable_numbers(self):
        with self.assertRaisesRegex(ValueError, "performance_score"):
            CandidateEvaluationResult(
                10**1000,
                {"loss": 1.0},
                1.0,
                1,
                1,
                1,
            )
        with self.assertRaisesRegex(ValueError, "raw metric loss"):
            CandidateEvaluationResult(
                50.0,
                {"loss": 10**1000},
                1.0,
                1,
                1,
                1,
            )

    def test_prediction_storage_limit_is_derived_from_the_request_shape(self):
        self.assertEqual(
            prediction_storage_limit(examples=2048, vocab_size=50257),
            {
                "max_bytes": 786 * 1024 * 1024,
                "max_inodes": 128,
                "artifact_max_bytes": 2048 * 50257 * 4 + PREDICTION_MANIFEST_MAX_BYTES,
                "artifact_max_entries": 2,
                "logits_bytes": 2048 * 50257 * 4,
                "manifest_max_bytes": PREDICTION_MANIFEST_MAX_BYTES,
            },
        )
        small = prediction_storage_limit(examples=3, vocab_size=3)
        self.assertEqual(small["max_bytes"], 16 * 1024 * 1024)
        self.assertEqual(small["artifact_max_bytes"], 36 + 65536)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            prediction_storage_limit(examples=True, vocab_size=3)

    def test_command_result_types_are_strict(self):
        with self.assertRaisesRegex(TypeError, "exit_code"):
            CommandResult(True, "", "")
        with self.assertRaisesRegex(TypeError, "stdout and stderr"):
            CommandResult(0, b"bytes", "")
        with self.assertRaisesRegex(TypeError, "timed_out"):
            CommandResult(0, "", "", timed_out=1)

    def test_hidden_dataset_request_contains_identity_but_no_location(self):
        request = HiddenDatasetRequest("test", "a" * 64)

        self.assertEqual(request.name, "test")
        self.assertEqual(request.expected_sha256, "a" * 64)
        self.assertEqual(request.identity_algorithm, HIDDEN_BUNDLE_IDENTITY_ALGORITHM)
        self.assertFalse(hasattr(request, "path"))
        self.assertFalse(hasattr(request, "url"))

    def test_hidden_dataset_contract_rejects_invalid_identities(self):
        with self.assertRaisesRegex(ValueError, "name"):
            HiddenDatasetRequest("val")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            HiddenDatasetRequest("test", "not-a-digest")
        with self.assertRaisesRegex(ValueError, "identity algorithm"):
            HiddenDatasetRequest("test", "a" * 64, "sha256")
        with self.assertRaisesRegex(ValueError, "resolution_id"):
            ResolvedHiddenDataset("test", "a" * 64, "")
        with self.assertRaisesRegex(ValueError, "opaque identifier"):
            ResolvedHiddenDataset("test", "a" * 64, "/private/test.json")
        with self.assertRaisesRegex(ValueError, "identity algorithm"):
            ResolvedHiddenDataset("test", "a" * 64, "opaque", "sha256")

    def test_identity_mismatch_carries_only_a_semantic_resolution(self):
        resolved = ResolvedHiddenDataset("analysis", "b" * 64, "opaque-1")

        mismatch = HiddenDatasetIdentityMismatch(resolved)

        self.assertIs(mismatch.receipt, resolved)
        self.assertNotIn("/", str(mismatch))

    def test_workload_resolution_is_semantic_and_path_free(self):
        request = WorkloadRequest("training", "example/training@sha256:" + "a" * 64)
        resolved = ResolvedWorkload(
            role="training",
            requested_identity=request.requested_identity,
            effective_format="sif",
            effective_sha256="b" * 64,
            transformation_sha256="c" * 64,
            resolution_id="opaque-resolution",
        )
        acceptance = WorkloadResolutionAcceptance(
            **resolved.evidence_manifest(),
        )

        self.assertTrue(acceptance.matches(resolved))
        serialized = json.dumps(resolved.evidence_manifest()).lower()
        for forbidden in ("path", "url", "mount", "scheduler", "container_id"):
            self.assertNotIn(forbidden, serialized)

    def test_workload_resolution_rejects_invalid_or_unbound_evidence(self):
        with self.assertRaisesRegex(ValueError, "workload role"):
            WorkloadRequest("agent", "example/agent")
        with self.assertRaisesRegex(ValueError, "effective workload format"):
            ResolvedWorkload(
                "training",
                "example/training",
                "Apptainer SIF",
                "a" * 64,
                None,
                "opaque",
            )
        with self.assertRaisesRegex(ValueError, "opaque identifier"):
            ResolvedWorkload(
                "training",
                "example/training",
                "sif",
                "a" * 64,
                None,
                "/private/cache/image.sif",
            )

    def test_capabilities_are_separate_from_deployment_trust(self):
        assessment = assess_worker(
            descriptor(),
            BASELINE_REQUIREMENTS,
        )
        self.assertTrue(assessment.satisfied)
        self.assertFalse(assessment.trust_satisfied)
        self.assertFalse(assessment.conformance_satisfied)
        self.assertFalse(assessment.leaderboard_eligible)

    def test_certified_conformant_worker_can_be_eligible(self):
        assessment = assess_worker(
            descriptor(trust=TrustLevel.CERTIFIED),
            BASELINE_REQUIREMENTS,
            conformance_receipt=conformance_receipt(),
        )
        self.assertTrue(assessment.leaderboard_eligible)

    def test_conformance_passed_must_be_a_boolean(self):
        for invalid in ("false", "true", 0, 1, None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(TypeError, "must be a boolean"):
                    conformance_receipt(passed=invalid)

    def test_conformance_receipt_identity_fields_are_strict(self):
        for invalid_version in (True, 1.0, 2):
            with self.subTest(invalid_version=invalid_version):
                with self.assertRaisesRegex(ValueError, "receipt version"):
                    ConformanceReceipt(
                        receipt_version=invalid_version,
                        suite_version=WORKER_CONFORMANCE_SUITE_VERSION,
                        worker_id="test-deployment",
                        authority="test-organizer",
                        report_sha256="c" * 64,
                        passed=True,
                    )
        with self.assertRaisesRegex(ValueError, "worker_id"):
            ConformanceReceipt(
                receipt_version=CONFORMANCE_RECEIPT_VERSION,
                suite_version=WORKER_CONFORMANCE_SUITE_VERSION,
                worker_id="",
                authority="test-organizer",
                report_sha256="c" * 64,
                passed=True,
            )
        with self.assertRaisesRegex(ValueError, "report_sha256"):
            ConformanceReceipt(
                receipt_version=CONFORMANCE_RECEIPT_VERSION,
                suite_version=WORKER_CONFORMANCE_SUITE_VERSION,
                worker_id="test-deployment",
                authority="test-organizer",
                report_sha256="not-a-digest",
                passed=True,
            )

    def test_stale_conformance_suite_cannot_certify_a_new_protocol(self):
        assessment = assess_worker(
            descriptor(trust=TrustLevel.CERTIFIED),
            BASELINE_REQUIREMENTS,
            conformance_receipt=conformance_receipt(suite_version="worker-v8"),
        )

        self.assertFalse(assessment.conformance_satisfied)
        self.assertFalse(assessment.leaderboard_eligible)

    def test_conformance_receipt_is_bound_to_the_assessed_worker(self):
        assessment = assess_worker(
            descriptor(trust=TrustLevel.CERTIFIED),
            BASELINE_REQUIREMENTS,
            conformance_receipt=conformance_receipt(worker_id="another-worker"),
        )

        self.assertFalse(assessment.conformance_satisfied)
        self.assertFalse(assessment.leaderboard_eligible)

    def test_worker_descriptor_cannot_self_certify(self):
        with self.assertRaisesRegex(TypeError, "unexpected keyword"):
            WorkerDescriptor(
                worker_id="test-deployment",
                trust_level=TrustLevel.CERTIFIED,
                roles=("research", "evaluation"),
                capabilities={},
                conformance=conformance_receipt(),
            )

    def test_missing_and_insufficient_capabilities_never_silently_pass(self):
        worker = descriptor(
            overrides={
                "training_network_isolation": Capability(EnforcementLevel.BEST_EFFORT)
            }
        )
        requirements = tuple(BASELINE_REQUIREMENTS) + (
            CapabilityRequirement("site_attestation"),
        )
        assessment = assess_worker(worker, requirements)
        self.assertFalse(assessment.satisfied)
        self.assertEqual(assessment.missing, ("site_attestation",))
        self.assertEqual(assessment.insufficient, ("training_network_isolation",))

    def test_descriptor_manifest_contains_no_platform_schema(self):
        manifest = descriptor().public_manifest()
        self.assertEqual(manifest["protocol_version"], 16)
        self.assertEqual(manifest["roles"], ["research", "evaluation"])
        self.assertNotIn("conformance", manifest)
        serialized = json.dumps(manifest).lower()
        self.assertNotIn("docker", serialized)
        self.assertNotIn("slurm", serialized)
        self.assertNotIn("kubernetes", serialized)

    def test_descriptor_rejects_unknown_protocol_version(self):
        with self.assertRaisesRegex(ValueError, "protocol version"):
            WorkerDescriptor(
                worker_id="future-worker",
                protocol_version=14,
                trust_level=TrustLevel.DEVELOPMENT,
                roles=("research",),
                capabilities={},
            )

    def test_a_capability_required_by_both_roles_is_reported_once(self):
        """`IMMUTABLE_WORKLOAD_IDENTITY` belongs to both requirement sets.

        A deployment self-check assesses itself against the combined baseline,
        so an undeduplicated shortfall list reports the same gap twice.
        """

        assessment = assess_worker(
            descriptor(
                overrides={
                    IMMUTABLE_WORKLOAD_IDENTITY: Capability(EnforcementLevel.UNVERIFIED)
                }
            ),
            BASELINE_REQUIREMENTS,
        )

        self.assertEqual(assessment.insufficient, (IMMUTABLE_WORKLOAD_IDENTITY,))
        self.assertEqual(
            assessment.public_manifest()["insufficient_capabilities"],
            [IMMUTABLE_WORKLOAD_IDENTITY],
        )

    def test_a_capability_missing_from_both_roles_is_reported_once(self):
        capabilities = {
            requirement.name: Capability(EnforcementLevel.ENFORCED)
            for requirement in BASELINE_REQUIREMENTS
            if requirement.name != IMMUTABLE_WORKLOAD_IDENTITY
        }
        incomplete = WorkerDescriptor(
            worker_id="test-deployment",
            trust_level=TrustLevel.DEVELOPMENT,
            roles=("research", "evaluation"),
            capabilities=capabilities,
        )

        assessment = assess_worker(incomplete, BASELINE_REQUIREMENTS)

        self.assertEqual(assessment.missing, (IMMUTABLE_WORKLOAD_IDENTITY,))


if __name__ == "__main__":
    unittest.main()
