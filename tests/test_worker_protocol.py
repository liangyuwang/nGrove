import json
import unittest

from autotrain_bench.worker_protocol import (
    BASELINE_REQUIREMENTS,
    IMMUTABLE_WORKLOAD_IDENTITY,
    Capability,
    CapabilityRequirement,
    CandidateChannelRequirements,
    CommandResult,
    ConformanceReceipt,
    EnforcementLevel,
    HiddenDatasetIdentityMismatch,
    HiddenDatasetRequest,
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


def descriptor(*, trust=TrustLevel.DEVELOPMENT, conformance=None, overrides=None):
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
        conformance=conformance,
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

    def test_candidate_channel_describes_prefix_visibility_precisely(self):
        manifest = CandidateChannelRequirements(
            protocol="next_token_logits_v1"
        ).public_manifest()

        self.assertEqual(manifest["request_content"], "token_prefixes")
        self.assertEqual(manifest["hidden_artifact_visibility"], "denied")
        self.assertEqual(manifest["current_target_visibility"], "denied")
        self.assertEqual(manifest["future_token_visibility"], "denied")
        self.assertEqual(manifest["candidate_egress"], "denied")

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
        self.assertFalse(hasattr(request, "path"))
        self.assertFalse(hasattr(request, "url"))

    def test_hidden_dataset_contract_rejects_invalid_identities(self):
        with self.assertRaisesRegex(ValueError, "name"):
            HiddenDatasetRequest("val")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            HiddenDatasetRequest("test", "not-a-digest")
        with self.assertRaisesRegex(ValueError, "resolution_id"):
            ResolvedHiddenDataset("test", "a" * 64, "")
        with self.assertRaisesRegex(ValueError, "opaque identifier"):
            ResolvedHiddenDataset("test", "a" * 64, "/private/test.json")

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
            descriptor(
                trust=TrustLevel.CERTIFIED,
                conformance=ConformanceReceipt(WORKER_CONFORMANCE_SUITE_VERSION, True),
            ),
            BASELINE_REQUIREMENTS,
        )
        self.assertTrue(assessment.leaderboard_eligible)

    def test_conformance_passed_must_be_a_boolean(self):
        for invalid in ("false", "true", 0, 1, None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(TypeError, "must be a boolean"):
                    ConformanceReceipt(WORKER_CONFORMANCE_SUITE_VERSION, invalid)

    def test_stale_conformance_suite_cannot_certify_a_new_protocol(self):
        assessment = assess_worker(
            descriptor(
                trust=TrustLevel.CERTIFIED,
                conformance=ConformanceReceipt("worker-v7", True),
            ),
            BASELINE_REQUIREMENTS,
        )

        self.assertFalse(assessment.conformance_satisfied)
        self.assertFalse(assessment.leaderboard_eligible)

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
        self.assertEqual(manifest["protocol_version"], 8)
        self.assertEqual(manifest["roles"], ["research", "evaluation"])
        serialized = json.dumps(manifest).lower()
        self.assertNotIn("docker", serialized)
        self.assertNotIn("slurm", serialized)
        self.assertNotIn("kubernetes", serialized)

    def test_descriptor_rejects_unknown_protocol_version(self):
        with self.assertRaisesRegex(ValueError, "protocol version"):
            WorkerDescriptor(
                worker_id="future-worker",
                protocol_version=9,
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
