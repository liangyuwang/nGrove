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
    TrustLevel,
    WORKER_CONFORMANCE_SUITE_VERSION,
    WorkerDescriptor,
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
                conformance=ConformanceReceipt("worker-v2", True),
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
        self.assertEqual(manifest["protocol_version"], 4)
        self.assertEqual(manifest["roles"], ["research", "evaluation"])
        serialized = json.dumps(manifest).lower()
        self.assertNotIn("docker", serialized)
        self.assertNotIn("slurm", serialized)
        self.assertNotIn("kubernetes", serialized)

    def test_descriptor_rejects_unknown_protocol_version(self):
        with self.assertRaisesRegex(ValueError, "protocol version"):
            WorkerDescriptor(
                worker_id="future-worker",
                protocol_version=5,
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
