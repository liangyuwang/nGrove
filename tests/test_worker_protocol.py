import json
import unittest

from autotrain_bench.worker_protocol import (
    BASELINE_REQUIREMENTS,
    Capability,
    CapabilityRequirement,
    ConformanceReceipt,
    EnforcementLevel,
    TrustLevel,
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
                conformance=ConformanceReceipt("worker-v2", True),
            ),
            BASELINE_REQUIREMENTS,
        )
        self.assertTrue(assessment.leaderboard_eligible)

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
        self.assertEqual(manifest["protocol_version"], 2)
        self.assertEqual(manifest["roles"], ["research", "evaluation"])
        serialized = json.dumps(manifest).lower()
        self.assertNotIn("docker", serialized)
        self.assertNotIn("slurm", serialized)
        self.assertNotIn("kubernetes", serialized)

    def test_descriptor_rejects_unknown_protocol_version(self):
        with self.assertRaisesRegex(ValueError, "protocol version"):
            WorkerDescriptor(
                worker_id="future-worker",
                protocol_version=3,
                trust_level=TrustLevel.DEVELOPMENT,
                roles=("research",),
                capabilities={},
            )


if __name__ == "__main__":
    unittest.main()
