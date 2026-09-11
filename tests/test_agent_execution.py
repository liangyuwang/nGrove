import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from benchcore.agent_execution import AgentExecutionEvidence
from benchcore.config import AgentConfig
from benchcore.local_agent_execution import describe_local_agent_execution
from benchcore.worker_protocol import EnforcementLevel, TrustLevel


class AgentExecutionEvidenceTests(unittest.TestCase):
    def test_unverified_mechanism_is_not_compliant_without_limitations(self):
        evidence = AgentExecutionEvidence(
            source="test-launcher",
            mechanism="container",
            enforcement=EnforcementLevel.UNVERIFIED,
            trust_level=TrustLevel.CERTIFIED,
            workload_identity="example/agent@sha256:" + "a" * 64,
            immutable_workload_identity=True,
        )

        self.assertFalse(evidence.policy_compliant)
        self.assertFalse(evidence.leaderboard_eligible)

    def test_local_container_mechanism_does_not_imply_compliance_or_trust(self):
        with tempfile.TemporaryDirectory() as temporary:
            agent = AgentConfig(
                workspace=Path(temporary),
                runner="container",
                command=["agent"],
                initial_workspace="empty",
                image="example/agent@sha256:" + "a" * 64,
                egress="denied",
            ).validate()

            evidence = describe_local_agent_execution(agent)

            self.assertFalse(evidence.policy_compliant)
            self.assertFalse(evidence.leaderboard_eligible)
            self.assertEqual(evidence.trust_level, TrustLevel.DEVELOPMENT)
            self.assertEqual(
                evidence.limitations,
                ("agent_writable_storage_unbounded",),
            )
            protected = describe_local_agent_execution(
                agent, operator_storage_bounded=True
            )
            self.assertNotIn("agent_writable_storage_unbounded", protected.limitations)
            self.assertFalse(protected.policy_compliant)
            self.assertFalse(protected.leaderboard_eligible)
            certified = replace(
                evidence,
                enforcement=EnforcementLevel.ENFORCED,
                trust_level=TrustLevel.CERTIFIED,
                limitations=(),
            )
            self.assertTrue(certified.leaderboard_eligible)

    def test_host_process_reports_the_boundary_limitation(self):
        with tempfile.TemporaryDirectory() as temporary:
            agent = AgentConfig(
                workspace=Path(temporary),
                runner="host_cli",
                command=["agent"],
                initial_workspace="empty",
            ).validate()

            evidence = describe_local_agent_execution(agent)

            self.assertFalse(evidence.policy_compliant)
            self.assertEqual(
                evidence.limitations,
                (
                    "agent_execution_boundary",
                    "agent_writable_storage_unbounded",
                ),
            )
            self.assertIn(
                "agent_writable_storage_unbounded",
                describe_local_agent_execution(
                    agent, operator_storage_bounded=True
                ).limitations,
            )


if __name__ == "__main__":
    unittest.main()
