from envcore.benchmark import agent_policy_compliant, agent_benchmark_eligible
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from envcore.agent_execution import AgentExecutionEvidence
from envcore.config import AgentConfig
from envcore.local_agent_execution import describe_local_agent_execution
from envcore.worker_protocol import EnforcementLevel, TrustLevel


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

        self.assertFalse(agent_policy_compliant(evidence))
        self.assertFalse(agent_benchmark_eligible(evidence))

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

            self.assertFalse(agent_policy_compliant(evidence))
            self.assertFalse(agent_benchmark_eligible(evidence))
            self.assertEqual(evidence.trust_level, TrustLevel.DEVELOPMENT)
            self.assertEqual(
                evidence.limitations,
                ("agent_writable_storage_unbounded",),
            )
            protected = describe_local_agent_execution(
                agent, operator_storage_bounded=True
            )
            self.assertNotIn("agent_writable_storage_unbounded", protected.limitations)
            self.assertFalse(agent_policy_compliant(protected))
            self.assertFalse(agent_benchmark_eligible(protected))
            certified = replace(
                evidence,
                enforcement=EnforcementLevel.ENFORCED,
                trust_level=TrustLevel.CERTIFIED,
                limitations=(),
            )
            self.assertTrue(agent_benchmark_eligible(certified))

    def test_host_process_reports_the_boundary_limitation(self):
        with tempfile.TemporaryDirectory() as temporary:
            agent = AgentConfig(
                workspace=Path(temporary),
                runner="host_cli",
                command=["agent"],
                initial_workspace="empty",
            ).validate()

            evidence = describe_local_agent_execution(agent)

            self.assertFalse(agent_policy_compliant(evidence))
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
