import copy
import unittest
from dataclasses import replace

from envcore.gateway_execution import (
    GatewayArtifactIdentity,
    GatewayExecutionReceipt,
    GatewayExecutionRequest,
    gateway_eligibility_violations,
    inference_policy_sha256,
)
from envcore.results import ResultSchemaError, normalize_result_report
from envcore.worker_protocol import EnforcementLevel, TrustLevel
from tests.test_results import _finalized


class GatewayExecutionTests(unittest.TestCase):
    def setUp(self):
        self.request = GatewayExecutionRequest(
            episode_id="episode-1",
            requested_identity="example/gateway@sha256:" + "a" * 64,
            core_revision="b" * 40,
            inference_registration_sha256="c" * 64,
            inference_policy_sha256=inference_policy_sha256(
                "openai_compatible", ["model"], ["/v1/responses"]
            ),
        )
        self.receipt = GatewayExecutionReceipt(
            receipt_id="execution:1",
            request_sha256=self.request.sha256,
            source="trusted-launcher",
            artifact=GatewayArtifactIdentity("complete-bundle-v1", "d" * 64),
            enforcement=EnforcementLevel.ENFORCED,
            trust_level=TrustLevel.CERTIFIED,
        )

    def report(self):
        return _finalized(
            agent_system={
                "runner": "container",
                "egress": "denied",
                "inference": {
                    "enabled": True,
                    "gateway_image": self.request.requested_identity,
                    "protocol": "openai_compatible",
                    "allowed_models": ["model"],
                    "allowed_paths": ["/v1/responses"],
                },
            },
            artifact_identities={"core_git": {"revision": self.request.core_revision}},
            gateway_execution={
                "request": self.request.private_manifest(),
                "receipt": self.receipt.private_manifest(),
            },
        )

    def test_request_and_receipt_round_trip_without_order_sensitive_hashing(self):
        self.assertEqual(
            inference_policy_sha256("openai_compatible", ["a", "b"], ["/x", "/y"]),
            inference_policy_sha256("openai_compatible", ["b", "a"], ["/y", "/x"]),
        )
        request = GatewayExecutionRequest.from_manifest(
            dict(reversed(list(self.request.private_manifest().items())))
        )
        self.assertEqual(request.sha256, self.request.sha256)
        receipt = GatewayExecutionReceipt.from_manifest(self.receipt.private_manifest())
        receipt.validate_request(request)
        self.assertEqual(receipt, self.receipt)
        self.assertEqual(gateway_eligibility_violations(receipt, receipt.artifact), [])

    def test_episode_workload_core_and_inference_registration_are_all_bound(self):
        for changes in (
            {"episode_id": "another-episode"},
            {"requested_identity": "example/other@sha256:" + "a" * 64},
            {"core_revision": "e" * 40},
            {"inference_registration_sha256": "f" * 64},
            {"inference_policy_sha256": "e" * 64},
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(
                ValueError, "does not match"
            ):
                self.receipt.validate_request(replace(self.request, **changes))

    def test_trust_alone_does_not_certify_enforcement_or_limitations(self):
        for changes in (
            {"trust_level": TrustLevel.DEVELOPMENT},
            {"trust_level": TrustLevel.CALIBRATION},
            {"enforcement": EnforcementLevel.BEST_EFFORT},
            {"enforcement": EnforcementLevel.UNVERIFIED},
            {"limitations": ("mutable_source_mount",)},
        ):
            with self.subTest(changes=changes):
                receipt = replace(self.receipt, **changes)
                self.assertEqual(
                    gateway_eligibility_violations(receipt, receipt.artifact),
                    ["inference_gateway_execution_not_certified"],
                )
        self.assertEqual(
            gateway_eligibility_violations(
                self.receipt, replace(self.receipt.artifact, format="different-format")
            ),
            ["inference_gateway_artifact_mismatch"],
        )

    def test_malformed_receipts_are_rejected(self):
        for field, value in (
            ("receipt_version", True),
            ("receipt_version", 2),
            ("receipt_id", ""),
            ("request_sha256", "not-a-digest"),
            ("source", " "),
            ("artifact", {"format": "x", "sha256": "y"}),
            ("enforcement", "certified"),
            ("trust_level", "self-certified"),
            ("limitations", "none"),
            ("limitations", ["same", "same"]),
            ("limitations", [None]),
        ):
            value_manifest = {**self.receipt.private_manifest(), field: value}
            with self.subTest(field=field, value=value), self.assertRaises(
                (TypeError, ValueError)
            ):
                GatewayExecutionReceipt.from_manifest(value_manifest)
        value_manifest = {**self.receipt.private_manifest(), "certified": True}
        with self.assertRaises(ValueError):
            GatewayExecutionReceipt.from_manifest(value_manifest)

    def test_report_revalidates_binding_instead_of_trusting_recorded_flags(self):
        valid = self.report()
        original = copy.deepcopy(valid)
        self.assertEqual(
            normalize_result_report(valid)["gateway_execution"],
            original["gateway_execution"],
        )
        self.assertEqual(valid, original)
        for field, value in (
            ("episode_id", "other-episode"),
            ("requested_identity", "example/substitution"),
            ("core_revision", "e" * 40),
            ("inference_registration_sha256", "e" * 64),
        ):
            payload = copy.deepcopy(valid)
            payload["gateway_execution"]["request"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ResultSchemaError, "Gateway"
            ):
                normalize_result_report(payload)
        for mutate in (
            lambda value: value.update(episode_id="other-episode"),
            lambda value: value["agent_system"]["inference"].update(
                gateway_image="different-gateway"
            ),
            lambda value: value["artifact_identities"]["core_git"].update(
                revision="e" * 40
            ),
        ):
            payload = copy.deepcopy(valid)
            mutate(payload)
            with self.assertRaisesRegex(
                ResultSchemaError, "does not match the episode"
            ):
                normalize_result_report(payload)

    def test_enabled_inference_requires_request_but_missing_receipt_is_explicit(self):
        payload = self.report()
        payload["gateway_execution"]["receipt"] = None
        self.assertIsNone(
            normalize_result_report(payload)["gateway_execution"]["receipt"]
        )
        payload["gateway_execution"] = None
        with self.assertRaisesRegex(ResultSchemaError, "requires a Gateway request"):
            normalize_result_report(payload)
        del payload["gateway_execution"]
        with self.assertRaisesRegex(ResultSchemaError, "gateway_execution is required"):
            normalize_result_report(payload)
        payload = self.report()
        payload["agent_system"]["inference"]["enabled"] = False
        with self.assertRaisesRegex(ResultSchemaError, "requires enabled inference"):
            normalize_result_report(payload)

    def test_legacy_versions_remain_readable_without_new_evidence(self):
        for version in (12, 13, 14):
            with self.subTest(version=version):
                payload = _finalized(result_schema_version=version)
                self.assertNotIn("gateway_execution", normalize_result_report(payload))
                payload["gateway_execution"] = None
                with self.assertRaisesRegex(ResultSchemaError, "version 15"):
                    normalize_result_report(payload)
                del payload["gateway_execution"]
                payload["formal_release_verification"] = {
                    "inference_gateway_artifact": self.receipt.artifact.private_manifest()
                }
                with self.assertRaisesRegex(ResultSchemaError, "version 15"):
                    normalize_result_report(payload)


if __name__ == "__main__":
    unittest.main()
