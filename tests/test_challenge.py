import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from autotrain_bench.challenge import (
    FORMAL_ARTIFACT_DIGEST_FIELDS,
    FORMAL_ARTIFACT_IDENTITY_ALGORITHMS,
    ChallengeRelease,
    ReleaseVerificationReceipt,
)
from autotrain_bench.config import ConfigError, TaskConfig
from autotrain_bench.worker_protocol import (
    ResourceEnvelope,
    WorkloadResolutionAcceptance,
)


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tests/fixtures/task/task.json"
LM_PRETRAIN_PREVIEW = ROOT / "challenges/lm-pretrain-v1/challenge.json"


class ChallengeReleaseTests(unittest.TestCase):
    def _write_release(self, directory: str) -> Path:
        payload = {
            "schema_version": 2,
            "challenge_id": "test-byte-lm-v1",
            "task_id": "test-byte-lm-v1",
            "task": "task.json",
            "release_stage": "preview",
            "leaderboard_eligible": False,
            "track_configs": {
                "a-1h": {
                    "track": "A",
                    "total_seconds": 3600,
                    "provisional": True,
                },
                "b-3x10m": {
                    "track": "B",
                    "max_leases": 3,
                    "lease_seconds": 600,
                    "provisional": True,
                },
            },
            "runtime_profiles": {
                "h100-80gb-x4": {
                    "host_count": 1,
                    "provisional": True,
                    "accelerator": {
                        "vendor": "nvidia",
                        "model": "NVIDIA H100 80GB",
                        "model_match": "H100",
                        "count": 4,
                        "minimum_memory_mib": 80000,
                    },
                }
            },
        }
        (Path(directory) / "task.json").write_text("{}", encoding="utf-8")
        path = Path(directory) / "challenge.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_preview_manifest_has_a_finite_public_profile_map(self):
        release = ChallengeRelease.load(LM_PRETRAIN_PREVIEW)
        public = release.public_manifest()
        self.assertEqual(public["challenge_id"], "lm-pretrain-v1")
        self.assertEqual(public["task"], "task.json")
        self.assertEqual(public["release_stage"], "preview")
        self.assertFalse(public["leaderboard_eligible"])
        self.assertEqual(
            public["comparison_key_fields"],
            ["challenge_id", "track", "track_config", "runtime_profile"],
        )
        self.assertEqual(len(public["track_configs"]), 2)
        self.assertEqual(len(public["runtime_profiles"]), 4)

    def test_public_catalog_does_not_fingerprint_private_manifest_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_release(directory)
            original = ChallengeRelease.load(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["organizer_only"] = {"context_lengths": [1, 3]}
            path.write_text(json.dumps(payload), encoding="utf-8")
            (Path(directory) / "task.json").write_text(
                '{"evaluation": {"context_lengths": [1, 3]}}', encoding="utf-8"
            )
            changed = ChallengeRelease.load(path)
            self.assertNotEqual(original.manifest_sha256, changed.manifest_sha256)
            self.assertNotEqual(
                original.task_manifest_sha256, changed.task_manifest_sha256
            )
            self.assertEqual(original.public_manifest(), changed.public_manifest())
            self.assertNotIn("task_manifest_sha256", changed.public_manifest())
            self.assertNotIn("challenge_manifest_sha256", changed.public_manifest())

    def test_release_resolves_budget_resources_and_comparison_key(self):
        with tempfile.TemporaryDirectory() as directory:
            release = ChallengeRelease.load(self._write_release(directory))
            task = replace(
                TaskConfig.load(TASK), task_kind="benchmark_challenge"
            ).validate()
            resolved = release.resolve(
                task,
                track_config_id="b-3x10m",
                runtime_profile_id="h100-80gb-x4",
            )
            self.assertEqual(resolved.track, "B")
            self.assertEqual(resolved.task.track_b_max_leases, 3)
            self.assertEqual(resolved.task.track_b_lease_seconds, 600)
            self.assertEqual(resolved.resources.nvidia_gpu_count, 4)
            self.assertEqual(resolved.resources.required_gpu_model, "H100")
            self.assertEqual(resolved.resources.minimum_gpu_memory_mib, 80000)
            resource_manifest = resolved.resource_envelope.public_manifest()
            self.assertEqual(
                [item["name"] for item in resource_manifest["normative"]],
                [
                    "accelerator.count",
                    "accelerator.memory_mib_per_device",
                    "accelerator.model",
                    "accelerator.vendor",
                    "host.count",
                ],
            )
            self.assertTrue(
                all(
                    item["minimum_enforcement"] == "best_effort"
                    for item in resource_manifest["normative"]
                )
            )
            self.assertEqual(
                resolved.release_manifest["resource_envelope"],
                resource_manifest,
            )
            self.assertEqual(
                resolved.comparison_key.public_manifest(),
                {
                    "challenge_id": "test-byte-lm-v1",
                    "track": "B",
                    "track_config": "b-3x10m",
                    "runtime_profile": "h100-80gb-x4",
                },
            )

    def test_nonleaderboard_release_accepts_development_task(self):
        with tempfile.TemporaryDirectory() as directory:
            release = ChallengeRelease.load(self._write_release(directory))
            task = TaskConfig.load(TASK)
            resolved = release.resolve(
                task,
                track_config_id="a-1h",
                runtime_profile_id="h100-80gb-x4",
            )
            self.assertEqual(resolved.task.task_kind, "development_smoke")
            with self.assertRaisesRegex(ConfigError, "unknown track config"):
                release.resolve(
                    task,
                    track_config_id="missing",
                    runtime_profile_id="h100-80gb-x4",
                )

    def test_leaderboard_release_rejects_development_task(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_release(directory)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["leaderboard_eligible"] = True
            payload["release_stage"] = "official"
            for value in payload["track_configs"].values():
                value["provisional"] = False
            for value in payload["runtime_profiles"].values():
                value["provisional"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            release = ChallengeRelease.load(path)
            with self.assertRaisesRegex(ConfigError, "development smoke"):
                release.resolve(
                    TaskConfig.load(TASK),
                    track_config_id="a-1h",
                    runtime_profile_id="h100-80gb-x4",
                )

    def test_runtime_profile_can_explicitly_require_no_accelerator(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_release(directory)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["runtime_profiles"]["cpu-local"] = {
                "host_count": 1,
                "provisional": True,
                "accelerator": None,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            release = ChallengeRelease.load(path)
            profile = release.public_manifest()["runtime_profiles"]["cpu-local"]
            self.assertIsNone(profile["accelerator"])
            resolved = release.resolve(
                TaskConfig.load(TASK),
                track_config_id="a-1h",
                runtime_profile_id="cpu-local",
            )
            self.assertEqual(resolved.resources.nvidia_gpu_count, 0)
            self.assertIsNone(resolved.resources.required_gpu_model)

    def test_runtime_profile_accepts_release_defined_resource_properties(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_release(directory)
            payload = json.loads(path.read_text(encoding="utf-8"))
            profile = payload["runtime_profiles"]["h100-80gb-x4"]
            profile["resource_contract"] = {
                "schema_version": 1,
                "normative": [
                    {
                        "name": "host.count",
                        "relation": "exact",
                        "value": 1,
                        "minimum_enforcement": "enforced",
                    },
                    {
                        "name": "accelerator.count",
                        "relation": "exact",
                        "value": 4,
                        "minimum_enforcement": "enforced",
                    },
                    {
                        "name": "accelerator.vendor",
                        "relation": "exact",
                        "value": "nvidia",
                        "minimum_enforcement": "enforced",
                    },
                    {
                        "name": "accelerator.model",
                        "relation": "contains",
                        "value": "H100",
                        "minimum_enforcement": "enforced",
                    },
                    {
                        "name": "accelerator.memory_mib_per_device",
                        "relation": "at_least",
                        "value": 80000,
                        "minimum_enforcement": "enforced",
                    },
                    {
                        "name": "site.scratch.capacity_bytes",
                        "relation": "at_least",
                        "value": 1000,
                        "minimum_enforcement": "enforced",
                    },
                ],
                "observational": ["site.fabric.class"],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

            release = ChallengeRelease.load(path)
            resolved = release.resolve(
                replace(
                    TaskConfig.load(TASK), task_kind="benchmark_challenge"
                ).validate(),
                track_config_id="a-1h",
                runtime_profile_id="h100-80gb-x4",
            )

            self.assertEqual(
                resolved.resource_envelope.public_manifest(),
                ResourceEnvelope.from_manifest(
                    profile["resource_contract"]
                ).public_manifest(),
            )

    def test_resource_contract_cannot_erase_profile_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_release(directory)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["runtime_profiles"]["h100-80gb-x4"]["resource_contract"] = {
                "schema_version": 1,
                "normative": [],
                "observational": [],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ConfigError, "must include 'host.count'"):
                ChallengeRelease.load(path)

    def test_resource_contract_cannot_conflict_with_profile_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_release(directory)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["runtime_profiles"]["h100-80gb-x4"]["resource_contract"] = {
                "schema_version": 1,
                "normative": [
                    {
                        "name": "host.count",
                        "relation": "exact",
                        "value": 2,
                        "minimum_enforcement": "best_effort",
                    },
                    {
                        "name": "accelerator.count",
                        "relation": "exact",
                        "value": 4,
                        "minimum_enforcement": "best_effort",
                    },
                    {
                        "name": "accelerator.vendor",
                        "relation": "exact",
                        "value": "nvidia",
                        "minimum_enforcement": "best_effort",
                    },
                    {
                        "name": "accelerator.model",
                        "relation": "contains",
                        "value": "H100",
                        "minimum_enforcement": "best_effort",
                    },
                    {
                        "name": "accelerator.memory_mib_per_device",
                        "relation": "at_least",
                        "value": 80000,
                        "minimum_enforcement": "best_effort",
                    },
                ],
                "observational": [],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ConfigError, "conflicts.*'host.count'"):
                ChallengeRelease.load(path)

    def test_release_rejects_unsupported_runtime_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_release(directory)
            payload = json.loads(path.read_text(encoding="utf-8"))
            profile = payload["runtime_profiles"]["h100-80gb-x4"]
            profile["host_count"] = 2
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "host_count must be 1"):
                ChallengeRelease.load(path)

    def test_leaderboard_release_must_be_official_and_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_release(directory)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["leaderboard_eligible"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "release_stage='official'"):
                ChallengeRelease.load(path)

            payload["release_stage"] = "official"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "provisional"):
                ChallengeRelease.load(path)

    def test_leaderboard_release_requires_enforced_normative_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_release(directory)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["leaderboard_eligible"] = True
            payload["release_stage"] = "official"
            for value in payload["track_configs"].values():
                value["provisional"] = False
            for value in payload["runtime_profiles"].values():
                value["provisional"] = False
                value["resource_contract"] = {
                    "schema_version": 1,
                    "normative": [
                        {
                            "name": "host.count",
                            "relation": "exact",
                            "value": 1,
                            "minimum_enforcement": "best_effort",
                        },
                        {
                            "name": "accelerator.count",
                            "relation": "exact",
                            "value": 4,
                            "minimum_enforcement": "enforced",
                        },
                        {
                            "name": "accelerator.vendor",
                            "relation": "exact",
                            "value": "nvidia",
                            "minimum_enforcement": "enforced",
                        },
                        {
                            "name": "accelerator.model",
                            "relation": "contains",
                            "value": "H100",
                            "minimum_enforcement": "enforced",
                        },
                        {
                            "name": "accelerator.memory_mib_per_device",
                            "relation": "at_least",
                            "value": 80000,
                            "minimum_enforcement": "enforced",
                        },
                    ],
                    "observational": [],
                }
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ConfigError, "enforced normative"):
                ChallengeRelease.load(path)

    def test_formal_release_receipt_requires_every_trusted_artifact(self):
        digests = {field: "a" * 64 for field in FORMAL_ARTIFACT_DIGEST_FIELDS}
        receipt = ReleaseVerificationReceipt(
            authority="test-organizer",
            release_id="test-byte-lm-v1",
            core_revision="b" * 40,
            artifact_sha256=digests,
            artifact_identity_algorithms=FORMAL_ARTIFACT_IDENTITY_ALGORITHMS,
        )
        self.assertIs(receipt.validate(), receipt)
        self.assertEqual(
            receipt.private_manifest()["artifact_identity_algorithms"],
            FORMAL_ARTIFACT_IDENTITY_ALGORITHMS,
        )

        missing = dict(digests)
        missing.pop("test_sha256")
        with self.assertRaisesRegex(ConfigError, "test_sha256"):
            replace(receipt, artifact_sha256=missing).validate()
        with self.assertRaisesRegex(ConfigError, "identity algorithms"):
            replace(
                receipt,
                artifact_identity_algorithms={
                    "file": "sha256",
                    "hidden_evaluation": "sha256",
                },
            ).validate()

    def test_formal_release_receipt_binds_accepted_workload_transformation(self):
        acceptance = WorkloadResolutionAcceptance(
            role="training",
            requested_identity="example/training@sha256:" + "a" * 64,
            effective_format="sif",
            effective_sha256="b" * 64,
            transformation_sha256="c" * 64,
        )
        receipt = ReleaseVerificationReceipt(
            authority="test-organizer",
            release_id="test-byte-lm-v1",
            core_revision="d" * 40,
            artifact_sha256={
                field: "a" * 64 for field in FORMAL_ARTIFACT_DIGEST_FIELDS
            },
            artifact_identity_algorithms=FORMAL_ARTIFACT_IDENTITY_ALGORITHMS,
            workload_resolution_acceptance={"training": acceptance},
        )

        manifest = receipt.validate().private_manifest()

        self.assertEqual(
            manifest["workload_resolution_acceptance"]["training"],
            acceptance.public_manifest(),
        )
        with self.assertRaisesRegex(ConfigError, "role does not match"):
            replace(
                receipt,
                workload_resolution_acceptance={"candidate": acceptance},
            ).validate()


if __name__ == "__main__":
    unittest.main()
