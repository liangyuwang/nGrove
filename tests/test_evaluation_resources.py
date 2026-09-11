"""Evaluation allocation evidence and cost boundaries, without Docker or GPUs."""

import copy
import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from benchcore.challenge import ChallengeRelease
from benchcore.config import ConfigError, TaskConfig, TrainingResources
from benchcore.results import ResultSchemaError, normalize_result_report
from benchcore.worker_protocol import (
    EnforcementLevel,
    EvaluationResourceFailure,
    EvaluationResourceReceipt,
    EvaluationTaskSpec,
    ResourceAllocationReceipt,
    ResourceEnvelope,
    ResourceObservation,
    ResourcePropertyEvidence,
    ResourceRequirement,
    WorkerFailure,
    WorkerTeardownFailure,
    assess_evaluation_resources,
)
from tests import test_challenge, test_results, test_supervisor
from tests.test_runtime import CapturingDockerRuntime, GPU_ONE, TASK, gpu_inventory


def envelope(level=EnforcementLevel.BEST_EFFORT):
    return ResourceEnvelope(
        requirements=(
            ResourceRequirement("cpu.quota_cores", "exact", 1.5, level),
            ResourceRequirement("memory.limit_bytes", "exact", 64 * 1024**2, level),
        ),
        observational_properties=("cache.policy",),
    )


class EvaluationResourceProtocolTests(unittest.TestCase):
    def receipt(self):
        request = envelope()
        return EvaluationResourceReceipt(
            episode_id="episode",
            evaluation_id="evaluation:test",
            worker_id="worker",
            resource_request_sha256=request.sha256,
            receipt_id="receipt:test",
            properties=tuple(
                ResourcePropertyEvidence(
                    name=item.name,
                    allocator_allocated=ResourceObservation("unavailable"),
                    workload_visible=ResourceObservation(
                        "observed", item.value, "worker-cgroup"
                    ),
                    enforcement_level=EnforcementLevel.BEST_EFFORT,
                    enforcement_authority="worker",
                )
                for item in request.requirements
            ),
        )

    def test_receipt_cannot_be_replayed_for_another_operation_or_training(self):
        receipt = self.receipt()
        self.assertEqual(
            EvaluationResourceReceipt.from_manifest(receipt.public_manifest()), receipt
        )
        with self.assertRaises(ValueError):
            ResourceAllocationReceipt.from_manifest(receipt.public_manifest())
        for changes in (
            {"episode_id": "another"},
            {"evaluation_id": "evaluation:analysis"},
            {"worker_id": "another"},
            {"resource_request_sha256": "0" * 64},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                assess_evaluation_resources(
                    envelope(),
                    replace(receipt, **changes),
                    episode_id="episode",
                    evaluation_id="evaluation:test",
                    worker_id="worker",
                )

    def test_effective_limit_overrides_larger_host_allocation(self):
        receipt = self.receipt()
        cpu, memory = receipt.properties
        cpu = replace(
            cpu, allocator_allocated=ResourceObservation("observed", 64, "host")
        )
        receipt = replace(receipt, properties=(cpu, memory))

        def assess(value):
            return assess_evaluation_resources(
                envelope(),
                value,
                episode_id="episode",
                evaluation_id="evaluation:test",
                worker_id="worker",
            )

        self.assertTrue(assess(receipt).satisfied)
        bad = replace(
            cpu, workload_visible=ResourceObservation("observed", 0.5, "worker-cgroup")
        )
        self.assertEqual(
            assess(replace(receipt, properties=(bad, memory))).mismatched,
            ("cpu.quota_cores",),
        )


class EvaluationResourceReleaseTests(unittest.TestCase):
    def test_separate_evaluation_contract_roundtrips_and_requires_new_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = test_challenge.ChallengeReleaseTests()._write_release(directory)
            payload = json.loads(path.read_text())
            profile = payload["runtime_profiles"]["h100-80gb-x4"]
            profile["evaluation_resource_contract"] = envelope().public_manifest()
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ConfigError, "schema_version 3"):
                ChallengeRelease.load(path)
            payload["schema_version"] = 3
            path.write_text(json.dumps(payload))
            release = ChallengeRelease.load(path)
            resolved = release.resolve(
                TaskConfig.load(TASK),
                track_config_id="a-1h",
                runtime_profile_id="h100-80gb-x4",
            )
            self.assertEqual(
                resolved.release_manifest["evaluation_resource_envelope"],
                envelope().public_manifest(),
            )
            self.assertNotIn(
                "cpu.quota_cores",
                [item.name for item in resolved.resource_envelope.requirements],
            )
            self.assertEqual(
                release.public_manifest()["runtime_profiles"]["h100-80gb-x4"][
                    "evaluation_resource_contract"
                ],
                envelope().public_manifest(),
            )

    def test_formal_release_requires_its_own_enforced_evaluation_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = test_challenge.ChallengeReleaseTests()._write_release(directory)
            payload = json.loads(path.read_text())
            payload.update(
                schema_version=3, leaderboard_eligible=True, release_stage="official"
            )
            for group in ("track_configs", "runtime_profiles"):
                for item in payload[group].values():
                    item["provisional"] = False
            for value in (None, ResourceEnvelope(()), envelope()):
                for profile in payload["runtime_profiles"].values():
                    profile.pop("evaluation_resource_contract", None)
                    if value is not None:
                        profile["evaluation_resource_contract"] = (
                            value.public_manifest()
                        )
                path.write_text(json.dumps(payload))
                with self.subTest(contract=value), self.assertRaisesRegex(
                    ConfigError, "evaluation_resource_contract"
                ):
                    ChallengeRelease.load(path)
            for profile in payload["runtime_profiles"].values():
                profile["evaluation_resource_contract"] = envelope(
                    EnforcementLevel.ENFORCED
                ).public_manifest()
            path.write_text(json.dumps(payload))
            ChallengeRelease.load(path)


class EvaluationResourceSupervisorTests(unittest.TestCase):
    setUp = test_supervisor.SupervisorTests.setUp
    tearDown = test_supervisor.SupervisorTests.tearDown
    supervisor = test_supervisor.SupervisorTests.supervisor
    formal_supervisor = test_supervisor.SupervisorTests.formal_supervisor
    finalize_after_agent_stop = (
        test_supervisor.SupervisorTests.finalize_after_agent_stop
    )

    def finish(self, transform=None, *, contract=True, formal=False):
        supervisor = (
            self.formal_supervisor()[0]
            if formal
            else self.supervisor(
                challenge_release={
                    "evaluation_resource_envelope": envelope().public_manifest()
                    if contract
                    else None,
                }
            )
        )
        supervisor.start_lease()
        candidate = supervisor.commit_lease("/output/candidate")
        supervisor.declare_incumbent(candidate["candidate_id"])
        supervisor.finish()
        original = self.worker.evaluate_candidate

        def evaluate(**kwargs):
            result = original(**kwargs)
            return transform(result, kwargs) if transform else result

        with patch.object(self.worker, "evaluate_candidate", side_effect=evaluate):
            self.finalize_after_agent_stop(supervisor)
        return supervisor.state

    def test_reordered_release_contract_has_the_same_meaning(self):
        request = replace(
            envelope(), observational_properties=("cache.policy", "host.exclusivity")
        )
        reordered = replace(
            request,
            requirements=tuple(reversed(request.requirements)),
            observational_properties=tuple(reversed(request.observational_properties)),
        )
        self.assertEqual(request.sha256, reordered.sha256)
        supervisor = self.supervisor(
            challenge_release={
                "evaluation_resource_envelope": request.public_manifest()
            },
            evaluation_resource_envelope=reordered,
        )
        self.assertEqual(
            supervisor.state["evaluation_resource_envelope"], request.public_manifest()
        )

    def test_changed_release_requirement_still_conflicts(self):
        request = envelope()
        for index, changed in enumerate(
            (
                replace(request.requirements[0], value=2),
                replace(request.requirements[0], relation="at_most"),
                replace(
                    request.requirements[0],
                    minimum_enforcement=EnforcementLevel.ENFORCED,
                ),
            )
        ):
            self.episode = self.episode.parent / f"episode-{index}"
            explicit = replace(
                request, requirements=(changed, *request.requirements[1:])
            )
            with self.subTest(changed=changed), self.assertRaisesRegex(
                ValueError, "differs from the release"
            ):
                self.supervisor(
                    challenge_release={
                        "evaluation_resource_envelope": request.public_manifest()
                    },
                    evaluation_resource_envelope=explicit,
                )

    def test_reordered_formal_release_preserves_eligibility_and_report_validation(self):
        constructor = test_supervisor.Supervisor

        def reordered_release(**kwargs):
            release = copy.deepcopy(kwargs["challenge_release"])
            raw = copy.deepcopy(release["evaluation_resource_envelope"])
            release["evaluation_resource_envelope"] = raw
            kwargs["evaluation_resource_envelope"] = ResourceEnvelope.from_manifest(raw)
            raw["normative"].reverse()
            kwargs["challenge_release"] = release
            return constructor(**kwargs)

        with patch.object(test_supervisor, "Supervisor", side_effect=reordered_release):
            state = self.finish(formal=True)
        self.assertTrue(state["leaderboard_eligible"])
        self.assertEqual(state["leaderboard_eligibility_violations"], [])
        normalize_result_report(state)
        changed = copy.deepcopy(state)
        changed["challenge_release"]["evaluation_resource_envelope"]["normative"][0][
            "value"
        ] = 99
        with self.assertRaisesRegex(ResultSchemaError, "must match the frozen release"):
            normalize_result_report(changed)

    def assert_only_analysis_resources_failed(self, state):
        self.assertEqual(state["status"], "finalized")
        self.assertEqual(state["outcome"], "success")
        self.assertIsNotNone(state["test_cost"])
        self.assertFalse(state["analysis_complete"])
        self.assertEqual(
            state["analysis_failure"], "analysis_evaluation_resource_contract_failure"
        )
        self.assertEqual(
            state["analysis_trajectory"][0]["analysis_error"],
            "evaluation_resource_contract_failure",
        )
        self.assertEqual(state["deployment_teardown_failures"], [])
        self.assertTrue(state["policy_compliant"])
        self.assertTrue(state["leaderboard_eligible"])
        self.assertEqual(state["leaderboard_eligibility_violations"], [])
        normalize_result_report(state)

    def test_missing_analysis_receipt_preserves_formal_eligibility(self):
        def transform(result, kwargs):
            return (
                replace(result, resource_receipt=None)
                if kwargs["hidden_dataset"].name == "analysis"
                else result
            )

        self.assert_only_analysis_resources_failed(self.finish(transform, formal=True))

    def test_real_analysis_teardown_failure_still_revokes_eligibility(self):
        def transform(result, kwargs):
            if kwargs["hidden_dataset"].name == "analysis":
                raise WorkerTeardownFailure("Candidate teardown is unconfirmed")
            return result

        state = self.finish(transform, formal=True)
        self.assertEqual(state["outcome"], "success")
        self.assertIsNotNone(state["test_cost"])
        self.assertFalse(state["leaderboard_eligible"])
        self.assertFalse(state["policy_compliant"])
        self.assertTrue(state["deployment_teardown_failures"])
        self.assertIn(
            "deployment_teardown_incomplete",
            state["leaderboard_eligibility_violations"],
        )
        normalize_result_report(state)

    def fail_gpu_query(self, split):
        runtime = CapturingDockerRuntime()
        resources = TrainingResources(nvidia_gpu_count=1, required_gpu_model="H100")
        runtime._nvidia_selection = (GPU_ONE,)
        runtime._nvidia_request = runtime._nvidia_request_key(resources)

        def transform(result, kwargs):
            if kwargs["hidden_dataset"].name == split:
                runtime._evaluation_resource_plan(
                    episode_id=kwargs["episode_id"],
                    evaluation_id=kwargs["evaluation_id"],
                    envelope=kwargs["resource_envelope"],
                    resources=resources,
                    device="cuda",
                    timeout_seconds=10,
                )
            return result

        with patch.object(
            runtime,
            "_host_nvidia_gpus",
            side_effect=WorkerFailure("NVIDIA GPU inventory query failed"),
        ):
            return self.finish(transform, formal=True)

    def test_test_gpu_query_failure_is_infrastructure_failure(self):
        state = self.fail_gpu_query("test")
        self.assertEqual(state["status"], "infrastructure_failure")
        self.assertEqual(state["failure"], "evaluation_resource_contract_failure")
        self.assertIsNone(state["test_cost"])
        self.assertIsNone(state["test_performance_score"])
        self.assertEqual(self.worker.evaluations, ["test"])
        self.assertFalse(state["leaderboard_eligible"])
        normalize_result_report(state)

    def test_analysis_gpu_query_failure_preserves_formal_eligibility(self):
        self.assert_only_analysis_resources_failed(self.fail_gpu_query("analysis"))

    def test_contract_passes_to_each_invocation_and_report_revalidates_it(self):
        state = self.finish()
        self.assertEqual(state["outcome"], "success")
        records = state["evaluation_resource_allocations"]
        self.assertEqual([item["split"] for item in records], ["test", "analysis"])
        self.assertNotEqual(records[0]["evaluation_id"], records[1]["evaluation_id"])
        self.assertTrue(all(item["accepted"] for item in records))
        normalize_result_report(state)
        for field, changed in (("receipt", records[0]["receipt"]), ("accepted", False)):
            tampered = copy.deepcopy(state)
            tampered["evaluation_resource_allocations"][1][field] = changed
            with self.subTest(field=field), self.assertRaises(ResultSchemaError):
                normalize_result_report(tampered)

    def test_missing_receipt_is_infrastructure_failure_without_a_score(self):
        state = self.finish(lambda result, _: replace(result, resource_receipt=None))
        self.assertEqual(state["status"], "infrastructure_failure")
        self.assertIsNone(state["test_cost"])
        self.assertEqual(self.worker.evaluations, ["test"])
        normalize_result_report(state)

    def test_mismatched_effective_allocation_is_rejected(self):
        def transform(result, _):
            receipt = result.resource_receipt
            first, *rest = receipt.properties
            first = replace(
                first, workload_visible=ResourceObservation("observed", 0.25, "worker")
            )
            return replace(
                result, resource_receipt=replace(receipt, properties=(first, *rest))
            )

        state = self.finish(transform)
        self.assertEqual(state["status"], "infrastructure_failure")
        self.assertFalse(
            state["evaluation_resource_allocations"][0]["assessment"]["satisfied"]
        )
        normalize_result_report(state)

    def test_analysis_cannot_reuse_test_receipt_or_invalidate_test_score(self):
        receipts = []

        def transform(result, kwargs):
            if kwargs["hidden_dataset"].name == "test":
                receipts.append(result.resource_receipt)
                return result
            return replace(result, resource_receipt=receipts[0])

        state = self.finish(transform)
        self.assertEqual(state["outcome"], "success")
        self.assertIsNotNone(state["test_cost"])
        self.assertFalse(state["analysis_complete"])
        self.assertIsNone(state["analysis_trajectory"][0]["analysis_cost"])
        normalize_result_report(state)

    def test_receipt_ids_must_not_be_reused_even_with_a_new_evaluation_id(self):
        state = self.finish(
            lambda result, _: replace(
                result,
                resource_receipt=replace(result.resource_receipt, receipt_id="reused"),
            )
        )
        self.assertEqual(state["outcome"], "success")
        self.assertFalse(state["analysis_complete"])

    def test_unspecified_preview_is_recorded_without_an_acceptance_claim(self):
        state = self.finish(contract=False)
        self.assertEqual(state["outcome"], "success")
        self.assertIsNone(state["evaluation_resource_envelope"])
        self.assertTrue(
            all(
                item["receipt"] is None and not item["accepted"]
                for item in state["evaluation_resource_allocations"]
            )
        )
        normalize_result_report(state)


class EvaluationResourceDockerTests(unittest.TestCase):
    def test_gpu_probe_errors_are_resource_failures(self):
        resources = TrainingResources(nvidia_gpu_count=1, required_gpu_model="H100")
        runtime = CapturingDockerRuntime()
        runtime._nvidia_selection = (GPU_ONE,)
        runtime._nvidia_request = runtime._nvidia_request_key(resources)
        for response in (
            subprocess.TimeoutExpired("nvidia-smi", 10),
            subprocess.CompletedProcess(["nvidia-smi"], 1, "", "driver unavailable"),
            gpu_inventory(f"{GPU_ONE}, NVIDIA A100, 81920, Disabled"),
        ):
            options = (
                {"side_effect": response}
                if isinstance(response, Exception)
                else {"return_value": response}
            )
            with self.subTest(response=response), patch(
                "benchcore.runtime.shutil.which", return_value="nvidia-smi"
            ), patch("benchcore.runtime.subprocess.run", **options):
                with self.assertRaises(EvaluationResourceFailure):
                    runtime._evaluation_resource_plan(
                        episode_id="episode",
                        evaluation_id="evaluation:test",
                        envelope=envelope(),
                        resources=resources,
                        device="cuda",
                        timeout_seconds=10,
                    )

    def runtime(self, *, limits=None, supported=True):
        class Runtime(CapturingDockerRuntime):
            now = 10.0

            def __init__(self):
                super().__init__(monotonic=lambda: self.now)

            def _run(self, args, **kwargs):
                if args[0] == "info":
                    return subprocess.CompletedProcess(
                        args, 0, json.dumps([supported] * 3), ""
                    )
                if args[0] == "inspect" and "{{json .HostConfig}}" in args:
                    self.calls.append(list(args))
                    self.now += 7  # Trusted verification must not inflate cost.
                    return subprocess.CompletedProcess(
                        args,
                        0,
                        json.dumps(
                            limits
                            or {
                                "NanoCpus": 1500000000,
                                "Memory": 64 * 1024**2,
                                "MemorySwap": 64 * 1024**2,
                                "DeviceRequests": [],
                            }
                        ),
                        "",
                    )
                if "benchcore.candidate_offline_runner" in args:
                    self.now += 3
                if "--mode" in args and args[args.index("--mode") + 1] == "score":
                    mount = next(item for item in args if item.endswith("dst=/output"))
                    output = Path(mount.split("src=", 1)[1].split(",dst=", 1)[0])
                    (output / "test.json").write_text(
                        json.dumps(
                            {
                                "performance_score": 72.5,
                                "raw_metrics": {"mean_nll": 1.25},
                                "examples": 3,
                                "input_tokens": 6,
                                "prediction_bytes": 64,
                            }
                        )
                    )
                return super()._run(args, **kwargs)

        return Runtime()

    def evaluate(self, runtime, request, *, resources=None, device=None):
        task = TaskConfig.load(TASK)
        evaluation_task = EvaluationTaskSpec.from_task(task)
        if device is not None:
            evaluation_task = replace(evaluation_task, device=device)
        with tempfile.TemporaryDirectory() as directory:
            return runtime._run_isolated_evaluator(
                episode_id="episode",
                evaluation_id="evaluation:test",
                resource_envelope=request,
                task=evaluation_task,
                resources=resources or TrainingResources(),
                candidate_dir=TASK.parent / "agent",
                split="test",
                data_path=task.test_path,
                output_dir=Path(directory) / "evaluation",
                candidate_image=task.candidate_image,
                evaluator_image=task.evaluator_image,
                timeout_seconds=60,
            )

    def test_candidate_limits_and_cost_exclude_trusted_inspection(self):
        runtime = self.runtime()
        result = self.evaluate(runtime, envelope())
        self.assertEqual(result.candidate_wall_seconds, 3)
        command = next(
            call
            for call in runtime.calls
            if "benchcore.candidate_offline_runner" in call
        )
        self.assertEqual(command[command.index("--cpus") + 1], "1.5")
        self.assertEqual(command[command.index("--memory") + 1], str(64 * 1024**2))
        properties = {item.name: item for item in result.resource_receipt.properties}
        self.assertEqual(
            properties["cpu.quota_cores"].workload_visible.status, "unavailable"
        )
        self.assertEqual(
            properties["cache.policy"].enforcement_level, EnforcementLevel.UNSUPPORTED
        )
        self.assertEqual(runtime._pending_evaluations, {})

    def test_unsupported_normative_conditions_fail_before_candidate_launch(self):
        for request in (
            envelope(EnforcementLevel.ENFORCED),
            ResourceEnvelope((ResourceRequirement("cache.policy", "exact", "cold"),)),
        ):
            runtime = self.runtime()
            with self.subTest(request=request), self.assertRaises(
                EvaluationResourceFailure
            ):
                self.evaluate(runtime, request)
            self.assertFalse(
                any(
                    "benchcore.candidate_offline_runner" in call
                    for call in runtime.calls
                )
            )

    def test_missing_backend_support_rejects_the_requested_limits(self):
        with self.assertRaises(EvaluationResourceFailure):
            self.evaluate(self.runtime(supported=False), envelope())

    def test_initial_gpu_preflight_errors_also_belong_to_infrastructure(self):
        for contract in (None, envelope()):
            for step in ("validate_hardware", "_nvidia_grant"):
                runtime = self.runtime()
                with self.subTest(contract=contract, step=step), patch.object(
                    runtime, "validate_hardware"
                ), patch.object(
                    runtime,
                    step,
                    side_effect=WorkerFailure("GPU unavailable before launch"),
                ):
                    with self.assertRaises(EvaluationResourceFailure):
                        self.evaluate(
                            runtime,
                            contract,
                            resources=TrainingResources(nvidia_gpu_count=1),
                            device="cuda",
                        )
                self.assertFalse(
                    any(
                        "benchcore.candidate_offline_runner" in call
                        for call in runtime.calls
                    )
                )

    def test_cpu_evaluation_records_no_gpu_request_despite_training_grant(self):
        request = ResourceEnvelope(
            (
                ResourceRequirement(
                    "accelerator.count", "exact", 0, EnforcementLevel.BEST_EFFORT
                ),
            )
        )
        runtime = self.runtime()
        with patch.object(runtime, "_host_nvidia_gpus") as inventory:
            _, receipt = runtime._evaluation_resource_plan(
                episode_id="episode",
                evaluation_id="evaluation:test",
                envelope=request,
                resources=TrainingResources(nvidia_gpu_count=4),
                device="cpu",
                timeout_seconds=10,
            )
        inventory.assert_not_called()
        self.assertEqual(receipt.properties[0].allocator_allocated.value, 0)
        self.assertEqual(receipt.properties[0].workload_visible.status, "unavailable")

    def test_gpu_inspection_rejects_a_different_or_unbounded_grant(self):
        gpu = "GPU-00000000-0000-0000-0000-000000000001"
        good = {"Count": 0, "DeviceIDs": [gpu], "Capabilities": [["gpu"]]}
        for changes in (
            {},
            {"Count": -1},
            {"Count": False},
            {"DeviceIDs": ["another"]},
        ):
            runtime = self.runtime(limits={"DeviceRequests": [{**good, **changes}]})
            if changes:
                with self.subTest(changes=changes), self.assertRaises(
                    EvaluationResourceFailure
                ):
                    runtime._verify_evaluation_resource_limits(
                        "candidate", [], gpu_ids=(gpu,), timeout_seconds=10
                    )
            else:
                runtime._verify_evaluation_resource_limits(
                    "candidate", [], gpu_ids=(gpu,), timeout_seconds=10
                )

    def test_late_resource_inspection_cannot_return_a_measurement(self):
        # The fake inspection takes seven seconds; its successful response must
        # not turn a timed-out operation into a measured evaluation.
        runtime = self.runtime()
        original = runtime._verify_evaluation_resource_limits

        def verify(*args, **kwargs):
            original(*args, **kwargs)
            runtime.now += 100

        with patch.object(
            runtime, "_verify_evaluation_resource_limits", side_effect=verify
        ):
            with self.assertRaisesRegex(WorkerFailure, "isolated evaluation failed"):
                self.evaluate(runtime, envelope())
        self.assertFalse(any("score" in call for call in runtime.calls))
        self.assertEqual(runtime._pending_evaluations, {})

    def test_actual_limit_mismatch_discards_score_and_cleans_resources(self):
        runtime = self.runtime(limits={"NanoCpus": 0})
        with self.assertRaises(EvaluationResourceFailure):
            self.evaluate(runtime, envelope())
        self.assertFalse(any("score" in call for call in runtime.calls))
        self.assertEqual(runtime._pending_evaluations, {})


class EvaluationResourceLegacyTests(unittest.TestCase):
    def test_legacy_results_remain_readable_without_synthetic_evidence(self):
        for version in (12, 13):
            value = test_results._finalized(result_schema_version=version)
            report = normalize_result_report(value)
            self.assertNotIn("evaluation_resource_envelope", report)
            value["evaluation_resource_envelope"] = envelope().public_manifest()
            with self.assertRaisesRegex(ResultSchemaError, "version 14"):
                normalize_result_report(value)
