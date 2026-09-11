"""GPU selection regressions at the local Worker's command boundary, no hardware."""

import csv
import json
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from benchcore.challenge import ChallengeRelease
from benchcore.config import TaskConfig, TrainingResources
from benchcore.runtime import DockerRuntime, RuntimeFailure
from benchcore.worker_protocol import (
    EnforcementLevel,
    ResearchTaskSpec,
    ResourceAllocationReceipt,
    WorkloadRequest,
    WorkerTeardownFailure,
    assess_resource_allocation,
    assess_resource_properties,
    v0_resource_envelope,
)
from tests.test_runtime import (
    CapturingDockerRuntime,
    GPU_ONE,
    GPU_TWO,
    TASK,
    gpu_inventory,
    publish_agent_api,
)


GPU_THREE = "GPU-00000000-0000-0000-0000-000000000003"
GPU_FOUR = "GPU-00000000-0000-0000-0000-000000000004"
H100 = TrainingResources(
    nvidia_gpu_count=1, required_gpu_model="H100", minimum_gpu_memory_mib=80000
)
GOOD = f"{GPU_ONE}, NVIDIA H100, 81559, Disabled"


class GpuRuntime(CapturingDockerRuntime):
    def __init__(self, visible=GOOD, *, cleanup_fails=False):
        super().__init__()
        self.visible = visible
        self.cleanup_fails = cleanup_fails
        self.stops = []

    def _run(self, args, **kwargs):
        if args[0] == "exec" and "nvidia-smi" in args:
            self.calls.append(list(args))
            return subprocess.CompletedProcess(args, 0, self.visible, "")
        return super()._run(args, **kwargs)

    def stop_container(self, name, **kwargs):
        self.stops.append(name)
        if self.cleanup_fails:
            raise WorkerTeardownFailure("fixture removal unconfirmed")


class DockerGpuTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        task_path = self.root / "task.json"
        task_path.write_text("{}")
        publish_agent_api(task_path)
        self.task = ResearchTaskSpec.from_task(
            TaskConfig.load(TASK), self.root / "agent_api"
        )
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.smi = self.patches.enter_context(
            patch("benchcore.runtime.shutil.which", return_value="nvidia-smi")
        )
        self.query = self.patches.enter_context(
            patch(
                "benchcore.runtime.subprocess.run",
                return_value=gpu_inventory(GOOD),
            )
        )

    def start(self, runtime, *, resources=H100, lease="lease-1"):
        workload = runtime.resolve_workload(
            episode_id="episode",
            workload=WorkloadRequest("training", self.task.training_image),
            timeout_seconds=20,
        )
        return runtime.start_lease(
            episode_id="episode",
            lease_id=lease,
            duration_seconds=60,
            task=self.task,
            workspace=self.root,
            resources=resources,
            lease_dir=self.root / lease,
            workload=workload,
        )

    def test_heterogeneous_host_grants_only_matching_full_gpus(self):
        self.query.return_value = gpu_inventory(
            f"{GPU_FOUR}, NVIDIA H100, 81559, Disabled",
            f"{GPU_ONE}, NVIDIA A100, 81920, Disabled",
            f"{GPU_TWO}, NVIDIA H100, 65536, Disabled",
            f"{GPU_THREE}, NVIDIA H100, 81559, Disabled",
        )
        runtime = GpuRuntime(
            f"{GPU_FOUR}, NVIDIA H100, 81559, Disabled\n{GPU_THREE}, NVIDIA H100, 81559, Disabled"
        )
        self.start(runtime, resources=replace(H100, nvidia_gpu_count=2))
        launch = next(call for call in runtime.calls if call[0] == "run")
        # A single CSV field protects the inner comma in Docker's --gpus parser.
        self.assertEqual(
            next(csv.reader([launch[launch.index("--gpus") + 1]])),
            [f"device={GPU_THREE},{GPU_FOUR}"],
        )
        evidence = json.loads((self.root / "lease-1/gpu-binding.json").read_text())
        self.assertEqual(evidence["selected_uuids"], [GPU_THREE, GPU_FOUR])
        self.assertEqual(evidence["observed_uuids"], [GPU_FOUR, GPU_THREE])
        self.assertEqual(
            (self.root / "lease-1/gpu-binding.json").stat().st_mode & 0o777, 0o600
        )
        self.assertEqual(self.query.call_args.kwargs["timeout"], 20)

    def test_release_gpu_matrix_binds_training_and_cuda_evaluation_resources(self):
        release = ChallengeRelease.load(
            Path(__file__).resolve().parents[1]
            / "challenges/lm-pretrain/challenge.json"
        )
        for profile_id, profile in release.runtime_profiles.items():
            with self.subTest(profile=profile_id):
                resources = profile.training_resources()
                count = resources.nvidia_gpu_count
                rows = [
                    f"GPU-00000000-0000-0000-0000-{index:012d}, {profile.accelerator_model}, {profile.minimum_memory_mib + 1000}, Disabled"
                    for index in range(1, 9)
                ]
                self.query.return_value = gpu_inventory(*reversed(rows))
                runtime = GpuRuntime("\n".join(reversed(rows[:count])))
                receipt = self.start(runtime, resources=resources, lease=profile_id)
                envelope = profile.resource_envelope(
                    minimum_enforcement=EnforcementLevel.BEST_EFFORT
                )
                receipt = ResourceAllocationReceipt.from_manifest(
                    receipt.public_manifest()
                )
                self.assertTrue(
                    assess_resource_allocation(
                        envelope,
                        receipt,
                        episode_id="episode",
                        lease_id=profile_id,
                        worker_id=receipt.worker_id,
                    ).satisfied
                )
                launch = next(call for call in runtime.calls if call[0] == "run")
                devices = [row.split(",", 1)[0] for row in rows[:count]]
                self.assertEqual(
                    next(csv.reader([launch[launch.index("--gpus") + 1]])),
                    ["device=" + ",".join(devices)],
                )
                runtime.terminate_lease(episode_id="episode", lease_id=profile_id)
                _, properties = runtime._evaluation_resource_properties(
                    envelope=envelope,
                    resources=resources,
                    device="cuda",
                    timeout_seconds=20,
                )
                self.assertTrue(
                    assess_resource_properties(envelope, tuple(properties)).satisfied
                )
                self.assertEqual(runtime._nvidia_devices(resources), tuple(devices))

    def test_exact_model_matching_rejects_variants_and_cannot_be_relaxed(self):
        resources = TrainingResources(
            nvidia_gpu_count=1,
            required_gpu_model="NVIDIA H200",
            minimum_gpu_memory_mib=140000,
            gpu_model_match_mode="exact",
        )
        good = f"{GPU_TWO}, NVIDIA H200, 143771, Disabled"
        variant = f"{GPU_ONE}, NVIDIA H200 NVL, 143771, Disabled"
        self.query.return_value = gpu_inventory(variant, good)
        runtime = GpuRuntime(good)
        receipt = self.start(runtime, resources=resources)
        self.assertEqual(runtime._nvidia_devices(resources), (GPU_TWO,))
        envelope = v0_resource_envelope(
            resources, minimum_enforcement=EnforcementLevel.BEST_EFFORT
        )
        # A Worker that ignores exact matching still cannot supply an accepted
        # receipt for another model, even when its GPU count and memory match.
        wrong_properties = tuple(
            replace(
                item,
                workload_visible=replace(
                    item.workload_visible, value=("NVIDIA H200 NVL",)
                ),
            )
            if item.name == "accelerator.model"
            else item
            for item in receipt.properties
        )
        self.assertEqual(
            assess_resource_properties(envelope, wrong_properties).mismatched,
            ("accelerator.model",),
        )
        with self.assertRaisesRegex(RuntimeFailure, "requirements changed"):
            runtime.validate_hardware(
                replace(resources, gpu_model_match_mode="contains")
            )
        for row in (
            variant,
            good.replace("143771", "80000"),
            good.replace("Disabled", "Enabled"),
        ):
            self.query.return_value = gpu_inventory(row)
            with self.subTest(row=row), self.assertRaises(RuntimeFailure):
                GpuRuntime(row).validate_hardware(resources)
        self.query.return_value = gpu_inventory(good)
        for count in (2, 4, 8):
            with self.subTest(count=count), self.assertRaises(RuntimeFailure):
                GpuRuntime(good).validate_hardware(
                    replace(resources, nvidia_gpu_count=count)
                )

        # The same UUID reporting another variant must also fail inside a Lease.
        runtime = GpuRuntime(good.replace("NVIDIA H200", "NVIDIA H200 NVL"))
        with self.assertRaisesRegex(RuntimeFailure, "equal to"):
            self.start(runtime, resources=resources, lease="wrong-variant")
        self.assertEqual(len(runtime.stops), 1)

    def test_wrong_workload_devices_or_properties_reject_lease_and_clean_up(self):
        for visible in (
            GOOD.replace(GPU_ONE, GPU_TWO),
            "",
            GOOD + "\n" + GOOD,
            GOOD + "\n" + GOOD.replace(GPU_ONE, GPU_TWO),
            GOOD.replace("H100", "A100"),
            GOOD.replace("81559", "40000"),
            GOOD.replace("Disabled", "Enabled"),
        ):
            with self.subTest(visible=visible):
                runtime = GpuRuntime(visible)
                with self.assertRaises(RuntimeFailure):
                    self.start(runtime)
                self.assertEqual(len(runtime.stops), 1)
                self.assertEqual(runtime._worker_leases, {})
                self.assertFalse((self.root / "lease-1/gpu-binding.json").exists())

    def test_mismatch_with_failed_cleanup_keeps_retryable_handle(self):
        runtime = GpuRuntime(GOOD.replace(GPU_ONE, GPU_TWO), cleanup_fails=True)
        with self.assertRaises(WorkerTeardownFailure):
            self.start(runtime)
        self.assertIn(("episode", "lease-1"), runtime._worker_leases)
        runtime.cleanup_fails = False
        runtime.terminate_lease(episode_id="episode", lease_id="lease-1")
        self.assertEqual(runtime._worker_leases, {})

    def test_frozen_selection_is_not_replaced_or_reconfigured(self):
        runtime = GpuRuntime()
        runtime.validate_hardware(H100)
        self.query.return_value = gpu_inventory(GOOD.replace(GPU_ONE, GPU_TWO))
        with self.assertRaisesRegex(RuntimeFailure, "refusing substitution"):
            runtime.validate_hardware(H100)
        with self.assertRaisesRegex(RuntimeFailure, "requirements changed"):
            self.start(runtime, resources=replace(H100, nvidia_gpu_count=2))
        with self.assertRaisesRegex(RuntimeFailure, "requirements changed"):
            self.start(runtime, resources=TrainingResources())
        self.assertFalse(any(call[0] == "run" for call in runtime.calls))

    def test_later_leases_keep_the_selected_uuid(self):
        runtime = GpuRuntime()
        self.start(runtime)
        runtime.terminate_lease(episode_id="episode", lease_id="lease-1")
        self.query.side_effect = AssertionError("must not silently select another GPU")
        self.start(runtime, lease="lease-2")
        launches = [call for call in runtime.calls if call[0] == "run"]
        self.assertEqual(
            [call[call.index("--gpus") + 1] for call in launches],
            [f'"device={GPU_ONE}"'] * 2,
        )

    def test_candidate_reuses_frozen_selection_after_host_inventory_changes(self):
        second = GOOD.replace(GPU_ONE, GPU_TWO)
        self.query.return_value = gpu_inventory(second)
        runtime = GpuRuntime(second)
        self.start(runtime)
        runtime.terminate_lease(episode_id="episode", lease_id="lease-1")
        # A newly visible matching device sorts before the original selection.
        self.query.return_value = gpu_inventory(GOOD, second)
        runtime.calls.clear()
        config = replace(TaskConfig.load(TASK), evaluator_device="cuda")
        with self.assertRaises(RuntimeFailure):  # Fixture produces no scored result.
            runtime.run_evaluator(
                config=config,
                resources=H100,
                candidate_dir=TASK.parent / "agent",
                split="test",
                output_dir=self.root / "evaluation",
                timeout_seconds=7,
            )
        candidate = next(
            call
            for call in runtime.calls
            if "benchcore.candidate_offline_runner" in call
        )
        self.assertEqual(
            candidate[candidate.index("--gpus") + 1], f'"device={GPU_TWO}"'
        )
        self.assertLessEqual(self.query.call_args.kwargs["timeout"], 7)

        runtime.calls.clear()
        self.query.return_value = gpu_inventory(GOOD)
        with self.assertRaises(RuntimeFailure):
            runtime.run_evaluator(
                config=config,
                resources=H100,
                candidate_dir=TASK.parent / "agent",
                split="analysis",
                output_dir=self.root / "later-evaluation",
            )
        self.assertFalse(
            any("benchcore.candidate_offline_runner" in call for call in runtime.calls)
        )

    def test_environment_records_selection_separately_from_host_inventory(self):
        runtime = GpuRuntime()
        runtime.validate_hardware(H100)
        self.query.return_value = gpu_inventory(
            f"NVIDIA H100, {GPU_ONE}, 555.1, 81559",
            f"NVIDIA H100, {GPU_TWO}, 555.1, 81559",
        )
        manifest = runtime.environment_manifest(TaskConfig.load(TASK), resources=H100)
        self.assertEqual(manifest["selected_nvidia_gpu_uuids"], [GPU_ONE])
        self.assertEqual(len(manifest["gpus"]), 2)

    def test_invalid_inventory_or_mig_does_not_fall_back_to_a_count_grant(self):
        for inventory in (
            "",
            GOOD + "\n" + GOOD,
            GOOD.replace(GPU_ONE, "GPU-malformed"),
            GOOD.replace("81559", "-1"),
            GOOD.replace("Disabled", "Unknown"),
            GOOD.replace("Disabled", "Enabled"),
            GOOD.replace("81559", "N/A"),
        ):
            with self.subTest(inventory=inventory):
                self.query.return_value = gpu_inventory(inventory)
                runtime = GpuRuntime()
                with self.assertRaises(RuntimeFailure):
                    self.start(runtime)
                self.assertFalse(any(call[0] == "run" for call in runtime.calls))

    def test_inventory_timeout_is_typed_and_cpu_does_not_query_gpu(self):
        self.query.side_effect = subprocess.TimeoutExpired(["nvidia-smi"], 20)
        with self.assertRaisesRegex(RuntimeFailure, "inventory query failed"):
            DockerRuntime().validate_hardware(H100)
        self.query.reset_mock()
        DockerRuntime().validate_hardware(TrainingResources())
        self.query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
