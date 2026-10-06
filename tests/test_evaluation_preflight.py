"""Reject unavailable cost environments before spending a research budget."""

import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from envcore.cli import build_parser, command_start
from envcore.config import TaskConfig, TrainingResources
from envcore.worker_protocol import (
    EnforcementLevel,
    EvaluationResourceFailure,
    ResourceEnvelope,
    ResourceRequirement,
)
from tests import test_challenge, test_evaluation_resources
from tests.test_runtime import GPU_ONE, GPU_TWO, TASK, gpu_inventory


class EvaluationPreflightTests(unittest.TestCase):
    def setUp(self):
        self.config = TaskConfig.load(TASK)
        self.fixture = test_evaluation_resources.EvaluationResourceDockerTests()
        self.runtime = self.fixture.runtime()
        self.request = test_evaluation_resources.envelope()

    def preflight(self, **kwargs):
        return self.runtime.preflight_evaluation_resources(
            self.config,
            resources=TrainingResources(),
            resource_envelope=self.request,
            **kwargs,
        )

    def test_preflight_is_read_only_and_does_not_issue_a_receipt(self):
        calls = []
        original = self.runtime._run

        def read_only(args, **kwargs):
            calls.append(args)
            self.assertEqual(args[0], "info")
            return original(args, **kwargs)

        with patch.object(self.runtime, "_run", side_effect=read_only), patch(
            "envcore.runtime.EvaluationResourceReceipt",
            side_effect=AssertionError(
                "preflight must not issue an allocation receipt"
            ),
        ), patch.object(self.runtime, "resolve_hidden_dataset") as hidden:
            self.assertIsNone(self.preflight())
        self.assertEqual(len(calls), 1)
        hidden.assert_not_called()
        self.assertEqual(self.runtime._pending_evaluations, {})

    def test_unspecified_preview_does_not_add_resource_probes(self):
        self.request = None
        with patch.object(self.runtime, "_run") as run, patch.object(
            self.runtime, "_host_nvidia_gpus"
        ) as gpu:
            self.assertIsNone(self.preflight())
        run.assert_not_called()
        gpu.assert_not_called()

    def test_unavailable_observations_do_not_block_preflight(self):
        self.request = ResourceEnvelope((), ("cache.policy", "host.exclusivity"))
        with patch.object(self.runtime, "_run") as run:
            self.assertIsNone(self.preflight())
        run.assert_not_called()

    def test_backend_change_after_preflight_is_rechecked_before_candidate(self):
        self.preflight()
        original = self.runtime._run

        def changed(args, **kwargs):
            if args[0] == "info":
                return subprocess.CompletedProcess(args, 0, "[false,false,false]", "")
            return original(args, **kwargs)

        with patch.object(self.runtime, "_run", side_effect=changed):
            with self.assertRaises(EvaluationResourceFailure):
                self.fixture.evaluate(self.runtime, self.request)
        self.assertFalse(
            any(
                "envcore.candidate_offline_runner" in call
                for call in self.runtime.calls
            )
        )
        self.assertEqual(self.runtime._pending_evaluations, {})

    def test_gpu_queries_share_the_preflight_deadline_even_during_initial_selection(
        self,
    ):
        config = replace(self.config, evaluator_device="cuda")
        resources = TrainingResources(nvidia_gpu_count=1, required_gpu_model="H100")
        timeouts = []

        def query(*args, **kwargs):
            timeouts.append(kwargs["timeout"])
            self.runtime.now += 2
            return gpu_inventory(f"{GPU_ONE}, NVIDIA H100, 81920, Disabled")

        with patch("envcore.runtime.shutil.which", return_value="nvidia-smi"), patch(
            "envcore.runtime.subprocess.run", side_effect=query
        ):
            with self.assertRaisesRegex(EvaluationResourceFailure, "deadline"):
                self.runtime.preflight_evaluation_resources(
                    config,
                    resources=resources,
                    resource_envelope=self.request,
                    timeout_seconds=3,
                )
        self.assertEqual(timeouts, [3, 1])
        self.assertEqual(self.runtime.calls, [])

    def test_preflight_gpu_selection_is_not_substituted_when_it_disappears(self):
        config = replace(self.config, evaluator_device="cuda")
        resources = TrainingResources(nvidia_gpu_count=1, required_gpu_model="H100")
        with patch("envcore.runtime.shutil.which", return_value="nvidia-smi"), patch(
            "envcore.runtime.subprocess.run",
            return_value=gpu_inventory(f"{GPU_ONE}, NVIDIA H100, 81920, Disabled"),
        ) as query:
            self.runtime.preflight_evaluation_resources(
                config,
                resources=resources,
                resource_envelope=self.request,
            )
            self.assertEqual(self.runtime._nvidia_selection, (GPU_ONE,))
            query.return_value = gpu_inventory(
                f"{GPU_TWO}, NVIDIA H100, 81920, Disabled"
            )
            with self.assertRaisesRegex(
                EvaluationResourceFailure, "refusing substitution"
            ):
                self.fixture.evaluate(
                    self.runtime, self.request, resources=resources, device="cuda"
                )
        self.assertFalse(
            any(
                "envcore.candidate_offline_runner" in call
                for call in self.runtime.calls
            )
        )

    def launch_args(self, root, request):
        release = test_challenge.ChallengeReleaseTests()._write_release(str(root))
        payload = json.loads(release.read_text())
        payload["schema_version"] = 3
        payload["runtime_profiles"] = {
            "cpu-preview": {
                "host_count": 1,
                "accelerator": None,
                "provisional": True,
                "evaluation_resource_contract": request.public_manifest(),
            }
        }
        release.write_text(json.dumps(payload))
        return build_parser().parse_args(
            [
                "start",
                "--challenge",
                str(release),
                "--track-config",
                "b-3x10m",
                "--runtime-profile",
                "cpu-preview",
                "--episode-dir",
                str(root / "episode"),
                "--agent-runner",
                "container",
                "--agent-image",
                "example/agent@sha256:" + "a" * 64,
                "--agent-command",
                "participant-agent",
                "--operator-max-storage-bytes",
                str(64 * 1024**2),
            ]
        )

    def test_cli_rejects_unsupported_contract_before_any_episode_setup(self):
        for request, supported in (
            (
                ResourceEnvelope(
                    (ResourceRequirement("cache.policy", "exact", "cold"),)
                ),
                True,
            ),
            (test_evaluation_resources.envelope(EnforcementLevel.ENFORCED), True),
            (self.request, False),
        ):
            with self.subTest(
                request=request, supported=supported
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = self.launch_args(root, request)
                runtime = self.fixture.runtime(supported=supported)
                with patch(
                    "envcore.cli.TaskConfig.load", return_value=self.config
                ), patch("envcore.cli.DockerRuntime", return_value=runtime), patch(
                    "envcore.cli.Supervisor"
                ) as supervisor, patch(
                    "envcore.cli._initialize_agent_workspace"
                ) as workspace, patch.object(
                    runtime, "configure_operator_storage"
                ) as storage, patch.object(runtime, "prepare") as prepare, patch.object(
                    runtime, "start_agent"
                ) as agent, redirect_stdout(io.StringIO()):
                    with self.assertRaises(EvaluationResourceFailure):
                        command_start(args)
                supervisor.assert_not_called()
                workspace.assert_not_called()
                storage.assert_not_called()
                prepare.assert_not_called()
                agent.assert_not_called()
                self.assertFalse((root / "episode").exists())
                self.assertEqual(runtime._pending_evaluations, {})

    def test_cli_passes_the_selected_contract_from_preflight_to_supervisor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.launch_args(root, self.request)
            with patch("envcore.cli.TaskConfig.load", return_value=self.config), patch(
                "envcore.cli.DockerRuntime", return_value=self.runtime
            ), patch.object(
                self.runtime,
                "preflight_evaluation_resources",
                wraps=self.runtime.preflight_evaluation_resources,
            ) as preflight, patch.object(
                self.runtime, "configure_operator_storage"
            ), patch.object(self.runtime, "prepare"), patch.object(
                self.runtime, "enable_recovery"
            ), patch.object(
                self.runtime, "environment_manifest", return_value={}
            ), patch(
                "envcore.cli.Supervisor",
                side_effect=RuntimeError("stop before reveal"),
            ) as supervisor, redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "stop before reveal"):
                    command_start(args)
            preflight.assert_called_once()
            self.assertEqual(
                preflight.call_args.kwargs["resource_envelope"], self.request
            )
            self.assertEqual(
                supervisor.call_args.kwargs["evaluation_resource_envelope"],
                self.request,
            )
            self.assertEqual(
                supervisor.call_args.kwargs["challenge_release"][
                    "evaluation_resource_envelope"
                ],
                self.request.public_manifest(),
            )
            self.assertFalse((root / "episode").exists())
