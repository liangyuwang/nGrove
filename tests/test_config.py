import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from autotrain_bench.config import (
    AGENT_PROTOCOL_VERSION,
    AgentConfig,
    ConfigError,
    InferenceConfig,
    TaskConfig,
    TrainingResources,
    resolve_operator_research_limit,
    resolve_operator_storage_limit,
)


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tests/fixtures/task/task.json"


class ConfigTests(unittest.TestCase):
    def test_only_effective_protection_defaults_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.json"
            defaults = task.with_name("operator-defaults.json")
            for invalid in (None, -1, True, "invalid"):
                defaults.write_text(
                    json.dumps(
                        {
                            "track_b_max_research_seconds": 1800,
                            "max_episode_storage_bytes": invalid,
                        }
                    )
                )
                self.assertEqual(resolve_operator_research_limit(task, "B"), 1800)
                self.assertIsNone(resolve_operator_storage_limit(task, disabled=True))
                self.assertEqual(
                    resolve_operator_storage_limit(task, override=4096), 4096
                )
                with self.assertRaises(ConfigError):
                    resolve_operator_storage_limit(task)
                defaults.write_text(
                    json.dumps(
                        {
                            "track_b_max_research_seconds": invalid,
                            "max_episode_storage_bytes": 2**30,
                        }
                    )
                )
                self.assertEqual(resolve_operator_storage_limit(task), 2**30)
                self.assertEqual(
                    resolve_operator_research_limit(task, "B", override=30), 30
                )
                self.assertIsNone(
                    resolve_operator_research_limit(task, "B", disabled=True)
                )
                self.assertIsNone(resolve_operator_research_limit(task, "A"))
                with self.assertRaises(ConfigError):
                    resolve_operator_research_limit(task, "B")

    def test_storage_defaults_remain_private_and_independent_of_track_time(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self._write_task(directory)
            original = TaskConfig.load(task)
            self.assertIsNone(resolve_operator_storage_limit(task))
            defaults = task.with_name("operator-defaults.json")
            defaults.write_text(
                json.dumps(
                    {
                        "track_b_max_research_seconds": 1800,
                        "max_episode_storage_bytes": 2 * 2**30,
                    }
                )
            )
            self.assertEqual(resolve_operator_storage_limit(task), 2 * 2**30)
            self.assertEqual(resolve_operator_storage_limit(task, override=4096), 4096)
            self.assertIsNone(resolve_operator_storage_limit(task, disabled=True))
            self.assertEqual(resolve_operator_research_limit(task, "B"), 1800)
            self.assertIsNone(resolve_operator_research_limit(task, "A"))
            self.assertEqual(TaskConfig.load(task), original)
            for track in ("A", "B"):
                self.assertEqual(
                    TaskConfig.load(task).public_manifest(track),
                    original.public_manifest(track),
                )
            for value in (True, 0, -1, 1.5, "1024", None):
                with self.subTest(value=value):
                    defaults.write_text(
                        json.dumps({"max_episode_storage_bytes": value})
                    )
                    with self.assertRaises(ConfigError):
                        resolve_operator_storage_limit(task)
                    self.assertIsNone(
                        resolve_operator_storage_limit(task, disabled=True)
                    )
                    self.assertEqual(
                        resolve_operator_storage_limit(task, override=2048), 2048
                    )
            with self.assertRaises(ConfigError):
                resolve_operator_storage_limit(task, override=1024, disabled=True)

    def test_operator_defaults_are_private_and_can_be_overridden_or_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self._write_task(directory)
            original = TaskConfig.load(task)
            self.assertIsNone(resolve_operator_research_limit(task, "B"))
            defaults = task.with_name("operator-defaults.json")
            for seconds in (1800, 86400):
                defaults.write_text(
                    json.dumps({"track_b_max_research_seconds": seconds}),
                    encoding="utf-8",
                )
                self.assertEqual(resolve_operator_research_limit(task, "B"), seconds)
                self.assertEqual(
                    resolve_operator_research_limit(task, "B", override=60), 60
                )
                self.assertIsNone(
                    resolve_operator_research_limit(task, "B", disabled=True)
                )
                self.assertIsNone(resolve_operator_research_limit(task, "A"))
                loaded = TaskConfig.load(task)
                self.assertEqual(loaded, original)
                self.assertEqual(
                    loaded.public_manifest("B"), original.public_manifest("B")
                )

    def test_invalid_operator_defaults_do_not_silently_disable_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.json"
            defaults = task.with_name("operator-defaults.json")
            for content in (
                "not json",
                "[]",
                "{}",
                '{"misspelled_limit": 1800}',
                *(
                    json.dumps({"track_b_max_research_seconds": invalid})
                    for invalid in (
                        None,
                        True,
                        0,
                        -1,
                        "1800",
                        float("inf"),
                        float("nan"),
                    )
                ),
            ):
                with self.subTest(content=content):
                    defaults.write_text(content, encoding="utf-8")
                    with self.assertRaises(ConfigError):
                        resolve_operator_research_limit(task, "B")
                    # Explicit run choices do not depend on the defaults file.
                    self.assertEqual(
                        resolve_operator_research_limit(task, "B", override=30), 30
                    )
                    self.assertIsNone(
                        resolve_operator_research_limit(task, "B", disabled=True)
                    )
                    self.assertIsNone(resolve_operator_research_limit(task, "A"))

    def _write_task(self, directory):
        payload = json.loads(TASK.read_text(encoding="utf-8"))
        base = TASK.parent
        for key, value in payload["paths"].items():
            payload["paths"][key] = str((base / value).resolve())
        payload["evaluation"]["entrypoint"] = str(
            (base / payload["evaluation"]["entrypoint"]).resolve()
        )
        destination = Path(directory) / "task.json"
        destination.write_text(json.dumps(payload), encoding="utf-8")
        return destination

    def test_public_task_manifest_hides_private_splits_and_has_no_participant(self):
        config = TaskConfig.load(TASK)
        public = config.public_manifest("B")
        serialized = json.dumps(public)
        self.assertNotIn(str(config.analysis_path), serialized)
        self.assertNotIn(str(config.test_path), serialized)
        self.assertNotIn("agent_environment", public)
        self.assertNotIn("agent", public)
        self.assertNotIn("hardware", public)
        self.assertEqual(public["task_kind"], "development_smoke")
        self.assertFalse(public["leaderboard_eligible"])
        self.assertNotIn("official_val_queries_per_lease", public["budget"])
        self.assertEqual(public["validation"]["organizer_feedback"], "none")
        self.assertEqual(public["analysis"]["feedback"], "never_agent_visible")
        self.assertEqual(config.context_lengths, (1, 2, 3))
        self.assertEqual(config.evaluator_device, "cpu")
        self.assertEqual(config.evaluator_backend, "isolated_candidate")
        self.assertTrue(config.evaluator_policy_compliant)
        boundary = public["evaluation"]["candidate_boundary"]
        self.assertEqual(boundary["target_artifact_mount"], "none")
        self.assertEqual(boundary["context_artifact_access"], "read_only")
        self.assertEqual(
            boundary["prediction_artifact_access"],
            "candidate_read_write_isolated",
        )
        prediction_manifest = boundary["prediction_manifest"]
        self.assertEqual(
            prediction_manifest["exact_fields"],
            [
                "schema_version",
                "candidate_protocol",
                "dataset_id",
                "examples",
                "vocab_size",
                "dtype",
                "logits_file",
            ],
        )
        self.assertFalse(prediction_manifest["additional_fields_allowed"])
        self.assertFalse(prediction_manifest["json_member_order_significant"])
        self.assertEqual(
            prediction_manifest["fields"]["candidate_protocol"]["constant"],
            "offline_next_token_logits_v3",
        )
        self.assertEqual(
            prediction_manifest["fields"]["dataset_id"]["source"],
            "contexts.json.dataset_id",
        )
        self.assertEqual(
            prediction_manifest["fields"]["examples"]["source"],
            "len(contexts.json.examples)",
        )
        self.assertEqual(
            boundary["public_helper_api"]["module"],
            "autotrain_bench.candidate_protocol",
        )
        self.assertEqual(boundary["prediction_storage"]["artifact_max_entries"], 2)
        self.assertEqual(boundary["prediction_storage"]["max_inodes"], 128)
        self.assertEqual(
            boundary["prediction_artifact_retention"],
            "ephemeral_until_evaluation_teardown",
        )
        self.assertEqual(boundary["channel_mechanism"], "worker_materialized_files")
        self.assertEqual(boundary["serving_timing"]["clock"], "trusted_monotonic_wall")
        limits = public["evaluation"]["candidate_artifact_limits"]
        self.assertEqual(limits, {"max_bytes": 1048576, "max_files": 100})
        self.assertEqual(
            public["evaluation"]["performance_score"],
            {
                "definition_id": "uniform_normalized_log_score_v1",
                "range": {"minimum": 0.0, "maximum": 100.0},
                "direction": "maximize",
                "comparison_scope": "benchmark_release",
                "raw_metrics": ["mean_nll"],
            },
        )
        self.assertEqual(public["evaluation"]["max_context_tokens"], 3)
        self.assertNotIn("context_lengths", public["evaluation"])
        self.assertNotIn("context_length", public["evaluation"])
        self.assertEqual(
            public["evaluation"]["cost"],
            {
                "definition_id": "candidate_end_to_end_time_per_task_v1",
                "task_unit": "next_token_prediction",
                "primary_metric": "time_per_task",
                "unit": "seconds_per_task",
                "direction": "minimize",
                "aggregation": "candidate_wall_seconds_divided_by_task_count",
                "comparison_scope": "benchmark_release",
                "comparison_requires": [
                    "hidden_dataset_identity",
                    "runtime_profile",
                ],
                "submetrics": [
                    "candidate_wall_seconds",
                    "task_count",
                    "input_tokens",
                    "prediction_bytes",
                    "tasks_per_second",
                    "input_tokens_per_second",
                ],
            },
        )

    def test_context_lengths_are_frozen_and_do_not_exceed_the_input_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self._write_task(directory)
            payload = json.loads(task.read_text(encoding="utf-8"))

            for invalid in ([1, 1, 3], [2, 1, 3], [1, 2, 4], [0, 2, 3]):
                with self.subTest(invalid=invalid):
                    payload["evaluation"]["context_lengths"] = invalid
                    task.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(ConfigError, "context_length"):
                        TaskConfig.load(task)

    def test_hidden_length_choices_do_not_change_the_public_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self._write_task(directory)
            payload = json.loads(task.read_text(encoding="utf-8"))
            original = TaskConfig.load(task)
            for lengths in ([1], [2], [1, 3], [1, 2, 3]):
                with self.subTest(lengths=lengths):
                    payload["evaluation"]["context_lengths"] = lengths
                    task.write_text(json.dumps(payload), encoding="utf-8")
                    config = TaskConfig.load(task)
                    self.assertEqual(config.context_lengths, tuple(lengths))
                    for track in ("A", "B"):
                        self.assertEqual(
                            config.public_manifest(track),
                            original.public_manifest(track),
                        )

    def test_task_rejects_participant_owned_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self._write_task(directory)
            payload = json.loads(task.read_text(encoding="utf-8"))
            payload["agent"] = {"runner": "container"}
            task.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "participant-owned"):
                TaskConfig.load(task)

            payload.pop("agent")
            payload["paths"]["agent_workspace"] = str(TASK.parent / "agent")
            task.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "participant-owned"):
                TaskConfig.load(task)

            payload["paths"].pop("agent_workspace")
            payload["images"]["agent"] = "python:test"
            task.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "participant-owned"):
                TaskConfig.load(task)

            payload["images"].pop("agent")
            payload["hardware"] = {"gpu_count": 1, "gpu_model": "test"}
            task.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "episode-owned"):
                TaskConfig.load(task)

    def test_task_classification_is_explicit_and_smoke_is_never_ranked(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self._write_task(directory)
            payload = json.loads(task.read_text(encoding="utf-8"))

            payload.pop("task_kind")
            task.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "task_kind"):
                TaskConfig.load(task)

            payload["task_kind"] = "development_smoke"
            payload["leaderboard_eligible"] = True
            task.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "may not be leaderboard"):
                TaskConfig.load(task)

            payload["task_kind"] = "benchmark_challenge"
            payload["leaderboard_eligible"] = False
            task.write_text(json.dumps(payload), encoding="utf-8")
            config = TaskConfig.load(task)
            self.assertEqual(config.task_kind, "benchmark_challenge")
            self.assertFalse(config.leaderboard_eligible)

    def test_task_requires_the_generic_performance_score_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self._write_task(directory)
            payload = json.loads(task.read_text(encoding="utf-8"))

            payload["evaluation"].pop("performance_score")
            task.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "performance_score"):
                TaskConfig.load(task)

            payload["evaluation"]["performance_score"] = {
                "definition_id": "accuracy_v2",
                "range": {"minimum": 0.0, "maximum": 100.0},
                "direction": "maximize",
                "comparison_scope": "benchmark_release",
                "raw_metrics": ["accuracy"],
            }
            task.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                TaskConfig.load(task).performance_score.definition_id,
                "accuracy_v2",
            )

            payload["evaluation"]["performance_score"] = {
                "definition_id": "example_score_v1",
                "range": {"minimum": 0.0, "maximum": 1.0},
                "direction": "minimize",
                "comparison_scope": "global",
                "raw_metrics": ["loss"],
            }
            task.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "performance_score"):
                TaskConfig.load(task)

    def test_task_requires_the_average_cost_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self._write_task(directory)
            payload = json.loads(task.read_text(encoding="utf-8"))

            payload["evaluation"].pop("cost")
            task.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "evaluation.cost"):
                TaskConfig.load(task)

    def test_leaderboard_task_requires_digest_pinned_images(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self._write_task(directory)
            payload = json.loads(task.read_text(encoding="utf-8"))
            payload["task_kind"] = "benchmark_challenge"
            payload["leaderboard_eligible"] = True
            task.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "digest-pinned"):
                TaskConfig.load(task)

            digest = "@sha256:" + "a" * 64
            for name in payload["images"]:
                payload["images"][name] = f"example/{name}{digest}"
            task.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(TaskConfig.load(task).leaderboard_eligible)

            payload["images"]["training"] = f"example/training image{digest}"
            task.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "digest-pinned"):
                TaskConfig.load(task)

    def test_hidden_datasets_must_be_distinct_and_outside_train(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self._write_task(directory)
            payload = json.loads(task.read_text(encoding="utf-8"))
            payload["paths"]["analysis"] = str(TASK.parent / "train/train.json")
            task.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "outside.*TRAIN"):
                TaskConfig.load(task)

            payload["paths"]["analysis"] = payload["paths"]["test"]
            task.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "distinct locations"):
                TaskConfig.load(task)

    def test_hidden_datasets_need_not_exist_when_task_is_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self._write_task(directory)
            payload = json.loads(task.read_text(encoding="utf-8"))
            payload["paths"]["analysis"] = str(Path(directory) / "late/analysis")
            payload["paths"]["test"] = str(Path(directory) / "late/test")
            task.write_text(json.dumps(payload), encoding="utf-8")

            config = TaskConfig.load(task)

            self.assertFalse(config.analysis_path.exists())
            self.assertFalse(config.test_path.exists())

    def test_public_task_manifest_declares_the_current_agent_protocol(self):
        config = TaskConfig.load(TASK)
        manifest = config.public_manifest("A")

        # The Agent reads this manifest and the `agent status` response.  A
        # rename in either surface must move this version with it.
        self.assertEqual(manifest["protocol_version"], AGENT_PROTOCOL_VERSION)
        self.assertEqual(AGENT_PROTOCOL_VERSION, 12)

    def test_prior_task_storage_semantics_are_not_silently_reinterpreted(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self._write_task(directory)
            payload = json.loads(task.read_text(encoding="utf-8"))
            payload["schema_version"] = 6
            task.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "schema_version must be 7"):
                TaskConfig.load(task)

    def test_agent_egress_environment_and_host_runner_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            container = AgentConfig(
                workspace=workspace,
                runner="container",
                command=["python", "/workspace/agent.py"],
                initial_workspace="participant_snapshot",
                image="python:test",
                egress="denied",
            ).validate()
            self.assertEqual(container.network, "disabled")
            self.assertEqual(
                container.public_manifest(
                    "python:gateway",
                    policy_compliant=True,
                    control_transport="test_transport",
                )["workspace_scope"],
                "episode_snapshot",
            )

            host = AgentConfig(
                workspace=workspace,
                runner="host_cli",
                command=["example-agent"],
                initial_workspace="participant_snapshot",
            ).validate()
            self.assertEqual(host.network, "host_managed")

            with self.assertRaises(ConfigError):
                replace(container, egress="host").validate()
            with self.assertRaises(ConfigError):
                replace(container, environment=["HOME"]).validate()
            with self.assertRaises(ConfigError):
                replace(host, egress="denied").validate()

    def test_registered_inference_belongs_to_agent_run_not_task(self):
        inference = InferenceConfig.load(
            {
                "base_url": "https://api.openai.com",
                "api_key_env": "OPENAI_API_KEY",
                "allowed_models": ["researcher-a", "researcher-b"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            agent = AgentConfig(
                workspace=Path(directory),
                runner="container",
                command=["agent"],
                initial_workspace="participant_snapshot",
                image="agent:image",
                egress="denied",
                inference=inference,
            ).validate()
            public = agent.public_manifest(
                "gateway:image",
                policy_compliant=True,
                control_transport="test_transport",
            )["inference"]
            self.assertTrue(public["enabled"])
            self.assertEqual(
                public["local_base_url"], "http://inference-gateway:8765/v1"
            )
            self.assertNotIn("OPENAI_API_KEY", json.dumps(public))
            self.assertNotIn("api.openai.com", json.dumps(public))

            with self.assertRaises(ConfigError):
                replace(agent, environment=["OPENAI_API_KEY"]).validate()
            with self.assertRaises(ConfigError):
                AgentConfig(
                    workspace=Path(directory),
                    runner="host_cli",
                    command=["agent"],
                    initial_workspace="participant_snapshot",
                    inference=inference,
                ).validate()

    def test_inference_gateway_allows_auth_free_local_endpoint(self):
        inference = InferenceConfig.load(
            {
                "base_url": "http://host.docker.internal:8000",
                "api_key_env": None,
                "allowed_models": ["local-model"],
            }
        )
        self.assertIsNotNone(inference)
        self.assertIsNone(inference.api_key_env)

    def test_gpu_evaluator_device_is_public_and_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self._write_task(directory)
            payload = json.loads(task.read_text(encoding="utf-8"))
            payload["evaluation"]["device"] = "cuda"
            task.write_text(json.dumps(payload), encoding="utf-8")
            config = TaskConfig.load(task)
            evaluation = config.public_manifest("A")["evaluation"]
            self.assertNotIn("batch_size", evaluation)
            self.assertEqual(evaluation["device"], "cuda")

            TrainingResources(nvidia_gpu_count=1).validate_for(config)
            with self.assertRaises(ConfigError):
                TrainingResources().validate_for(config)
            with self.assertRaises(ConfigError):
                replace(config, evaluator_device="mps").validate()

    def test_training_resources_are_episode_owned_and_validated(self):
        task = TaskConfig.load(TASK)
        resources = TrainingResources(
            nvidia_gpu_count=2,
            required_gpu_model="H100",
            minimum_gpu_memory_mib=80000,
            devices=["/dev/accel0"],
        ).validate_for(task)
        self.assertEqual(resources.public_manifest()["selection"], "episode_runtime")
        self.assertEqual(resources.public_manifest()["minimum_gpu_memory_mib"], 80000)
        with self.assertRaises(ConfigError):
            TrainingResources(required_gpu_model="H100").validate()
        with self.assertRaises(ConfigError):
            TrainingResources(minimum_gpu_memory_mib=80000).validate()
        with self.assertRaises(ConfigError):
            TrainingResources(devices=["relative-device"]).validate()

    def test_evaluator_backend_is_explicit_and_never_core_defaulted(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self._write_task(directory)
            payload = json.loads(task.read_text(encoding="utf-8"))
            payload["evaluation"].pop("backend")
            task.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ConfigError):
                TaskConfig.load(task)

            payload["evaluation"]["backend"] = "official_container"
            task.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ConfigError):
                TaskConfig.load(task)


if __name__ == "__main__":
    unittest.main()
