import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from autotrain_bench.config import ConfigError, TaskConfig


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_public_manifest_hides_private_splits(self):
        config = TaskConfig.load(ROOT / "examples/toy/task.json")
        serialized = json.dumps(config.public_manifest("B"))
        self.assertNotIn(str(config.val_path), serialized)
        self.assertNotIn(str(config.test_path), serialized)
        self.assertEqual(
            config.public_manifest("B")["budget"]["official_val_queries_per_lease"],
            1,
        )
        self.assertEqual(
            config.public_manifest("B")["agent_environment"]["network"],
            "disabled",
        )
        self.assertEqual(
            config.public_manifest("B")["agent_environment"]["mode"],
            "sandboxed",
        )
        self.assertEqual(
            config.public_manifest("B")["agent_environment"]["workspace_scope"],
            "episode_copy",
        )

    def _write_task(self, directory, *, mode="sandboxed", environment=None, inference=None):
        source = ROOT / "examples/toy/task.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        base = source.parent
        for key, value in payload["paths"].items():
            payload["paths"][key] = str((base / value).resolve())
        payload["agent"].pop("network", None)
        payload["agent"]["mode"] = mode
        payload["agent"]["environment"] = environment or []
        if inference is not None:
            payload["agent"]["inference"] = inference
        else:
            payload["agent"].pop("inference", None)
        destination = Path(directory) / "task.json"
        destination.write_text(json.dumps(payload), encoding="utf-8")
        return destination

    def test_agent_mode_and_environment_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self._write_task(
                directory,
                mode="unrestricted-dev",
                environment=["AGENT_TOKEN", "AGENT_TOKEN"],
            )
            config = TaskConfig.load(task)
            self.assertEqual(config.agent_mode, "unrestricted-dev")
            self.assertFalse(config.policy_compliant)
            self.assertEqual(config.agent_environment, ["AGENT_TOKEN"])

            invalid_mode = self._write_task(directory, mode="host")
            with self.assertRaises(ConfigError):
                TaskConfig.load(invalid_mode)

            reserved = self._write_task(directory, environment=["HOME"])
            with self.assertRaises(ConfigError):
                TaskConfig.load(reserved)

    def test_inference_gateway_registration_and_secret_isolation(self):
        inference = {
            "protocol": "openai_compatible",
            "base_url": "https://api.openai.com",
            "api_key_env": "OPENAI_API_KEY",
            "allowed_models": ["researcher-a", "researcher-b"],
        }
        with tempfile.TemporaryDirectory() as directory:
            task = self._write_task(directory, inference=inference)
            config = TaskConfig.load(task)
            self.assertIsNotNone(config.inference)
            self.assertEqual(config.inference.allowed_models, ("researcher-a", "researcher-b"))
            public = config.public_manifest("A")["agent_environment"]["inference"]
            self.assertTrue(public["enabled"])
            self.assertEqual(
                public["local_base_url"], "http://inference-gateway:8765/v1"
            )
            self.assertIn(config.inference_gateway_image, config.images)
            self.assertNotIn("OPENAI_API_KEY", json.dumps(public))
            self.assertNotIn("api.openai.com", json.dumps(public))

            leaked = self._write_task(
                directory,
                environment=["OPENAI_API_KEY"],
                inference=inference,
            )
            with self.assertRaises(ConfigError):
                TaskConfig.load(leaked)

            # CLI overrides are applied after load, so final validation must
            # enforce the same cross-field credential boundary.
            with self.assertRaises(ConfigError):
                replace(
                    config,
                    agent_environment=["OPENAI_API_KEY"],
                ).validate()

    def test_legacy_network_configuration_migrates(self):
        with tempfile.TemporaryDirectory() as directory:
            source = ROOT / "examples/toy/task.json"
            payload = json.loads(source.read_text(encoding="utf-8"))
            base = source.parent
            for key, value in payload["paths"].items():
                payload["paths"][key] = str((base / value).resolve())
            payload["agent"].pop("mode", None)
            payload["agent"]["network"] = "none"
            destination = Path(directory) / "task.json"
            destination.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(TaskConfig.load(destination).agent_mode, "sandboxed")
            payload["agent"]["network"] = "open"
            destination.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(TaskConfig.load(destination).agent_mode, "unrestricted-dev")


if __name__ == "__main__":
    unittest.main()
