import json
import tempfile
import unittest
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
            "none",
        )
        self.assertEqual(
            config.public_manifest("B")["agent_environment"]["workspace_scope"],
            "episode_copy",
        )

    def _write_task(self, directory, *, network="none", environment=None):
        source = ROOT / "examples/toy/task.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        base = source.parent
        for key, value in payload["paths"].items():
            payload["paths"][key] = str((base / value).resolve())
        payload["agent"]["network"] = network
        payload["agent"]["environment"] = environment or []
        destination = Path(directory) / "task.json"
        destination.write_text(json.dumps(payload), encoding="utf-8")
        return destination

    def test_agent_network_and_environment_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self._write_task(
                directory,
                network="open",
                environment=["OPENAI_API_KEY", "OPENAI_API_KEY", "AGENT_TOKEN"],
            )
            config = TaskConfig.load(task)
            self.assertEqual(config.agent_network, "open")
            self.assertEqual(
                config.agent_environment, ["OPENAI_API_KEY", "AGENT_TOKEN"]
            )

            invalid_network = self._write_task(directory, network="host")
            with self.assertRaises(ConfigError):
                TaskConfig.load(invalid_network)

            reserved = self._write_task(directory, environment=["HOME"])
            with self.assertRaises(ConfigError):
                TaskConfig.load(reserved)

    def test_legacy_agent_configuration_defaults_to_closed_network(self):
        with tempfile.TemporaryDirectory() as directory:
            task = self._write_task(directory)
            payload = json.loads(task.read_text(encoding="utf-8"))
            payload["agent"].pop("network")
            payload["agent"].pop("environment")
            task.write_text(json.dumps(payload), encoding="utf-8")

            config = TaskConfig.load(task)
            self.assertEqual(config.agent_network, "none")
            self.assertEqual(config.agent_environment, [])


if __name__ == "__main__":
    unittest.main()
