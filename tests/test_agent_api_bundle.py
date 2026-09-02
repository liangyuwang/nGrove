import tempfile
import unittest
from pathlib import Path

from autotrain_bench.agent_api_bundle import (
    PUBLIC_AGENT_API_FILES,
    publish_agent_api_bundle,
)


class AgentApiBundleTests(unittest.TestCase):
    def test_publishes_only_the_explicit_public_allowlist(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "agent-api"

            published = publish_agent_api_bundle(destination)

            package = published / "autotrain_bench"
            self.assertEqual(
                {path.name for path in package.iterdir()},
                set(PUBLIC_AGENT_API_FILES.values()) | {"__init__.py"},
            )
            self.assertFalse((package / "supervisor.py").exists())
            self.assertEqual(published.stat().st_mode & 0o777, 0o555)
            self.assertEqual(package.stat().st_mode & 0o777, 0o555)
            self.assertTrue(
                all(path.stat().st_mode & 0o777 == 0o444 for path in package.iterdir())
            )

    def test_refuses_to_replace_an_existing_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "agent-api"
            publish_agent_api_bundle(destination)

            with self.assertRaises(FileExistsError):
                publish_agent_api_bundle(destination)


if __name__ == "__main__":
    unittest.main()
