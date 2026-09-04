import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from autotrain_bench.agent_api_bundle import (
    PUBLIC_AGENT_API_FILES,
    publish_agent_api_bundle,
)


class AgentApiBundleTests(unittest.TestCase):
    def test_published_preflight_works_without_the_core_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            published = publish_agent_api_bundle(root / "agent-api")
            candidate = root / "candidate"
            candidate.mkdir()
            fixture = Path(__file__).parent / "fixtures/task/agent/candidate.py"
            shutil.copyfile(fixture, candidate / "candidate.py")
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(published)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "autotrain_bench.candidate_preflight",
                    "--candidate",
                    str(candidate),
                    "--device",
                    "cpu",
                    "--vocab-size",
                    "10",
                    "--context-lengths",
                    "1",
                    "2",
                    "--examples",
                    "4",
                ],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "public_abi_passed")

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
