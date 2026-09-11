import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchcore.agent_api_bundle import (
    PUBLIC_AGENT_API_FILES,
    publish_agent_api_bundle,
)


class AgentApiBundleTests(unittest.TestCase):
    def test_published_protocol_helpers_work_without_the_core_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            published = publish_agent_api_bundle(root / "agent-api")
            candidate = root / "candidate"
            candidate.mkdir()
            fixture = Path(__file__).parent / "fixtures/task/agent/candidate.py"
            shutil.copyfile(fixture, candidate / "candidate.py")
            (root / "contexts.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "candidate_protocol": "offline_next_token_logits_v3",
                        "dataset_id": "public-api-test",
                        "vocab_size": 10,
                        "context_lengths": [1, 2],
                        "examples": [
                            {"example_id": "one", "input_ids": [1]},
                            {"example_id": "two", "input_ids": [1, 2]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(published)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path\n"
                    "from benchcore.candidate_protocol import load_candidate\n"
                    "candidate = Path('candidate').resolve()\n"
                    "load_candidate(candidate).run_offline(\n"
                    "    candidate, Path('contexts.json').resolve(),\n"
                    "    Path('predictions').resolve(), 'cpu')\n",
                ],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            predictions = root / "predictions"
            manifest = json.loads(
                (predictions / "predictions.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["dataset_id"], "public-api-test")
            self.assertEqual(manifest["examples"], 2)
            self.assertEqual(manifest["vocab_size"], 10)
            self.assertEqual((predictions / "logits.f32").stat().st_size, 2 * 10 * 4)

    def test_publishes_only_the_explicit_public_allowlist(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "agent-api"

            published = publish_agent_api_bundle(destination)

            package = published / "benchcore"
            self.assertEqual(
                {path.name for path in package.iterdir()},
                set(PUBLIC_AGENT_API_FILES.values()) | {"__init__.py"},
            )
            self.assertFalse((package / "supervisor.py").exists())
            self.assertFalse((package / "operator_status.py").exists())
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
