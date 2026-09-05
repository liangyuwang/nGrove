import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from autotrain_bench.config import TaskConfig
from autotrain_bench.worker_protocol import CandidateChannelRequirements


ROOT = Path(__file__).resolve().parents[1]


class CandidateEntrypointTests(unittest.TestCase):
    def _task_config(self, task):
        source = ROOT / task / "task.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        # Exercise each real task declaration without preparing its datasets.
        payload["paths"] = {
            name: str((source.parent / value).resolve())
            for name, value in payload["paths"].items()
        }
        payload["paths"]["train"] = str(ROOT / "tests/fixtures/task/train")
        payload["evaluation"]["entrypoint"] = str(
            (source.parent / payload["evaluation"]["entrypoint"]).resolve()
        )
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "task.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            return TaskConfig.load(manifest)

    def test_public_candidate_contracts_agree_with_worker_for_both_tasks_and_tracks(
        self,
    ):
        for task in ("dev_tasks/smoke_lm", "challenges/lm-pretrain-v1"):
            config = self._task_config(task)
            worker = CandidateChannelRequirements(
                protocol=config.candidate_protocol
            ).public_manifest()
            for track in ("A", "B"):
                with self.subTest(task=task, track=track):
                    evaluation = config.public_manifest(track)["evaluation"]
                    self.assertEqual(
                        evaluation["max_context_tokens"], config.context_length
                    )
                    self.assertNotIn("context_lengths", evaluation)
                    self.assertNotIn(
                        "evaluation.context_lengths", json.dumps(evaluation)
                    )
                    boundary = evaluation["candidate_boundary"]
                    for section in (
                        "entrypoint",
                        "context_manifest",
                        "prediction_manifest",
                        "public_helper_api",
                    ):
                        self.assertEqual(boundary[section], worker[section])
                    context = boundary["context_manifest"]
                    self.assertEqual(
                        context["exact_fields"],
                        [
                            "schema_version",
                            "candidate_protocol",
                            "dataset_id",
                            "vocab_size",
                            "context_lengths",
                            "examples",
                        ],
                    )
                    self.assertFalse(context["additional_fields_allowed"])
                    self.assertNotIn("coverage", context)
                    self.assertEqual(
                        context["fields"]["context_lengths"]["value_visibility"],
                        "candidate_only_after_research_sealing",
                    )
                    example = context["fields"]["examples"]["items"]
                    self.assertEqual(
                        example["exact_fields"], ["example_id", "input_ids"]
                    )
                    self.assertFalse(example["additional_fields_allowed"])
                    self.assertEqual(
                        example["fields"]["input_ids"]["json_type"], "array"
                    )
                    self.assertEqual(
                        example["fields"]["input_ids"]["max_items_source"],
                        "evaluation.max_context_tokens",
                    )
                    self.assertEqual(
                        boundary["public_helper_api"]["request_reader_return"][
                            "contract"
                        ],
                        "evaluation.candidate_boundary.context_manifest",
                    )
                    entrypoint = boundary["entrypoint"]
                    self.assertEqual(entrypoint["module_file"], "candidate.py")
                    self.assertEqual(
                        entrypoint["module_location"], "candidate_artifact_root"
                    )
                    self.assertEqual(entrypoint["function"], "run_offline")
                    self.assertEqual(
                        entrypoint["signature"],
                        "run_offline(candidate_dir, contexts_path, output_dir, device)",
                    )
                    self.assertEqual(
                        entrypoint["argument_passing"], "positional_in_declared_order"
                    )
                    self.assertEqual(
                        [argument["name"] for argument in entrypoint["arguments"]],
                        ["candidate_dir", "contexts_path", "output_dir", "device"],
                    )
                    self.assertEqual(
                        [
                            argument["access"]
                            for argument in entrypoint["arguments"][:3]
                        ],
                        ["read_only", "read_only", "read_write"],
                    )
                    device = entrypoint["arguments"][3]
                    self.assertEqual(device["value_source"], "evaluation.device")
                    self.assertEqual(device["allowed_values"], ["cpu", "cuda"])

    def test_real_offline_runner_matches_the_published_invocation_contract(self):
        config = TaskConfig.load(ROOT / "tests/fixtures/task/task.json")
        entrypoint = config.public_manifest("B")["evaluation"]["candidate_boundary"][
            "entrypoint"
        ]
        for device in entrypoint["arguments"][3]["allowed_values"]:
            with self.subTest(
                device=device
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                candidate = root / "artifact with spaces"
                candidate.mkdir()
                (candidate / entrypoint["module_file"]).write_text(
                    "import json\n"
                    "from pathlib import Path\n"
                    "assert __name__ != '__main__'\n"
                    f"def {entrypoint['function']}(*args):\n"
                    "    observation = {\n"
                    "        'types': ['pathlib.Path' if isinstance(arg, Path)\n"
                    "                  else type(arg).__name__ for arg in args],\n"
                    "        'values': [str(arg) for arg in args],\n"
                    "    }\n"
                    "    (args[2] / 'call.json').write_text(json.dumps(observation))\n"
                    "    return {'ignored': True}\n",
                    encoding="utf-8",
                )
                contexts = root / "contexts.json"
                contexts.write_text("{}\n", encoding="utf-8")
                output = root / "predictions"
                output.mkdir()
                environment = dict(os.environ)
                environment["PYTHONPATH"] = str(ROOT)
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "autotrain_bench.candidate_offline_runner",
                        "--candidate",
                        candidate.name,
                        "--contexts",
                        contexts.name,
                        "--output",
                        output.name,
                        "--device",
                        device,
                    ],
                    cwd=root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                observation = json.loads((output / "call.json").read_text())
                self.assertEqual(
                    observation["types"],
                    [argument["python_type"] for argument in entrypoint["arguments"]],
                )
                self.assertEqual(
                    observation["values"],
                    [str(candidate), str(contexts), str(output), device],
                )


if __name__ == "__main__":
    unittest.main()
