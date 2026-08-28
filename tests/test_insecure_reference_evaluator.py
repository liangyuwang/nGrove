import importlib.util
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = (
    ROOT / "examples/evaluators/insecure_reference/evaluator.py"
)
SPEC = importlib.util.spec_from_file_location(
    "insecure_reference_evaluator", EVALUATOR
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - repository invariant
    raise RuntimeError(f"cannot import evaluator at {EVALUATOR}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
evaluate = MODULE.evaluate


class InsecureReferenceEvaluatorTests(unittest.TestCase):
    def test_candidate_abi_produces_finite_nll(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory)
            shutil.copy2(
                ROOT / "examples/toy/agent/candidate_template.py",
                candidate / "candidate.py",
            )
            uniform = math.log(1 / 3)
            (candidate / "model.json").write_text(
                json.dumps(
                    {
                        "transition_logits": [
                            [uniform, uniform, uniform],
                            [uniform, uniform, uniform],
                            [uniform, uniform, uniform],
                        ]
                    }
                ),
                encoding="utf-8",
            )
            loss = evaluate(
                candidate, ROOT / "examples/toy/private/val.json", vocab_size=3
            )
            self.assertAlmostEqual(loss, math.log(3), places=7)

    def test_prefixes_are_batched_without_exposing_future_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate"
            candidate.mkdir()
            (candidate / "candidate.py").write_text(
                """import json
from pathlib import Path

def load(candidate_dir, device):
    return {"root": Path(candidate_dir), "device": device}

def next_token_logits(model, input_ids):
    path = model["root"] / "calls.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"device": model["device"], "prefixes": input_ids}) + "\\n")
    return [[[[0.0, 0.0, 0.0] for _ in prefix][index] for index in range(len(prefix))] for prefix in input_ids]
""",
                encoding="utf-8",
            )
            data = root / "data.json"
            data.write_text(
                json.dumps({"sequences": [[0, 1, 2, 0], [2, 1, 0, 2]]}),
                encoding="utf-8",
            )

            loss = evaluate(candidate, data, vocab_size=3, batch_size=3, device="cpu")

            self.assertAlmostEqual(loss, math.log(3), places=7)
            calls = [
                json.loads(line)
                for line in (candidate / "calls.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([len(call["prefixes"]) for call in calls], [3, 3])
            self.assertTrue(all(call["device"] == "cpu" for call in calls))
            self.assertEqual(calls[0]["prefixes"], [[0], [0, 1], [0, 1, 2]])


if __name__ == "__main__":
    unittest.main()
