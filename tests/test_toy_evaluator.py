import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from autotrain_bench.toy_evaluator import evaluate


ROOT = Path(__file__).resolve().parents[1]


class ToyEvaluatorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

