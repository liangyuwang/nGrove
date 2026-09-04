import io
import sys
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

from autotrain_bench import candidate_preflight
from autotrain_bench.candidate_protocol import CandidateProtocolError


class CandidatePreflightTests(unittest.TestCase):
    def test_public_contract_failure_is_one_actionable_line(self):
        candidate = SimpleNamespace(run_offline=lambda *args: None)
        diagnostic = (
            'prediction manifest field "dtype" must be the string constant '
            '"float32-le"'
        )
        stderr = io.StringIO()

        with (
            patch.object(
                sys,
                "argv",
                [
                    "candidate_preflight",
                    "--candidate",
                    "/output/candidate",
                    "--device",
                    "cpu",
                    "--vocab-size",
                    "257",
                    "--context-lengths",
                    "8",
                    "16",
                ],
            ),
            patch.object(candidate_preflight, "load_candidate", return_value=candidate),
            patch.object(
                candidate_preflight,
                "validate_prediction_artifact",
                side_effect=CandidateProtocolError(diagnostic),
            ),
            redirect_stderr(stderr),
        ):
            with self.assertRaises(SystemExit) as raised:
                candidate_preflight.main()

        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(
            stderr.getvalue(), f"candidate preflight failed: {diagnostic}\n"
        )
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
