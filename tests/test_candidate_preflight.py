import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from autotrain_bench import candidate_preflight
from autotrain_bench.candidate_protocol import CandidateProtocolError


class CandidatePreflightTests(unittest.TestCase):
    def test_scratch_inside_candidate_is_rejected_before_recursive_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            with self.assertRaisesRegex(
                CandidateProtocolError, "outside the Candidate"
            ):
                candidate_preflight._copy_read_only_candidate(source, source / "copy")
            self.assertFalse((source / "copy").exists())

    def test_public_contract_failure_is_one_actionable_line(self):
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
            patch.object(
                candidate_preflight,
                "check_candidate",
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

    def test_synthetic_inputs_are_reproducible_balanced_and_varied(self):
        kwargs = dict(
            vocab_size=257, context_lengths=(8, 16, 32, 64, 128), examples=512, seed=0
        )
        request = candidate_preflight.synthetic_request(**kwargs)
        self.assertEqual(request, candidate_preflight.synthetic_request(**kwargs))
        self.assertNotEqual(
            request, candidate_preflight.synthetic_request(**{**kwargs, "seed": 1})
        )
        rows = request["examples"]
        self.assertEqual(len(rows), 512)
        lengths = [len(row["input_ids"]) for row in rows]
        self.assertNotEqual(lengths, sorted(lengths))
        self.assertLessEqual(
            max(lengths.count(n) for n in set(lengths))
            - min(lengths.count(n) for n in set(lengths)),
            1,
        )
        self.assertEqual(
            {token for row in rows for token in row["input_ids"]}, set(range(257))
        )
        self.assertTrue(all(set(row) == {"example_id", "input_ids"} for row in rows))

    def _run_candidate(self, source, *, examples=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "original"
            candidate.mkdir()
            (candidate / "candidate.py").write_text(source, encoding="utf-8")
            scratch = root / "scratch"
            scratch.mkdir()
            environment = dict(os.environ)
            environment["AUTOTRAIN_SCRATCH_DIR"] = str(scratch)
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
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
                    "257",
                    "--context-lengths",
                    "8",
                    "16",
                    "32",
                    "64",
                    "128",
                ]
                + ([] if examples is None else ["--examples", str(examples)]),
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(
                (candidate / "candidate.py").read_text(encoding="utf-8"), source
            )
            self.assertTrue(candidate.stat().st_mode & 0o200)
            self.assertEqual(
                {entry.name for entry in candidate.iterdir()}, {"candidate.py"}
            )
            self.assertEqual(list(scratch.iterdir()), [])
            return result

    def test_real_subprocess_uses_artifact_cwd_readonly_copy_and_public_argv(self):
        source = """
import os, stat, sys
from pathlib import Path
from autotrain_bench.candidate_protocol import load_context_request, write_prediction_artifact
def run_offline(candidate_dir, contexts_path, output_dir, device):
    assert Path.cwd() == candidate_dir
    assert "--contexts" in sys.argv and "--output" in sys.argv
    assert not (candidate_dir.stat().st_mode & 0o222)
    assert not ((candidate_dir / "candidate.py").stat().st_mode & 0o222)
    request = load_context_request(str(contexts_path))
    print("participant debug output")
    write_prediction_artifact(str(output_dir), dataset_id=request["dataset_id"],
        logits=([0.0] * 257 for _ in request["examples"]),
        examples=len(request["examples"]), vocab_size=257)
"""
        for examples, expected in ((None, 512), (32, 32)):
            with self.subTest(examples=examples):
                result = self._run_candidate(source, examples=examples)
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = json.loads(result.stdout)
                self.assertEqual(payload["examples"], expected)
                self.assertEqual(payload["status"], "public_abi_passed")
                self.assertFalse(payload["hidden_evaluation_guaranteed"])
                self.assertIn("candidate_image", payload["not_checked"])
                self.assertIn("participant debug output", result.stderr)

    def test_large_batch_failure_preserves_public_input_traceback(self):
        result = self._run_candidate(
            """
from autotrain_bench.candidate_protocol import load_context_request
def run_offline(candidate_dir, contexts_path, output_dir, device):
    request = load_context_request(contexts_path)
    assert len(request["examples"]) < 512, "batch too large"
""",
            examples=512,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("batch too large", result.stderr)
        self.assertIn("offline Candidate process exited with status", result.stderr)

    @unittest.skipIf(
        os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
        "mode-bit write rejection requires an unprivileged POSIX user",
    )
    def test_writing_into_candidate_copy_fails_without_changing_original(self):
        result = self._run_candidate("""
def run_offline(candidate_dir, contexts_path, output_dir, device):
    (candidate_dir / "cache.bin").write_bytes(b"not allowed")
""")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PermissionError", result.stderr)


if __name__ == "__main__":
    unittest.main()
