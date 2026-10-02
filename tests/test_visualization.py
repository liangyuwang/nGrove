import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.test_results import _finalized


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "visualization" / "build.py"
spec = importlib.util.spec_from_file_location("visualization_build", SCRIPT)
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


def saved_result(**overrides):
    return _finalized(
        track="A",
        task_id="synthetic-test-task",
        comparison_key={
            "challenge_id": "synthetic-test-release",
            "track": "A",
            "track_config": "synthetic-a",
            "runtime_profile": "synthetic-cpu",
        },
        artifact_identities={"hidden_evaluation": {"test_sha256": "synthetic-test"}},
        agent_control={"cleanup_complete": True},
        agent_system={
            "termination_reason": "agent_finished",
            "command": ["do-not-export-this-command"],
        },
        **overrides,
    )


class VisualizationTests(unittest.TestCase):
    def test_demo_build_is_offline_and_preserves_missing_points(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site with spaces"
            subprocess.run(
                [sys.executable, str(SCRIPT), "--demo", "--output-dir", str(output)],
                cwd=directory,
                check=True,
                capture_output=True,
            )
            data = json.loads((output / "data.json").read_text())
            self.assertTrue(data["synthetic"])
            b = next(row for row in data["rows"] if row["track"] == "B")
            self.assertEqual(len(b["points"]), 3)
            self.assertIsNone(b["points"][1]["nll"])
            self.assertIsNone(b["score"])
            html = (output / "index.html").read_text()
            self.assertIn('href="data.json"', html)
            self.assertNotIn("__RSI_DATA__", html)
            self.assertNotIn("xiehuanyi", html)

    def test_native_result_export_is_read_only_and_omits_private_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(json.dumps(saved_result()))
            before = path.read_bytes()
            data = viewer.report_data(
                [path], ["Arbitrary Agent </script><script>bad()</script>"]
            )
            row = data["rows"][0]
            self.assertEqual(row["score"], 100)
            self.assertEqual(row["points"][0]["nll"], 0)
            self.assertIsNone(row["points"][0]["leases_used"])
            self.assertIsNone(row["total_tokens"])
            self.assertNotIn("do-not-export-this-command", json.dumps(data))
            output = viewer.build(data, Path(directory) / "output", [path])
            self.assertNotIn(
                "</script><script>bad()", (output / "index.html").read_text()
            )
            self.assertEqual(before, path.read_bytes())

    def test_failed_final_result_is_retained_without_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed.json"
            path.write_text(
                json.dumps(
                    saved_result(
                        outcome="failed",
                        test_status="failed",
                        failure="invalid_metric",
                        test_cost=None,
                        test_raw_metrics=None,
                        test_performance_score=None,
                    )
                )
            )
            row = viewer.report_data([path], None)["rows"][0]
            self.assertEqual(row["outcome"], "failed")
            self.assertIsNone(row["test_nll"])
            self.assertIsNone(row["score"])

    def test_missing_and_null_display_metadata_can_be_combined(self):
        with tempfile.TemporaryDirectory() as directory:
            first = saved_result()
            first.pop("task_id")
            first["comparison_key"].pop("runtime_profile")
            second = copy.deepcopy(first)
            second["track"] = "B"
            second["comparison_key"]["track"] = "B"
            second["comparison_key"]["track_config"] = "synthetic-b"
            second["comparison_key"]["runtime_profile"] = None
            second["task_id"] = None
            paths = [Path(directory) / "missing.json", Path(directory) / "null.json"]
            for path, result in zip(paths, [first, second]):
                path.write_text(json.dumps(result))
            data = viewer.report_data(paths, None)
            self.assertEqual(len(data["rows"]), 2)
            self.assertEqual(data["task"], "Not recorded")
            self.assertEqual(data["runtime"], "Not recorded")

    def test_saved_core_revision_survives_page_export(self):
        with tempfile.TemporaryDirectory() as directory:
            result = saved_result()
            result["artifact_identities"]["core_git"] = {
                "revision": "a" * 40,
                "working_directory": "do-not-export-this-path",
            }
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(result))
            data = viewer.report_data([path], None)
            output = viewer.build(data, Path(directory) / "site", [path])
            exported = json.loads((output / "data.json").read_text())
            self.assertEqual(exported["core_revision"], "a" * 40)
            self.assertNotIn("do-not-export-this-path", json.dumps(exported))

    def test_demo_has_no_claim_of_a_recorded_core_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            source = ROOT / "examples/research-dynamics/fixtures/demo.json"
            output = viewer.build(viewer.demo_data(source), Path(directory), [source])
            data = json.loads((output / "data.json").read_text())
            self.assertEqual(data["core_revision"], "Not recorded")

    def test_unconfirmed_cleanup_and_old_identity_are_rejected(self):
        for change in ("cleanup", "identity"):
            with self.subTest(
                change=change
            ), tempfile.TemporaryDirectory() as directory:
                result = saved_result()
                if change == "cleanup":
                    result["agent_control"]["cleanup_complete"] = False
                else:
                    result["artifact_identity_algorithms"]["candidate_tree"] = (
                        "old-tree-format"
                    )
                path = Path(directory) / "state.json"
                path.write_text(json.dumps(result))
                with self.assertRaises(ValueError):
                    viewer.report_data([path], None)

    def test_mixed_or_incomplete_comparison_conditions_are_rejected(self):
        for change in ("runtime", "missing"):
            with self.subTest(
                change=change
            ), tempfile.TemporaryDirectory() as directory:
                first = saved_result()
                second = copy.deepcopy(first)
                if change == "runtime":
                    second["comparison_key"]["runtime_profile"] = "other-hardware"
                else:
                    first["comparison_key"].pop("challenge_id")
                    second["comparison_key"].pop("challenge_id")
                paths = [Path(directory) / "one.json", Path(directory) / "two.json"]
                for path, result in zip(paths, [first, second]):
                    path.write_text(json.dumps(result))
                with self.assertRaises(ValueError):
                    viewer.report_data(paths, None)

    def test_output_cannot_overlap_source_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.json"
            source.write_text("{}")
            with self.assertRaises(ValueError):
                viewer.build({}, Path(directory), [source])
            self.assertEqual(source.read_text(), "{}")


if __name__ == "__main__":
    unittest.main()
