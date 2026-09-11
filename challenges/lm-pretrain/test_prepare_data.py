"""Challenge-local regressions for source scale and holdouts; no corpus download."""

import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchcore.training_data import verify_training_data


PREPARE = Path(__file__).with_name("prepare_data.py")
SPEC = importlib.util.spec_from_file_location("lm_pretrain_holdout_regression", PREPARE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def collectors(target_examples=MODULE.PREVIEW_EVAL_EXAMPLES):
    return {
        name: MODULE.EvaluationCollector(
            target_examples=target_examples,
            context_lengths=MODULE.EVAL_CONTEXT_LENGTHS,
            boundary_token_id=MODULE.EOT_TOKEN_ID,
        )
        for name in ("analysis", "test")
    }


class PreviewHoldoutTests(unittest.TestCase):
    def test_default_hash_pool_fills_quotas_where_the_old_pool_exhausts(self):
        # Synthetic row identities and document lengths, not FineWeb statistics.
        # Both pools use the actual preview_split and EvaluationCollector code.
        old = collectors()
        current = collectors()
        tokens = [MODULE.EOT_TOKEN_ID] + [1] * 8_192
        for index in range(500_000):
            key = f"000_00000.parquet:{index}"
            split = MODULE.preview_split(key)
            if split == "train":
                continue
            current[split].add_document(key, tokens)
            with patch.object(MODULE, "SPLIT_BUCKETS", 4_096):
                old_split = MODULE.preview_split(key)
            if old_split != "train":
                old[old_split].add_document(key, tokens)

        for name in current:
            self.assertFalse(old[name].complete, name)
            self.assertTrue(current[name].complete, name)
            self.assertEqual(len(current[name].examples), 2_048)
            self.assertEqual(
                current[name].context_length_histogram(),
                {
                    "64": 342,
                    "128": 342,
                    "256": 341,
                    "512": 341,
                    "1024": 341,
                    "2048": 341,
                },
            )
        analysis_sources = {item["source_id"] for item in current["analysis"].examples}
        test_sources = {item["source_id"] for item in current["test"].examples}
        self.assertTrue(analysis_sources.isdisjoint(test_sources))
        MODULE._require_complete_holdouts(current["analysis"], current["test"])
        with self.assertRaises(MODULE.PreparationError):
            MODULE._require_complete_holdouts(old["analysis"], old["test"])

    def test_shortfall_diagnostic_identifies_the_missing_long_context(self):
        groups = collectors(target_examples=6)
        # Earlier short windows consume the only long document. Sufficient
        # document count alone does not establish capacity for every quota.
        for index, length in enumerate(reversed(MODULE.EVAL_CONTEXT_LENGTHS)):
            groups["analysis"].add_document(
                f"analysis-{index}", [MODULE.EOT_TOKEN_ID] + [1] * length
            )
        groups["test"].add_document("test-0", [MODULE.EOT_TOKEN_ID] + [1] * 8_192)

        with self.assertRaises(MODULE.PreparationError) as failure:
            MODULE._require_complete_holdouts(groups["analysis"], groups["test"])
        self.assertIn(
            "source sample did not fill preview ANALYSIS/TEST", str(failure.exception)
        )
        shortfalls = json.loads(str(failure.exception).split("shortfalls=", 1)[1])
        self.assertEqual(
            shortfalls, {"analysis": {"2048": {"actual": 0, "required": 1}}}
        )


def source_manifest(path, files=None):
    payload = {
        "schema_version": 1,
        "dataset": MODULE.DATASET,
        "revision": MODULE.DATASET_REVISION,
        "upstream_sample": MODULE.UPSTREAM_SAMPLE,
        "files": [
            {
                "file": f"{MODULE.UPSTREAM_SAMPLE}/{name}",
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in (files or {"part.parquet": b"source"}).items()
        ],
    }
    path.write_text(json.dumps(payload))
    return path


class PreviewScaleTests(unittest.TestCase):
    def test_builder_cli_defaults_to_full_source_and_accepts_explicit_limit(self):
        for arguments, limit in (([], None), (["--target-train-tokens", "4"], 4)):
            with self.subTest(arguments=arguments):
                with patch("sys.argv", [str(PREPARE), *arguments]):
                    with patch.object(MODULE, "prepare") as prepare:
                        MODULE.main()
                self.assertEqual(prepare.call_args.kwargs["target_train_tokens"], limit)

    def _prepare_fixture(self, root, *, target_train_tokens=None, rows=None):
        class FakeTable:
            @staticmethod
            def from_pylist(records, schema=None):
                del schema
                return list(records)

        class FakePyArrow:
            Table = FakeTable

            @staticmethod
            def string():
                return "string"

            @staticmethod
            def large_string():
                return "large_string"

            @staticmethod
            def float64():
                return "float64"

            @staticmethod
            def field(name, kind, nullable=True):
                return name, kind, nullable

            @staticmethod
            def schema(fields):
                return fields

        class FakeParquetWriter:
            def __init__(self, path, schema, compression):
                del schema, compression
                self.path = path
                self.records = []

            def write_table(self, records):
                self.records.extend(records)

            def close(self):
                self.path.write_text(
                    json.dumps(self.records, sort_keys=True), encoding="utf-8"
                )

        class FakeParquet:
            ParquetWriter = FakeParquetWriter

        class FakeEncoding:
            n_vocab = 50_257
            eot_token = 50_256
            _mergeable_ranks = {b"a": 0, b"b": 1}
            _special_tokens = {"<|endoftext|>": 50_256}

            @staticmethod
            def encode_ordinary(text):
                return [ord(character) for character in text]

        class FakeTiktoken:
            @staticmethod
            def get_encoding(name):
                self.assertEqual(name, "gpt2")
                return FakeEncoding()

        default_rows = [
            {"text": "vv", "url": "analysis", "_source_file": "part", "_source_row": 0},
            {"text": "tt", "url": "test", "_source_file": "part", "_source_row": 1},
            {"text": "ab", "url": "train-1", "_source_file": "part", "_source_row": 2},
            {"text": "cd", "url": "train-2", "_source_file": "part", "_source_row": 3},
        ]

        rows = default_rows if rows is None else rows

        def split(source_key):
            if source_key.endswith(":0"):
                return "analysis"
            if source_key.endswith(":1"):
                return "test"
            return "train"

        source = root / "source"
        source.mkdir()
        (source / "part.parquet").write_bytes(b"source")
        pinned = source_manifest(root / "source-files.json")
        output = root / "generated"
        dependencies = (
            None,
            FakePyArrow(),
            FakeParquet(),
            FakeTiktoken(),
        )
        with patch.object(MODULE, "SOURCE_MANIFEST_PATH", pinned), patch.object(
            MODULE, "_load_dependencies", return_value=dependencies
        ):
            with patch.object(MODULE, "iter_documents", return_value=iter(rows)):
                with patch.object(MODULE, "preview_split", side_effect=split):
                    with patch("builtins.print"):
                        manifest = MODULE.prepare(
                            source_dir=source,
                            output_dir=output,
                            target_train_tokens=target_train_tokens,
                            shard_tokens=4,
                            eval_examples=2,
                            eval_context_lengths=(1,),
                            accept_large_download=False,
                            force=False,
                        )

        return manifest, output

    def test_full_source_below_nominal_scale_records_exact_size_and_completeness(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, output = self._prepare_fixture(Path(directory))
            metadata = manifest["tokenized_train"]
            self.assertEqual(metadata["nominal_scale_tokens"], 10_000_000_000)
            self.assertEqual(metadata["exact_tokens"], 6)
            self.assertIsNone(metadata["requested_token_limit"])
            self.assertTrue(metadata["source_exhausted"])
            self.assertFalse(metadata["train_truncated_by_token_limit"])
            self.assertNotIn("target_scale_reached", metadata)
            self.assertNotIn("target_scale_tokens", metadata)
            for relative in (
                "manifest.json",
                "public/manifest.json",
                "public/tokenized/manifest.json",
            ):
                saved = json.loads((output / relative).read_bytes())
                self.assertEqual(saved["tokenized_train"], metadata)
            verify_training_data(output / "public")

    def test_nominal_scale_does_not_cap_the_default_build(self):
        # Synthetic tokens make crossing the nominal scale cheap to exercise.
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(MODULE, "NOMINAL_TRAIN_TOKENS", 2):
                manifest, _ = self._prepare_fixture(Path(directory))
            metadata = manifest["tokenized_train"]
            self.assertEqual(metadata["exact_tokens"], 6)
            self.assertEqual(metadata["documents"], 2)
            self.assertTrue(metadata["source_exhausted"])
            self.assertFalse(metadata["train_truncated_by_token_limit"])

    def test_explicit_limit_preserves_whole_documents_and_reports_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, output = self._prepare_fixture(
                Path(directory), target_train_tokens=2
            )
            metadata = manifest["tokenized_train"]
            self.assertEqual(metadata["requested_token_limit"], 2)
            self.assertEqual(metadata["exact_tokens"], 3)
            self.assertEqual(metadata["documents"], 1)
            self.assertFalse(metadata["source_exhausted"])
            self.assertTrue(metadata["train_truncated_by_token_limit"])
            documents = [
                json.loads(line)
                for line in (output / "public/tokenized/documents.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual([item["doc_id"] for item in documents], ["part:2"])
            verify_training_data(output / "public")

    def test_source_exhaustion_does_not_hide_rows_skipped_before_holdouts_fill(self):
        rows = [
            {"text": "ab", "_source_file": "part", "_source_row": 2},
            {"text": "cd", "_source_file": "part", "_source_row": 3},
            {"text": "vv", "_source_file": "part", "_source_row": 0},
            {"text": "tt", "_source_file": "part", "_source_row": 1},
        ]
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = self._prepare_fixture(
                Path(directory), target_train_tokens=2, rows=rows
            )
            metadata = manifest["tokenized_train"]
            self.assertEqual(metadata["exact_tokens"], 3)
            self.assertTrue(metadata["source_exhausted"])
            self.assertTrue(metadata["train_truncated_by_token_limit"])
            hidden = manifest["preview_hidden_evaluation"]
            self.assertEqual(hidden["analysis_target_predictions"], 2)
            self.assertEqual(hidden["test_target_predictions"], 2)

    def test_explicit_limit_does_not_imply_truncation_if_all_train_rows_fit(self):
        for limit in (6, 100):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as directory:
                manifest, _ = self._prepare_fixture(
                    Path(directory), target_train_tokens=limit
                )
                metadata = manifest["tokenized_train"]
                self.assertEqual(metadata["exact_tokens"], 6)
                self.assertEqual(metadata["requested_token_limit"], limit)
                self.assertTrue(metadata["source_exhausted"])
                self.assertFalse(metadata["train_truncated_by_token_limit"])

    def test_incomplete_holdouts_still_fail_without_a_train_token_limit(self):
        rows = [{"text": "ab", "_source_file": "part", "_source_row": 2}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(MODULE.PreparationError, "did not fill"):
                self._prepare_fixture(root, rows=rows)
            self.assertFalse((root / "generated").exists())
            self.assertEqual(list(root.glob(".generated.stage-*")), [])

    def test_explicit_token_limit_must_remain_a_positive_integer(self):
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(MODULE.PreparationError, "positive"):
                    self._prepare_fixture(Path(directory), target_train_tokens=limit)


if __name__ == "__main__":
    unittest.main()
