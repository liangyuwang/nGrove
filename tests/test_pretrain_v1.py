import importlib.util
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchcore.config import TaskConfig
from benchcore.training_data import verify_training_data


ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_ROOT = ROOT / "challenges/lm-pretrain-v1"
TASK = PRETRAIN_ROOT / "task.json"
PREPARE = PRETRAIN_ROOT / "prepare_data.py"

SPEC = importlib.util.spec_from_file_location("lm_pretrain_v1_prepare", PREPARE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def recognized_manifest():
    return {
        "builder": {"id": MODULE.BUILDER_ID},
        "source": {
            "dataset": MODULE.DATASET,
            "revision": MODULE.DATASET_REVISION,
            "upstream_sample": MODULE.UPSTREAM_SAMPLE,
        },
        "tokenizer": {
            "name": MODULE.TOKENIZER,
            "vocab_size": MODULE.VOCAB_SIZE,
            "document_separator_token_id": MODULE.EOT_TOKEN_ID,
        },
        "layout": {},
        "tokenized_train": {},
    }


def source_manifest(path, files=None):
    payload = {
        "schema_version": 1,
        "dataset": MODULE.DATASET,
        "revision": MODULE.DATASET_REVISION,
        "upstream_sample": MODULE.UPSTREAM_SAMPLE,
        "files": [
            {
                "file": f"sample/10BT/{name}",
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in (files or {"part.parquet": b"source"}).items()
        ],
    }
    path.write_text(json.dumps(payload))
    return path


class PretrainV1Tests(unittest.TestCase):
    def test_pinned_manifest_covers_all_upstream_shards(self):
        payload = json.loads(MODULE.SOURCE_MANIFEST_PATH.read_bytes())
        self.assertEqual(payload["revision"], MODULE.DATASET_REVISION)
        self.assertEqual(
            [item["file"] for item in payload["files"]],
            [f"sample/10BT/{index:03d}_00000.parquet" for index in range(14)],
        )

    def test_local_source_requires_exact_pinned_bytes_and_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            pinned = source_manifest(root / "source-files.json")
            path = source / "part.parquet"
            with patch.object(MODULE, "SOURCE_MANIFEST_PATH", pinned):
                for layout in (source, source / "sample/10BT"):
                    layout.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        path.unlink()
                    path = layout / "part.parquet"
                    path.write_bytes(b"source")
                    files, receipt, _ = MODULE.verify_source(source)
                    self.assertEqual(files, [path])
                    self.assertEqual(receipt["status"], "verified")
                for content in (b"sourc", b"SOURCE"):
                    path.write_bytes(content)
                    with self.assertRaisesRegex(MODULE.PreparationError, "mismatch"):
                        MODULE.verify_source(source)
                path.write_bytes(b"source")
                extra = source / "extra.parquet"
                extra.write_bytes(b"source")
                with self.assertRaisesRegex(MODULE.PreparationError, "unexpected"):
                    MODULE.verify_source(source)
                extra.unlink()
                duplicate = source / "part.parquet"
                duplicate.write_bytes(b"source")
                with self.assertRaisesRegex(MODULE.PreparationError, "duplicate"):
                    MODULE.verify_source(source)
                duplicate.unlink()
                path.unlink()
                with self.assertRaises(MODULE.PreparationError):
                    MODULE.verify_source(source)

    def test_bad_source_fails_before_loading_dependencies_or_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "part.parquet").write_bytes(b"SOURCE")
            pinned = source_manifest(root / "source-files.json")
            with patch.object(MODULE, "SOURCE_MANIFEST_PATH", pinned):
                with patch.object(MODULE, "_load_dependencies") as dependencies:
                    with self.assertRaisesRegex(MODULE.PreparationError, "checksum"):
                        MODULE.prepare(
                            source_dir=source,
                            output_dir=root / "output",
                            target_train_tokens=1,
                            shard_tokens=1,
                            eval_examples=1,
                            eval_context_lengths=(1,),
                            accept_large_download=False,
                            force=False,
                        )
                    dependencies.assert_not_called()
            self.assertEqual(
                {p.name for p in root.iterdir()}, {"source", "source-files.json"}
            )

    def test_force_refuses_unrecognized_or_overlapping_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            victim = root / "victim"
            victim.mkdir()
            (victim / "important.txt").write_text("keep", encoding="utf-8")
            arguments = {
                "source_dir": source,
                "target_train_tokens": 1,
                "shard_tokens": 1,
                "eval_examples": 1,
                "eval_context_lengths": (1,),
                "accept_large_download": False,
                "force": True,
            }
            with self.assertRaisesRegex(MODULE.PreparationError, "recognizable"):
                MODULE.prepare(output_dir=victim, **arguments)
            self.assertEqual(
                (victim / "important.txt").read_text(encoding="utf-8"), "keep"
            )

            with self.assertRaisesRegex(MODULE.PreparationError, "overlap"):
                MODULE.prepare(output_dir=source, **arguments)

            download_arguments = dict(arguments)
            download_arguments.pop("source_dir")
            with patch.object(MODULE, "DEFAULT_RAW_DIR", source):
                with self.assertRaisesRegex(MODULE.PreparationError, "overlap"):
                    MODULE.prepare(
                        source_dir=None,
                        output_dir=source,
                        **download_arguments,
                    )

    def test_output_target_may_not_be_a_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            target = root / "target"
            target.mkdir()
            link = root / "generated"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(MODULE.PreparationError, "symbolic link"):
                MODULE.prepare(
                    source_dir=source,
                    output_dir=link,
                    target_train_tokens=1,
                    shard_tokens=1,
                    eval_examples=1,
                    eval_context_lengths=(1,),
                    accept_large_download=False,
                    force=True,
                )

    def test_force_recognizer_requires_the_exact_task_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "generated"
            target.mkdir()
            manifest = {
                "builder": {"id": MODULE.BUILDER_ID},
                "source": {
                    "dataset": MODULE.DATASET,
                    "revision": "not-the-pinned-revision",
                    "upstream_sample": MODULE.UPSTREAM_SAMPLE,
                },
                "tokenizer": {
                    "name": MODULE.TOKENIZER,
                    "vocab_size": MODULE.VOCAB_SIZE,
                    "document_separator_token_id": MODULE.EOT_TOKEN_ID,
                },
                "layout": {},
                "tokenized_train": {},
            }
            (target / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (target / "important.txt").write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(
                MODULE.PreparationError, "belongs to another dataset build"
            ):
                MODULE._validate_existing_output(target)

            self.assertEqual(
                (target / "important.txt").read_text(encoding="utf-8"), "keep"
            )

    def test_force_identity_covers_every_directory_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "generated"
            target.mkdir()
            (target / "manifest.json").write_text(
                json.dumps(recognized_manifest()), encoding="utf-8"
            )
            before = MODULE._validate_existing_output(target)

            (target / "concurrent-file").write_text("new", encoding="utf-8")
            after_addition = MODULE._validate_existing_output(target)
            (target / "concurrent-file").write_text("old", encoding="utf-8")
            after_same_size_change = MODULE._validate_existing_output(target)

            self.assertNotEqual(before, after_addition)
            self.assertNotEqual(after_addition, after_same_size_change)

    def test_force_replacement_restores_the_old_build_on_install_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "generated"
            target.mkdir()
            (target / "manifest.json").write_text(
                json.dumps(recognized_manifest()), encoding="utf-8"
            )
            (target / "old-data").write_text("preserve", encoding="utf-8")
            expected = MODULE._validate_existing_output(target)
            staging = root / ".generated.stage-test"
            staging.mkdir()
            (staging / "new-data").write_text("new", encoding="utf-8")
            real_replace = MODULE.os.replace

            def fail_install(source, destination):
                if Path(source) == staging:
                    raise OSError("simulated install failure")
                return real_replace(source, destination)

            with patch.object(MODULE.os, "replace", side_effect=fail_install):
                with self.assertRaisesRegex(OSError, "simulated install failure"):
                    MODULE._replace_output_directory(
                        staging=staging,
                        output_dir=target,
                        expected_identity=expected,
                    )

            self.assertEqual(
                (target / "old-data").read_text(encoding="utf-8"), "preserve"
            )
            self.assertTrue(staging.is_dir())
            self.assertEqual(list(root.glob(".generated.replaced-*")), [])

    def test_task_uses_fixed_gpt2_ids_and_public_train_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = root / "public"
            private = root / "private"
            public.mkdir()
            private.mkdir()
            analysis = private / "analysis"
            test = private / "test"
            analysis.mkdir()
            test.mkdir()
            payload = json.loads(TASK.read_text(encoding="utf-8"))
            payload["paths"] = {
                "train": str(public),
                "analysis": str(analysis),
                "test": str(test),
            }
            payload["evaluation"]["entrypoint"] = str(
                PRETRAIN_ROOT / "evaluator/evaluator.py"
            )
            task_path = root / "task.json"
            task_path.write_text(json.dumps(payload), encoding="utf-8")
            task = TaskConfig.load(task_path)
            self.assertEqual(task.task_id, "lm-pretrain-v1")
            self.assertEqual(task.task_kind, "benchmark_challenge")
            self.assertEqual(task.vocab_size, 50_257)
            self.assertEqual(task.context_length, 2_048)
            self.assertEqual(
                task.context_lengths,
                (64, 128, 256, 512, 1_024, 2_048),
            )
            self.assertEqual(task.train_dir, public.resolve())

    def test_uint16_shards_preserve_document_spans(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = MODULE.TokenShardWriter(
                Path(directory) / "tokenized", maximum_tokens=3
            )
            segments = writer.add_document([50_256, 1, 2, 3, 4])
            shards = writer.close()
            self.assertEqual(
                segments,
                [
                    {"shard": 0, "token_offset": 0, "token_count": 3},
                    {"shard": 1, "token_offset": 0, "token_count": 2},
                ],
            )
            first = Path(directory) / "tokenized/tokens-00000.bin"
            second = Path(directory) / "tokenized/tokens-00001.bin"
            self.assertEqual(struct.unpack("<3H", first.read_bytes()), (50_256, 1, 2))
            self.assertEqual(struct.unpack("<2H", second.read_bytes()), (3, 4))
            self.assertEqual([shard["tokens"] for shard in shards], [3, 2])

    def test_preview_evaluation_is_balanced_and_non_overlapping(self):
        collector = MODULE.EvaluationCollector(
            target_examples=5,
            context_lengths=(1, 3),
            boundary_token_id=50_256,
        )
        collector.add_document("doc-1", [50_256, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertTrue(collector.complete)
        self.assertEqual(
            collector.context_length_histogram(),
            {"1": 3, "3": 2},
        )
        self.assertEqual(
            [example["target"] for example in collector.examples],
            [1, 4, 5, 8, 9],
        )
        self.assertEqual(
            [
                (example["source_start"], example["source_end"])
                for example in collector.examples
            ],
            [(0, 1), (1, 4), (4, 5), (5, 8), (8, 9)],
        )

    def test_builder_keeps_raw_and_tokenized_train_but_not_holdouts(self):
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

        rows = [
            {"text": "vv", "url": "analysis", "_source_file": "part", "_source_row": 0},
            {"text": "tt", "url": "test", "_source_file": "part", "_source_row": 1},
            {"text": "ab", "url": "train-1", "_source_file": "part", "_source_row": 2},
            {"text": "cd", "url": "train-2", "_source_file": "part", "_source_row": 3},
        ]

        def split(source_key):
            if source_key.endswith(":0"):
                return "analysis"
            if source_key.endswith(":1"):
                return "test"
            return "train"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
                                target_train_tokens=4,
                                shard_tokens=4,
                                eval_examples=2,
                                eval_context_lengths=(1,),
                                accept_large_download=False,
                                force=False,
                            )

            raw_path = output / "public/raw/documents-00000.parquet"
            raw_records = json.loads(raw_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [record["url"] for record in raw_records],
                ["train-1", "train-2"],
            )
            self.assertTrue((output / "public/tokenized/tokens-00000.bin").is_file())
            for split in ("analysis", "test"):
                self.assertTrue((output / f"private/{split}/contexts.json").is_file())
                self.assertTrue((output / f"private/{split}/targets.json").is_file())
            self.assertEqual(manifest["raw_train"]["documents"], 2)
            self.assertEqual(manifest["tokenized_train"]["documents"], 2)
            verified = verify_training_data(output / "public")
            self.assertEqual(verified["files"], 5)
            public = json.loads((output / "public/manifest.json").read_bytes())
            self.assertEqual(public["source"]["verification"]["status"], "verified")
            self.assertNotIn("preview_hidden_evaluation", public)
            self.assertFalse(
                any("private" in item["file"] for item in public["integrity"]["files"])
            )


if __name__ == "__main__":
    unittest.main()
