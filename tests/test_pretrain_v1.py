import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autotrain_bench.config import TaskConfig


ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_ROOT = ROOT / "challenges/lm-pretrain-v1"
TASK = PRETRAIN_ROOT / "task.json"
PREPARE = PRETRAIN_ROOT / "prepare_data.py"

SPEC = importlib.util.spec_from_file_location("lm_pretrain_v1_prepare", PREPARE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PretrainV1Tests(unittest.TestCase):
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
                "eval_target_tokens": 1,
                "eval_sequence_targets": 1,
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
                    eval_target_tokens=1,
                    eval_sequence_targets=1,
                    accept_large_download=False,
                    force=True,
                )

    def test_task_uses_fixed_gpt2_ids_and_public_train_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = root / "public"
            private = root / "private"
            public.mkdir()
            private.mkdir()
            val = private / "val.json"
            test = private / "test.json"
            val.write_text('{"sequences":[[50256,1]]}', encoding="utf-8")
            test.write_text('{"sequences":[[50256,2]]}', encoding="utf-8")
            payload = json.loads(TASK.read_text(encoding="utf-8"))
            payload["paths"] = {
                "train": str(public),
                "val": str(val),
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

    def test_preview_evaluation_cap_is_exact(self):
        collector = MODULE.EvaluationCollector(target_tokens=7, sequence_targets=3)
        collector.add_document([50_256, 1, 2, 3, 4, 5, 6, 7, 8])
        self.assertTrue(collector.complete)
        self.assertEqual(collector.token_count, 7)
        self.assertEqual(
            collector.sequences,
            [[50_256, 1, 2, 3], [50_256, 4, 5, 6], [50_256, 7]],
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
            {"text": "vv", "url": "val", "_source_file": "part", "_source_row": 0},
            {"text": "tt", "url": "test", "_source_file": "part", "_source_row": 1},
            {"text": "ab", "url": "train-1", "_source_file": "part", "_source_row": 2},
            {"text": "cd", "url": "train-2", "_source_file": "part", "_source_row": 3},
        ]

        def split(source_key):
            if source_key.endswith(":0"):
                return "val"
            if source_key.endswith(":1"):
                return "test"
            return "train"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "part.parquet").write_bytes(b"source")
            output = root / "generated"
            dependencies = (
                None,
                FakePyArrow(),
                FakeParquet(),
                FakeTiktoken(),
            )
            with patch.object(MODULE, "_load_dependencies", return_value=dependencies):
                with patch.object(MODULE, "iter_documents", return_value=iter(rows)):
                    with patch.object(MODULE, "preview_split", side_effect=split):
                        with patch("builtins.print"):
                            manifest = MODULE.prepare(
                                source_dir=source,
                                output_dir=output,
                                target_train_tokens=4,
                                shard_tokens=4,
                                eval_target_tokens=2,
                                eval_sequence_targets=2,
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
            self.assertTrue((output / "private/val.json").is_file())
            self.assertTrue((output / "private/test.json").is_file())
            self.assertEqual(manifest["raw_train"]["documents"], 2)
            self.assertEqual(manifest["tokenized_train"]["documents"], 2)


if __name__ == "__main__":
    unittest.main()
