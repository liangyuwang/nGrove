import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autotrain_bench import training_data
from autotrain_bench.training_data import (
    TrainingDataError,
    verify_training_data,
    write_training_manifest,
)


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "train"
        self.root.mkdir()
        (self.root / "raw").mkdir()
        (self.root / "raw/data").write_bytes(b"raw text")
        (self.root / "tokens").write_bytes(b"\x01\x00\x02\x00")
        (self.root / "documents.jsonl").write_text('{"offset":0}\n')
        self.manifest = self.root / "manifest.json"
        write_training_manifest(self.root, {"source": {"revision": "fixture"}})

    def test_verifies_every_file_and_binds_manifest_without_hidden_data(self):
        # An unrelated, unreadable hidden location is outside the inventory.
        (self.root.parent / "private").symlink_to("missing-hidden-data")
        receipt = verify_training_data(self.root)
        self.assertEqual(receipt["status"], "verified")
        self.assertEqual(receipt["files"], 3)
        self.assertEqual(receipt["bytes"], 25)
        self.assertEqual(
            receipt["manifest_sha256"],
            hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
        )

    def test_rejects_missing_extra_truncated_and_same_size_changed_files(self):
        for name in ("raw/data", "tokens", "documents.jsonl"):
            path = self.root / name
            original = path.read_bytes()
            for mutation in ("missing", "truncated", "same_size"):
                with self.subTest(file=name, mutation=mutation):
                    if mutation == "missing":
                        path.unlink()
                    elif mutation == "truncated":
                        path.write_bytes(original[:-1])
                    else:
                        path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                    with self.assertRaises(TrainingDataError):
                        verify_training_data(self.root)
                    path.write_bytes(original)
        (self.root / "unlisted").write_bytes(b"extra")
        with self.assertRaisesRegex(TrainingDataError, "file set mismatch"):
            verify_training_data(self.root)

    def test_rejects_legacy_manifest_and_unsafe_or_duplicate_inventory(self):
        original = json.loads(self.manifest.read_bytes())
        cases = [{"source": "legacy"}]
        for name in (
            "../private/test",
            "/absolute",
            "a//b",
            "a\\b",
            "C:/data",
            "manifest.json",
        ):
            payload = json.loads(json.dumps(original))
            payload["integrity"]["files"][0]["file"] = name
            cases.append(payload)
        duplicate = json.loads(json.dumps(original))
        duplicate["integrity"]["files"].append(duplicate["integrity"]["files"][0])
        cases.append(duplicate)
        for payload in cases:
            with self.subTest(payload=payload):
                self.manifest.write_text(json.dumps(payload))
                with self.assertRaises(TrainingDataError):
                    verify_training_data(self.root)
        self.manifest.unlink()
        with self.assertRaisesRegex(TrainingDataError, "prepare the data again"):
            verify_training_data(self.root)

    def test_rejects_symlinks_and_hard_links(self):
        path = self.root / "tokens"
        outside = self.root.parent / "outside"
        path.rename(outside)
        for kind in ("symlink", "hardlink"):
            with self.subTest(kind=kind):
                if kind == "symlink":
                    path.symlink_to(outside)
                else:
                    os.link(outside, path)
                with self.assertRaisesRegex(TrainingDataError, "link or special"):
                    verify_training_data(self.root)
                path.unlink()

    def test_detects_changes_to_already_verified_files(self):
        real_identity = training_data.file_identity

        def change_after_hash(path):
            identity = real_identity(path)
            if path.name == "tokens":
                (self.root / "raw/data").write_bytes(b"changed!")
            return identity

        with patch.object(
            training_data, "file_identity", side_effect=change_after_hash
        ):
            with self.assertRaisesRegex(
                TrainingDataError, "changed during verification"
            ):
                verify_training_data(self.root)


if __name__ == "__main__":
    unittest.main()
