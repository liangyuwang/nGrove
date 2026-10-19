import hashlib
import tempfile
import unittest
from pathlib import Path

from envcore.artifact_identity import (
    HIDDEN_BUNDLE_FILENAMES,
    HIDDEN_BUNDLE_IDENTITY_ALGORITHM,
    hidden_bundle_sha256,
)


class ArtifactIdentityTests(unittest.TestCase):
    def test_hidden_bundle_v1_matches_the_normative_byte_grammar(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "hidden"
            bundle.mkdir()
            payloads = {
                "contexts.json": b'{"visible":true}\n',
                "targets.json": b'{"hidden":true}\n',
            }
            for name, payload in payloads.items():
                (bundle / name).write_bytes(payload)

            expected = hashlib.sha256()
            expected.update(b"envcore-hidden-bundle-v1\0")
            for name in HIDDEN_BUNDLE_FILENAMES:
                encoded_name = name.encode("utf-8")
                payload = payloads[name]
                expected.update(len(encoded_name).to_bytes(8, "big"))
                expected.update(encoded_name)
                expected.update(len(payload).to_bytes(8, "big"))
                expected.update(payload)

            self.assertEqual(
                HIDDEN_BUNDLE_IDENTITY_ALGORITHM,
                "envcore-hidden-bundle-v1-sha256",
            )
            self.assertEqual(hidden_bundle_sha256(bundle), expected.hexdigest())

    def test_hidden_bundle_rejects_extra_entries_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "hidden"
            bundle.mkdir()
            (bundle / "contexts.json").write_text("{}", encoding="utf-8")
            (bundle / "targets.json").write_text("{}", encoding="utf-8")
            (bundle / "extra").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(OSError, "contain only"):
                hidden_bundle_sha256(bundle)

            (bundle / "extra").unlink()
            (bundle / "contexts.json").unlink()
            (bundle / "contexts.json").symlink_to(bundle / "targets.json")
            with self.assertRaisesRegex(OSError, "regular file"):
                hidden_bundle_sha256(bundle)


if __name__ == "__main__":
    unittest.main()
