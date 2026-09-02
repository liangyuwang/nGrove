import json
import tempfile
import unittest
from pathlib import Path

from autotrain_bench.local_mailbox import LocalFileMailbox, MailboxLimitExceeded


class LocalFileMailboxTests(unittest.TestCase):
    def test_dispatches_to_a_semantic_handler_without_a_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            mailbox = LocalFileMailbox(Path(temporary) / "control")
            mailbox.initialize()
            (mailbox.inbox / "status.json").write_text(
                json.dumps({"op": "status"}),
                encoding="utf-8",
            )

            result = mailbox.process_one(lambda request: {"echo": request["op"]})

            self.assertTrue(result.handled)
            response = json.loads(
                (mailbox.outbox / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(response, {"ok": True, "result": {"echo": "status"}})

    def test_pending_entry_limit_covers_inbox_and_outbox(self):
        with tempfile.TemporaryDirectory() as temporary:
            mailbox = LocalFileMailbox(
                Path(temporary) / "control",
                maximum_pending_entries=1,
            )
            mailbox.initialize()
            (mailbox.inbox / "request.tmp").write_text("", encoding="utf-8")
            (mailbox.outbox / "response.tmp").write_text("", encoding="utf-8")

            with self.assertRaisesRegex(MailboxLimitExceeded, "too many"):
                mailbox.next_request()

    def test_pending_byte_limit_covers_inbox_and_outbox(self):
        with tempfile.TemporaryDirectory() as temporary:
            mailbox = LocalFileMailbox(
                Path(temporary) / "control",
                maximum_pending_bytes=3,
            )
            mailbox.initialize()
            (mailbox.inbox / "request.tmp").write_bytes(b"ab")
            (mailbox.outbox / "response.tmp").write_bytes(b"cd")

            with self.assertRaisesRegex(MailboxLimitExceeded, "byte limit"):
                mailbox.next_request()


if __name__ == "__main__":
    unittest.main()
