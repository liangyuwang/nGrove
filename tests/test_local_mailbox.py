import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from envcore.local_mailbox import LocalFileMailbox, MailboxLimitExceeded


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

    def test_handler_value_error_is_a_trusted_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            mailbox = LocalFileMailbox(Path(temporary) / "control")
            mailbox.initialize()
            (mailbox.inbox / "mutate.json").write_text(
                json.dumps({"op": "mutate"}),
                encoding="utf-8",
            )

            def fail_after_dispatch(request):
                del request
                raise ValueError("trusted handler failed")

            result = mailbox.process_one(fail_after_dispatch)

            self.assertEqual(
                result.infrastructure_failure,
                "trusted handler failed",
            )
            response = json.loads(
                (mailbox.outbox / "mutate.json").read_text(encoding="utf-8")
            )
            self.assertEqual(response["error"]["code"], "infrastructure_failure")

    def test_request_read_io_error_is_a_trusted_transport_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            mailbox = LocalFileMailbox(Path(temporary) / "control")
            mailbox.initialize()
            (mailbox.inbox / "status.json").write_text(
                json.dumps({"op": "status"}),
                encoding="utf-8",
            )

            with patch.object(
                mailbox,
                "read_request",
                side_effect=OSError("private storage path: /organizer/state"),
            ):
                result = mailbox.process_one(lambda request: request)

            self.assertEqual(
                result.infrastructure_failure_code,
                "agent_control_transport_failure",
            )
            self.assertIn("/organizer/state", result.infrastructure_failure)
            response_text = (mailbox.outbox / "status.json").read_text(encoding="utf-8")
            self.assertNotIn("/organizer/state", response_text)
            response = json.loads(response_text)
            self.assertEqual(response["error"]["code"], "infrastructure_failure")

    def test_malformed_json_remains_untrusted_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            mailbox = LocalFileMailbox(Path(temporary) / "control")
            mailbox.initialize()
            (mailbox.inbox / "malformed.json").write_text("{", encoding="utf-8")

            result = mailbox.process_one(lambda request: request)

            self.assertIsNone(result.infrastructure_failure)
            response = json.loads(
                (mailbox.outbox / "malformed.json").read_text(encoding="utf-8")
            )
            self.assertEqual(response["error"]["code"], "invalid_request")

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

    def test_hostile_directory_is_not_traversed_and_is_reported_at_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            mailbox = LocalFileMailbox(
                Path(temporary) / "control",
                maximum_pending_bytes=1,
            )
            mailbox.initialize()
            unseen = mailbox.inbox / "unseen.tmp"
            unseen.mkdir()
            (unseen / "nested").write_bytes(b"0123456789")

            self.assertIsNone(mailbox.next_request())
            cleanup = mailbox.cleanup()

            self.assertTrue(cleanup.complete)
            self.assertEqual(cleanup.agent_policy_violations, 1)
            self.assertEqual(cleanup.infrastructure_failures, 0)


if __name__ == "__main__":
    unittest.main()
