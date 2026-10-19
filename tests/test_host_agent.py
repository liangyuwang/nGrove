import errno
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from envcore.host_agent import (
    _BoundedLogSink,
    start_host_agent_process,
    summarize_host_agent_trace,
)


class FailingLogStream:
    """Inject filesystem errors without filling the host's actual filesystem."""

    def __init__(self, stream, operation, error_number):
        self.stream = stream
        self.operation = operation
        self.error_number = error_number
        self.armed = threading.Event()

    @property
    def closed(self):
        return self.stream.closed

    def fail(self, operation):
        if self.armed.is_set() and self.operation == operation:
            raise OSError(self.error_number, "injected log filesystem error")

    def write(self, data):
        self.fail("write")
        if self.armed.is_set() and self.operation == "short_write":
            return self.stream.write(data[: len(data) // 2])
        return self.stream.write(data)

    def flush(self):
        self.fail("flush")
        self.stream.flush()

    def close(self):
        self.stream.close()
        self.fail("close_sink")


class HostAgentLogTests(unittest.TestCase):
    def test_missing_log_is_marked_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "missing.log"
            usage = summarize_host_agent_trace(trace)
            self.assertFalse(usage["usage_complete"])
            self.assertEqual(usage["total_tokens"], 0)
            self.assertEqual(usage["malformed_records"], 0)
            self.assertFalse(trace.exists())

    def test_storage_failure_drains_real_pipes_and_marks_lost_usage(self):
        for channel in ("stdout", "stderr"):
            for operation, number in (
                ("write", errno.ENOSPC),
                ("flush", getattr(errno, "EDQUOT", 122)),
                ("close_sink", errno.EIO),
                ("short_write", errno.EIO),
            ):
                with self.subTest(
                    channel=channel, operation=operation
                ), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    streams = []

                    def make_sink(path):
                        sink = _BoundedLogSink(path)
                        if path.name == f"{channel}.log":
                            sink.stream = FailingLogStream(
                                sink.stream, operation, number
                            )
                            streams.append(sink.stream)
                        return sink

                    with patch("envcore.host_agent._BoundedLogSink", make_sink):
                        handle = self.start(
                            root,
                            "import json, sys, time\n"
                            "from pathlib import Path\n"
                            "record = json.dumps({'type': 'turn.completed', 'usage': "
                            "{'input_tokens': 20, 'output_tokens': 5}}) + '\\n'\n"
                            "sys.stdout.write(record); sys.stdout.flush()\n"
                            "sys.stderr.write('ready\\n'); sys.stderr.flush()\n"
                            "while not Path('release').exists(): time.sleep(0.01)\n"
                            f"sys.{channel}.write(record * 4096); sys.{channel}.flush()\n"
                            "Path('done').touch()\n",
                        )
                    try:
                        self.wait_for(
                            lambda: handle.stdout_path.stat().st_size > 0
                            and handle.stderr_path.stat().st_size > 0
                        )
                        # A valid early usage record does not prove a running
                        # process's transcript has finished capture.
                        self.assertFalse(handle.summarize_usage()["usage_complete"])
                        streams[0].armed.set()
                        (root / "release").touch()
                        handle.process.wait(timeout=5)
                        self.assertEqual(handle.returncode, 0)
                        self.assertTrue((root / "done").exists())
                    finally:
                        handle.stop()

                    summary = handle.summarize_usage()
                    self.assertEqual(summary["usage_complete"], channel == "stderr")
                    self.assertGreaterEqual(summary["total_tokens"], 25)
                    capture = summary["log_capture"][channel]
                    self.assertTrue(capture["finished"])
                    self.assertTrue(capture["eof"])
                    self.assertFalse(capture["complete"])
                    self.assertFalse(capture["truncated"])
                    self.assertEqual(
                        capture["error"],
                        {
                            "operation": "write"
                            if operation == "short_write"
                            else operation,
                            "error_type": "OSError",
                            "errno": number,
                        },
                    )
                    other = "stdout" if channel == "stderr" else "stderr"
                    self.assertTrue(summary["log_capture"][other]["complete"])
                    self.assertTrue(handle._closed)
                    self.assertTrue(handle.process.stdout.closed)
                    self.assertTrue(handle.process.stderr.closed)
                    self.assertTrue(handle._stdout_sink.stream.closed)
                    self.assertTrue(handle._stderr_sink.stream.closed)
                    # Some failures leave a perfectly parseable prefix (or
                    # fail only on close). File-only parsing cannot see loss.
                    if channel == "stdout" and operation in ("write", "close_sink"):
                        self.assertTrue(
                            summarize_host_agent_trace(handle.stdout_path)[
                                "usage_complete"
                            ]
                        )

    def test_missing_truncation_marker_still_makes_capture_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "stdout.log"
            record = b'{"type":"turn.completed","usage":{"input_tokens":20}}\n'
            sink = _BoundedLogSink(trace)
            sink.limit = len(record)
            stream = FailingLogStream(sink.stream, "write", errno.ENOSPC)
            sink.stream = stream

            class Source(io.BytesIO):
                def read1(self, size):
                    if self.tell():
                        stream.armed.set()
                    return super().read1(len(record))

            sink.drain(Source(record * 3))
            self.assertEqual(trace.read_bytes(), record)
            self.assertTrue(sink.eof)
            self.assertTrue(sink.truncated)
            self.assertEqual(sink.error["errno"], errno.ENOSPC)
            self.assertFalse(sink.capture_status()["complete"])

    def test_read_and_close_errors_are_retained_and_both_resources_close(self):
        for operation in ("read", "close_source"):
            with self.subTest(
                operation=operation
            ), tempfile.TemporaryDirectory() as directory:
                sink = _BoundedLogSink(Path(directory) / "stdout.log")

                class Source(io.BytesIO):
                    def read1(self, size):
                        if operation == "read":
                            raise OSError(errno.EIO, "injected pipe failure")
                        return super().read1(size)

                    def close(self):
                        super().close()
                        raise OSError(errno.EBADF, "injected close failure")

                source = Source(b"{}\n")
                sink.drain(source)
                self.assertEqual(sink.error["operation"], operation)
                self.assertFalse(sink.capture_status()["complete"])
                self.assertTrue(source.closed)
                self.assertTrue(sink.stream.closed)

    def test_exited_process_usage_remains_incomplete_until_capture_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            closing = threading.Event()
            release = threading.Event()

            def make_sink(path):
                sink = _BoundedLogSink(path)
                if path.name == "stdout.log":
                    close = sink.close

                    def delayed_close(source=None):
                        closing.set()
                        release.wait(timeout=5)
                        close(source)

                    sink.close = delayed_close
                return sink

            with patch("envcore.host_agent._BoundedLogSink", make_sink):
                handle = self.start(
                    root,
                    'print(\'{"type":"turn.completed","usage":{"input_tokens":20}}\')',
                )
            try:
                handle.process.wait(timeout=5)
                self.assertTrue(closing.wait(timeout=5))
                self.assertFalse(handle.running)
                self.assertTrue(
                    summarize_host_agent_trace(handle.stdout_path)["usage_complete"]
                )
                summary = handle.summarize_usage()
                self.assertFalse(summary["usage_complete"])
                self.assertFalse(summary["log_capture"]["stdout"]["finished"])
                self.assertTrue(summary["log_capture"]["stdout"]["eof"])
            finally:
                release.set()
                handle.stop()
            self.assertTrue(handle.summarize_usage()["usage_complete"])

    def test_launch_failure_closes_both_logs_despite_first_close_error(self):
        with tempfile.TemporaryDirectory() as directory:
            sinks = []

            def make_sink(path):
                sink = _BoundedLogSink(path)
                if path.name == "stdout.log":
                    sink.stream = FailingLogStream(sink.stream, "close_sink", errno.EIO)
                    sink.stream.armed.set()
                sinks.append(sink)
                return sink

            with patch("envcore.host_agent._BoundedLogSink", make_sink), patch(
                "envcore.host_agent.subprocess.Popen",
                side_effect=RuntimeError("launch failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "launch failed"):
                    self.start(Path(directory), "pass")
            self.assertEqual(len(sinks), 2)
            self.assertTrue(all(sink.stream.closed for sink in sinks))
            self.assertEqual(sinks[0].error["operation"], "close_sink")

    def start(self, root, script):
        task = root / "task.json"
        task.write_text("{}", encoding="utf-8")
        return start_host_agent_process(
            command=[sys.executable, "-c", script],
            workspace=root,
            control_dir=root,
            public_task_path=task,
            package_root=root,
            forwarded_environment=(),
            stdout_path=root / "stdout.log",
            stderr_path=root / "stderr.log",
        )

    def wait_for(self, condition):
        deadline = time.monotonic() + 5
        while not condition():
            if time.monotonic() >= deadline:
                self.fail("child output did not become visible before the deadline")
            time.sleep(0.01)

    def test_short_stdout_and_stderr_are_visible_before_child_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handle = self.start(
                root,
                "import json, sys, time\n"
                "from pathlib import Path\n"
                "print(json.dumps({'type': 'probe.ready'}), flush=True)\n"
                "print('short diagnostic', file=sys.stderr, flush=True)\n"
                "Path('ready').touch()\n"
                "while not Path('release').exists(): time.sleep(0.01)\n"
                "print(json.dumps({'type': 'turn.completed', 'usage': "
                "{'input_tokens': 20, 'cached_input_tokens': 12, "
                "'output_tokens': 5, 'reasoning_output_tokens': 2}}), flush=True)\n",
            )
            try:
                self.wait_for(lambda: (root / "ready").exists())
                self.wait_for(
                    lambda: handle.stdout_path.stat().st_size > 0
                    and handle.stderr_path.stat().st_size > 0
                )
                self.assertTrue(handle.running)
                self.assertEqual(
                    json.loads(handle.stdout_path.read_text()), {"type": "probe.ready"}
                )
                self.assertEqual(handle.stderr_path.read_text(), "short diagnostic\n")
                (root / "release").touch()
                handle.process.wait(timeout=5)
            finally:
                handle.stop()
            usage = summarize_host_agent_trace(handle.stdout_path)
            self.assertTrue(usage["usage_complete"])
            self.assertEqual(usage["total_tokens"], 25)
            self.assertTrue(handle.summarize_usage()["usage_complete"])
            for status in handle.summarize_usage()["log_capture"].values():
                self.assertTrue(status["complete"])
            self.assertTrue(handle._closed)

    def test_excess_output_is_drained_without_exceeding_either_log_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("envcore.host_agent.MAX_HOST_AGENT_LOG_BYTES", 1024):
                handle = self.start(
                    root,
                    "import sys\n"
                    "sys.stdout.buffer.write(b'x' * (256 * 1024))\n"
                    "sys.stdout.flush()\n"
                    "sys.stderr.buffer.write(b'y' * (256 * 1024))\n"
                    "sys.stderr.flush()\n",
                )
            try:
                handle.process.wait(timeout=5)
                self.assertEqual(handle.returncode, 0)
            finally:
                handle.stop()
            for path in (handle.stdout_path, handle.stderr_path):
                self.assertLessEqual(path.stat().st_size, 1024)
                self.assertIn(b"[log truncated by Envcore]", path.read_bytes())
            self.assertTrue(handle._stdout_sink.truncated)
            self.assertTrue(handle._stderr_sink.truncated)
            self.assertFalse(
                summarize_host_agent_trace(handle.stdout_path)["usage_complete"]
            )
            self.assertTrue(handle._closed)

    def test_forced_stop_drains_partial_record_and_keeps_usage_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handle = self.start(
                root,
                "import sys, time\n"
                'sys.stdout.write(\'{"type":"turn.completed",\')\n'
                "sys.stdout.flush()\n"
                "time.sleep(30)\n",
            )
            try:
                self.wait_for(lambda: handle.stdout_path.stat().st_size > 0)
                self.assertTrue(handle.running)
            finally:
                handle.stop(timeout_seconds=1)
            self.assertFalse(handle.running)
            self.assertTrue(handle._closed)
            usage = summarize_host_agent_trace(handle.stdout_path)
            self.assertFalse(usage["usage_complete"])
            self.assertEqual(usage["malformed_records"], 1)


if __name__ == "__main__":
    unittest.main()
