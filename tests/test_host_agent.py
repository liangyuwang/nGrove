import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from autotrain_bench.host_agent import (
    start_host_agent_process,
    summarize_host_agent_trace,
)


class HostAgentLogTests(unittest.TestCase):
    def test_missing_log_is_marked_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "missing.log"
            usage = summarize_host_agent_trace(trace)
            self.assertFalse(usage["usage_complete"])
            self.assertEqual(usage["total_tokens"], 0)
            self.assertEqual(usage["malformed_records"], 0)
            self.assertFalse(trace.exists())

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
            self.assertTrue(handle._closed)

    def test_excess_output_is_drained_without_exceeding_either_log_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("autotrain_bench.host_agent.MAX_HOST_AGENT_LOG_BYTES", 1024):
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
                self.assertIn(b"[log truncated by AutoTrain-Bench]", path.read_bytes())
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
