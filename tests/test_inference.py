import http.client
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from benchcore.agent_gateway import create_gateway_server
from benchcore.config import InferenceConfig
from benchcore.inference import summarize_inference_trace


class _UpstreamHandler(BaseHTTPRequestHandler):
    mode = "json"
    redirect_target = ""
    seen_auth = None
    seen_body = None
    first_sse_sent = threading.Event()
    finish_sse = threading.Event()

    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        type(self).seen_auth = self.headers.get("Authorization")
        length = int(self.headers.get("Content-Length", "0"))
        type(self).seen_body = self.rfile.read(length)
        if type(self).mode == "redirect":
            self.send_response(302)
            self.send_header("Location", type(self).redirect_target)
            self.end_headers()
            return
        if type(self).mode == "sse":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(
                b'data: {"type":"response.output_text.delta","delta":"a"}\n\n'
            )
            self.wfile.flush()
            type(self).first_sse_sent.set()
            type(self).finish_sse.wait(timeout=3)
            self.wfile.write(
                b'data: {"type":"response.completed","response":{"usage":'
                b'{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}\n\n'
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
            return

        body = json.dumps(
            {
                "id": "response",
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "total_tokens": 5,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _RedirectSink(BaseHTTPRequestHandler):
    seen_auth = None

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        type(self).seen_auth = self.headers.get("Authorization")
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    do_POST = do_GET


class InferenceGatewayTests(unittest.TestCase):
    def setUp(self):
        _UpstreamHandler.mode = "json"
        _UpstreamHandler.seen_auth = None
        _UpstreamHandler.seen_body = None
        _UpstreamHandler.first_sse_sent.clear()
        _UpstreamHandler.finish_sse.clear()
        _RedirectSink.seen_auth = None

        self.temporary = tempfile.TemporaryDirectory()
        self.audit_path = Path(self.temporary.name) / "inference.jsonl"
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
        self.upstream_thread = threading.Thread(
            target=self.upstream.serve_forever, daemon=True
        )
        self.upstream_thread.start()
        host, port = self.upstream.server_address
        self.config = InferenceConfig(
            protocol="openai_compatible",
            base_url=f"http://{host}:{port}",
            api_key_env="TEST_GATEWAY_KEY",
            allowed_models=("researcher-a",),
            allowed_paths=("/v1/responses", "/v1/chat/completions"),
            timeout_seconds=10,
        )
        self.gateway = create_gateway_server(
            "127.0.0.1", 0, self.config, self.audit_path
        )
        self.gateway_thread = threading.Thread(
            target=self.gateway.serve_forever, daemon=True
        )
        self.gateway_thread.start()

    def tearDown(self):
        _UpstreamHandler.finish_sse.set()
        self.gateway.shutdown()
        self.gateway.server_close()
        self.gateway_thread.join(timeout=2)
        self.upstream.shutdown()
        self.upstream.server_close()
        self.upstream_thread.join(timeout=2)
        self.temporary.cleanup()

    def request(self, payload, path="/v1/responses"):
        body = json.dumps(payload).encode()
        connection = http.client.HTTPConnection(*self.gateway.server_address, timeout=3)
        connection.request(
            "POST", path, body=body, headers={"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def wait_for_trace(self, event_type):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if self.audit_path.exists() and event_type in self.audit_path.read_text(
                encoding="utf-8"
            ):
                return
            time.sleep(0.01)
        self.fail(f"trace did not contain {event_type}")

    def test_forwards_registered_model_and_records_usage(self):
        with patch.dict(os.environ, {"TEST_GATEWAY_KEY": "host-secret"}):
            status, _, body = self.request({"model": "researcher-a", "input": "hello"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["id"], "response")
        self.assertEqual(_UpstreamHandler.seen_auth, "Bearer host-secret")
        self.assertNotIn(b"host-secret", _UpstreamHandler.seen_body)
        self.wait_for_trace("INFERENCE_COMPLETED")
        summary = summarize_inference_trace(self.audit_path)
        self.assertEqual(summary["model_calls"], 1)
        self.assertEqual(summary["input_tokens"], 3)
        self.assertEqual(summary["output_tokens"], 2)
        self.assertTrue(summary["usage_complete"])

    def test_rejects_unregistered_model_path_and_hosted_tool(self):
        status, _, _ = self.request({"model": "researcher-b"})
        self.assertEqual(status, 403)
        status, _, _ = self.request({"model": "researcher-a"}, path="/v1/files")
        self.assertEqual(status, 403)
        status, _, _ = self.request(
            {"model": "researcher-a", "tools": [{"type": "web_search"}]}
        )
        self.assertEqual(status, 403)
        summary = summarize_inference_trace(self.audit_path)
        self.assertEqual(summary["model_calls"], 0)
        self.assertEqual(summary["rejected_requests"], 3)

    def test_rejects_provider_state_and_remote_attachments(self):
        payloads = [
            {"model": "researcher-a", "previous_response_id": "resp_123"},
            {"model": "researcher-a", "conversation": "conv_123"},
            {"model": "researcher-a", "prompt": {"id": "pmpt_123"}},
            {
                "model": "researcher-a",
                "input": [{"type": "input_file", "file_id": "file_123"}],
            },
            {
                "model": "researcher-a",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.test/x.png"},
                            }
                        ],
                    }
                ],
            },
            {
                "model": "researcher-a",
                "attachments": [{"file_id": "file_456"}],
            },
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                status, _, body = self.request(payload)
                self.assertEqual(status, 403)
                self.assertEqual(
                    json.loads(body)["error"]["code"],
                    "external_context_not_allowed",
                )
        self.assertIsNone(_UpstreamHandler.seen_body)

    def test_forbids_redirect_without_leaking_credential(self):
        sink = ThreadingHTTPServer(("127.0.0.1", 0), _RedirectSink)
        sink_thread = threading.Thread(target=sink.serve_forever, daemon=True)
        sink_thread.start()
        try:
            _UpstreamHandler.mode = "redirect"
            _UpstreamHandler.redirect_target = (
                f"http://127.0.0.1:{sink.server_address[1]}/stolen"
            )
            with patch.dict(os.environ, {"TEST_GATEWAY_KEY": "host-secret"}):
                status, _, body = self.request({"model": "researcher-a"})
            self.assertEqual(status, 502)
            self.assertEqual(
                json.loads(body)["error"]["code"],
                "inference_upstream_redirect_forbidden",
            )
            self.assertIsNone(_RedirectSink.seen_auth)
        finally:
            sink.shutdown()
            sink.server_close()
            sink_thread.join(timeout=2)

    def test_sse_is_forwarded_before_upstream_finishes_and_usage_is_parsed(self):
        _UpstreamHandler.mode = "sse"
        body = json.dumps({"model": "researcher-a", "stream": True}).encode()
        connection = http.client.HTTPConnection(*self.gateway.server_address, timeout=2)
        with patch.dict(os.environ, {"TEST_GATEWAY_KEY": "host-secret"}):
            connection.request(
                "POST",
                "/v1/responses",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            self.assertTrue(_UpstreamHandler.first_sse_sent.wait(timeout=1))
            first_line = response.readline()
            self.assertIn(b"response.output_text.delta", first_line)
            _UpstreamHandler.finish_sse.set()
            remaining = response.read()
        connection.close()
        self.assertIn(b"response.completed", remaining)
        self.wait_for_trace("INFERENCE_COMPLETED")
        summary = summarize_inference_trace(self.audit_path)
        self.assertEqual(summary["total_tokens"], 5)
        self.assertTrue(summary["usage_complete"])


if __name__ == "__main__":
    unittest.main()
