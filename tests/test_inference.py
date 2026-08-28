import base64
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from autotrain_bench.config import InferenceConfig
from autotrain_bench.inference import InferenceFailure, forward_inference_request


class _Handler(BaseHTTPRequestHandler):
    seen_auth = None
    seen_body = None

    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        type(self).seen_auth = self.headers.get("Authorization")
        length = int(self.headers.get("Content-Length", "0"))
        type(self).seen_body = self.rfile.read(length)
        body = json.dumps({"id": "response", "usage": {"input_tokens": 3}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.config = InferenceConfig(
            protocol="openai_compatible",
            base_url=f"http://{host}:{port}",
            api_key_env="TEST_GATEWAY_KEY",
            allowed_models=("researcher-a",),
            allowed_paths=("/v1/responses", "/v1/chat/completions"),
            timeout_seconds=10,
        )

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, payload, path="/v1/responses"):
        body = json.dumps(payload).encode()
        return forward_inference_request(
            inference_config=self.config,
            method="POST",
            path=path,
            headers={"content-type": "application/json"},
            body_base64=base64.b64encode(body).decode("ascii"),
        )

    def test_forwards_registered_model_with_host_side_credential(self):
        with patch.dict(os.environ, {"TEST_GATEWAY_KEY": "host-secret"}):
            result = self.request({"model": "researcher-a", "input": "hello"})
        self.assertEqual(result["status"], 200)
        self.assertEqual(_Handler.seen_auth, "Bearer host-secret")
        self.assertNotIn(b"host-secret", _Handler.seen_body)

    def test_rejects_unregistered_model_and_unconfigured_path(self):
        with self.assertRaises(InferenceFailure) as caught:
            self.request({"model": "researcher-b", "input": "hello"})
        self.assertEqual(caught.exception.code, "model_not_registered")

        with self.assertRaises(InferenceFailure) as caught:
            self.request({"model": "researcher-a"}, path="/v1/files")
        self.assertEqual(caught.exception.code, "path_not_allowed")

    def test_allows_function_tools_but_rejects_provider_hosted_web_tools(self):
        with patch.dict(os.environ, {"TEST_GATEWAY_KEY": "host-secret"}):
            result = self.request(
                {
                    "model": "researcher-a",
                    "tools": [{"type": "function", "function": {"name": "shell"}}],
                }
            )
        self.assertEqual(result["status"], 200)

        with self.assertRaises(InferenceFailure) as caught:
            self.request(
                {"model": "researcher-a", "tools": [{"type": "web_search"}]}
            )
        self.assertEqual(caught.exception.code, "external_tool_not_allowed")


if __name__ == "__main__":
    unittest.main()
