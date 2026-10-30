"""Local HTTP protocol tests. No remote Qwen service or paid endpoint is used."""

import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import unittest

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from qwen_api_client import QwenAPIError, QwenClient


class Handler(BaseHTTPRequestHandler):
    request_body = None
    auth_seen = None

    def log_message(self, *args):
        pass

    def send_json(self, value, status=200):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self.send_json({"data": [{"id": "test-qwen-vl"}]})
        else:
            self.send_json({}, 404)

    def do_POST(self):
        Handler.auth_seen = self.headers.get("Authorization")
        Handler.request_body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if Handler.request_body["model"] == "http-error-fixture":
            self.send_json({"error": self.headers.get("Authorization")}, 401)
        elif Handler.request_body["model"] == "empty-response-fixture":
            self.send_json({"choices": [{"message": {"content": None}}]})
        else:
            self.send_json({"choices": [{"message": {"content": "OK"}}], "usage": {"total_tokens": 10}})


class QwenProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = "http://127.0.0.1:" + str(cls.server.server_port) + "/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join(timeout=2)

    def client(self, model="test-qwen-vl"):
        return QwenClient(self.url, model=model, api_key="LOCAL_UNIT_TEST_DUMMY", timeout_s=3)

    def test_model_listing_requires_no_inference(self):
        self.assertEqual(self.client().models(), ["test-qwen-vl"])

    def test_text_request_and_auth_header(self):
        client = self.client()
        self.assertEqual(client.chat([{"role": "user", "content": "OK"}], max_tokens=16), "OK")
        self.assertEqual(Handler.auth_seen, "Bearer LOCAL_UNIT_TEST_DUMMY")
        self.assertEqual(Handler.request_body["model"], "test-qwen-vl")
        self.assertFalse(Handler.request_body["stream"])
        self.assertEqual(client.last_usage["total_tokens"], 10)

    def test_vision_request_contains_real_image_bytes(self):
        with tempfile.TemporaryDirectory(prefix="l17_qwen_") as tmp:
            path = Path(tmp) / "fixture.png"
            Image.new("RGB", (16, 16), (15, 100, 200)).save(path)
            self.assertEqual(self.client().describe([path], "Describe"), "OK")
            blocks = Handler.request_body["messages"][0]["content"]
            data_url = blocks[1]["image_url"]["url"]
            self.assertTrue(data_url.startswith("data:image/png;base64,"))
            decoded = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))
            self.assertEqual(decoded.getpixel((0, 0)), (15, 100, 200))

    def test_msgnav_callback_translates_text_and_image(self):
        encoded = base64.b64encode(b"fixture_image_bytes").decode("ascii")
        self.assertEqual(self.client().msgnav_completion("system", [("select",), ("view", encoded)]), "OK")
        messages = Handler.request_body["messages"]
        self.assertEqual(messages[0]["content"], "system")
        self.assertEqual(len(messages[1]["content"]), 3)
        self.assertTrue(messages[1]["content"][2]["image_url"]["url"].endswith(encoded))

    def test_error_does_not_echo_authentication(self):
        with self.assertRaises(QwenAPIError) as result:
            self.client("http-error-fixture").chat([{"role": "user", "content": "hello"}])
        self.assertIn("401", str(result.exception))
        self.assertNotIn("LOCAL_UNIT_TEST_DUMMY", str(result.exception))

    def test_no_content_is_not_reported_as_success(self):
        with self.assertRaises(QwenAPIError):
            self.client("empty-response-fixture").chat([{"role": "user", "content": "hello"}])

    def test_embedded_credentials_and_wrong_route_rejected(self):
        for url in ["http://name:password@localhost:8000/v1", "http://localhost:8000/v1/chat/completions"]:
            with self.assertRaises(QwenAPIError):
                QwenClient(url, model="test")


if __name__ == "__main__":
    unittest.main(verbosity=2)
