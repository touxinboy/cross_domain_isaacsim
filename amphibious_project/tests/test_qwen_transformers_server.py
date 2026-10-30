"""Real loopback HTTP tests with a CPU fixture backend, never a loaded model."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import threading
import unittest

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from qwen_api_client import QwenAPIError, QwenClient

spec = importlib.util.spec_from_file_location("l17_chat_server", PROJECT / "deploy/qwen_a100/qwen_transformers_server.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FixtureBackend:
    def __init__(self):
        self.calls = []

    def generate(self, messages, max_tokens, temperature):
        self.calls.append((messages, max_tokens, temperature))
        return "OK", {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}


class PreparedServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = FixtureBackend()
        cls.server = module.ChatServer(("127.0.0.1", 0), cls.backend, "LOCAL_TEST_ONLY")
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = "http://127.0.0.1:" + str(cls.server.server_port) + "/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        self.backend.calls.clear()
        self.client = QwenClient(self.url, module.MODEL_ID, api_key="LOCAL_TEST_ONLY", timeout_s=3)

    def test_model_listing_does_not_infer(self):
        self.assertEqual(self.client.models(), [module.MODEL_ID])
        self.assertEqual(self.backend.calls, [])

    def test_text_and_usage_match_existing_client(self):
        self.assertEqual(self.client.chat([{"role": "user", "content": "OK"}], max_tokens=16), "OK")
        self.assertEqual(self.client.last_usage["total_tokens"], 6)
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(self.backend.calls[0][0][0]["content"], [{"type": "text", "text": "OK"}])

    def test_system_text_also_uses_typed_processor_blocks(self):
        self.client.chat([{"role": "system", "content": "Choose only observed objects."},
                          {"role": "user", "content": "Choose."}])
        messages = self.backend.calls[0][0]
        self.assertTrue(all(isinstance(message["content"], list) for message in messages))
        self.assertEqual(messages[0]["content"][0]["type"], "text")

    def test_real_base64_image_is_passed_as_rgb_to_processor_backend(self):
        with tempfile.TemporaryDirectory(prefix="l17_qwen_server_") as directory:
            path = Path(directory) / "observed.png"
            Image.new("RGB", (32, 32), (22, 99, 170)).save(path)
            self.assertEqual(self.client.describe([path], "Describe"), "OK")
        image = self.backend.calls[0][0][0]["content"][1]["image"]
        self.assertEqual(image.getpixel((0, 0)), (22, 99, 170))

    def test_bad_key_never_reaches_model(self):
        client = QwenClient(self.url, module.MODEL_ID, api_key="WRONG_LOCAL_TEST_KEY", timeout_s=3)
        with self.assertRaises(QwenAPIError) as error:
            client.models()
        self.assertIn("401", str(error.exception))
        self.assertEqual(self.backend.calls, [])

    def test_streaming_is_rejected_without_inference(self):
        with self.assertRaises(QwenAPIError) as error:
            self.client._request("POST", "/chat/completions", {"model": module.MODEL_ID, "stream": True})
        self.assertIn("400", str(error.exception))
        self.assertEqual(self.backend.calls, [])

    def test_busy_server_rejects_without_concurrent_gpu_work(self):
        self.server.inference_lock.acquire()
        try:
            with self.assertRaises(QwenAPIError) as error:
                self.client.chat([{"role": "user", "content": "OK"}])
            self.assertIn("503", str(error.exception))
        finally:
            self.server.inference_lock.release()
        self.assertEqual(self.backend.calls, [])

    def test_action_checkpoint_name_is_not_accepted_as_chat_model(self):
        client = QwenClient(self.url, "LingBot-VLA", api_key="LOCAL_TEST_ONLY", timeout_s=3)
        with self.assertRaises(QwenAPIError):
            client.chat([{"role": "user", "content": "OK"}])
        self.assertEqual(self.backend.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
