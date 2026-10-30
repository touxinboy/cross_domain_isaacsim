"""Own Qwen3-VL text/image API using existing read-only weights/environment.

Prepared locally, not launched on the shared server. --check-only never loads a
model or writes runtime files; --serve requires a separate Slurm GPU allocation.
This exposes the models/chat routes used by qwen_api_client.py, not a full API.
"""
import argparse
import base64
import binascii
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
from io import BytesIO
import json
import math
import os
from pathlib import Path
import secrets
import sys
import threading
import time
import uuid

from PIL import Image

MODEL_ID = "l17-qwen3-vl-4b"
MODEL_DEFAULT = "/workspace/runs/individual/wangjiong/lingbo/models/Qwen3-VL-4B-Instruct"
MAX_REQUEST_BYTES = 32_000_000


class InputError(ValueError):
    pass


def preflight(model_dir):
    model = Path(model_dir).expanduser().resolve(strict=True)
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl" or "Qwen3VLForConditionalGeneration" not in config.get("architectures", []):
        raise InputError("Use the original Qwen3-VL checkpoint, not the LingBot action checkpoint")
    index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))
    names = set(index["weight_map"].values())
    for name in names:
        path = (model / name).resolve()
        if not path.is_relative_to(model) or not path.is_file() or path.stat().st_size <= 0:
            raise InputError("Model shard is absent or outside the checkpoint directory")
    if not names:
        raise InputError("Model index contains no weight shards")
    versions = {name: importlib.metadata.version(name) for name in ["torch", "transformers", "accelerate", "safetensors", "Pillow"]}
    return {"model_dir": str(model), "model_type": config["model_type"], "shard_count": len(names),
            "weight_bytes": sum((model / name).stat().st_size for name in names), "package_versions": versions,
            "model_id": MODEL_ID, "model_loaded": False, "files_written": 0}


def normalize_messages(messages):
    if not isinstance(messages, list) or not 1 <= len(messages) <= 40:
        raise InputError("Use 1 to 40 chat messages")
    converted, image_count = [], 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ["system", "user", "assistant"]:
            raise InputError("Unsupported chat role")
        content = message.get("content")
        if isinstance(content, str):
            # Qwen3-VL's 4.57 processor iterates content as typed blocks even
            # for text-only messages (including the system message).
            converted.append({"role": message["role"], "content": [{"type": "text", "text": content}]})
            continue
        if not isinstance(content, list):
            raise InputError("Chat content must be text or a list of text/image blocks")
        blocks = []
        for block in content:
            if not isinstance(block, dict):
                raise InputError("Invalid content block")
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                blocks.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image_url":
                image_count += 1
                if image_count > 16:
                    raise InputError("At most 16 images per request; compress the observed graph explicitly")
                value = block.get("image_url", {})
                url = value.get("url", "") if isinstance(value, dict) else ""
                if not isinstance(url, str) or not url.startswith(("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,")):
                    raise InputError("Use embedded PNG/JPEG/WebP data URLs")
                try:
                    raw = base64.b64decode(url.split(",", 1)[1], validate=True)
                    if len(raw) > 12_000_000:
                        raise InputError("Image exceeds 12 MB")
                    with Image.open(BytesIO(raw)) as image:
                        if image.width * image.height > 16_000_000:
                            raise InputError("Image exceeds the decoded pixel limit")
                        blocks.append({"type": "image", "image": image.convert("RGB")})
                except (binascii.Error, OSError, ValueError):
                    raise InputError("Image data could not be decoded within the limits") from None
            else:
                raise InputError("Only text and embedded image_url blocks are supported")
        converted.append({"role": message["role"], "content": blocks})
    return converted


class TransformersBackend:
    def __init__(self, model_dir):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one allocated CUDA device to this independent service")
        self.torch = torch
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa",
            trust_remote_code=False, local_files_only=True, use_safetensors=True).eval()
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False,
                                                       min_pixels=56*56, max_pixels=512*512)

    def generate(self, messages, max_tokens, temperature):
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        count = inputs["input_ids"].shape[-1]
        if count + max_tokens > 16384:
            raise InputError("Prompt plus requested generation exceeds the 16384-token experiment limit")
        options = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            options["temperature"] = temperature
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **options)
        tail = generated[0, count:]
        text = self.processor.decode(tail, skip_special_tokens=True)
        return text, {"prompt_tokens": count, "completion_tokens": len(tail), "total_tokens": count + len(tail)}


class ChatServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, backend, api_key):
        if not api_key:
            raise ValueError("An API key is required")
        self.backend, self.api_key = backend, api_key
        self.inference_lock = threading.Lock()
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, value):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authorized(self):
        value = self.headers.get("Authorization", "").encode("utf-8")
        if not hmac.compare_digest(value, ("Bearer " + self.server.api_key).encode("utf-8")):
            self.reply(401, {"error": {"message": "Authentication required"}})
            return False
        return True

    def do_GET(self):
        if self.path == "/healthz":
            self.reply(200, {"status": "ready"})
        elif self.authorized():
            if self.path == "/v1/models":
                self.reply(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "L17"}]})
            else:
                self.reply(404, {"error": {"message": "Unknown route"}})

    def do_POST(self):
        if not self.authorized():
            return
        if self.path != "/v1/chat/completions":
            self.reply(404, {"error": {"message": "Unknown route"}})
            return
        locked = False
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise InputError("Provide a JSON body no larger than 32 MB")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict) or data.get("model") != MODEL_ID:
                raise InputError("Unknown model ID; query /v1/models")
            if data.get("stream", False):
                raise InputError("This experiment interface uses non-streaming responses")
            limit = data.get("max_tokens", 512)
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 4096:
                raise InputError("max_tokens must be an integer between 1 and 4096")
            temperature = float(data.get("temperature", 0.))
            if not math.isfinite(temperature) or not 0 <= temperature <= 2:
                raise InputError("temperature must be finite and between 0 and 2")
            messages = normalize_messages(data.get("messages"))
            locked = self.server.inference_lock.acquire(blocking=False)
            if not locked:
                self.reply(503, {"error": {"message": "The own inference service is busy"}})
                return
            text, usage = self.server.backend.generate(messages, limit, temperature)
            self.reply(200, {"id": "chatcmpl-l17-" + uuid.uuid4().hex, "object": "chat.completion",
                             "created": int(time.time()), "model": MODEL_ID,
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                          "finish_reason": "length" if usage["completion_tokens"] >= limit else "stop"}],
                             "usage": usage})
        except (InputError, ValueError, TypeError, KeyError):
            self.reply(400, {"error": {"message": "Invalid request, image, token budget or context length"}})
        except Exception as error:
            print("own_inference_error_type=" + type(error).__name__, file=sys.stderr, flush=True)
            self.reply(500, {"error": {"message": "Own model inference failed; inspect its log"}})
        finally:
            if locked:
                self.server.inference_lock.release()


def own_runtime():
    allowed = (Path.home() / "L17").resolve()
    root = Path(os.environ.get("L17_QWEN_ROOT", str(allowed / "qwen_service"))).resolve()
    if root == allowed or not root.is_relative_to(allowed):
        raise RuntimeError("Runtime files must stay in a subdirectory of this account's ~/L17")
    os.umask(0o077)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, suffix in {"HF_HOME": "cache/huggingface", "XDG_CACHE_HOME": "cache", "TMPDIR": "tmp",
                         "PYTHONPYCACHEPREFIX": "cache/pycache", "CUDA_CACHE_PATH": "cache/nvidia",
                         "TORCH_EXTENSIONS_DIR": "cache/torch_extensions"}.items():
        path = (root / suffix).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError("Runtime cache symlink leaves own service directory")
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.environ[name] = str(path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    credentials = root / "credentials"
    if credentials.is_symlink():
        raise RuntimeError("Use an own regular credentials directory")
    credentials.mkdir(exist_ok=True, mode=0o700)
    credentials.chmod(0o700)
    path = credentials / "api_key"
    if path.is_symlink():
        raise RuntimeError("Use an own regular API key file")
    if not path.exists():
        with path.open("x", encoding="utf-8") as stream:
            stream.write(secrets.token_urlsafe(32) + "\n")
    path.chmod(0o600)
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("The own API key file is empty")
    return root, key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-only", action="store_true")
    action.add_argument("--serve", action="store_true")
    parser.add_argument("--checkpoint", default=os.environ.get("QWEN_MODEL_DIR", MODEL_DEFAULT))
    parser.add_argument("--port", type=int, default=int(os.environ.get("QWEN_PORT", "18080")))
    args = parser.parse_args()
    status = preflight(args.checkpoint)
    if args.check_only:
        print(json.dumps(status, indent=2))
        return 0
    if not 1024 <= args.port <= 65535:
        raise RuntimeError("Use a non-privileged service port")
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("CUDA_VISIBLE_DEVICES", "") in ["", "-1", "NoDevFiles"]:
        raise RuntimeError("Start with a separate single-GPU Slurm allocation, not the login shell")
    root, api_key = own_runtime()
    os.chdir(root)
    backend = TransformersBackend(status["model_dir"])
    with ChatServer(("127.0.0.1", args.port), backend, api_key) as server:
        print(json.dumps({"event": "own_qwen_service_ready", "host": "127.0.0.1", "port": args.port,
                          "model": MODEL_ID}), flush=True)
        server.serve_forever()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, RuntimeError, importlib.metadata.PackageNotFoundError) as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        raise SystemExit(3)
