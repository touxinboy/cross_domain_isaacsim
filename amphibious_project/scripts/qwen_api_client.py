"""Small HTTP client for the user's Qwen-compatible service, without SDK dependencies.

Uses QWEN_BASE_URL, QWEN_MODEL, QWEN_API_KEY. Never writes or prints keys.
Importing this module performs no network requests.
"""

import base64
import json
import math
import mimetypes
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


class QwenAPIError(RuntimeError):
    pass


class QwenClient:
    def __init__(self, base_url=None, model=None, api_key=None, timeout_s=120.):
        value = (base_url or os.environ.get("QWEN_BASE_URL", "")).rstrip("/")
        if not value:
            raise QwenAPIError("Set QWEN_BASE_URL to the A100 service URL ending in /v1")
        parsed = urlsplit(value)
        if parsed.scheme not in ["http", "https"] or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise QwenAPIError("Use a plain HTTP(S) base URL; keep credentials in QWEN_API_KEY")
        if not parsed.path.rstrip("/").endswith("/v1"):
            raise QwenAPIError("QWEN_BASE_URL must end in /v1 (not /chat/completions)")
        self.base_url = value
        self.model = model or os.environ.get("QWEN_MODEL", "")
        self.api_key = api_key if api_key is not None else os.environ.get("QWEN_API_KEY", "")
        self.timeout_s = float(timeout_s)
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise QwenAPIError("Timeout must be a positive finite number")
        # Requests are made from the caller to its configured service. In particular,
        # localhost SSH tunnels must not be diverted through a system HTTP proxy.
        self.opener = build_opener(ProxyHandler({}))
        self.last_usage = None
        self.last_latency_s = None

    def _request(self, method, endpoint, payload=None):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        body = json.dumps(payload, allow_nan=False).encode("utf-8") if payload is not None else None
        request = Request(self.base_url + endpoint, data=body, headers=headers, method=method)
        started = time.monotonic()
        try:
            with self.opener.open(request, timeout=self.timeout_s) as response:
                data = response.read(20_000_001)
            if len(data) > 20_000_000:
                raise QwenAPIError("Qwen response exceeds the expected size")
            parsed = json.loads(data)
            if not isinstance(parsed, dict):
                raise QwenAPIError("Qwen response must be a JSON object")
            return parsed
        except HTTPError as error:
            # Do not print server echo bodies, request headers, or credentials.
            status = error.code
            error.close()
            raise QwenAPIError("Qwen returned HTTP " + str(status) + "; check service route, model name and authentication") from None
        except (URLError, TimeoutError, ConnectionError, OSError):
            raise QwenAPIError("Could not reach Qwen within the timeout; check address, port or SSH tunnel") from None
        except (UnicodeError, json.JSONDecodeError):
            raise QwenAPIError("Qwen returned invalid JSON") from None
        finally:
            self.last_latency_s = time.monotonic() - started

    def models(self):
        data = self._request("GET", "/models").get("data", [])
        if not isinstance(data, list):
            raise QwenAPIError("Invalid model listing")
        return [item["id"] for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)]

    def chat(self, messages, max_tokens=512, temperature=0.):
        if not self.model:
            raise QwenAPIError("Set QWEN_MODEL to an id returned by --models")
        if not 1 <= int(max_tokens) <= 4096:
            raise QwenAPIError("max_tokens must be between 1 and 4096")
        result = self._request("POST", "/chat/completions", {
            "model": self.model, "messages": messages, "max_tokens": int(max_tokens),
            "temperature": float(temperature), "stream": False,
        })
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise QwenAPIError("Qwen response has no choices[0].message.content") from None
        if not isinstance(content, str) or not content.strip():
            raise QwenAPIError("Qwen returned no text content; check reasoning mode and generation budget")
        self.last_usage = result.get("usage")
        return content

    def describe(self, image_paths, prompt, max_tokens=512):
        content = [{"type": "text", "text": prompt}]
        for path in image_paths:
            content.append({"type": "image_url", "image_url": {"url": image_data_url(path)}})
        return self.chat([{"role": "user", "content": content}], max_tokens=max_tokens)

    def msgnav_completion(self, system_prompt, contents):
        """Exact callback shape used by upstream MSGNav's explore_utils functions."""
        blocks = []
        for item in contents:
            blocks.append({"type": "text", "text": str(item[0])})
            if len(item) == 2:
                encoded = item[1]
                if not isinstance(encoded, str):
                    raise QwenAPIError("MSGNav image content must be base64 text")
                blocks.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}})
        return self.chat([{"role": "system", "content": system_prompt}, {"role": "user", "content": blocks}],
                         max_tokens=2048, temperature=.1)


def image_data_url(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 12_000_000:
        raise QwenAPIError("Use an existing image under 12 MB")
    mime = mimetypes.guess_type(str(path))[0]
    if mime not in ["image/png", "image/jpeg", "image/webp"]:
        raise QwenAPIError("Use PNG, JPEG or WebP images")
    return "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode("ascii")
