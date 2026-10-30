"""Check the configured A100 Qwen service from the computer running this command."""

import argparse
import json
import sys

from qwen_api_client import QwenAPIError, QwenClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--models", action="store_true", help="GET /v1/models; no inference")
    actions.add_argument("--text", action="store_true", help="One small text inference request")
    actions.add_argument("--image", help="One image inference request to verify Qwen-VL support")
    parser.add_argument("--timeout", type=float, default=30.)
    args = parser.parse_args()
    client = QwenClient(timeout_s=args.timeout)
    if args.models:
        result = {"model_ids": client.models(), "vision_support_verified": False}
    elif args.text:
        result = {"response": client.chat([{"role": "user", "content": "Reply with exactly OK."}], max_tokens=16),
                  "vision_support_verified": False}
    else:
        result = {"response": client.describe([args.image], "Describe the visible objects and their colors in one sentence.", max_tokens=120),
                  "vision_request_sent": True,
                  "vision_support_verified": "Review the description against the supplied image; HTTP 200 alone is insufficient"}
    result["latency_s"] = client.last_latency_s
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QwenAPIError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(3)
