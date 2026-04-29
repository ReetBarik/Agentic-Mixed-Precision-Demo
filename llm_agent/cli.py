#!/usr/bin/env python3
"""Smoke test for Anthropic SDK connectivity via the Argo proxy."""

import argparse
import json
import sys

from llm_agent.client import make_client
from llm_agent import config


def main():
    ap = argparse.ArgumentParser(description="Anthropic SDK proxy smoke test")
    ap.add_argument("--base-url", default=None, help="Proxy base URL (default: ANTHROPIC_BASE_URL env or http://127.0.0.1:8083/argoapi/)")
    ap.add_argument("--model", default=config.DEFAULT_MODEL)
    ap.add_argument("--prompt", default="Reply with OK")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--json", action="store_true", help="Print full JSON response")
    args = ap.parse_args()

    client = make_client(base_url=args.base_url)
    response = client.messages.create(
        model=args.model,
        max_tokens=args.max_tokens,
        messages=[{"role": "user", "content": args.prompt}],
    )
    if args.json:
        print(json.dumps(response.model_dump(), indent=2))
    else:
        for block in response.content:
            if hasattr(block, "text"):
                print(block.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
