#!/usr/bin/env python3
"""CLI smoke test for direct Argo access from JLSE proxy."""

import argparse
import json
import sys

from llm_agent.argo_client import ArgoClient


def main():
    ap = argparse.ArgumentParser(description="Direct Argo chat smoke test")
    ap.add_argument("--base-url", default=None, help="Proxy base URL, e.g. http://127.0.0.1:8092")
    ap.add_argument("--model", default="claudeopus46", help="Primary Argo model id")
    ap.add_argument("--fallback-model", default="gpt4turbo", help="Fallback model id")
    ap.add_argument("--user", default=None, help="Argo username (default: ARGO_USERNAME)")
    ap.add_argument("--prompt", default="Reply with OK", help="User prompt")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON response instead of extracted text",
    )
    args = ap.parse_args()

    client = ArgoClient(
        base_url=args.base_url,
        user=args.user,
        model=args.model,
        fallback_model=args.fallback_model,
    )
    messages = [{"role": "user", "content": args.prompt}]
    resp = client.chat(
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    if args.json:
        print(json.dumps(resp, indent=2))
    else:
        print(client.extract_text(resp))
    return 0


if __name__ == "__main__":
    sys.exit(main())

