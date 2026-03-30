#!/usr/bin/env python3
"""Minimal direct Argo chat client (no MCP dependencies)."""

import json
import os


class ArgoClient(object):
    """HTTP client for Argo chat-completions endpoint."""

    def __init__(
        self,
        base_url=None,
        user=None,
        model="claudeopus46",
        fallback_model="gpt4turbo",
        timeout=60,
    ):
        env_base = os.environ.get("ARGO_PROXY_BASE_URL")
        if not base_url:
            if env_base:
                base_url = env_base
            else:
                port = os.environ.get("ARGO_PROXY_PORT", "8092")
                base_url = "http://127.0.0.1:{0}".format(port)
        self.base_url = base_url.rstrip("/")
        self.chat_url = self.base_url + "/v1/chat/completions"
        self.user = user or os.environ.get("ARGO_USERNAME")
        self.model = model
        self.fallback_model = fallback_model
        self.timeout = int(timeout)
        if not self.user:
            raise ValueError("ARGO_USERNAME is required")
        try:
            import requests as _requests
        except ImportError:
            raise ImportError(
                "requests is required for llm_agent. Install with: "
                "pip install -r requirements-argo-agent.txt"
            )
        self._requests = _requests

    def _payload(self, model, messages, temperature=None, max_tokens=None):
        payload = {
            "model": model,
            "user": self.user,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        # Some Argo model adapters reject null or unsupported max_tokens.
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    def _post(self, payload):
        response = self._requests.post(self.chat_url, json=payload, timeout=self.timeout)
        if response.status_code >= 400:
            raise RuntimeError(
                "Argo chat call failed ({0}): {1}".format(
                    response.status_code, response.text[:1200]
                )
            )
        return response.json()

    def chat(self, messages, model=None, temperature=None, max_tokens=None, allow_fallback=True):
        chosen = model or self.model
        payload = self._payload(chosen, messages, temperature=temperature, max_tokens=max_tokens)
        try:
            return self._post(payload)
        except RuntimeError as exc:
            if not allow_fallback:
                raise
            if not self.fallback_model or chosen == self.fallback_model:
                raise
            text = str(exc).lower()
            if "invalid model" not in text and "model" not in text:
                raise
            payload = self._payload(
                self.fallback_model,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return self._post(payload)

    @staticmethod
    def extract_text(response_json):
        choices = response_json.get("choices") or []
        if not choices:
            return json.dumps(response_json)
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if content is None:
            return json.dumps(response_json)
        return content

