"""Anthropic SDK client factory pointing at the Argo proxy."""

import anthropic
from llm_agent import config


def make_client(base_url: str = None, api_key: str = None) -> anthropic.Anthropic:
    """Return an Anthropic client aimed at the Argo proxy.

    By default reads ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN / ARGO_USERNAME
    from the environment. Pass base_url / api_key to override per call.
    """
    return anthropic.Anthropic(
        base_url=base_url or config.PROXY_BASE_URL,
        api_key=api_key or config.AUTH_TOKEN,
    )
