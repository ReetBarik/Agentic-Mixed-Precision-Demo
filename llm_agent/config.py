"""Centralized configuration from environment variables."""

import os

# Proxy URL for ai-agents-at-anl/claude-argo-proxy.py (port 8083, /argoapi/ path)
PROXY_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8083/argoapi/")

# Auth token: prefer ANTHROPIC_AUTH_TOKEN, fall back to ARGO_USERNAME for compatibility
AUTH_TOKEN = (
    os.environ.get("ANTHROPIC_AUTH_TOKEN")
    or os.environ.get("ARGO_USERNAME", "")
)

DEFAULT_MODEL = os.environ.get("ARGO_MODEL", "claudeopus46")
DEFAULT_MIN_DIGITS = 10.0
DEFAULT_BATCH = 10
DEFAULT_SEED = 123
MAX_ITERATIONS_PER_VAR = 3
MAX_PROPOSE_RETRIES = 2
