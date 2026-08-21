"""Thin AMP wrapper over ``tracked_tools.interop.llm`` (librarized).

LLM generation plumbing (anthropic-SDK streaming, target-header embedding,
bounded generate→accept retries) moved into the Tracked library
(``third_party/tracked/tools``); logic-identical.  See the library module for
the full docs.
"""

from __future__ import annotations

from tracked_tools.interop.llm import (  # noqa: F401
    HEADER_EMBED_CAP,
    INCLUDE_RE,
    collect_target_headers,
    embed_file,
    generate_with_retries,
    rel,
    resolve_local_include,
    stream_llm,
    stream_shim,
    strip_code_fences,
)
