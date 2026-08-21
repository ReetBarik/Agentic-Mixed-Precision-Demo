"""Thin AMP wrapper over ``tracked_tools.interop.cache`` (librarized).

The SOURCE_HASH staleness discipline moved into the Tracked library
(``third_party/tracked/tools``); logic-identical, so every committed shim's
``SOURCE_HASH`` is preserved byte-for-byte (the ruleset hash is pinned
upstream too).  See the library module for the full docs.
"""

from __future__ import annotations

from tracked_tools.interop.cache import (  # noqa: F401
    HEADER_SUFFIXES,
    SOURCE_HASH_PENDING,
    SOURCE_HASH_RE,
    apply_source_hash,
    compute_region_hash,
    compute_source_hash,
    extract_source_hash,
    hash_header_dir,
    ruleset_hash,
)
