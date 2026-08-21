"""Thin AMP wrapper over ``tracked_tools.boundary`` (librarized C8).

The compiler-error-driven int<->tracked boundary mapper moved into the
Tracked library (``third_party/tracked/tools``, console script
``tracked-boundary-patch``); logic-identical and already parameterized on the
instrumented scalar spelling.  See the library module for the full docs.
"""

from __future__ import annotations

from tracked_tools.boundary import (  # noqa: F401
    _C8_ARGNOTE_RE,
    _C8_ASSIGN_EQ_RE,
    _C8_ERR_RE,
    _C8_NOEQ_RE,
    _Q,
    _TypePatterns,
    _caret_span,
    _file_line,
    _is_int_tracked_flavored,
    _iter_error_blocks,
    _map_assign,
    _map_noeq,
    _map_refbind,
    _parse_c8_errors,
    _rel_in,
    derive_c8_patch,
    extract_call_arg,
    is_within,
    split_top_level,
    synthesize_patch,
)
