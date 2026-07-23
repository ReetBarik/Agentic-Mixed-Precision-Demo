"""Phase 2a — deterministic variant naming from a caller path + integral.

When the Patcher fan-out (see :mod:`agents.patcher.fanout`) accepts a modification
to a function ``f`` reached from the integral's entry point via a caller chain,
each function on that chain gets a *per-caller-path variant*.  The variant name is
built here so the whole scheme has one definition and can be swapped for a
hashed-with-a-manifest scheme later (if path depth ever makes the flat names
unwieldy) without touching the fan-out logic.

Naming scheme (design §"Variant naming")
----------------------------------------
Full path names, **bottom-up**, separated by underscores, with the integral suffix
last.  For the path ``entry -> h -> g -> f`` (entry is the call-graph root):

* ``f`` → ``f_g_h_B1``   (f, then its callers g, h — root ``entry`` omitted)
* ``g`` → ``g_h_B1``
* ``h`` → ``h_B1``
* ``entry`` (BO) is the cascade ceiling: **never renamed** — its call site lives in
  the shared, read-only driver, so it keeps its name and only its *body* is edited
  to call the first-level variant.  :func:`variant_name` therefore rejects the
  degenerate "name the root" request (``callers_above == []`` with the function
  being the root) at the call site, not here — here it simply produces ``h_B1``
  for a single-caller-above chain and ``f`` (unchanged, empty suffix) is never
  requested because the root is handled specially by the fan-out.

Determinism + collision-freeness
--------------------------------
The name is a pure function of ``(func, callers_above, integral)`` — no counters,
no hashing, no globals — so two runs over the same call graph produce byte-identical
names (a hard requirement for the workflow-cache / resume story and for diffing a
Phase-2a run against Phase-1).  Because a caller path from the root to a target is
unique in an acyclic static call graph, distinct paths yield distinct names; the
fan-out asserts this via :func:`assert_no_collisions` as a belt-and-suspenders
check (it "shouldn't happen if paths are unique").
"""

from __future__ import annotations

import re

# Characters allowed in a C++ identifier segment.  A variant name must itself be a
# legal C++ identifier, so every path element and the integral tag are validated
# against this before they are joined.
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class VariantNameError(ValueError):
    """A path element or integral tag that cannot form a legal variant name."""


def _check_ident(tok: str, what: str) -> str:
    if not isinstance(tok, str) or not _IDENT_RE.match(tok):
        raise VariantNameError(
            f"{what} {tok!r} is not a legal C++ identifier segment")
    return tok


def variant_name(func: str, callers_above: list[str], integral: str) -> str:
    """Deterministic variant name for ``func`` reached via ``callers_above``.

    ``callers_above`` is the caller chain **from the immediate caller upward**,
    *excluding* the (never-renamed) entry-point root — i.e. for the path
    ``entry -> h -> g -> f`` the variant of ``f`` is requested with
    ``callers_above=["g", "h"]`` and yields ``f_g_h_<integral>``.

    A request with ``callers_above == []`` names ``func`` itself with only the
    integral suffix (``func_<integral>``) — used for the first function *below* the
    root (``h`` in the example, whose only above-root caller is the root).  The
    root itself is never passed here (the fan-out edits it in place); the empty
    suffix that would collide a variant with the bare function name is thus never
    produced.
    """
    _check_ident(func, "function name")
    _check_ident(integral, "integral tag")
    parts = [func] + [_check_ident(c, "caller name") for c in callers_above]
    return "_".join(parts) + "_" + integral


def variant_names_for_path(path: list[str], integral: str) -> dict[str, str]:
    """Map every non-root function on ``path`` to its variant name.

    ``path`` is the full caller chain **root-first**: ``[entry, h, g, f]``.  The
    root (``path[0]``) is the cascade ceiling and is deliberately **absent** from
    the returned map — it is never renamed.  Every other function ``x`` maps to the
    variant built from ``x`` plus the chain of callers between ``x`` and the root
    (exclusive), reversed to bottom-up order.

    Example — ``path=["entry","h","g","f"], integral="B1"``::

        {"h": "h_B1", "g": "g_h_B1", "f": "f_g_h_B1"}
    """
    if len(path) < 2:
        # Only the root (or empty): nothing below the ceiling to rename.
        return {}
    out: dict[str, str] = {}
    for idx in range(1, len(path)):
        func = path[idx]
        # callers strictly between the root and func, nearest-caller first (bottom-up)
        callers_above = list(reversed(path[1:idx]))
        out[func] = variant_name(func, callers_above, integral)
    return out


def assert_no_collisions(name_maps: list[dict[str, str]]) -> None:
    """Assert that no two *distinct* (function, path) mappings share a variant name.

    Each entry of ``name_maps`` is a per-path map from :func:`variant_names_for_path`.
    A collision means two different call paths produced the same variant name for
    *different* original functions — impossible for unique paths in an acyclic
    graph, so it signals a naming bug (or a cycle the call-graph builder should have
    rejected).  A single original function legitimately keeping the *same* variant
    name across passes (byte-identical over-generation) is **not** a collision and
    is allowed.
    """
    owner: dict[str, str] = {}   # variant_name -> original function it names
    for m in name_maps:
        for func, vname in m.items():
            prev = owner.get(vname)
            if prev is not None and prev != func:
                raise VariantNameError(
                    f"variant-name collision: {vname!r} names both {prev!r} and "
                    f"{func!r} (paths should be unique in an acyclic call graph)")
            owner[vname] = func
