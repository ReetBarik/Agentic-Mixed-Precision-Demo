"""Tracked-integrator agent — shared service (no LangGraph node).

Owns the responsibility of integrating the ``Tracked<T>`` error-propagation
datatype into an arbitrary scientific application.  Given a target library's
headers and a driver source that exercises them, it produces a single C++
interop shim header (``<app>_interop.hpp``) that makes the library callable with
``T = Tracked<double>`` (and ``Complex<Tracked<double>>`` where applicable), so
its floating-point computations emit a condition-number / error-propagation
journal.  The generated header is self-auditing via comments; there is no
separate manifest file.

Structurally symmetric with ``agents/build_run/agent.py``: this module exposes a
plain callable (:func:`integrate`) rather than a LangGraph node, and is intended
to be invoked from ``build_and_run(...)`` as a prerequisite step whenever a
target uses Tracked and no up-to-date shim exists.

**Refactored onto** :mod:`agents.integrator_base` (2026-07): the SOURCE_HASH
cache, the Anthropic streaming shim, the target-header embedding helpers, the
bounded retry loop, and the C8 compiler-error-driven boundary patcher all live in
the shared base now, so a dd / ff integrator can reuse them.  What stays here is
this integrator's *target-specific* surface: its classification ruleset (the
system prompt, in :data:`_SYSTEM_PROMPT`, loaded from ``system_prompt.txt``), the
Tracked-flavored user-message assembly, and the ``tracked::Tracked`` scalar type
name it hands to the C8 patcher.  The generation is driven by an LLM call when a
``cfg`` is supplied; when ``cfg`` is ``None`` (the offline path exercised by the
scaffold smoke tests) it falls back to writing a benign placeholder header.  The
signature and the hash/caching contract are unchanged, and the tracked shim's
``SOURCE_HASH`` (``25f2b895…``) is preserved byte-for-byte.
"""

from __future__ import annotations

from pathlib import Path

from agents.integrator_base import cache, llm
from agents.integrator_base import c8 as _c8

# ---------------------------------------------------------------------------
# Target-specific ruleset (the LLM system prompt).  Loaded from a sibling text
# file so its bytes are the single source of truth for the ruleset hash folded
# into every shim's SOURCE_HASH (see cache.compute_source_hash).  The nine
# classification rules + the C1–C7 Tracked-API rules are load-bearing: every
# generated element must cite the rule that justified it, and Rule 9 is a
# deliberate hard-#error escape hatch.
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.txt").read_text(encoding="utf-8")

# The concrete instrumented scalar type name handed to the C8 boundary patcher.
_TRACKED_TYPE_NAME = "tracked::Tracked"

# Max output tokens for the generation call.  The reference B13 shim is ~480
# lines; give generous headroom so a full shim is never truncated mid-file.
_MAX_OUTPUT_TOKENS = 32000

# --- Back-compat re-exports of internals other modules/tests reach for -------
# The C8 helpers moved to the shared base; keep the historical tracked-integrator
# names resolving so callers (and tests/tracked_integrator/test_c8_patch.py) that
# reference ``ti._split_top_level`` / ``ti._extract_call_arg`` keep working.
_split_top_level = _c8.split_top_level
_extract_call_arg = _c8.extract_call_arg
_synthesize_patch = _c8.synthesize_patch
_SOURCE_HASH_PENDING = cache.SOURCE_HASH_PENDING


def integrate(
    target_library_headers,
    driver_source_path,
    tracked_repo_path=None,
    existing_shim=None,
    *,
    cfg=None,
    out_path=None,
    app_name=None,
) -> Path:
    """Produce (or reuse) the ``<app>_interop.hpp`` shim for a target library.

    Parameters
    ----------
    target_library_headers:
        Path to the target library's header directory.  Its contents are hashed
        into the shim's ``// SOURCE_HASH:`` line for staleness detection.
    driver_source_path:
        Path to the driver source file that exercises the library.  Named
        ``driver_source_path`` (not ``driver_source``) to disambiguate from
        ``driver_gen.driver_source``, which is C++ *text*, not a path.  When no
        explicit ``out_path`` is given, the generated shim is written alongside
        this file.
    tracked_repo_path:
        Path to the Tracked upstream checkout.  Defaults to the vendored subtree
        at ``third_party/tracked`` when ``None``.
    existing_shim:
        Path to a pre-existing shim to extend / refresh in place.  If it exists
        and its embedded ``SOURCE_HASH`` matches the freshly computed hash, the
        shim is considered up to date and returned untouched (cache hit).
    cfg:
        Optional :class:`~agents.config.PipelineConfig`.  With a cfg the LLM
        drives generation (``cfg.model``); without one the benign placeholder is
        written (offline / scaffold path).
    out_path:
        Optional explicit output path for the shim.  Overrides the default
        (``<driver_dir>/<app>_interop.hpp``) and ``existing_shim`` for placement.
    app_name:
        Optional application name used to build the default filename.  Derived
        from the header directory name when ``None``.

    Returns
    -------
    Path
        The path to the up-to-date shim (freshly written, or the cached one).
    """
    headers_dir = Path(target_library_headers).resolve()
    driver_path = Path(driver_source_path).resolve()
    if tracked_repo_path is None:
        tracked_repo_path = Path(__file__).parent.parent.parent / "third_party" / "tracked"
    tracked_repo_path = Path(tracked_repo_path)

    if not headers_dir.is_dir():
        raise NotADirectoryError(
            f"target_library_headers is not a directory: {headers_dir}"
        )

    source_hash = cache.compute_source_hash(headers_dir, _SYSTEM_PROMPT)

    resolved_app_name = app_name or _derive_app_name(headers_dir)

    # Resolve the output path.  Precedence: explicit out_path > existing_shim
    # (refresh in place) > default alongside the driver.
    if out_path is not None:
        shim_path = Path(out_path).resolve()
    elif existing_shim is not None:
        shim_path = Path(existing_shim).resolve()
    else:
        shim_path = driver_path.parent / f"{resolved_app_name}_interop.hpp"

    # Staleness check: an existing shim whose embedded hash matches the current
    # header contents is up to date — return it without rewriting.
    cache_candidate = Path(existing_shim).resolve() if existing_shim is not None else shim_path
    if cache_candidate.exists():
        cached_hash = cache.extract_source_hash(cache_candidate.read_text(encoding="utf-8"))
        if cached_hash == source_hash:
            return cache_candidate

    # (Re)generate.  With a cfg we drive the LLM; without one (scaffold /
    # offline path) we fall back to the benign placeholder so callers that don't
    # wire up an LLM still get a compilable no-op shim with a valid SOURCE_HASH.
    shim_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg is None:
        shim_text = cache.apply_source_hash(
            _render_placeholder(resolved_app_name, cache.SOURCE_HASH_PENDING),
            source_hash,
        )
    else:
        raw = _generate_shim(
            headers_dir=headers_dir,
            driver_path=driver_path,
            tracked_repo_path=tracked_repo_path,
            existing_shim=Path(existing_shim).resolve() if existing_shim else None,
            app_name=resolved_app_name,
            cfg=cfg,
        )
        # Post-process: the model emits `// SOURCE_HASH: PENDING`; stamp the real
        # hash computed above.
        shim_text = cache.apply_source_hash(raw, source_hash)

    shim_path.write_text(shim_text, encoding="utf-8")
    return shim_path


def derive_c8_patch(compile_stderr: str, headers_dir, repo_root) -> str | None:
    """Map int<->tracked compile diagnostics to a git-apply-able library patch.

    Thin wrapper over :func:`agents.integrator_base.c8.derive_c8_patch` bound to
    this integrator's scalar type name (``tracked::Tracked``).  See the base for
    the C8 (a)/(b)/(c) crossing patterns and the ``C8_UNCLASSIFIED_ERROR``
    contract.
    """
    return _c8.derive_c8_patch(
        compile_stderr, headers_dir, repo_root,
        tracked_type_name=_TRACKED_TYPE_NAME,
    )


# ---------------------------------------------------------------------------
# LLM generation (target-specific message assembly; streaming via the base)
# ---------------------------------------------------------------------------

def _generate_shim(
    headers_dir: Path,
    driver_path: Path,
    tracked_repo_path: Path,
    existing_shim: Path | None,
    app_name: str,
    cfg,
) -> str:
    """Assemble the Tracked-flavored user turn and stream the shim from the LLM."""
    user_message = _build_user_message(
        headers_dir=headers_dir,
        driver_path=driver_path,
        tracked_repo_path=tracked_repo_path,
        existing_shim=existing_shim,
        app_name=app_name,
    )
    return llm.stream_llm(_SYSTEM_PROMPT, user_message, cfg, _MAX_OUTPUT_TOKENS)


def _build_user_message(
    headers_dir: Path,
    driver_path: Path,
    tracked_repo_path: Path,
    existing_shim: Path | None,
    app_name: str,
) -> str:
    """Assemble the user turn: Tracked API + target headers + driver + shim.

    The Tracked public API is included (not just the URL from the system prompt)
    so the model calls the real ``tracked::`` signatures rather than
    hallucinating them.  Target headers are split into the driver's transitive
    local-include closure (embedded in full) and the rest (listed by name), per
    the spec's "prefer full contents of headers the driver #includes".
    """
    parts: list[str] = []
    parts.append(
        f"Generate the Tracked interop shim `{app_name}_interop.hpp` for the "
        f"target library below, so its driver can be instantiated with "
        f"`T = Tracked<double>` / `Complex<Tracked<double>>`.\n"
    )

    # --- Tracked public API (do not modify; call these) ---
    tracked_inc = tracked_repo_path / "include" / "tracked"
    tracked_headers = sorted(tracked_inc.glob("*.hpp")) if tracked_inc.is_dir() else []
    if tracked_headers:
        parts.append(
            "## Tracked datatype public API (do NOT modify; call these exact "
            "signatures)\n"
        )
        for hp in tracked_headers:
            parts.append(llm.embed_file(hp, hp.name))

    # --- Target library headers: closure (full) + others (names only) ---
    driver_text = driver_path.read_text(encoding="utf-8", errors="replace")
    closure, others = llm.collect_target_headers(headers_dir, driver_text)

    parts.append(
        "## Target library headers (the shim makes these callable with "
        "Tracked types)\n"
    )
    for hp in closure:
        rel = llm.rel(hp, headers_dir)
        parts.append(llm.embed_file(hp, rel))

    if others:
        listing = "\n".join(f"  - {llm.rel(hp, headers_dir)}" for hp in others)
        parts.append(
            "### Other headers on the include path (transitively available; "
            "contents omitted — request-by-name only)\n" + listing + "\n"
        )

    # --- Driver source ---
    parts.append(
        "## Driver source (exercises the library — generate a shim for every "
        "library symbol it instantiates)\n"
    )
    parts.append(llm.embed_file(driver_path, driver_path.name, text=driver_text))

    # --- Existing shim to extend, if any ---
    if existing_shim is not None and existing_shim.exists():
        parts.append(
            "## Existing shim to extend/refresh (preserve what still applies, "
            "add what the current driver/headers now require)\n"
        )
        parts.append(llm.embed_file(existing_shim, existing_shim.name))

    # --- Output contract ---
    parts.append(
        f"## Output\n"
        f"Emit ONLY the complete contents of `{app_name}_interop.hpp` — no "
        f"prose, no markdown fences. Include a `// SOURCE_HASH: PENDING` line "
        f"near the top; the caller replaces PENDING with the real hash. Every "
        f"generated overload, specialization, and annotation must carry a "
        f"comment naming the rule that justified it."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# SOURCE_HASH wrappers (bound to this integrator's ruleset) + internal helpers
# ---------------------------------------------------------------------------

def _ruleset_hash() -> str:
    """SHA-256 of this integrator's ruleset (the LLM system prompt)."""
    return cache.ruleset_hash(_SYSTEM_PROMPT)


def _compute_source_hash(headers_dir: Path) -> str:
    """The shim's staleness key for THIS integrator (headers ⊕ tracked ruleset)."""
    return cache.compute_source_hash(Path(headers_dir), _SYSTEM_PROMPT)


def _extract_source_hash(text: str) -> str | None:
    return cache.extract_source_hash(text)


def _apply_source_hash(text: str, source_hash: str) -> str:
    return cache.apply_source_hash(text, source_hash)


def _derive_app_name(headers_dir: Path) -> str:
    """Best-effort application name from the header directory name.

    Strips common packaging suffixes (``qcdloop_headers`` -> ``qcdloop``); falls
    back to the raw directory name.
    """
    name = headers_dir.name
    for suffix in ("_headers", "-headers", "_include", "_includes", "-include", "_inc"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)] or name
    return name


def _render_placeholder(app_name: str, source_hash: str) -> str:
    """A benign, valid header standing in for a not-yet-generated shim.

    Deliberately *not* an ``#error``: the scaffold placeholder must be a compilable
    no-op so wiring it into ``build_and_run`` cannot break an unrelated build.  The
    LLM ruleset's Rule 9 escape hatch is what emits ``#error`` for genuinely
    unclassifiable functions.
    """
    return (
        f"// {app_name}_interop.hpp — Tracked<T> interop shim (SCAFFOLD PLACEHOLDER)\n"
        f"//\n"
        f"// Generated by agents/tracked_integrator (offline/scaffold path).\n"
        f"// LLM-driven shim generation runs only when a cfg is supplied; this is a\n"
        f"// compilable no-op so the caching/staleness plumbing can be exercised.\n"
        f"// SOURCE_HASH: {source_hash}\n"
        f"#pragma once\n"
    )
