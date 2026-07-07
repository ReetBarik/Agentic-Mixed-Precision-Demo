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
target uses Tracked and no up-to-date shim exists (see revision #2 of the task
spec: shared service, no new graph edges, no ``PipelineState`` changes).

**Part 2 (LLM logic).**  :func:`integrate` now drives the generation with an LLM
call (system prompt below, model from ``cfg.model`` — no hardcoded names) when a
``cfg`` is supplied.  It still performs the same staleness bookkeeping first
(hash the target-library header directory, compare against the ``// SOURCE_HASH:``
line in an existing shim) and short-circuits on a cache hit.  When ``cfg`` is
``None`` — the structural/offline path exercised by the scaffold smoke tests — it
falls back to writing the benign placeholder header instead of calling the LLM.
The signature and the hash/caching contract are unchanged from Part 1.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# LLM system prompt (Part 2).  Embedded verbatim per the task spec; the nine
# classification rules are load-bearing — every generated element must cite the
# rule that justified it, and Rule 9 is a deliberate hard-#error escape hatch.
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are a software engineer responsible for integrating the Tracked<T> \
error-propagation datatype \
(https://github.com/ReetBarik/Tracked-Error-Propagation-Datatype-Demo) \
into scientific applications so their floating-point computations produce \
a condition-number and error-propagation journal.

Your deliverable is a single C++ interop shim header (<app>_interop.hpp). \
Given (a) the target library's headers, (b) the driver source that \
exercises them, and (c) any pre-existing shim to extend, produce a shim \
that makes the library callable with T=Tracked<double> (and \
Complex<Tracked<double>> where applicable) without introducing type-safety \
bugs or losing precision-tracking fidelity.

Core principle: keep integer computations in integer land, keep \
floating-point computations in tracked land, convert only at the boundary.

Classification rules (apply mechanically; every generated overload, \
specialization, and annotation carries a comment naming which rule \
justified it):

Rule 1. Discrete return type (sign, count, index, comparison result) -> \
  shim returns raw int / bool / native discrete type. Never Tracked<int>, \
  never Tracked<bool>. The shim body unwraps: `return x.value() > 0 ? 1 \
  : ...`.

Rule 2. Floating-point return participating in downstream error \
  propagation -> shim returns Tracked<T>. The shim body computes in \
  tracked arithmetic.

Rule 3. Container of floating-point (complex, array, tuple) -> shim \
  returns the container of the tracked type: Complex<Tracked<T>>, not \
  Tracked<Complex<T>>.

Rule 4. Integer literal in a floating-point expression -> promote via \
  the scalar Tracked(T v) ctor, which synthesizes a _lit@?#N id.

Rule 5. Named constant (any spelling) -> wrap via \
  tracked::constant("<name>", value). A named constant is any \
  compile-time-known scalar that appears in the driver as a designator \
  (identifier, macro, or template accessor) rather than as an inline \
  literal. Common patterns include Constants<T>::name(), \
  std::numbers::name_v<T>, namespace::name, top-level constexpr values, \
  and #define macros. Detect the pattern the target library uses and \
  generate the corresponding named-wrapper (a Constants specialization, \
  a namespace-scoped constant() call, a wrapper function, etc.) as \
  appropriate for that library's dispatch mechanism. Preserve the source \
  identifier as the tracked-constant name.

Rule 6. Anonymous inline literal (bare number in expression) -> \
  tracked::literal(value). No name to preserve.

Rule 7. Comparison operators on tracked values -> return bool by \
  comparing .value(). Never lift comparisons into tracked booleans.

Rule 8. Shim called inside a parallel dispatch construct -> apply the \
  execution-space annotations that construct's programming model \
  requires. Detect the programming model from the driver's #includes \
  and dispatch syntax:
    - Kokkos parallel_for/parallel_reduce -> KOKKOS_INLINE_FUNCTION
    - CUDA/HIP kernel launch -> __host__ __device__
    - SYCL command group -> no annotation needed
    - OpenMP #pragma omp parallel -> no annotation needed
    - std::execution::par -> no annotation needed
    - Plain host code -> no annotation needed
  If the dispatch flavor is unrecognized, apply Rule 9.

Rule 9. Escape hatch. If you cannot confidently classify a function's \
  return type, cannot identify the execution-space annotation, or cannot \
  determine the target library's constant-dispatch idiom, do NOT guess. \
  Emit:

    // UNCLASSIFIED: <name>
    // Rule N unclear because: <one-line reason>
    // Human review needed before this shim can compile.
    #error "Tracked Datatype Integrator: <name> requires manual classification"

  This is a hard build failure by design - it surfaces the ambiguity \
  directly to whoever runs Build/Run.

Output: a single header file containing:
  - Necessary #includes (Tracked headers, target library headers)
  - Every function shim required by the driver, ordered as they appear \
    in the target library
  - Every Constants specialization (or equivalent named-constant wrapper) \
    required by the driver
  - Every execution-space annotation appropriate to the driver's dispatch \
    model
  - A `// SOURCE_HASH: PENDING` placeholder line (the caller replaces \
    PENDING with the actual sha256 after generation)
  - Rule-justification comments on every generated element

Integration clarifications (apply alongside the rules above):

These clarifications are library-agnostic: they name only the Tracked API \
(which is always provided to you) and generic C++ constructs. Discover every \
target-library-specific name (types, aliases, functions, traits templates, \
namespaces) from the headers and driver you are given — never assume them.

C1. Tracked type spellings (clarifies Rules 2/3). Discover the Tracked \
  library's OWN type spellings from the provided Tracked API headers and use \
  them verbatim; never introduce your own nesting. If the library provides a \
  tracked complex type whose components are ALREADY the tracked scalar (e.g. \
  `tracked::Complex<T>` documented as a complex of two `Tracked<T>` reals), \
  then the tracked complex is spelled `tracked::Complex<T>` with T the \
  underlying real scalar (e.g. double) — NOT `tracked::Complex<tracked::Tracked<T>>`, \
  which double-wraps and will not match. Read the driver's own type aliases for \
  its scalar and complex working types and make every shim overload accept and \
  return exactly those spellings.

C2. Literal / constant promotion (clarifies Rules 4/5/6). Use only the \
  factories and constructors the Tracked API actually defines — e.g. \
  `tracked::constant("name", T(v))`, `tracked::literal(T(v))`, and the \
  `Tracked<T>(T v)` ctor. The underlying real scalar is the element type the \
  library computes in (e.g. double, or the template parameter of your tracked \
  specialization). Do NOT invent helper names (e.g. a `Raw(...)` alias) — none \
  exists. Call only free functions the Tracked API declares (there may be no \
  `tracked::pow`, for instance — implement integer powers as a multiply loop \
  over the tracked `operator*`).

C3. Missing operators. If a target-library template applies an operator to a \
  tracked value that the Tracked type does not define (e.g. unary `operator+` \
  on a tracked scalar), add that operator as a free function in the Tracked \
  library's namespace (found by ADL) rather than editing the Tracked library. \
  An identity operator introduces no rounding and should emit no journal record.

C4. Execution-space annotation follows the DRIVER (clarifies Rule 8). Choose \
  the annotation from how the DRIVER dispatches the shimmed calls, not from \
  annotations the target library's own functions happen to carry. If the driver \
  invokes the tracked computation from a plain host loop (not inside a \
  parallel_for / kernel launch), emit NO execution-space annotation: the \
  Tracked ops are host-only (they use std::string / journaling), so a device \
  annotation is unnecessary and wrong.

C5. Specializing a class template the target library owns (clarifies Rule 5). \
  When you specialize a class template that the TARGET library defines (e.g. a \
  numeric-traits or named-constants template) and your shim is included BEFORE \
  the library header that defines it (check the driver's include order), you \
  MUST forward-declare the primary template first, inside the library's own \
  namespace — `namespace <lib> { template <typename T> struct <Name>; }` — so \
  your specialization parses; the library supplies the full primary definition \
  later in the same translation unit. Prefer a PARTIAL specialization keyed on \
  the tracked scalar (`template <class T> struct <Name><tracked::Tracked<T>>`) \
  over a full explicit specialization, so it covers the tracked type \
  generically and its members return the tracked scalar. Mirror the FULL member \
  interface of the library's primary template (every accessor the driver's call \
  graph can reach), routing each named leaf scalar through \
  `tracked::constant("<name>", T(value))` so no library symbol is lost and \
  every constant keeps its name in the journal.

C6. Discrete vs floating-point is decided by USE, not by name (disambiguates \
  Rules 1 vs 2). Before returning a raw int/bool under Rule 1, check how the \
  target library actually consumes the result. A helper that yields a numeric \
  +/-1 or 0/1 (a sign / step / heaviside-style function) whose result then \
  flows into floating-point arithmetic — assigned to the library's \
  floating-point/tracked working type, multiplied or added into a tracked \
  expression, or used to build a complex — is a FLOATING-POINT return and MUST \
  return the tracked type (Rule 2), preserving provenance. Reserve raw int/bool \
  (Rule 1) for results consumed ONLY as discrete selectors: array indices, loop \
  counts, or branch/boolean conditions (e.g. a zero-test used solely inside an \
  `if`). Apply the test per overload: a complex sign used as `z / |z|` returns \
  the tracked complex container (Rule 3).
"""

# Max output tokens for the generation call.  The reference B13 shim is ~480
# lines; give generous headroom so a full shim is never truncated mid-file.
_MAX_OUTPUT_TOKENS = 32000

# Per-file char cap when embedding a header's contents in the user message, so a
# single pathological header can't blow the context budget.  The B13 tree is
# tiny (~4k lines total); this only bites on very large libraries.
_HEADER_EMBED_CAP = 60000

_INCLUDE_RE = re.compile(r'#\s*include\s*[<"]([^">]+)[">]')

# Files under the target-library header directory that participate in the
# SOURCE_HASH.  Kept broad on purpose: any C/C++ header flavor invalidates the
# cached shim when its bytes change.  Non-header files (README, etc.) are
# ignored so documentation churn does not force regeneration.
_HEADER_SUFFIXES = {".h", ".hpp", ".hh", ".hxx", ".ipp", ".inc", ".cuh", ".tcc"}

_SOURCE_HASH_RE = re.compile(r"//\s*SOURCE_HASH:\s*(\S+)")

# Written verbatim by Part 1; Part 2's post-processing replaces PENDING with the
# real hash.  The scaffold writes the real hash directly (no LLM round-trip).
_SOURCE_HASH_PENDING = "PENDING"


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
        at ``third_party/tracked`` when ``None``.  Unused by the scaffold body;
        Part 2 embeds the Tracked headers' include path.
    existing_shim:
        Path to a pre-existing shim to extend / refresh in place.  If it exists
        and its embedded ``SOURCE_HASH`` matches the freshly computed hash, the
        shim is considered up to date and returned untouched (cache hit).
    cfg:
        Optional :class:`~agents.config.PipelineConfig`.  Unused by the scaffold;
        Part 2 reads ``cfg.model`` for the LLM call (no hardcoded model names).
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

    source_hash = _hash_header_dir(headers_dir)

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
        cached_hash = _extract_source_hash(cache_candidate.read_text(encoding="utf-8"))
        if cached_hash == source_hash:
            return cache_candidate

    # (Re)generate.  With a cfg we drive the LLM; without one (scaffold /
    # offline path — e.g. the structural smoke tests) we fall back to the
    # benign placeholder so callers that don't wire up an LLM still get a
    # compilable no-op shim with a valid SOURCE_HASH.
    shim_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg is None:
        shim_text = _render_placeholder(resolved_app_name, source_hash)
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
        # hash computed above (step 4 of the spec).
        shim_text = _apply_source_hash(raw, source_hash)

    shim_path.write_text(shim_text, encoding="utf-8")
    return shim_path


# ---------------------------------------------------------------------------
# LLM generation (Part 2)
# ---------------------------------------------------------------------------

def _generate_shim(
    headers_dir: Path,
    driver_path: Path,
    tracked_repo_path: Path,
    existing_shim: Path | None,
    app_name: str,
    cfg,
) -> str:
    """Call the LLM (cfg.model, via the anthropic SDK) and return shim text.

    Imported lazily so importing this module (and the scaffold smoke tests)
    never requires the anthropic client or a live endpoint.
    """
    import anthropic

    user_message = _build_user_message(
        headers_dir=headers_dir,
        driver_path=driver_path,
        tracked_repo_path=tracked_repo_path,
        existing_shim=existing_shim,
        app_name=app_name,
    )

    client = anthropic.Anthropic(base_url=cfg.base_url, api_key=cfg.auth_token)
    # Stream: a full shim can approach _MAX_OUTPUT_TOKENS, and the SDK refuses a
    # non-streaming request whose worst-case duration exceeds 10 minutes.
    with client.messages.stream(
        model=cfg.model,
        max_tokens=_MAX_OUTPUT_TOKENS,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        final = stream.get_final_message()

    text = "".join(
        block.text for block in final.content
        if getattr(block, "type", None) == "text"
    ).strip()
    if not text:
        raise RuntimeError("tracked_integrator: LLM returned no text content")
    return _strip_code_fences(text)


def _build_user_message(
    headers_dir: Path,
    driver_path: Path,
    tracked_repo_path: Path,
    existing_shim: Path | None,
    app_name: str,
) -> str:
    """Assemble the user turn: Tracked API + target headers + driver + shim.

    The Tracked public API is included (not just the URL from the system
    prompt) so the model calls the real ``tracked::`` signatures rather than
    hallucinating them.  Target headers are split into the driver's transitive
    local-include closure (embedded in full) and the rest (listed by name),
    per the spec's "prefer full contents of headers the driver #includes".
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
            parts.append(_embed_file(hp, hp.name))

    # --- Target library headers: closure (full) + others (names only) ---
    driver_text = driver_path.read_text(encoding="utf-8", errors="replace")
    closure, others = _collect_target_headers(headers_dir, driver_text)

    parts.append(
        "## Target library headers (the shim makes these callable with "
        "Tracked types)\n"
    )
    for hp in closure:
        rel = _rel(hp, headers_dir)
        parts.append(_embed_file(hp, rel))

    if others:
        listing = "\n".join(f"  - {_rel(hp, headers_dir)}" for hp in others)
        parts.append(
            "### Other headers on the include path (transitively available; "
            "contents omitted — request-by-name only)\n" + listing + "\n"
        )

    # --- Driver source ---
    parts.append(
        "## Driver source (exercises the library — generate a shim for every "
        "library symbol it instantiates)\n"
    )
    parts.append(_embed_file(driver_path, driver_path.name, text=driver_text))

    # --- Existing shim to extend, if any ---
    if existing_shim is not None and existing_shim.exists():
        parts.append(
            "## Existing shim to extend/refresh (preserve what still applies, "
            "add what the current driver/headers now require)\n"
        )
        parts.append(_embed_file(existing_shim, existing_shim.name))

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


def _collect_target_headers(
    headers_dir: Path, driver_text: str
) -> tuple[list[Path], list[Path]]:
    """Split the target header tree into (driver include-closure, everything else).

    Starting from the driver's local ``#include`` lines, BFS over local includes
    that resolve inside ``headers_dir``.  System includes (``<Kokkos_Core.hpp>``,
    ``<cmath>``, ``<tracked/...>``) never resolve here, so they are naturally
    skipped.  Returns both lists sorted by path for deterministic prompts.
    """
    all_headers = {
        p.resolve()
        for p in headers_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _HEADER_SUFFIXES
    }

    seen: set[Path] = set()
    queue: list[Path] = []

    def _seed(text: str) -> None:
        for inc in _INCLUDE_RE.findall(text):
            resolved = _resolve_local_include(inc, headers_dir)
            if resolved is not None and resolved not in seen:
                queue.append(resolved)

    _seed(driver_text)
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        try:
            _seed(current.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass

    closure = sorted(seen)
    others = sorted(all_headers - seen)
    return closure, others


def _resolve_local_include(inc: str, headers_dir: Path) -> Path | None:
    """Resolve an ``#include`` target to a file under ``headers_dir``, or None.

    Tries the path as written relative to ``headers_dir`` first, then falls back
    to a basename match anywhere in the tree (the B13 CMake adds both the tree
    root and ``box/`` to the include path, so ``#include "B2m.h"`` and
    ``#include "box/B2m.h"`` both need to resolve).
    """
    candidate = (headers_dir / inc)
    if candidate.is_file():
        return candidate.resolve()
    base = Path(inc).name
    matches = sorted(p for p in headers_dir.rglob(base) if p.is_file())
    return matches[0].resolve() if matches else None


def _embed_file(path: Path, label: str, text: str | None = None) -> str:
    """Render one file as a fenced ``### label`` section, capped in length."""
    if text is None:
        text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > _HEADER_EMBED_CAP:
        text = (
            text[:_HEADER_EMBED_CAP]
            + f"\n// ... [truncated {len(text) - _HEADER_EMBED_CAP} chars] ...\n"
        )
    return f"### `{label}`\n```cpp\n{text}\n```\n"


def _rel(path: Path, headers_dir: Path) -> str:
    try:
        return path.resolve().relative_to(headers_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def _strip_code_fences(text: str) -> str:
    """Strip a leading/trailing markdown code fence if the model added one."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    # Drop the opening fence line (``` or ```cpp) and a trailing fence line.
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _apply_source_hash(text: str, source_hash: str) -> str:
    """Stamp the real hash onto the shim's ``// SOURCE_HASH:`` line.

    Replaces a ``PENDING`` placeholder (or any prior value) with ``source_hash``.
    If the model omitted the line entirely, inject one after the first line so
    the caching contract still holds on the next run.
    """
    if _SOURCE_HASH_RE.search(text):
        return _SOURCE_HASH_RE.sub(f"// SOURCE_HASH: {source_hash}", text, count=1)
    lines = text.splitlines()
    insert_at = 1 if lines else 0
    lines.insert(insert_at, f"// SOURCE_HASH: {source_hash}")
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hash_header_dir(headers_dir: Path) -> str:
    """SHA-256 over the header files under ``headers_dir`` (recursive).

    The digest folds in each header's path relative to ``headers_dir`` and its
    bytes, walked in sorted order, so a rename, move, edit, add, or delete of any
    header changes the hash.  Non-header files (see :data:`_HEADER_SUFFIXES`) are
    skipped so documentation churn does not invalidate the cached shim.
    """
    h = hashlib.sha256()
    files = sorted(
        p for p in headers_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _HEADER_SUFFIXES
    )
    for path in files:
        rel = path.relative_to(headers_dir).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _extract_source_hash(text: str) -> str | None:
    """Return the hash on the ``// SOURCE_HASH:`` line, or ``None`` if absent.

    A ``PENDING`` placeholder is treated as "no hash" so it never counts as a
    cache hit.
    """
    m = _SOURCE_HASH_RE.search(text)
    if not m:
        return None
    value = m.group(1)
    return None if value == _SOURCE_HASH_PENDING else value


def _derive_app_name(headers_dir: Path) -> str:
    """Best-effort application name from the header directory name.

    Strips common packaging suffixes (``qcdloop_headers`` -> ``qcdloop``); falls
    back to the raw directory name.  Part 2 may refine this from the driver's
    includes, but the scaffold only needs a stable, sensible default.
    """
    name = headers_dir.name
    for suffix in ("_headers", "-headers", "_include", "_includes", "-include", "_inc"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)] or name
    return name


def _render_placeholder(app_name: str, source_hash: str) -> str:
    """A benign, valid header standing in for the not-yet-implemented shim.

    Deliberately *not* an ``#error``: the scaffold placeholder must be a compilable
    no-op so wiring it into ``build_and_run`` cannot break an unrelated build.
    Part 2's escape hatch (Rule 9) is what emits ``#error`` for genuinely
    unclassifiable functions.
    """
    return (
        f"// {app_name}_interop.hpp — Tracked<T> interop shim (SCAFFOLD PLACEHOLDER)\n"
        f"//\n"
        f"// Generated by agents/tracked_integrator (structure-only pass, Part 1).\n"
        f"// LLM-driven shim generation is not implemented yet (Part 2); this is a\n"
        f"// compilable no-op so the caching/staleness plumbing can be exercised.\n"
        f"// SOURCE_HASH: {source_hash}\n"
        f"#pragma once\n"
    )
