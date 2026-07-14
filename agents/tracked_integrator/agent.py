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

import difflib
import hashlib
import json
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

Then — after the complete header — the C8 patch section: the sentinel line \
`===C8PATCH===` followed by a single JSON array of library-patch records (or `[]` \
if there are none), exactly as specified in rule C8. Emit nothing after the JSON.

Tracked-API classification rules C1-C7 (permanent; apply on equal footing with \
Rules 1-9):

These are not target-specific workarounds — each follows from a property of the \
Tracked API surface itself (its own tracked scalar and container types, its \
factory functions, the arithmetic operators it does and does not define, how a \
named-constants traits template is specialized, and its host-only execution \
model). Because that surface is the same for every integration, these rules hold \
for ANY target library, not just any one. They name ONLY the Tracked API (which \
is always provided to you) and generic C++ placeholders (`<lib>`, `<Name>`, `T`); \
they contain no target-library identifier. Discover every target-specific name \
(types, aliases, functions, traits templates, namespaces) from the headers and \
driver you are given — never assume them.

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
  An identity operator introduces no rounding and should emit no journal record. \
  Scope by STATIC INSTANTIATION, not by run-time path: the operators (and, more \
  generally, the overloads) you must supply are those applied ANYWHERE in the \
  statically-instantiated call graph of the driver's calls — NOT only along the \
  branch the driver's specific inputs happen to select at run time. A dispatcher \
  template that routes to sub-cases by kinematics/flags/values instantiates EVERY \
  branch at compile time regardless of the driver's data, so an operator or \
  overload used in any branch MUST be provided even if this driver's inputs never \
  reach that branch at run time. Never omit an overload with reasoning like "the \
  path we exercise doesn't need it" — if the translation unit compiles the \
  branch, you must shim what that branch uses.

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
  every constant keeps its name in the journal. (This rule governs CLASS \
  templates, where partial SPECIALIZATION on the tracked scalar is the \
  mechanism. For FUNCTION templates that shadow a library primary of the same \
  name, see C7 — partial ORDERING, not partial specialization, is what makes \
  your overload win.)

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

C7. Outrank a library's OWN same-named function template via partial ordering \
  (clarifies Rules 2/3/8; the function-template counterpart of C5). When the \
  target library declares its OWN function template of a name your shim must \
  override — so a qualified `<lib>::foo<...>(x)` call at the library's own \
  definition sites resolves to your tracked overload rather than the library's \
  generic one — every shim overload MUST be STRICTLY MORE SPECIALIZED than that \
  library primary under C++ overload partial ordering. Achieve this by \
  constraining the VALUE parameter to the concrete tracked type \
  (`const tracked::Tracked<T>&` / `const tracked::Complex<T>&`), while carrying \
  whatever LEADING explicit template parameters the call site names — mirror the \
  primary's leading explicit-parameter arity exactly, e.g. for a library \
  `template <class TOutput, class TMass, class TScale> R foo(TMass const&, ...)` \
  emit `template <class TOutput, class TMass, class TScale, class T> \
  tracked::Tracked<T> foo(const tracked::Tracked<T>&, ...)`. A concrete tracked \
  value parameter is more specialized than the library's bare template parameter, \
  so partial ordering picks your overload unambiguously. NEVER introduce a \
  catch-all forwarder `template <..., class Base> foo(const Base&, ...)`: a bare \
  template type parameter in the value position is NOT more specialized than the \
  library's own generic parameter — the two tie under partial ordering and EVERY \
  qualified call becomes ambiguous ("call of overloaded ... is ambiguous"). \
  Provide one constrained overload per concrete tracked argument shape (scalar, \
  complex) instead. Critically, the SAME constrained overload must serve BOTH \
  the library's qualified `foo<A,B,C>(x)` call sites AND any unqualified `foo(x)` \
  sites: carry the leading explicit parameters ON the constrained overload (they \
  may go unused in the body) so an explicit-argument call binds to it directly. \
  Do NOT emit a deduced-only pair plus a separate forwarder — that forwarder IS \
  the catch-all this rule forbids. \
  WRONG (deduced pair + `Base` forwarder; the forwarder ties with the library \
  primary): `template <class T> Tracked<T> foo(const Tracked<T>&, const int&);` \
  and `template <class TOutput,class TMass,class TScale,class Base> auto \
  foo(const Base&, const int&) -> decltype(foo(...));`  // AMBIGUOUS. \
  RIGHT (each constrained overload itself carries the leading explicit params; no \
  forwarder): `template <class TOutput,class TMass,class TScale,class T> \
  tracked::Tracked<T> foo(const tracked::Tracked<T>&, const int&);` and the \
  matching `tracked::Complex<T>` overload. \
  Root cause you must internalize: a deduced-only `template <class T> foo(const \
  tracked::Tracked<T>&, ...)` overload has ONE template parameter, so a qualified \
  `foo<A,B,C>(x)` call supplying THREE explicit arguments cannot select it (too \
  many explicit args) — the call then falls through to the library primary, \
  silently losing journaling; the reflex "add a forwarder to accept the 3 explicit \
  args" reintroduces the ambiguity. The ONLY correct shape is to put the leading \
  explicit parameters directly on each concrete-typed overload so the qualified \
  call binds to it AND it outranks the primary by partial ordering.

C8. Type-boundary annotations (library patch). Some target libraries contain, in \
  their OWN source, an int/bool <-> tracked crossing that a free-function shim \
  cannot bridge, because the Tracked scalar has an EXPLICIT scalar constructor and \
  defines no `operator int` / `operator bool` conversion (discover both from the \
  provided Tracked API headers). Three crossing patterns arise: \
    (a) a tracked scalar value assigned to (or used to initialize) an int/bool \
        lvalue in library code; \
    (b) an int/bool expression passed where a tracked scalar (by value or \
        `const&`) is expected, or bound to a tracked reference / temporary; \
    (c) a tracked scalar compared (`==` / `!=`) against an integer or boolean \
        LITERAL. \
  Do NOT work around these in the shim and do NOT change the Tracked API. Instead \
  emit a LIBRARY PATCH that makes each crossing explicit with an annotation that \
  is a NO-OP when the tracked scalar is a plain scalar (its underlying real type, \
  e.g. `double`) and a transparent boundary marker under the tracked build: \
    (a) wrap the tracked expression in `.value()`, plus `static_cast<int>( ... )` \
        (or the matching integral type) when the lvalue is integral; \
    (b) wrap the int/bool argument in the tracked scalar's explicit constructor \
        (`tracked::Tracked<T>( ... )`, spelled with whatever alias the library \
        uses for that type) at the call / bind site; \
    (c) rewrite `<tracked> == <intlit>` as `<tracked>.value() == <matching \
        floating literal>` (e.g. `== 0` becomes `.value() == 0.0`). \
  These annotations preserve exact semantics — the crossed values are discrete \
  branch tags with no rounding to track — while keeping every tracked<->discrete \
  transition visible in the source. Do NOT instead add a hidden conversion \
  operator or a broad `operator==(tracked, int)`: that would silently re-enable \
  int<->tracked mixing everywhere and erase the type discipline the Tracked \
  datatype exists to provide. Scope by STATIC INSTANTIATION exactly as C3: patch \
  every crossing in any branch the translation unit compiles, not only the \
  run-time path this driver's inputs happen to take. \
  OUTPUT MECHANISM: emit the shim header FIRST, then on its own line the sentinel \
  `===C8PATCH===`, then a SINGLE JSON array of patch records and nothing after it. \
  Each record is an object: \
    {"file": "<library header path relative to the target library header root, \
       e.g. <subdir>/<Header>.h>", \
     "pattern": "a" | "b" | "c", \
     "original": "<exact source substring to replace, copied VERBATIM from that \
       header>", \
     "replacement": "<the annotated substring>", \
     "rule": "<one-line justification, e.g. C8(a) tracked->int assignment>"}. \
  The `original` string MUST occur EXACTLY ONCE in its file — include the whole \
  statement, and the preceding line as well if needed, so it is unique BY \
  CONSTRUCTION (the caller HARD-FAILS if it matches zero or multiple times). Emit \
  NO line numbers and NO hunk headers; the caller synthesizes the unified diff \
  deterministically from `original`/`replacement`. If the target library has no \
  such crossing, emit `===C8PATCH===` followed by `[]`. If a crossing does not fit \
  pattern (a), (b), or (c), do NOT invent a patch: emit, in the shim body, \
  `#error "C8_UNCLASSIFIED_BOUNDARY: <site>"` (a hard build failure, like Rule 9) \
  and omit that site from the JSON array.
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

# C8 (type-boundary library patch).  The LLM emits the shim, then this sentinel on
# its own line, then a JSON array of patch records (see rule C8 in the system
# prompt).  The sentinel is an in-transit delimiter only — it is split off here and
# never written into either on-disk artifact.
_C8_SENTINEL = "===C8PATCH==="

# sha256 of the emitted <app>.patch, stamped into the shim so patch tampering or
# deletion invalidates the cache (a shim keyed only on the header/rule hash would
# otherwise be reused even after its companion patch was changed or removed).
# "NONE" means the target had no boundary sites and no patch file exists.
_PATCH_HASH_RE = re.compile(r"//\s*PATCH_HASH:\s*(\S+)")
_PATCH_HASH_NONE = "NONE"


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

    source_hash = _compute_source_hash(headers_dir)

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
    # header contents is up to date — return it without rewriting.  The check now
    # also validates the companion C8 patch (if the shim declares one): a matching
    # SOURCE_HASH but a missing/edited patch is treated as stale so the build never
    # applies a patch inconsistent with the shim that declared it.
    cache_candidate = Path(existing_shim).resolve() if existing_shim is not None else shim_path
    if cache_candidate.exists():
        cached_text = cache_candidate.read_text(encoding="utf-8")
        cached_hash = _extract_source_hash(cached_text)
        if cached_hash == source_hash and _patch_cache_valid(
            cached_text, cache_candidate, resolved_app_name
        ):
            return cache_candidate

    # (Re)generate.  With a cfg we drive the LLM; without one (scaffold /
    # offline path — e.g. the structural smoke tests) we fall back to the
    # benign placeholder so callers that don't wire up an LLM still get a
    # compilable no-op shim with a valid SOURCE_HASH.
    shim_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path = shim_path.with_name(f"{resolved_app_name}.patch")
    if cfg is None:
        shim_text = _render_placeholder(resolved_app_name, source_hash)
        shim_text = _apply_patch_hash(shim_text, _PATCH_HASH_NONE)
        _remove_if_exists(patch_path)
    else:
        raw = _generate_shim(
            headers_dir=headers_dir,
            driver_path=driver_path,
            tracked_repo_path=tracked_repo_path,
            existing_shim=Path(existing_shim).resolve() if existing_shim else None,
            app_name=resolved_app_name,
            cfg=cfg,
        )
        # Split the single LLM response into the shim header and the C8 patch
        # records (the `===C8PATCH===` sentinel is only an in-transit delimiter).
        shim_raw, records = _split_llm_response(raw)
        shim_text = _strip_code_fences(shim_raw)

        # Synthesize the library patch (deterministic unified diff) from the
        # records, if any.  git-apply-able paths are relative to the repo root.
        repo_root = _find_repo_root(headers_dir) or headers_dir
        patch_text = _synthesize_patch(records, headers_dir, repo_root)

        # Post-process: the model emits `// SOURCE_HASH: PENDING`; stamp the real
        # hash (step 4 of the spec), then stamp the patch hash so patch changes
        # invalidate the cache.
        shim_text = _apply_source_hash(shim_text, source_hash)
        if patch_text is not None:
            patch_bytes = patch_text.encode("utf-8")
            shim_text = _apply_patch_hash(
                shim_text, hashlib.sha256(patch_bytes).hexdigest()
            )
            patch_path.write_bytes(patch_bytes)
        else:
            shim_text = _apply_patch_hash(shim_text, _PATCH_HASH_NONE)
            _remove_if_exists(patch_path)

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
    # Return the raw response; the caller splits off the C8 patch section (on the
    # `===C8PATCH===` sentinel) and strips code fences from the shim part.
    return text


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
        f"Emit the complete contents of `{app_name}_interop.hpp` — no prose, no "
        f"markdown fences. Include a `// SOURCE_HASH: PENDING` line near the top; "
        f"the caller replaces PENDING with the real hash. Every generated "
        f"overload, specialization, and annotation must carry a comment naming "
        f"the rule that justified it.\n"
        f"Then, on its own line, emit the sentinel `===C8PATCH===` followed by a "
        f"single JSON array of C8 library-patch records (or `[]` if the library "
        f"has no int/bool<->tracked crossing), exactly as specified in rule C8. "
        f"Emit nothing after the JSON array."
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
# C8 library-patch synthesis (Part 2, type-boundary annotations)
# ---------------------------------------------------------------------------

def _split_llm_response(text: str) -> tuple[str, list[dict]]:
    """Split the raw LLM response into (shim text, C8 patch records).

    The response is ``<shim>`` then, optionally, the ``===C8PATCH===`` sentinel
    followed by a JSON array of patch records.  A response without the sentinel
    (older prompts / a model that emitted no C8 section) yields no records.
    """
    if _C8_SENTINEL not in text:
        return text, []
    shim_part, _, patch_part = text.partition(_C8_SENTINEL)
    return shim_part, _parse_patch_records(patch_part)


def _parse_patch_records(patch_part: str) -> list[dict]:
    """Parse the JSON array following the ``===C8PATCH===`` sentinel.

    A bare ``[]`` (or an empty section) means "no boundary sites".  Malformed
    JSON is a hard error — the C8 contract is machine-readable by design, so a
    parse failure surfaces the ambiguity rather than silently dropping patches.
    """
    body = _strip_code_fences(patch_part.strip())
    if not body:
        return []
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"tracked_integrator C8: could not parse patch JSON after "
            f"{_C8_SENTINEL!r}: {exc}"
        ) from exc
    if not isinstance(data, list):
        raise RuntimeError(
            "tracked_integrator C8: patch section must be a JSON array, got "
            f"{type(data).__name__}"
        )
    return data


def _synthesize_patch(
    records: list[dict], headers_dir: Path, repo_root: Path
) -> str | None:
    """Turn C8 edit records into a deterministic, git-apply-able unified diff.

    Each record names a library header (relative to ``headers_dir``), an exact
    ``original`` substring, and its annotated ``replacement``.  For robustness the
    model produces the *semantic* edit; Python owns the byte-exact diff:

    * ``original`` MUST occur EXACTLY ONCE in the clean file — 0 or >1 is a hard
      failure that surfaces an ambiguous / too-short ``original`` (never a silent
      first-match), the same "surface the ambiguity" discipline as the C-rule
      UNCLASSIFIED escape hatch.
    * the diff is produced by :func:`difflib.unified_diff` with ``a/<repo-rel>`` /
      ``b/<repo-rel>`` labels so ``git apply -p1`` from the repo root applies it.

    Returns the combined diff text, or ``None`` when there are no records.
    """
    if not records:
        return None

    edits_by_file: dict[str, list[tuple[str, str, str]]] = {}
    for rec in records:
        try:
            relfile = rec["file"]
            original = rec["original"]
            replacement = rec["replacement"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"tracked_integrator C8: malformed patch record {rec!r}: {exc}"
            ) from exc
        rule = rec.get("rule", "")
        edits_by_file.setdefault(relfile, []).append((original, replacement, rule))

    diff_chunks: list[str] = []
    for relfile, edits in sorted(edits_by_file.items()):
        target = (headers_dir / relfile).resolve()
        if not target.is_file():
            raise RuntimeError(
                f"tracked_integrator C8: patch target not found: {relfile} "
                f"(resolved {target})"
            )
        original_text = target.read_text(encoding="utf-8")
        patched_text = original_text
        for original, replacement, rule in edits:
            clean_count = original_text.count(original)
            if clean_count != 1:
                raise RuntimeError(
                    f"tracked_integrator C8: 'original' must occur exactly once in "
                    f"{relfile} (found {clean_count}) [{rule}]: {original!r}"
                )
            if patched_text.count(original) < 1:
                # count==1 in the clean file but already consumed by a prior edit
                # → duplicate / overlapping record.  Surface it, don't no-op.
                raise RuntimeError(
                    f"tracked_integrator C8: 'original' already consumed by an "
                    f"earlier edit in {relfile} [{rule}]: {original!r}"
                )
            patched_text = patched_text.replace(original, replacement, 1)

        rel = target.relative_to(repo_root).as_posix()
        diff = difflib.unified_diff(
            original_text.splitlines(keepends=True),
            patched_text.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )
        diff_chunks.append("".join(diff))

    combined = "".join(diff_chunks)
    return combined if combined.strip() else None


def _apply_patch_hash(text: str, patch_hash: str) -> str:
    """Stamp ``// PATCH_HASH:`` into the shim (after the SOURCE_HASH line).

    Replaces an existing PATCH_HASH line if present; otherwise appends the line
    immediately after the SOURCE_HASH line (which :func:`_apply_source_hash`
    guarantees exists by the time this runs).
    """
    line = f"// PATCH_HASH: {patch_hash}"
    if _PATCH_HASH_RE.search(text):
        return _PATCH_HASH_RE.sub(line, text, count=1)
    if _SOURCE_HASH_RE.search(text):
        return _SOURCE_HASH_RE.sub(lambda m: m.group(0) + "\n" + line, text, count=1)
    # No SOURCE_HASH anchor (shouldn't happen post _apply_source_hash) — prepend.
    lines = text.splitlines()
    insert_at = 1 if lines else 0
    lines.insert(insert_at, line)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _patch_cache_valid(shim_text: str, shim_path: Path, app_name: str) -> bool:
    """True iff the shim's declared PATCH_HASH matches the on-disk patch.

    A shim with no PATCH_HASH line predates C8 → nothing to validate.  A shim
    declaring ``NONE`` must have no companion patch (a stray one would be applied
    erroneously); a shim declaring a hash must have a matching ``<app>.patch``.
    Any mismatch returns False so the caller regenerates.
    """
    m = _PATCH_HASH_RE.search(shim_text)
    if not m:
        return True
    declared = m.group(1)
    patch_path = shim_path.with_name(f"{app_name}.patch")
    if declared == _PATCH_HASH_NONE:
        return not patch_path.exists()
    if not patch_path.exists():
        return False
    actual = hashlib.sha256(patch_path.read_bytes()).hexdigest()
    return actual == declared


def _find_repo_root(start: Path) -> Path | None:
    """Walk up from ``start`` to the nearest directory containing ``.git``."""
    start = Path(start).resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _remove_if_exists(path: Path) -> None:
    """Delete ``path`` if present (used to clear a now-stale companion patch)."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ruleset_hash() -> str:
    """SHA-256 of the LLM system prompt (the classification rule set).

    Folded into the shim's ``SOURCE_HASH`` so that a rule refinement (adding C8,
    tightening C7, etc.) invalidates every cached shim automatically.  Without
    this, a shim cached against an unchanged target-header tree would be reused
    verbatim even after the rules that generated it changed — silently serving a
    stale shim across a whole Stage-2 sweep and defeating any re-validation.
    """
    return hashlib.sha256(_SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def _compute_source_hash(headers_dir: Path) -> str:
    """The shim's staleness key: target headers AND the rule-set version.

    Combining both means the cache hits only when BOTH the target library's
    headers and the generating rule set are unchanged.  A change to either forces
    regeneration, so ``existing_shim=None`` first-time integrations and rule
    refinements both invalidate correctly.
    """
    h = hashlib.sha256()
    h.update(_hash_header_dir(headers_dir).encode("utf-8"))
    h.update(b"\0")
    h.update(_ruleset_hash().encode("utf-8"))
    return h.hexdigest()


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
