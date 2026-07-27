"""The ``clonable_leaf`` predicate — leaf-callee promotion gate (Subtask L2).

A **leaf callee** is a function called *from* a chain line whose own body holds no
chain line, so today the closure fixed point hits its call and refuses
(``chain_closure_escapes``).  Rule (d) (``chain_promote._apply_rule_d``) instead
*clones* such a leaf into a promoted per-integral frame — but only when it is
provably safe to do so.  This module owns that proof: a **pure, side-effect-free**
predicate encoding ``LEAF_CALLEE_PROMOTION_DESIGN.md`` §1.2 clauses (1)–(4).

    clonable_leaf(g) :=
       (1) g's body is available in the analysed source (not extern / vendored-binary);
     ∧ (2) g's body, reads promoted to dd, calls ONLY:
             - another clonable_leaf callee (recurse), OR
             - a dd TERMINATION-BOUNDARY symbol (§2.6):
                 (i)   a vendored ``quad::ddfun`` op (or a ``<cmath>`` op the Gap-A
                       bridge redirects onto the vendored surface),
                 (ii)  a Class-1 SYNTHESIZED wrapper (L1′ ``is_class1_synthesizable``),
                 (iii) a Class-2 / source symbol the source instantiates at dd;
     ∧ (3) g is NOT self-recursive under a SAME-NAME overload set a rename cannot
           separate (STOP #K guard — §3);
     ∧ (4) cloning g does not require widening a shared g PARAMETER a non-chain caller
           also binds — automatically satisfied for a *pure clone* (own params).

**Conservative-parser contract (§1.2, Appendix).**  A false negative (refuse a
clonable leaf) is *safe* — the leaf simply stays a ``chain_closure_escapes``
frontier.  A false positive (clone an un-instantiable leaf) is the **STOP #K
hard-fail** we must never ship, so every uncertainty here resolves to ``ok=False``.
The predicate emits nothing and mutates nothing; it only answers whether rule (d)
*may* clone ``g``.

No app-specific identifiers appear here.  Classification is by *structure* (call
targets, boundary membership, overload count) against injected surfaces — the L1′
``surface`` / ``is_class1_synthesizable``, a ``source_instantiates_at_dd`` query, a
``resolve_primary_body`` body lookup, and the ``CallGraph``.  ``ql``/``Lnrat``/
``kLog`` appear only in comments and tests as qcdloop-representative examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agents.integrator_base.regional import _MATH_FN_NAMES, _VENDORED_NS_ROOTS
from agents.integrator_base.shallow_wrapper import _template_param_names
from agents.shared import region_scan

# The design's rule-(d) recursion cap (§2.8): "recursion depth > 3" aborts.  The
# circuit breaker in ``chain_promote`` owns the frame-count threshold (8); this cap
# is the predicate's own backstop so a pathological source graph cannot recurse
# unboundedly even though ``seen`` already makes it cycle-safe.
_MAX_RECURSION_DEPTH = 3

# C++ keywords that take a ``(`` but are control flow / casts / operators, not callees.
# A ``keyword (`` is never a function call the clause-(2) classifier should inspect.
_CALL_KEYWORDS = frozenset({
    "if", "for", "while", "switch", "return", "sizeof", "static_cast",
    "dynamic_cast", "reinterpret_cast", "const_cast", "decltype", "noexcept",
    "alignof", "catch", "throw",
})


@dataclass
class ClonableLeafResult:
    """Outcome of :func:`clonable_leaf`.

    ``ok`` is True iff every §1.2 clause holds — rule (d) may clone the leaf.
    ``reason`` records the first failing clause (for the ``chain_closure_escapes``
    diagnostic) or a one-line success note.  ``transitive_deps`` are the inner
    *clonable leaves* discovered by clause (2) recursion (NOT the Class-1 wrappers
    or source symbols, which are synthesized / read at the boundary, never cloned) —
    rule (d) pulls each into ``F`` as its own frame.
    """

    ok: bool
    reason: str
    transitive_deps: list[str] = field(default_factory=list)


def _last_segment(qualified_name: str) -> str:
    """Final ``::`` segment of a (possibly-qualified) name (``ql::kLog`` -> ``kLog``)."""
    return qualified_name.rsplit("::", 1)[-1]


def _root_segment(qual: str) -> str:
    """Leading ``::`` segment of a qualifier chain (``ql::kLog`` -> ``ql``)."""
    return qual.split("::", 1)[0] if "::" in qual else qual


def is_dd_boundary(qual: str, last: str, *, surface, source_instantiates_at_dd,
                   is_class1_synthesizable, resolve_primary_body) -> bool:
    """True iff call target ``(qual, last)`` resolves at the dd termination boundary
    (§2.6) — a value/op the pipeline synthesizes or reads, NOT a frame to clone.

    The four boundary kinds (§2.6):

    * (i)  a vendored ``quad::ddfun`` op (qualifier root is a vendored namespace), or a
      ``<cmath>`` op the Gap-A bridge redirects onto the vendored surface;
    * (iii) a Class-2 / source symbol the source instantiates at dd (a constant or
      coefficient-table accessor — ``_ipio2`` / ``_half`` / ``_pi2o6`` / ``_C``);
    * (ii) a Class-1 synthesizable wrapper (L1′ ``is_class1_synthesizable``).

    Shared by :func:`clonable_leaf`'s clause-(2) classifier and rule (d)'s
    frame-discovery candidate filter so the two never disagree on what is a leaf.
    """
    root = _root_segment(qual)
    if root in _VENDORED_NS_ROOTS:
        return True
    if last in _MATH_FN_NAMES:
        return last in (surface.scalar_ops | surface.complex_ops)
    if source_instantiates_at_dd(last):
        return True
    body = resolve_primary_body(last)
    if body and is_class1_synthesizable(last, body, surface):
        return True
    return False


def _body_only(def_text: str) -> str:
    """The ``{...}`` body of a function definition (or ``def_text`` unchanged).

    A leaf's ``primary_body`` is its full definition text — signature included — so a
    naive call scan would read the function's OWN name in its signature
    (``TOutput Lnrat(...)``) as a self-call.  Return only the outermost brace body so
    the scan sees calls the body actually makes, not the declarator.  When no brace is
    present (a caller already passed a bare body), the text is returned unchanged.
    """
    open_i = def_text.find("{")
    if open_i < 0:
        return def_text
    depth = 0
    for i in range(open_i, len(def_text)):
        c = def_text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return def_text[open_i + 1:i]
    return def_text[open_i + 1:]


def scan_call_targets(body_text: str) -> list[tuple[str, str]]:
    """Every call target in ``body_text`` as ``(qualified_name, last_segment)``.

    A token scan (reusing :mod:`agents.shared.region_scan`) that recognizes BOTH a
    plain call ``Ns::g(...)`` and a template-id call ``Ns::g<...>(...)`` (qcdloop's
    dominant form for ``ql::ddilog<T,U,V>(x)`` and member accessors
    ``Constants<T>::_ipio2<...>()``), mirroring
    :func:`agents.patcher.chain_promote._scan_calls` but returning only the target
    identity the clause-(2) classifier needs (no argument analysis).  Scans only the
    function BODY (:func:`_body_only`), so the definition's own signature is never read
    as a self-call.  Over-detection of a stray comparison is harmless — a target is
    only *acted on* by the classifier when it is not already a boundary or frame.
    """
    toks = region_scan._tokenize(_body_only(body_text))
    n = len(toks)
    out: list[tuple[str, str]] = []
    i = 0
    while i < n:
        t = toks[i]
        paren_i: int | None = None
        if region_scan._is_ident_tok(t.text) and i + 1 < n:
            if toks[i + 1].text == "(":
                paren_i = i + 1
            elif toks[i + 1].text == "<":
                d = 0
                j2 = i + 1
                while j2 < n:
                    tx2 = toks[j2].text
                    if tx2 == "<":
                        d += 1
                    elif tx2 == ">":
                        d -= 1
                    elif tx2 == ">>":
                        d -= 2
                    j2 += 1
                    if d <= 0:
                        break
                if j2 < n and toks[j2].text == "(":
                    paren_i = j2
        if paren_i is not None and t.text not in _CALL_KEYWORDS:
            prefix: list[str] = []
            k = i - 1
            while k - 1 >= 0 and toks[k].text == "::" \
                    and region_scan._is_ident_tok(toks[k - 1].text):
                prefix.insert(0, toks[k - 1].text)
                k -= 2
            last = t.text
            qual = "::".join(prefix + [last]) if prefix else last
            out.append((qual, last))
            i += 1                       # advance ONE token: scan nested calls in turn
            continue
        i += 1
    return out


def clonable_leaf(
    qualified_name: str,
    primary_body: str | None,
    g_params=None,
    *,
    call_graph=None,
    surface,
    is_class1_synthesizable,
    source_instantiates_at_dd,
    resolve_primary_body,
    scalar_type: str,
    frame_names=frozenset(),
    type_tokens=frozenset(),
    binds_shared_param=None,
    seen: set[str] | None = None,
    depth: int = 0,
    max_depth: int = _MAX_RECURSION_DEPTH,
) -> ClonableLeafResult:
    """Whether rule (d) may clone leaf ``qualified_name`` (§1.2 clauses 1–4).

    Pure and side-effect-free: it reads source only through the injected callables and
    never mutates the tree.

    Parameters
    ----------
    qualified_name, primary_body, g_params:
        The leaf under test — its (possibly-qualified) name, its primary definition
        body text (``None`` when unavailable — a clause-(1) refusal), and its
        parameter list (``[(core_type, name), ...]`` or ``None``; used only by the
        clause-(4) shared-parameter check).
    call_graph:
        The :class:`~agents.patcher.call_graph.CallGraph` — consulted for clause (3)
        (does ``qualified_name`` name a SAME-NAME overload SET?) and, optionally,
        clause (1).  ``None`` disables the overload-set check (single-def assumption).
    surface, is_class1_synthesizable:
        The L1′ Class-1 machinery (``VendoredSurface`` + the pure manifest query).
    source_instantiates_at_dd:
        ``name -> bool`` — does the source instantiate this Class-2 / accessor symbol
        at dd (double primary at ``T=ddouble``, or the enriched dd source)?
    resolve_primary_body:
        ``name -> str | None`` — a callee's primary definition body text, for the
        Class-1 query and the clause-(2) recursion.  ``None`` result => body
        unavailable (the callee then classifies only if vendored / math / source).
    scalar_type:
        The dd scalar spelling (informational; the boundary is decided structurally).
    frame_names:
        Names ALREADY in the chain frame set ``F`` — a call to one of these is a
        chain-internal edge (rule (c) territory), not a leaf to classify.
    type_tokens:
        Type-name tokens whose ``T(...)`` call head is a functional CAST, not a
        callee — e.g. the chain's dd container tokens (``TOutput``/``TMass``).  The
        leaf's own ``template<...>`` parameters are added automatically, so a
        ``TOutput(x)`` cast in the body is never mistaken for a callee.
    binds_shared_param:
        Optional ``param_name -> bool`` — True iff a NON-chain caller binds this
        parameter of the *shared original* at caller precision, so widening it inward
        would corrupt that caller (clause 4).  A pure clone gets its own params, so
        this is satisfied by default (``None`` => every param OK).
    seen, depth:
        Recursion bookkeeping (cycle guard + depth cap).  Callers leave these at
        their defaults.
    """
    name_last = _last_segment(qualified_name)
    seen = set() if seen is None else seen

    # Cycle guard (§2.7 — the special-function call graph is a DAG, but a source
    # cycle must never spin): a leaf already on the recursion stack is treated as an
    # accepted internal edge (it is being cloned by an outer frame).
    if name_last in seen:
        return ClonableLeafResult(True, f"{name_last}: already on the clone stack "
                                  f"(cycle-safe internal edge)")

    # Predicate backstop for the design's rule-(d) recursion cap (§2.8).  Rule (d)
    # also enforces the 8-frame / depth-3 circuit breaker; this refusal lets it map a
    # runaway recursion to ``chain_closure_oversized`` rather than loop.
    if depth > max_depth:
        return ClonableLeafResult(
            False, f"{name_last}: rule-(d) recursion depth {depth} exceeds "
            f"{max_depth} (§2.8 circuit breaker)")

    # ---- clause (1): body available in the analysed source ---------------------
    if primary_body is None or not primary_body.strip():
        return ClonableLeafResult(
            False, f"{name_last}: no body available in the analysed source "
            f"(extern / vendored-binary symbol) — clause (1)")

    targets = scan_call_targets(primary_body)

    # Functional casts ``T(x)`` are not callees.  The leaf's own template parameters
    # (``TOutput``/``TMass``/``TScale``) plus the caller-supplied chain type tokens
    # are the cast set; a call whose target is one of them is skipped in clause (2).
    cast_tokens = set(type_tokens) | _template_param_names(primary_body) | {scalar_type}

    # ---- clause (3): self-recursion under a same-name overload set --------------
    # A clone renames g -> g_<integral> and rewrites in-body self-calls to the clone
    # name (§3.2), so a self-call is safe WHEN g has a single definition.  When g
    # names a same-name OVERLOAD SET, C++ re-selects a sibling by argument type
    # regardless of the explicit ``<...>`` — the STOP #K recursion pit a rename
    # cannot separate — so refuse.
    self_calls = [q for (q, last) in targets if last == name_last]
    if self_calls:
        n_defs = len(call_graph.defs.get(name_last, [])) if call_graph is not None else 1
        if n_defs > 1:
            return ClonableLeafResult(
                False, f"{name_last}: self-recursive under a same-name overload set "
                f"({n_defs} overloads) — a rename cannot separate the recursion "
                f"(STOP #K, clause 3)")

    # ---- clause (4): no inward widening of a shared parameter -------------------
    # A pure clone gets its own parameters, so a leaf never NEEDS to widen the shared
    # original's signature (§8.2 holds by construction).  Refuse only if the caller
    # supplies explicit evidence that a parameter cannot be given the clone's own
    # (widened) type because a non-chain caller binds the shared original at caller
    # precision on that same parameter.
    if binds_shared_param is not None and g_params:
        for entry in g_params:
            pname = entry[1] if isinstance(entry, (tuple, list)) else entry
            if binds_shared_param(pname):
                return ClonableLeafResult(
                    False, f"{name_last}: promotion demands inward dd on shared "
                    f"parameter {pname!r} a non-chain caller binds — clause (4)/§8.2")

    # ---- clause (2): every callee resolves at the dd termination boundary -------
    inner_seen = seen | {name_last}
    transitive_deps: list[str] = []

    def _is_synth_dep(inner: str) -> bool:
        """Transitive Class-1 dep query for :func:`is_class1_synthesizable`."""
        b = resolve_primary_body(inner)
        return bool(b) and is_class1_synthesizable(inner, b, surface)

    for (qual, last) in targets:
        if last == name_last:
            continue                       # self-call (clause 3 already ruled on it)
        if last in frame_names:
            continue                       # chain-internal frame edge (rule c)
        if "::" not in qual and last in cast_tokens:
            continue                       # functional cast ``T(x)``, not a callee
        root = _root_segment(qual)
        # (i) vendored quad::ddfun op — resolves at dd, no cloning.
        if root in _VENDORED_NS_ROOTS:
            continue
        # (i') <cmath> op the Gap-A bridge redirects onto the vendored surface — a
        # boundary iff the vendored surface actually provides it (STOP #S analogue).
        if last in _MATH_FN_NAMES:
            if last in (surface.scalar_ops | surface.complex_ops):
                continue
            return ClonableLeafResult(
                False, f"{name_last}: body calls math op {last!r} the vendored "
                f"surface does not provide for any operand kind (STOP #S) — clause (2)")
        # (iii) Class-2 / source symbol the source instantiates at dd (constants,
        # coefficient-table accessors: _ipio2 / _half / _pi2o6 / _C / _num_C).
        if source_instantiates_at_dd(last):
            continue
        # (ii) Class-1 synthesizable wrapper (L1′): the pipeline emits its dd overload.
        body = resolve_primary_body(last)
        if body and is_class1_synthesizable(last, body, surface,
                                            is_synth_dep=_is_synth_dep):
            continue
        # otherwise: is it itself a clonable leaf?  Recurse (§1.2 clause 2 first arm).
        if body:
            sub = clonable_leaf(
                last, body, None,
                call_graph=call_graph, surface=surface,
                is_class1_synthesizable=is_class1_synthesizable,
                source_instantiates_at_dd=source_instantiates_at_dd,
                resolve_primary_body=resolve_primary_body,
                scalar_type=scalar_type, frame_names=frame_names,
                type_tokens=type_tokens,
                binds_shared_param=binds_shared_param,
                seen=inner_seen, depth=depth + 1, max_depth=max_depth)
            if sub.ok:
                if last not in transitive_deps:
                    transitive_deps.append(last)
                for dep in sub.transitive_deps:
                    if dep not in transitive_deps:
                        transitive_deps.append(dep)
                continue
            return ClonableLeafResult(
                False, f"{name_last}: callee {last!r} is not clonable "
                f"({sub.reason}) — clause (2)")
        # no body, not vendored / math / source, not a resolvable leaf → refuse.
        return ClonableLeafResult(
            False, f"{name_last}: body names {last!r} which is neither vendored, a "
            f"synthesizable Class-1 wrapper, a source-instantiated symbol, nor a "
            f"clonable leaf with an available body — clause (2)")

    return ClonableLeafResult(
        True, f"{name_last}: clonable leaf (body available, all callees at the dd "
        f"boundary or clonable, no unseparable self-recursion, no inward param widen)",
        transitive_deps=transitive_deps)
