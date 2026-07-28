"""Shallow app-wrapper synthesis — the Class-1 extension of the Gap-A machinery.

The Gap-A bridge (:mod:`agents.integrator_base.regional`) already synthesizes
overloads that redirect a *namespace-qualified* ``<cmath>`` call
(``Ns::sqrt(promoted)``) onto the vendored ``quad::ffun`` / ``quad::ddfun`` surface,
so a promoted (extended-typed) operand is not narrowed to a built-in float.  This
module extends that machinery **one delegation hop**: an app-qualified call
``Ns::g(promoted)`` where ``g`` is NOT a ``<cmath>`` name, but whose *primary body*
is a shallow delegation to such an op (or a member accessor, or a scalar expression
over the parameter).  For those, the pipeline can synthesize ``g``'s extended
overload **mechanically** from the primary's own one-line body + the vendored
surface — no LLM, no vendored app-specific header.

This is the L1′ subtask of the leaf-callee promotion design (v3, §2.2 / §8):
``runs/qcdloop/LEAF_CALLEE_PROMOTION_DESIGN.md``.  It produces the machinery that
rule (d) (Subtask L2) will consume via :func:`is_class1_synthesizable`.

Design invariants (hard constraints from the subtask brief):

* **No app-specific identifiers.**  The recognizer works by inspecting a primary's
  *body shape*; the qcdloop wrappers (``kAbs``/``kLog``/``Real``/``Sign``/``iszero``)
  it happens to synthesize are an *emergent consequence*, never an enumerated list.
  (`[[feedback_no_placeholder_patterns]]`.)
* **``_MATH_FN_NAMES`` untouched.**  The standard-math vocabulary lives in
  :mod:`agents.integrator_base.regional`; this module *imports* it (the delegation
  target must be a ``<cmath>`` op) but never extends it with app names.
* **Conservative parser.**  A false negative (refusing a recognizable wrapper) is
  safe — the leaf falls back to the LLM hint path.  A false positive (emitting a
  broken overload for a non-shallow body) is the STOP #K / STOP #P hard-fail we
  must never ship.  Every clause rejects on the first sign of doubt.

The four recognized body shapes (all a single ``return <expr>;``):

1. **Delegation** — ``return <Ns>::<fn>(<arg>);`` where ``<fn> ∈ _MATH_FN_NAMES``
   and ``<Ns>`` is not the vendored root.  Transform: redirect the inner call to the
   vendored equivalent (``Kokkos::abs`` → ``quad::ddfun::abs``).
2. **Accessor** — ``return <arg>.<member>();`` where ``<member>`` is a container
   accessor (``real``/``imag``/…).  Transform: re-emit the accessor on the promoted
   parameter (the vendored complex provides ``.real()``/``.imag()``).
3. **Scalar-expression** — ``return <expr>;`` where ``<expr>`` is a scalar
   arithmetic/comparison over the parameter using only operators, literals, and
   functional casts *to the parameter's own type*.  Transform: substitute the
   promoted type for the parameter's type token throughout the body, so a
   ``double(0) < x`` becomes ``ddouble(0) < x`` (a bare re-emit does NOT compile —
   ``double(0) < ddouble`` has no operator; verified empirically, §6 probe).
4. **Transitive** — ``return <expr>;`` where ``<expr>``'s only non-boundary call is
   itself a Class-1-recognized wrapper (recurse).  Transform: the same param-type
   substitution; the inner wrapper's own overload must be emitted first / present.

All emitted overloads use ``auto`` return deduction, so the emitter needs **zero**
app-specific return-type knowledge (verified: ``auto`` deduces ``ddouble`` /
``ddcomplex`` / ``int`` / ``bool`` correctly for every shape).  A *template*-param
primary (``T kAbs(T)``) is instantiable at either the vendored scalar or complex, so
the emitter produces both overloads — each guarded by whether the vendored surface
actually provides that op for that operand kind (the STOP #S guard).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agents.integrator_base.regional import _MATH_FN_NAMES, _VENDORED_NS_ROOTS

# Recursion / size backstops (a shallow wrapper is trivial by construction; these
# only bound a pathological or adversarial body so recognition always terminates).
_MAX_TRANSITIVE_DEPTH = 8

# Body-shape classifications.
FORM_DELEGATION = "delegation"
FORM_ACCESSOR = "accessor"
FORM_SCALAR_EXPR = "scalar_expr"
FORM_TRANSITIVE = "transitive"

# Container member accessors that survive the promotion unchanged (the vendored
# complex provides ``.real()`` / ``.imag()``).  Framework-agnostic: the standard
# complex-accessor spellings, the same vocabulary as the ``real``/``imag`` free
# functions already in ``_MATH_FN_NAMES``.
_ACCESSOR_MEMBERS = frozenset({"real", "imag"})


class ShallowWrapperError(RuntimeError):
    """A recognizer/emitter invariant was violated (never a silent bad emission)."""


@dataclass(frozen=True)
class Recognition:
    """The result of classifying one primary body as a Class-1 shallow wrapper.

    ``form`` is one of the ``FORM_*`` constants.  ``inner_fn`` / ``inner_root`` are
    the delegated ``<cmath>`` op and its (non-vendored) qualifier for a delegation
    body; ``member`` is the accessor name for an accessor body; ``transitive_dep``
    is the name of the Class-1 wrapper an inner call delegates to for a transitive
    body.  ``param_type`` is the primary parameter's core type spelling (the token
    the emitter substitutes with the promoted type); ``param_is_template`` marks a
    generic ``T``-parameter primary (instantiable at both scalar and complex).
    ``param_name`` is the primary's own parameter spelling; ``body_expr`` is the
    return expression text (comment/whitespace-normalized) the emitter rewrites.
    """

    fn: str
    form: str
    param_type: str
    param_name: str
    body_expr: str
    param_is_template: bool = False
    inner_root: str | None = None
    inner_fn: str | None = None
    member: str | None = None
    transitive_dep: str | None = None


# --------------------------------------------------------------------------- #
# vendored-surface model (derived from the concrete C++ spellings, not baked in)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class VendoredSurface:
    """The vendored extended-precision op surface a shim can call.

    Derived from the ``RegionalSpec`` scalar/complex spellings so the recognizer /
    emitter carry no framework-specific knowledge.  ``root`` is the vendored
    namespace (``quad::ddfun``), ``scalar`` / ``complex`` the concrete types.
    ``scalar_ops`` / ``complex_ops`` are the ``_MATH_FN_NAMES`` names the vendored
    headers actually provide for a real vs complex operand — the STOP #S guard: a
    delegation to a ``<cmath>`` op the vendored surface does NOT provide for that
    operand kind is refused, never invented.
    """

    root: str
    scalar: str
    complex: str
    scalar_ops: frozenset[str]
    complex_ops: frozenset[str]


def surface_from_spelling(cpp_scalar: str, cpp_complex: str,
                          scalar_ops: frozenset[str] | None = None,
                          complex_ops: frozenset[str] | None = None) -> VendoredSurface:
    """Build a :class:`VendoredSurface` from the concrete C++ type spellings.

    ``cpp_scalar`` is e.g. ``quad::ddfun::ddouble``; the vendored root is everything
    before the last ``::``.  ``scalar_ops`` / ``complex_ops`` default to the full
    ``_MATH_FN_NAMES`` vocabulary ("assume the vendored surface provides every
    ``<cmath>`` op"); a caller that knows the vendored headers' real op set passes
    narrower sets so the STOP #S guard refuses a delegation to an unavailable op.
    """
    root = cpp_scalar.rsplit("::", 1)[0]
    full = frozenset(_MATH_FN_NAMES)
    return VendoredSurface(
        root=root, scalar=cpp_scalar, complex=cpp_complex,
        scalar_ops=scalar_ops if scalar_ops is not None else full,
        complex_ops=complex_ops if complex_ops is not None else full)


def scan_vendored_ops(header_texts: list[str], scalar: str,
                      complex_type: str) -> tuple[frozenset[str], frozenset[str]]:
    """``(scalar_ops, complex_ops)`` — the ``_MATH_FN_NAMES`` ops the vendored headers
    actually define, split by whether the op takes a scalar or complex argument.

    Scans free-function *definitions/declarations* ``<ret> <op>(<arg>)`` for each
    ``_MATH_FN_NAMES`` name and classifies by the first argument's type (a ``complex``
    core → ``complex_ops``, else ``scalar_ops``).  This is what grounds the STOP #S
    guard: a delegation to a ``<cmath>`` op the vendored surface does not provide for
    an operand kind is refused, not invented.  Framework-agnostic — the op names come
    from ``_MATH_FN_NAMES`` and the type spellings from the caller.
    """
    scalar_core = scalar.rsplit("::", 1)[-1]
    complex_core = complex_type.rsplit("::", 1)[-1]
    s_ops: set[str] = set()
    c_ops: set[str] = set()
    for raw in header_texts:
        src = strip_comments(raw)
        for op in _MATH_FN_NAMES:
            for m in re.finditer(
                    r'(?<![\w:])' + re.escape(op) + r'\s*\(([^);]*)', src):
                arg = m.group(1)
                # skip a preceding '.' (member access) — a free function only
                pre = src[:m.start()].rstrip()
                if pre.endswith(".") or pre.endswith("->"):
                    continue
                if complex_core in arg:
                    c_ops.add(op)
                elif scalar_core in arg:
                    s_ops.add(op)
    # ``real``/``imag`` are provided as complex accessors/free-fns even if declared
    # via a member; ensure they are available as complex ops when present anywhere.
    return frozenset(s_ops), frozenset(c_ops)


# --------------------------------------------------------------------------- #
# body isolation
# --------------------------------------------------------------------------- #

_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
# C++ control-flow / multi-statement keywords whose presence in a body means the
# wrapper is NOT a single straight-line return (clause-2 refusal).
_CONTROL_KEYWORDS = frozenset({
    "if", "else", "for", "while", "do", "switch", "case", "goto", "try", "catch",
})
# A functional cast / call head ``Ident(`` or ``Ns::Ident(`` (captures the leading
# qualified name so we can tell a math-op call from a cast from an accessor).
_CALL_HEAD_RE = re.compile(r'(?<![\w:])((?:[A-Za-z_]\w*\s*::\s*)*)([A-Za-z_]\w*)\s*\(')


def strip_comments(text: str) -> str:
    """Return ``text`` with C/C++ comments removed (whitespace preserved)."""
    text = _BLOCK_COMMENT_RE.sub(" ", text)
    text = _LINE_COMMENT_RE.sub("", text)
    return text


def _match_brace(text: str, open_idx: int) -> int | None:
    """Index of the ``}`` matching ``text[open_idx] == '{'`` (or ``None``)."""
    depth = 0
    for i in range(open_idx, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _balanced_args(text: str, open_paren: int) -> tuple[str, int]:
    """Args between ``text[open_paren] == '('`` and its match, plus the close index."""
    depth = 0
    for i in range(open_paren, len(text)):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren + 1:i], i
    return text[open_paren + 1:], len(text) - 1


def _split_top_level_semicolons(text: str) -> list[str]:
    """Split ``text`` on ``;`` not nested inside ``()``/``{}``/``[]``."""
    out: list[str] = []
    depth = 0
    start = 0
    for i, c in enumerate(text):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth = max(0, depth - 1)
        elif c == ";" and depth == 0:
            out.append(text[start:i])
            start = i + 1
    out.append(text[start:])
    return out


def extract_single_return(primary_body: str) -> str | None:
    """The return expression of a single-statement ``{ return <expr>; }`` body.

    Isolates the outermost function body (the first balanced ``{...}``), strips
    comments, and requires it to contain exactly one statement: a ``return``.
    Returns the (whitespace-collapsed) expression text, or ``None`` if the body is
    empty, multi-statement, has a control-flow keyword, or is not a lone return —
    every one of which is a conservative refusal (not Class-1).
    """
    src = strip_comments(primary_body)
    open_idx = src.find("{")
    if open_idx < 0:
        return None
    close_idx = _match_brace(src, open_idx)
    if close_idx is None:
        return None
    inner = src[open_idx + 1:close_idx].strip()
    if not inner:
        return None
    stmts = [s.strip() for s in _split_top_level_semicolons(inner) if s.strip()]
    if len(stmts) != 1:
        return None
    stmt = stmts[0]
    if not re.match(r"^return\b", stmt):
        return None
    for m in re.finditer(r"[A-Za-z_]\w*", stmt):
        if m.group(0) in _CONTROL_KEYWORDS:
            return None
    expr = stmt[len("return"):].strip()
    if not expr:
        return None
    return re.sub(r"\s+", " ", expr)


# --------------------------------------------------------------------------- #
# signature / parameter typing
# --------------------------------------------------------------------------- #

# ``<ret> <fn>(<one param>) [const] {`` head of a primary.  Captures the single
# parameter clause.  A multi-parameter primary is not a shallow wrapper shape we
# synthesize (refuse).
_SIGNATURE_RE = re.compile(
    r'([A-Za-z_][\w:<>,\s\*&]*?)\b([A-Za-z_]\w*)\s*\(([^)]*)\)\s*(?:const\s*)?\{')

_TYPE_QUALIFIERS = frozenset({"const", "volatile", "constexpr", "&", "&&", "inline",
                              "KOKKOS_INLINE_FUNCTION", "KOKKOS_FUNCTION", "static"})


def _template_param_names(primary_source: str) -> set[str]:
    """Type-parameter names declared in a leading ``template<...>`` clause."""
    src = strip_comments(primary_source)
    m = re.search(r'template\s*<([^>]*)>', src)
    if not m:
        return set()
    names: set[str] = set()
    for part in m.group(1).split(","):
        toks = re.findall(r"[A-Za-z_]\w*", part)
        # ``typename T`` / ``class T`` / ``int N`` — the last ident is the name
        if toks:
            names.add(toks[-1])
    return names


def _param_core_type_and_name(param_clause: str) -> tuple[str, str] | None:
    """Core type spelling + name of a single ``T const& x`` parameter clause.

    Returns ``("double", "x")`` for ``double const& x``.  A pointer / array / empty
    clause returns ``None`` (not a scalar/container operand → not a shallow wrapper).
    """
    clause = param_clause.strip()
    if not clause or "*" in clause or "[" in clause:
        return None
    toks = re.findall(r"[A-Za-z_]\w*|::|&&|&|<|>|,", clause)
    if not toks:
        return None
    name = None
    for t in reversed(toks):
        if re.match(r"^[A-Za-z_]\w*$", t) and t not in _TYPE_QUALIFIERS:
            name = t
            break
    if name is None:
        return None
    parts: list[str] = []
    for t in toks:
        if t == name and parts:
            break
        if t in _TYPE_QUALIFIERS or t in ("&", "&&"):
            if parts:
                break
            continue
        if t in ("<", ">", ","):
            break
        if t == "::":
            if parts:
                parts.append(t)
            continue
        if re.match(r"^[A-Za-z_]\w*$", t):
            if t == name:
                if not parts:
                    return None
                break
            parts.append(t)
    if not parts:
        return None
    return "".join(parts), name


def parse_signature(primary_source: str) -> tuple[str, str, str, bool] | None:
    """``(fn_name, param_core_type, param_name, param_is_template)`` of a
    single-param primary, or ``None``.

    ``None`` when the primary has zero or multiple parameters, a pointer param, or
    no recoverable signature — all conservative refusals.  ``param_is_template`` is
    True when the parameter's core type is a name declared in the ``template<...>``
    clause (a generic ``T`` parameter, instantiable at scalar or complex).
    """
    src = strip_comments(primary_source)
    m = _SIGNATURE_RE.search(src)
    if not m:
        return None
    fn = m.group(2)
    params = m.group(3).strip()
    if not params or "," in params:
        return None
    pt = _param_core_type_and_name(params)
    if pt is None:
        return None
    core, pname = pt
    tparams = _template_param_names(src[:m.start(2)])
    core_last = core.rsplit("::", 1)[-1]
    return fn, core, pname, core_last in tparams


# --------------------------------------------------------------------------- #
# recognizer
# --------------------------------------------------------------------------- #

def _top_level_calls(expr: str) -> list[tuple[str, str, str]]:
    """All ``(root_chain, fn, args)`` call heads in ``expr`` (any nesting depth)."""
    out: list[tuple[str, str, str]] = []
    for m in _CALL_HEAD_RE.finditer(expr):
        chain = re.sub(r"\s+", "", m.group(1)).rstrip(":")
        fn = m.group(2)
        args, _ = _balanced_args(expr, m.end() - 1)
        out.append((chain, fn, args))
    return out


def _idents(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_]\w*", text)


def recognize(primary_source: str, surface: VendoredSurface,
              *, is_synth_dep=None, _depth: int = 0) -> Recognition | None:
    """Classify ``primary_source`` as a Class-1 shallow wrapper, or ``None``.

    ``surface`` models the vendored op surface (derived from the concrete type
    spellings).  ``is_synth_dep(name) -> bool`` answers "is this inner callee itself
    a Class-1-synthesizable wrapper?" for the transitive form; when ``None``,
    transitive recognition is disabled (an inner app-call → refuse), the safe
    default.  Returns a :class:`Recognition`, or ``None`` for any body the parser is
    not fully confident is one of the four shallow shapes.
    """
    if _depth > _MAX_TRANSITIVE_DEPTH:
        return None
    sig = parse_signature(primary_source)
    if sig is None:
        return None
    fn, param_type, param_name, is_tmpl = sig
    expr = extract_single_return(primary_source)
    if expr is None:
        return None

    calls = _top_level_calls(expr)

    # ---- delegation: return <Ns>::<mathfn>(<arg using param>); ----
    if len(calls) == 1:
        chain, ifn, iargs = calls[0]
        lone = re.fullmatch(
            r'\(?\s*(?:[A-Za-z_]\w*\s*::\s*)*[A-Za-z_]\w*\s*\([^;]*\)\s*\)?', expr)
        if lone and ifn in _MATH_FN_NAMES and chain and \
                chain.split("::", 1)[0] not in _VENDORED_NS_ROOTS:
            if param_name not in _idents(iargs):
                return None
            # STOP #S guard: the vendored surface must provide this op for at least
            # one operand kind this primary can be instantiated at.
            if ifn not in (surface.scalar_ops | surface.complex_ops):
                return None
            return Recognition(fn=fn, form=FORM_DELEGATION, param_type=param_type,
                               param_name=param_name, body_expr=expr,
                               param_is_template=is_tmpl,
                               inner_root=chain.split("::", 1)[0], inner_fn=ifn)

    # ---- accessor: return <arg>.<member>(); ----
    acc = re.fullmatch(r'([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)\s*\(\s*\)', expr)
    if acc and acc.group(1) == param_name and acc.group(2) in _ACCESSOR_MEMBERS:
        return Recognition(fn=fn, form=FORM_ACCESSOR, param_type=param_type,
                           param_name=param_name, body_expr=expr,
                           param_is_template=is_tmpl, member=acc.group(2))

    # STOP #T guard (scalar-expr / transitive): a concrete emitted overload binds
    # ONLY the parameter type; if the body names any OTHER template parameter (e.g.
    # ``Constants<TScale>::_x<TOutput, TMass, TScale>()`` in a ``T``-generic wrapper),
    # the concrete overload cannot resolve it — the emitter does not do full
    # template-argument substitution.  Such a body needs a capability the emitter
    # doesn't have; refuse (conservative → falls to the LLM path).  Delegation /
    # accessor bodies never carry template-arg lists, so this only gates the
    # expression forms below.
    other_tparams = _template_param_names(strip_comments(primary_source)) - {param_type}
    if other_tparams and (set(_idents(expr)) & other_tparams):
        return None

    # ---- scalar-expression / transitive ----
    if calls:
        transitive_dep = None
        for chain, ifn, _a in calls:
            if not chain and ifn == param_type:
                continue                          # functional cast to the param type
            if is_synth_dep is not None and is_synth_dep(ifn):
                if transitive_dep is None:
                    transitive_dep = ifn
                continue
            return None                           # any other call → not Class-1
        if transitive_dep is not None:
            return Recognition(fn=fn, form=FORM_TRANSITIVE, param_type=param_type,
                               param_name=param_name, body_expr=expr,
                               param_is_template=is_tmpl,
                               transitive_dep=transitive_dep)
        return Recognition(fn=fn, form=FORM_SCALAR_EXPR, param_type=param_type,
                           param_name=param_name, body_expr=expr,
                           param_is_template=is_tmpl)

    # no calls at all: a pure operator/literal scalar expression over the param.
    return Recognition(fn=fn, form=FORM_SCALAR_EXPR, param_type=param_type,
                       param_name=param_name, body_expr=expr,
                       param_is_template=is_tmpl)


# --------------------------------------------------------------------------- #
# emitter
# --------------------------------------------------------------------------- #

def _param_core_is_complex(param_type: str) -> bool:
    return "complex" in param_type.rsplit("::", 1)[-1].lower()


def targets_for(recog: Recognition, surface: VendoredSurface) -> list[str]:
    """The promoted operand type(s) to emit an overload for.

    * accessor → the vendored complex only (a member accessor exists on the
      container);
    * a concrete-``complex`` parameter → the vendored complex;
    * a concrete scalar parameter → the vendored scalar;
    * a *template* ``T`` parameter → both the vendored scalar and complex (a generic
      wrapper is instantiable at either), each guarded for delegation by whether the
      vendored surface actually provides the op on that operand kind (STOP #S).
    """
    if recog.form == FORM_ACCESSOR:
        return [surface.complex]
    if not recog.param_is_template:
        return [surface.complex] if _param_core_is_complex(recog.param_type) \
            else [surface.scalar]
    # template parameter → both, filtered by op availability for a delegation.
    out: list[str] = []
    if recog.form == FORM_DELEGATION:
        if recog.inner_fn in surface.scalar_ops:
            out.append(surface.scalar)
        if recog.inner_fn in surface.complex_ops:
            out.append(surface.complex)
        return out
    # scalar-expr / transitive over a template param: emit at the scalar (the
    # complex instantiation, if ever needed, comes from a distinct concrete
    # overload in source — recognized on its own).  Conservative: scalar only.
    return [surface.scalar]


def _redirect_math_call(expr: str, inner_fn: str, surface: VendoredSurface) -> str:
    """Rewrite the delegated ``…::inner_fn(`` call onto the vendored root.

    ``Kokkos::abs(x)`` → ``quad::ddfun::abs(x)``.  Only the qualifier is replaced;
    the argument text is preserved verbatim (the arg is the promoted parameter).
    """
    pat = re.compile(
        r'(?<![\w:])(?:[A-Za-z_]\w*\s*::\s*)*' + re.escape(inner_fn) + r'\s*\(')
    return pat.sub(f"{surface.root}::{inner_fn}(", expr, count=1)


def _substitute_param_type(expr: str, param_type: str, target: str) -> str:
    """Replace whole-word ``param_type`` tokens with ``target`` in ``expr``.

    A scalar-expr body ``(double(0) < x) - (x < double(0))`` becomes
    ``(ddouble(0) < x) - (x < ddouble(0))`` — the functional casts to the
    parameter's own type must widen to the promoted type or the operators do not
    resolve (verified: ``double(0) < ddouble`` has no ``operator<``).
    """
    return re.sub(r'(?<![\w:])' + re.escape(param_type) + r'\b', target, expr)


def emit_overload(recog: Recognition, surface: VendoredSurface,
                  *, qualifier: str, target: str) -> str:
    """The overload text for one recognized shallow wrapper at one ``target`` type.

    Emitted *inside* ``namespace <qualifier> { … }`` so a qualified call resolves to
    it (the injection remedy ``_shim_bridges_qualifier`` sanctions).  Uses ``auto``
    return deduction (no app-specific return type).  Deterministic: the same
    ``(recog, target)`` yields byte-identical text.
    """
    if recog.form == FORM_DELEGATION:
        body = _redirect_math_call(recog.body_expr, recog.inner_fn, surface)
    elif recog.form == FORM_ACCESSOR:
        body = recog.body_expr            # ``z.real()`` — accessor survives verbatim
    else:                                 # scalar-expr / transitive
        body = _substitute_param_type(recog.body_expr, recog.param_type, target)

    src_note = {
        FORM_DELEGATION: f"delegation → {surface.root}::{recog.inner_fn}",
        FORM_ACCESSOR: f"accessor .{recog.member}()",
        FORM_SCALAR_EXPR: "scalar-expr (param-type widened)",
        FORM_TRANSITIVE: f"transitive → {recog.transitive_dep}",
    }[recog.form]
    param = f"{target} const& {recog.param_name}"
    return "\n".join([
        f"namespace {qualifier} {{",
        f"    // Subtask L1' shallow-wrapper synthesis: {recog.fn} "
        f"({src_note}); source primary re-emitted at {target}.",
        f"    KOKKOS_INLINE_FUNCTION auto {recog.fn}({param}) {{ return {body}; }}",
        f"}}",
    ])


# --------------------------------------------------------------------------- #
# de-duplicating overload set (Step 2: idempotent + deduplicating)
# --------------------------------------------------------------------------- #

@dataclass
class OverloadSet:
    """Accumulates synthesized overloads, deduplicated by ``(qualifier, fn, target)``.

    :meth:`add` is idempotent: adding the same recognized wrapper twice keeps one
    overload per target and yields byte-identical text.  :meth:`render` returns the
    overloads in insertion order joined by blank lines — a stable, retry-safe block.
    """

    surface: VendoredSurface
    _by_key: dict[tuple[str, str, str], str] = field(default_factory=dict)
    _order: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, recog: Recognition, *, qualifier: str) -> list[str]:
        texts: list[str] = []
        for target in targets_for(recog, self.surface):
            key = (qualifier, recog.fn, target)
            text = emit_overload(recog, self.surface, qualifier=qualifier,
                                 target=target)
            if key in self._by_key:
                if self._by_key[key] != text:
                    raise ShallowWrapperError(
                        f"non-idempotent emission for {key}: two distinct overloads")
                texts.append(text)
                continue
            self._by_key[key] = text
            self._order.append(key)
            texts.append(text)
        return texts

    def render(self) -> str:
        return "\n\n".join(self._by_key[k] for k in self._order)

    def keys(self) -> list[tuple[str, str, str]]:
        return list(self._order)

    def functions(self) -> set[str]:
        return {fn for (_q, fn, _t) in self._order}


# --------------------------------------------------------------------------- #
# primary-body resolution from source
# --------------------------------------------------------------------------- #

# ``[template<...>] <ret> fn(<params>) [const] {`` — a *definition* head for ``fn``.
def _find_primary_defs(fn: str, sources: list[str]) -> list[str]:
    """All ``[template<...>] <sig> { <body> }`` definition texts of ``fn`` in source.

    Returns the full signature-through-closing-brace text of each definition (a
    template primary and any concrete overloads).  A declaration with no body
    (``;`` before ``{``) is skipped.  Bounded, deterministic, comment-safe.
    """
    out: list[str] = []
    head = re.compile(r'(?<![\w:])' + re.escape(fn) + r'\s*\(')
    for raw in sources:
        src = strip_comments(raw)
        for m in head.finditer(src):
            paren = m.end() - 1
            _args, close_p = _balanced_args(src, paren)
            rest = src[close_p + 1:]
            # skip optional trailing ``const`` / whitespace to the body brace
            j = 0
            while j < len(rest) and (rest[j].isspace()):
                j += 1
            if rest[j:j + 5] == "const":
                j += 5
                while j < len(rest) and rest[j].isspace():
                    j += 1
            if j >= len(rest) or rest[j] != "{":
                continue                          # declaration / call, not a def
            body_open = close_p + 1 + j
            body_close = _match_brace(src, body_open)
            if body_close is None:
                continue
            # recover the definition head: back up to the start of this statement
            start = _def_head_start(src, m.start())
            out.append(src[start:body_close + 1])
    return out


def _def_head_start(src: str, name_pos: int) -> int:
    """Start index of the definition whose function name begins at ``name_pos``.

    Walks left past the return type / ``template<...>`` clause to the previous
    statement/scope boundary (``;`` ``{`` ``}``) so the captured head includes the
    ``template<...>`` and return type but not preceding code.
    """
    depth = 0
    i = name_pos - 1
    while i >= 0:
        c = src[i]
        if c in ">)":
            depth += 1
        elif c in "<(":
            depth = max(0, depth - 1)
        elif depth == 0 and c in ";{}":
            return i + 1
        i -= 1
    return 0


# --------------------------------------------------------------------------- #
# region-level synthesis pass (Step 3 — wired into the regional dispatch)
# --------------------------------------------------------------------------- #

# ``Root::...::fn`` — the qualified name; the call ``(`` / template-id ``<…>(`` that
# follows is located separately so a template-id call ``ql::g<T,U>(x)`` (qcdloop's
# dominant form) is recognized as readily as a plain ``ql::g(x)``.  Unlike the
# Gap-A math scan, an app wrapper is routinely called through a template-id, so we
# must NOT exclude the ``<`` here.
_QUALIFIED_NAME_RE = re.compile(
    r'(?<![\w:])((?:[A-Za-z_]\w*\s*::\s*)+)([A-Za-z_]\w*)\s*')


def _skip_template_args(text: str, lt_idx: int) -> int | None:
    """Index just past the ``>`` closing a template-arg list opened at ``text[lt_idx]``.

    Depth-tracks ``<``/``>`` so nested ``ql::f<ql::g<T>>`` is spanned.  Returns
    ``None`` if the ``<`` does not open a balanced template-arg list (e.g. a bare
    ``a < b`` comparison), so a comparison is never mistaken for a template-id.
    """
    depth = 0
    i = lt_idx
    n = len(text)
    while i < n:
        c = text[i]
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
            if depth == 0:
                return i + 1
        elif c in ";{}":                  # a statement boundary → not a template-id
            return None
        i += 1
    return None


def _contains_promoted(arg_text: str, promoted: frozenset[str]) -> bool:
    if not promoted:
        return False
    for m in re.finditer(r'[A-Za-z_]\w*', arg_text):
        if m.group(0) in promoted:
            return True
    return False


def find_qualified_app_calls(region_text: str, promoted: frozenset[str]):
    """App-qualified calls ``Ns::g(promoted)`` where ``g ∉ _MATH_FN_NAMES``.

    The complement of :func:`agents.integrator_base.regional.find_qualified_math_calls`
    (which returns only the ``<cmath>`` calls): these are the app-wrapper calls the
    shallow-wrapper recognizer inspects.  De-duplicated ``(root, fn, chain)`` list.
    """
    found: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for m in _QUALIFIED_NAME_RE.finditer(region_text):
        chain = re.sub(r"\s+", "", m.group(1))
        fn = m.group(2)
        root = chain.split("::", 1)[0]
        if fn in _MATH_FN_NAMES or root in _VENDORED_NS_ROOTS:
            continue
        # Locate the call ``(`` — directly, or after a balanced template-arg list
        # ``<…>`` for a template-id call ``ql::g<T,U>(x)``.
        j = m.end()
        n = len(region_text)
        while j < n and region_text[j].isspace():
            j += 1
        if j < n and region_text[j] == "<":
            past = _skip_template_args(region_text, j)
            if past is None:
                continue                  # ``a < b`` comparison, not a template-id
            j = past
            while j < n and region_text[j].isspace():
                j += 1
        if j >= n or region_text[j] != "(":
            continue                      # a name/type read, not a call
        args, _ = _balanced_args(region_text, j)
        if not _contains_promoted(args, promoted):
            continue
        key = (root, fn)
        if key in seen:
            continue
        seen.add(key)
        found.append((root, fn, chain.rstrip(":")))
    return found


@dataclass
class SynthesisResult:
    """Outcome of the region-level shallow-wrapper synthesis pass.

    ``overload_text`` is the rendered block of deterministically-emitted overloads
    (``""`` when none), ready to inject into the shim.  ``recognized`` is the list of
    ``(root, fn)`` app-calls handled deterministically (removed from what the LLM
    sees).  ``remaining`` is the app-calls the recognizer refused — they fall back to
    the existing LLM hint/lint path unchanged (conservative-parser contract).
    """

    overload_text: str
    recognized: list[tuple[str, str]]
    remaining: list[tuple[str, str, str]]
    overload_set: "OverloadSet"


def synthesize_for_region(region_text: str, promoted: frozenset[str],
                          sources: list[str], surface: VendoredSurface,
                          *, skip_source_provided: bool = False) -> SynthesisResult:
    """Recognize + emit Class-1 overloads for the region's app-wrapper calls.

    For each app-qualified call ``Ns::g(promoted)``, resolve ``g``'s primary
    definition(s) from ``sources`` and run the recognizer; a recognized wrapper is
    emitted into an :class:`OverloadSet` (transitively pulling in any Class-1 dep it
    names), and its call is recorded as ``recognized``.  A call whose primary is
    absent or not a shallow shape is left in ``remaining`` for the LLM path.  Pure,
    deterministic, idempotent (running twice yields byte-identical ``overload_text``).

    ``skip_source_provided`` (Resolution A, leaf-promotion L3-resume): when the
    analysed ``sources`` already provide a CONCRETE dd overload for a wrapper
    (:func:`source_provides_dd`), synthesizing it too would emit a second definition
    of the same overload — an ODR collision, and upstream the emitter's idempotence
    guard would raise on the two distinct primaries the enriched source now yields for
    one ``(ns, fn, type)`` key.  With this flag set, such a wrapper is a dd *boundary*:
    it is recorded as ``recognized`` (so the LLM path never re-adds it) but NO overload
    is emitted (the source's own definition resolves the call).  The default is
    ``False`` — every pre-enrichment / non-opted-in caller keeps its byte-identical
    behaviour (the flag never fires when the source has no concrete dd definition).
    """
    oset = OverloadSet(surface=surface)
    recognized: list[tuple[str, str]] = []
    remaining: list[tuple[str, str, str]] = []

    def _source_dd(name: str) -> bool:
        return skip_source_provided and source_provides_dd(name, sources, surface)

    # A transitive-dep predicate closed over ``sources``: is ``name`` itself a
    # Class-1 shallow wrapper?  (Used only to CLASSIFY the transitive form; the dep
    # is separately emitted below so the inner call resolves.)  A source-provided dd
    # dep is a boundary, never a synth dep, so it is excluded here too.
    def _dep(name: str) -> bool:
        if _source_dd(name):
            return False
        for defsrc in _find_primary_defs(name, sources):
            r = recognize(defsrc, surface)
            if r is not None and r.fn == name:
                return True
        return False

    for root, fn, chain in find_qualified_app_calls(region_text, promoted):
        # Resolution A: a wrapper the source already defines at dd is a boundary —
        # recognized (kept off the LLM path) but NOT emitted (no ODR duplicate).
        if _source_dd(fn):
            recognized.append((root, fn))
            continue
        defs = _find_primary_defs(fn, sources)
        emitted_any = False
        for defsrc in defs:
            r = recognize(defsrc, surface, is_synth_dep=_dep)
            if r is None or r.fn != fn:
                continue
            oset.add(r, qualifier=root)
            emitted_any = True
            # transitive: also emit the dep's own overload(s) — unless the source
            # already provides that dep at dd (then it is a boundary, not emitted).
            if r.form == FORM_TRANSITIVE and r.transitive_dep \
                    and not _source_dd(r.transitive_dep):
                _emit_dep(r.transitive_dep, root, sources, surface, oset, depth=1)
        if emitted_any:
            recognized.append((root, fn))
        else:
            remaining.append((root, fn, chain))

    return SynthesisResult(overload_text=oset.render(), recognized=recognized,
                           remaining=remaining, overload_set=oset)


def _emit_dep(fn: str, qualifier: str, sources: list[str], surface: VendoredSurface,
              oset: "OverloadSet", *, depth: int) -> None:
    """Emit a transitive dependency's own Class-1 overload(s) into ``oset``."""
    if depth > _MAX_TRANSITIVE_DEPTH:
        return

    def _dep(name: str) -> bool:
        for defsrc in _find_primary_defs(name, sources):
            r = recognize(defsrc, surface)
            if r is not None and r.fn == name:
                return True
        return False

    for defsrc in _find_primary_defs(fn, sources):
        r = recognize(defsrc, surface, is_synth_dep=_dep)
        if r is None or r.fn != fn:
            continue
        oset.add(r, qualifier=qualifier)
        if r.form == FORM_TRANSITIVE and r.transitive_dep:
            _emit_dep(r.transitive_dep, qualifier, sources, surface, oset,
                      depth=depth + 1)


# --------------------------------------------------------------------------- #
# source-provided dd boundary (Resolution A — leaf-promotion L3-resume)
# --------------------------------------------------------------------------- #

# Outcome sentinel for :func:`is_class1_synthesizable`: the analysed source already
# provides a CONCRETE dd definition for the queried wrapper, so it is a dd
# *boundary* the pipeline must NOT synthesize.  A pipeline-synthesized overlay for
# such a wrapper would be a second definition of the same overload — an ODR /
# ambiguating-redeclaration collision (STOP #K).  This mirrors how a source that
# defines its own dd constants dissolves the constant-synthesis obligation: a source
# that defines its own dd wrappers dissolves the wrapper-synthesis obligation.
# Truthy (a non-empty string) so every existing ``if is_class1_synthesizable(...)``
# boundary test keeps treating a recognized wrapper as a boundary; callers that must
# distinguish "synthesize" from "read from source" compare ``== SOURCE_PROVIDED``.
SOURCE_PROVIDED = "source_provided"


def source_provides_dd(fn: str, sources: list[str], surface: VendoredSurface) -> bool:
    """True iff ``sources`` already provide a CONCRETE dd overload/specialization of
    ``fn`` for the vendored surface's dd scalar or complex type.

    Structural and framework-agnostic: a *definition* of ``fn`` whose single
    parameter's CORE type token is the dd scalar/complex core (the last ``::``
    segment of ``surface.scalar`` / ``surface.complex``), and which is NOT a template
    *primary* over that parameter.  This recognizes BOTH an explicit
    ``template<>`` specialization at dd and a plain concrete dd overload, and it keys
    on the core type token only — so a source that spells the dd type through a
    DIFFERENT namespace root than the vendored surface (the ``ql::ddfun`` /
    ``quad::ddfun`` bridge) is still matched.

    A template primary (``T fn(T const&)``) has a template-parameter core and is
    NEVER a match, so under a source that provides only the primary (the pre-
    enrichment world) this is uniformly ``False`` — the synthesis obligation stands
    and every pre-enrichment behaviour is byte-identical.  Keys on the *signature*,
    not the body shape, so a source dd definition with a multi-statement body is
    recognized as a boundary exactly like a one-line one (the boundary is "the source
    owns the dd definition", never "the dd body matches the double body").

    Pure and side-effect-free.
    """
    dd_cores = {surface.scalar.rsplit("::", 1)[-1],
                surface.complex.rsplit("::", 1)[-1]}
    want = fn.rsplit("::", 1)[-1]
    for defsrc in _find_primary_defs(want, sources):
        sig = parse_signature(defsrc)
        if sig is None:
            continue
        pfn, param_core, _pname, is_tmpl = sig
        if pfn != want:
            continue
        if is_tmpl:
            continue                       # a template PRIMARY, not a concrete dd def
        if param_core.rsplit("::", 1)[-1] in dd_cores:
            return True
    return False


# --------------------------------------------------------------------------- #
# synthesis manifest API (Step 4 — consumed by L2's clonable_leaf predicate)
# --------------------------------------------------------------------------- #

def is_class1_synthesizable(qualified_name: str, primary_body_source: str,
                            surface: VendoredSurface,
                            *, is_synth_dep=None, source_provides_dd=None):
    """Classify ``qualified_name`` for L2/L3: ``True`` (the pipeline can synthesize
    its dd overload from ``primary_body_source``), :data:`SOURCE_PROVIDED` (the source
    already defines it at dd — a boundary, do NOT synthesize), or ``False`` (neither).

    This is the manifest-query the L2 ``clonable_leaf`` predicate calls to decide
    whether a leaf's promoted body names only synthesizable / source / vendored
    symbols.  ``qualified_name`` may be bare (``kAbs``) or qualified (``ql::kAbs``);
    only the final component is matched against the parsed primary's own name.
    ``is_synth_dep`` enables transitive recognition (pass a predicate closed over the
    caller's source map, or ``None`` to disable transitive deps).

    ``source_provides_dd`` (Resolution A) is an optional ``name -> bool`` predicate:
    when supplied and it reports the source already defines a concrete dd overload for
    this wrapper, the result is :data:`SOURCE_PROVIDED` — the wrapper is a dd boundary
    the pipeline reads from source rather than synthesizing (a synthesized overlay
    would ODR-collide with the source definition; STOP #K).  It is checked BEFORE
    recognition, so a wrapper whose double primary is *also* a recognizable shallow
    shape still defers to the source when the source provides dd.  When
    ``source_provides_dd`` is ``None`` (every pre-enrichment / non-opted-in caller)
    the result is the original ``bool`` and behaviour is byte-identical.

    Pure and side-effect-free — no emission, no shim mutation.  A ``False`` result is
    the conservative refusal L2 needs (the leaf falls to ``chain_closure_escapes``);
    ``True`` promises :func:`emit_overload` produces a compiling overload;
    :data:`SOURCE_PROVIDED` promises the source already provides a compiling dd one.
    """
    want = qualified_name.rsplit("::", 1)[-1]
    if source_provides_dd is not None and source_provides_dd(want):
        return SOURCE_PROVIDED
    recog = recognize(primary_body_source, surface, is_synth_dep=is_synth_dep)
    if recog is None:
        return False
    return recog.fn == want
