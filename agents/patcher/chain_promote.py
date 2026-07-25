"""Phase 2f — coordinated multi-region, multi-file dd promotion of a whole chain.

Where :func:`agents.patcher.fanout.fan_out_region` promotes ONE region per call,
this module promotes an entire **cancellation-cascade chain** — a set of
``(file, line)`` regions spanning several functions — to double-double as a single
coordinated envelope.  The chain-internal type contract is dd; the chain *boundary*
contract is caller precision (reads widened at entry, writes truncated at exit) —
the deterministic realization of the chain integrator's C9 rule.

Mechanism (all reused from the region fan-out; the manifest is the source of truth):

* For each chain region, resolve its enclosing function, derive reads/writes, and
  compute the per-region promotion via ``boundary.promote_region_block``.
* Regions in the entry point are promoted in place; the rest accumulate variant
  specs via the shared :func:`agents.patcher.fanout._accumulate_region_specs` into
  ONE ``new_specs`` / ``root_reroutes`` set across the whole chain.  Because variant
  names are a pure function of ``(path, integral)``, two chain regions reaching the
  same ancestor land on the SAME :class:`VariantSpec` — the caller variant carries
  both its own region promotion AND the reroute into the callee variant, so the
  chain-internal calls widen bottom-up (``_topo_order`` emits callee-before-caller).
* One ``_merge_into_file`` per touched file writes the AMP-FANOUT block.
* Variants are integral-scoped (``ddilog_B12`` vs ``ddilog_B14``); the ORIGINAL
  shared helper is never edited — a correctness requirement (Item 7 §3: promoting
  the original ``ddilog`` would re-measure benign B8/B9 floors).

Chain-scope gates (the envelope, not individual links):

* ``chain_promotion_no_op`` — fires iff NO region in the chain promotes anything.
  A single empty-payload link is NOT gated (its neighbours may promote, and its
  intra-chain writes stay wide for the next link to read).
* ``chain_write_truncation`` — fires iff any INTERIOR region of the chain truncates
  every widened write back to caller precision with no wider persistent sink.  The
  chain's OUTERMOST region(s) — those at the shallowest depth on the call graph, the
  last landing before the value returns to the shared driver — are EXEMPT: their
  store to the caller-precision output IS the chain's *designed* exit boundary (the
  final dd result rounded down to caller precision after the chain has done its work
  at dd), not evidence of inertness.  An interior write that truncates back to caller
  precision, by contrast, injects double roundoff *between* chain links and genuinely
  breaks the chain — that is what this gate catches.  Applying 2d-B's per-region gate
  to the outermost region is exactly the reasoning it would get wrong at chain scope:
  the per-region detector reads the exit-boundary rounding as inertness, when at chain
  scope it is the intended output handoff (Reet 2026-07-25, Fix 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agents.integrator_base import boundary
from agents.patcher.call_graph import CallGraph
from agents.patcher.fanout import (
    FanoutError, Promote, VariantSpec,
    _accumulate_region_specs, _complex_reads, _merge_into_file, _original_text,
    _pick_def, _promote_in_place, _reroute_in_function, _resolve_complex_binding,
    _resolve_root_file,
)
from agents.patcher.variant_naming import assert_no_collisions
from agents.shared import region_scan


@dataclass
class ChainManifest:
    """The input to :func:`chain_promote`: one cascade chain to widen to dd.

    ``lines`` are the chain's ``(file, line_start, line_end)`` regions (1-based
    inclusive), spanning one or more functions.  ``entry_point`` is the integral's
    call-graph root (matches the graph's ``root``); ``integral`` is the variant
    naming suffix (per-integral — ``B12``).  ``reads_by_region`` / ``writes_by_region``
    optionally override the source-derived operand sets, keyed by the ``(file,
    line_start, line_end)`` tuple; when absent they are derived from source
    (``region_scan``), exactly as the single-region fan-out does for reads.
    """

    chain_id: str
    integral: str
    entry_point: str
    lines: list[tuple[str, int, int]]
    reads_by_region: dict[tuple[str, int, int], list[str]] = field(default_factory=dict)
    writes_by_region: dict[tuple[str, int, int], list[str]] = field(default_factory=dict)


@dataclass
class ChainFanoutResult:
    """What one :func:`chain_promote` call produced (chain analogue of FanoutResult)."""

    declared_variants: list[str]
    files_touched: list[str]
    root_edited: bool = False
    in_place_regions: int = 0
    paths_enumerated: int = 0
    truncated: bool = False
    # Chain-scope 2c: True iff ANY region in the chain retyped something.  False =>
    # the whole chain's promotable payload is empty -> terminal promotion_no_op.
    promotion_applied: bool = False
    # Chain-scope 2d-B: True iff some INTERIOR region widens but truncates every
    # landing back to caller precision with no wider persistent sink -> an intra-chain
    # write injecting double roundoff between links -> terminal write_truncation.  The
    # chain's OUTERMOST region is EXEMPT (its truncation is the designed exit boundary).
    write_truncation: bool = False
    # Entry-point body reroutes applied (child -> variant), for silent-bypass telemetry.
    reroutes: dict[str, str] = field(default_factory=dict)
    # Per-region reads actually used (source-derived when not supplied), keyed by region.
    reads_used: dict[tuple[str, int, int], list[str]] = field(default_factory=dict)


def chain_promotion_no_op(per_region_promoted: list[bool]) -> bool:
    """Chain-scope 2c gate: fires iff NO region in the chain promotes anything.

    A single empty-payload link is NOT gated — the chain is widened as an envelope,
    so its neighbours may promote and its intra-chain writes stay wide for the next
    link.  Only a chain whose ENTIRE payload is empty is a true no-op.
    """
    return not any(per_region_promoted)


def chain_write_truncation(region_meta: list[dict], *, two_limb: bool,
                           caller_type: str = "double",
                           complex_tokens=frozenset(), caller_complex=None) -> bool:
    """Chain-scope 2d-B gate: applied to the chain's INTERIOR regions only.

    ``region_meta`` is the per-region list built by :func:`chain_promote`, each entry
    carrying ``depth`` (hops from the entry point — shallower = outer), ``region_text``,
    ``reads`` and ``writes``.  The chain's OUTERMOST region(s) — those at the minimum
    depth — are SKIPPED: their store to the caller-precision output is the chain's
    *designed* exit boundary (round the final dd result down to caller precision after
    the chain has done its work at dd), not evidence of inertness.  Every INTERIOR
    region is checked with the per-region
    :func:`agents.integrator_base.boundary.write_truncation_inert`; the gate fires iff
    ANY interior region trips it — an intra-chain write truncating back to caller
    precision injects double roundoff between chain links and genuinely breaks the
    chain.

    Returns ``False`` for a single-region chain (nothing is interior) — a lone region
    is its own designed exit boundary; the outermost exemption covers it.
    """
    if not region_meta:
        return False
    outermost_depth = min(m["depth"] for m in region_meta)
    for m in region_meta:
        if m["depth"] == outermost_depth:
            continue  # designed exit boundary — the exit-truncation is the design
        if boundary.write_truncation_inert(
                m["region_text"], m["reads"], m["writes"], two_limb,
                caller_type=caller_type, complex_tokens=complex_tokens,
                caller_complex=caller_complex):
            return True
    return False


# --------------------------------------------------------------------------- #
# Blocker A — carrier closure (BLOCKER_A_CARRIER_DESIGN.md §2, §4)
#
# A *carrier* is a variable declared OUTSIDE a chain's line set, written by one
# interior chain line and read by another — the value it carries crosses a
# chain-line boundary at caller precision, so the interior write truncates the
# widened (dd) value back to double and the 2d-B :func:`chain_write_truncation`
# gate correctly rejects the patch.  The fix (Subtasks 3-7) widens the carrier's
# DECLARATION alongside the line bodies.  This module owns only the *analysis*:
# given a chain's line set, classify its carriers as widenable / unwidenable /
# external.  It computes nothing that mutates the tree.
# --------------------------------------------------------------------------- #


@dataclass
class CarrierClosure:
    """The carrier classification of one chain (BLOCKER_A_CARRIER_DESIGN.md §4).

    * ``widenable`` — ``(file, decl_line, name, dd_type)`` per name whose
      declaration the emission layer will widen to the chain's internal dd type.
      A carrier's same-line multi-declarator siblings are included (§2: widening
      ``TMass Y, S, A;`` widens ``S`` too — an over-widened same-type sibling never
      truncates), so a name here need not itself be a strict carrier.
    * ``unwidenable_reasons`` — ``(name, reason)`` per strict carrier whose decl is
      a function parameter; v1 refuses to rewrite signatures (terminal
      ``chain_carrier_unwidenable``, wired in Subtask 5).
    * ``external_reasons`` — ``(name, reason)`` per strict carrier whose decl is a
      global / class member / output container; v1 refuses to widen shared state
      (terminal ``chain_carrier_external``, wired in Subtask 5).
    """

    widenable: list[tuple[str, int, str, str]] = field(default_factory=list)
    unwidenable_reasons: list[tuple[str, str]] = field(default_factory=list)
    external_reasons: list[tuple[str, str]] = field(default_factory=list)

    @property
    def carrier_names(self) -> set[str]:
        """Every name whose decl the emission layer widens (carriers + siblings)."""
        return {name for _f, _l, name, _t in self.widenable}


@dataclass
class _DeclStmt:
    """One statement-level local declaration recovered from a function body.

    ``names`` are all declarators sharing the leading type token (so a bare
    multi-declarator ``TMass Y, S, A;`` yields one ``_DeclStmt`` with three names);
    widening rewrites that one type token, widening every sibling (§2).
    """

    decl_line: int          # 1-based absolute file line of the leading type token
    core_type: str          # outermost type-name token (``TMass`` / ``TOutput``)
    names: list[str]


def compute_carrier_closure(
    *, manifest: ChainManifest, graph: CallGraph, scalar_type: str,
    complex_type: str | None = None, complex_tokens=frozenset(),
    max_paths: int = 1024,
) -> CarrierClosure:
    """Classify a chain's carriers (BLOCKER_A_CARRIER_DESIGN.md §4, steps 1-6).

    Pure, source-only analysis over ``manifest.lines`` — no tree mutation, no new
    source-analysis machinery (reuses :mod:`agents.shared.region_scan` +
    :class:`~agents.patcher.call_graph.CallGraph`).  A name is a **strict carrier**
    iff (1) written by an interior chain line, (2) read by another chain line so the
    value crosses a chain-line boundary, (3) its decl lies outside the chain's line
    set, and (4) it is not a write target of the outermost (min-depth) region (the
    designed exit boundary, §5).  Each surviving carrier's decl site is classified:
    a local var in a single chain function → widenable (with its multi-declarator
    siblings); a function parameter → unwidenable; a global / member / output
    container (or a name visible across >1 chain function) → external.

    Raises :class:`~agents.patcher.fanout.FanoutError` if a chain region is not
    inside any known function or its function is unreachable from the entry point —
    the same fail-loud contract as :func:`chain_promote`.
    """
    # -- 1/2. per-line read/write/depth, derived from source (region_scan) -------
    per_line: list[dict] = []
    func_src: dict[tuple[str, int, int], str] = {}   # (file, ls, le) -> source text
    func_line_start: dict[tuple[str, int, int], int] = {}
    chain_lineset: dict[str, set[int]] = {}          # file -> {chain line numbers}

    for (file, ls, le) in manifest.lines:
        fd = graph.enclosing_function(file, ls)
        if fd is None:
            raise FanoutError(
                f"carrier closure for chain {manifest.chain_id}: region "
                f"{file}:{ls}-{le} is not inside any known function "
                f"(call graph rooted at {graph.root!r})")
        fkey = (fd.file, fd.line_start, fd.line_end)
        if fkey not in func_src:
            func_src[fkey] = "\n".join(_original_text(fd.file, fd.line_start, fd.line_end))
            func_line_start[fkey] = fd.line_start
        region_text = "\n".join(_original_text(fd.file, ls, le))
        for ln in range(ls, le + 1):
            chain_lineset.setdefault(fd.file, set()).add(ln)
        per_line.append(dict(
            file=fd.file, line=ls, fd=fd, fkey=fkey,
            depth=_region_depth(graph, fd, manifest.chain_id, max_paths),
            reads=_names_read_in_region(region_text),
            writes=set(region_scan.region_writes_from_source(region_text))))

    if not per_line:
        return CarrierClosure()

    outermost_depth = min(p["depth"] for p in per_line)

    # -- 3. candidate carriers: written on an interior line AND read across lines -
    interior_writes: dict[str, set[int]] = {}    # name -> interior chain lines writing it
    all_writes: dict[str, set[int]] = {}         # name -> every chain line writing it
    all_reads: dict[str, set[int]] = {}          # name -> every chain line reading it
    outer_write_targets: set[str] = set()        # names written by the outermost region
    name_funcs: dict[str, set[tuple[str, int, int]]] = {}  # name -> chain funcs touching it

    for p in per_line:
        interior = p["depth"] != outermost_depth
        for w in p["writes"]:
            all_writes.setdefault(w, set()).add(p["line"])
            name_funcs.setdefault(w, set()).add(p["fkey"])
            if interior:
                interior_writes.setdefault(w, set()).add(p["line"])
            else:
                outer_write_targets.add(w)
        for r in p["reads"]:
            all_reads.setdefault(r, set()).add(p["line"])
            name_funcs.setdefault(r, set()).add(p["fkey"])

    out = CarrierClosure()
    widen_groups: dict[tuple[str, int], _DeclStmt] = {}   # (file, decl_line) -> stmt

    for name in sorted(interior_writes):
        wlines = interior_writes[name]                    # cond 1: interior write ✓
        rlines = all_reads.get(name, set())
        # cond 2: read by a chain line so the value crosses a chain-line boundary —
        # a strict write→read on different lines, or a same-line read-write plus at
        # least one other touching line (the OR clause of §2).
        touched = rlines | all_writes.get(name, set())
        if not (rlines and len(touched) >= 2):
            continue
        # cond 4: a write target of the outermost region is the designed exit
        # boundary (demoted on purpose) — excluded from the closure entirely (§5).
        if name in outer_write_targets:
            continue

        # a carrier local lives in ONE function; a name visible across >1 chain
        # function is not a plain local → global / class member / output container.
        funcs = name_funcs.get(name, set())
        if len(funcs) > 1:
            out.external_reasons.append((
                name, f"declared outside a single chain function (touched by "
                f"{len(funcs)} chain functions) — global / class member / output "
                f"container; v1 refuses to widen shared state"))
            continue

        fkey = next(iter(funcs))
        decls = _local_decls(func_src[fkey], func_line_start[fkey])
        param_cores = region_scan._param_name_core_types(
            region_scan._tokenize(func_src[fkey]))

        stmt = decls.get(name)
        if stmt is not None:
            # cond 3: a decl INSIDE the chain line set is already widened by the body
            # transform — out of scope (not returned).
            if stmt.decl_line in chain_lineset.get(fkey[0], set()):
                continue
            widen_groups.setdefault((fkey[0], stmt.decl_line), stmt)
        elif name in param_cores:
            out.unwidenable_reasons.append((
                name, f"declared as a parameter of the chain function at "
                f"{fkey[0]}:{fkey[1]} — v1 refuses to rewrite function signatures"))
        else:
            out.external_reasons.append((
                name, f"no local or parameter declaration found in the enclosing "
                f"chain function — global / class member / output container; v1 "
                f"refuses to widen shared state"))

    # -- 6/7. expand each widened decl line to ALL its declarators (siblings, §2) --
    for (dfile, dline), stmt in widen_groups.items():
        dd_type = _carrier_dd_type(stmt.core_type, scalar_type,
                                   complex_type, complex_tokens)
        for nm in stmt.names:
            out.widenable.append((dfile, dline, nm, dd_type))

    out.widenable.sort(key=lambda w: (w[0], w[1], w[2]))
    out.unwidenable_reasons.sort()
    out.external_reasons.sort()
    return out


def _region_depth(graph: CallGraph, fd, chain_id: str, max_paths: int) -> int:
    """Hops from the entry point to ``fd`` (shallower = outer), mirroring the depth
    :func:`chain_promote` records in ``region_meta``.  A region in the entry point is
    depth 0; any other function's depth is ``min(len(path)) - 1`` over the caller
    paths.  Raises :class:`FanoutError` for a function unreachable from the root."""
    if fd.name == graph.root:
        return 0
    paths, _ = graph.enumerate_paths(fd.name, max_paths=max_paths)
    if not paths:
        raise FanoutError(
            f"carrier closure for chain {chain_id}: function {fd.name!r} is "
            f"unreachable from entry point {graph.root!r}")
    return min(len(p) for p in paths) - 1


def _carrier_dd_type(core_type: str, scalar_type: str,
                     complex_type: str | None, complex_tokens) -> str:
    """The chain-internal dd type a carrier decl widens to: the complex container
    when the carrier's core type is a complex-bound token, else the scalar dd type
    (§7 — ``quad::ddfun::ddouble`` / ``ddcomplex``)."""
    if complex_type and core_type in complex_tokens:
        return complex_type
    return scalar_type


def _names_read_in_region(region_text: str) -> set[str]:
    """Identifiers *read* in a region — a raw token scan (not filtered to any
    scalar-name universe, so a carrier whose declared type is unrecognized, e.g.
    ``TMass``, is still detected).

    Excludes pure ``name =`` write targets (a compound ``name +=`` reads and the
    tokenizer emits ``+=`` whole, so it is kept), call targets (``name (``), and
    member / namespace-qualified references (``a.b`` / ``a->b`` / ``ql::x``).
    Over-inclusion is harmless: a name only becomes a carrier candidate if it is
    ALSO written on an interior chain line, so a stray type/constant read never
    promotes to a false carrier.
    """
    toks = region_scan._tokenize(region_text)
    n = len(toks)
    reads: set[str] = set()
    for i, t in enumerate(toks):
        if not region_scan._is_ident_tok(t.text):
            continue
        prev = toks[i - 1].text if i > 0 else ""
        nxt = toks[i + 1].text if i + 1 < n else ""
        if prev in (".", "->", "::"):
            continue                       # member / qualified reference lead
        if nxt in ("=", "(", "::"):
            continue                       # write target, call, or qualifier lead
        reads.add(t.text)
    return reads


def _local_decls(func_src: str, func_line_start: int) -> dict[str, _DeclStmt]:
    """Map each statement-level local name to its :class:`_DeclStmt`.

    Recognizes init (``T a = …``), bare (``T a;``) and bare multi-declarator
    (``TMass Y, S, A;``) forms — the last of which the boundary transform's
    ``_scan_decls`` (which requires ``<type> <name> =``) misses, so a dedicated
    scanner is needed (design §7).  A pure token scan over the enclosing function,
    consistent with :mod:`agents.shared.region_scan`; source-only, no AST type
    resolution (libclang cannot resolve a template region's types without its
    include context — it mislabels ``ddilog`` itself as a ``VAR_DECL``).
    """
    toks = region_scan._tokenize(func_src)
    out: dict[str, _DeclStmt] = {}
    stmt: list = []
    paren = 0

    def handle(stmt: list) -> None:
        if not stmt:
            return
        # skip leading storage/cv qualifiers
        k = 0
        while k < len(stmt) and stmt[k].text in region_scan._TYPE_QUALIFIERS:
            k += 1
        if k >= len(stmt) or not region_scan._is_ident_tok(stmt[k].text):
            return
        # tokenizer lines are 1-based within func_src; map to the absolute file line.
        type_line = func_line_start + stmt[k].line - 1
        # core type: leading ident, then (:: ident)* — reduced to its last segment
        j = k
        parts = [stmt[j].text]
        j += 1
        while j + 1 < len(stmt) and stmt[j].text == "::" \
                and region_scan._is_ident_tok(stmt[j + 1].text):
            parts.append(stmt[j + 1].text)
            j += 2
        core = parts[-1]
        # skip a template argument list ``<...>`` on the type
        if j < len(stmt) and stmt[j].text == "<":
            d = 0
            while j < len(stmt):
                tx = stmt[j].text
                if tx == "<":
                    d += 1
                elif tx == ">":
                    d -= 1
                elif tx == ">>":
                    d -= 2
                j += 1
                if d <= 0:
                    break
        while j < len(stmt) and stmt[j].text in ("&", "*"):
            j += 1
        while j < len(stmt) and stmt[j].text in region_scan._TYPE_QUALIFIERS:
            j += 1
        # declarators: name [= init] [ [extent] ] (, name …)* — bail on any shape
        # that is not a plain variable declaration (``name(`` = call/ctor/func decl).
        names: list[str] = []
        while j < len(stmt):
            if not region_scan._is_ident_tok(stmt[j].text):
                break
            nm = stmt[j].text
            nxt = stmt[j + 1].text if j + 1 < len(stmt) else ";"
            if nxt == "(":
                return                     # call / constructor-init / function decl
            if nxt not in (",", ";", "=", "["):
                return                     # not a plain declarator (e.g. ``a . b``)
            names.append(nm)
            j += 1
            if nxt in ("=", "["):          # skip initializer / array extent
                d = 0
                while j < len(stmt):
                    tx = stmt[j].text
                    if tx in "([{":
                        d += 1
                    elif tx in ")]}":
                        d -= 1
                    elif tx == "," and d == 0:
                        break
                    j += 1
            if j < len(stmt) and stmt[j].text == ",":
                j += 1
                continue
            break
        if not names or len(parts) < 1:
            return
        rec = _DeclStmt(decl_line=type_line, core_type=core, names=names)
        for nm in names:
            out.setdefault(nm, rec)

    for t in toks:
        tx = t.text
        if tx == "(":
            paren += 1
            stmt.append(t)
            continue
        if tx == ")":
            paren = max(0, paren - 1)
            stmt.append(t)
            continue
        if tx in "{}":
            handle(stmt)
            stmt = []
            continue
        if tx == ";" and paren == 0:
            handle(stmt)
            stmt = []
            continue
        stmt.append(t)
    return out


def chain_promote(*, manifest: ChainManifest, graph: CallGraph,
                  tree_root: str | Path, scalar_type: str, two_limb: bool,
                  shim_include: str | None, caller_type: str = "double",
                  complex_type: str | None = None, max_paths: int = 1024,
                  app_source_roots=()) -> ChainFanoutResult:
    """Widen a whole cascade chain to ``scalar_type`` (dd) as one coordinated envelope.

    Mutates the tree under ``tree_root`` in place (per-file AMP-FANOUT blocks + the
    entry-point body reroute) and returns a :class:`ChainFanoutResult` carrying the
    chain-scope 2c/2d gate verdicts.  Raises :class:`FanoutError` if a chain region
    is not inside a known function or its function is unreachable from the entry
    point.
    """
    tree = Path(tree_root).resolve()
    complex_tokens, caller_complex = _resolve_complex_binding(
        app_source_roots, caller_type, complex_type)

    new_specs: dict[str, dict[str, VariantSpec]] = {}
    root_reroutes: dict[str, str] = {}
    name_maps: list[dict[str, str]] = []

    per_region_promoted: list[bool] = []
    region_meta: list[dict] = []       # for outermost-by-depth gate computation
    reads_used: dict[tuple[str, int, int], list[str]] = {}
    touched: set[str] = set()
    in_place_regions = 0
    paths_enumerated = 0
    truncated = False

    for (file, ls, le) in manifest.lines:
        fd = graph.enclosing_function(file, ls)
        if fd is None:
            raise FanoutError(
                f"chain {manifest.chain_id}: region {file}:{ls}-{le} is not inside any "
                f"known function (call graph rooted at {graph.root!r}); cannot promote")
        rkey = (file, ls, le)

        func_src = "\n".join(_original_text(fd.file, fd.line_start, fd.line_end))
        region_text = "\n".join(_original_text(fd.file, ls, le))

        # Derive reads/writes from source when the manifest carries none (qcdloop's
        # template regions report empty operand sets — same recovery the fan-out uses).
        reads = list(manifest.reads_by_region.get(rkey, []))
        if not reads:
            reads = region_scan.region_reads_from_function(func_src, fd.line_start, ls, le)
        writes = list(manifest.writes_by_region.get(rkey, []))
        if not writes:
            writes = region_scan.region_writes_from_source(region_text)
        reads_used[rkey] = list(reads)

        complex_names = _complex_reads(func_src, reads, complex_tokens) if complex_type else []
        ckw = dict(complex_type=complex_type, complex_tokens=list(complex_tokens),
                   complex_names=complex_names, caller_complex=caller_complex)

        _, promoted = boundary.promote_region_block(
            region_text, reads, writes, scalar_type, caller_type, two_limb,
            complex_type=complex_type, complex_tokens=frozenset(complex_tokens),
            complex_names=frozenset(complex_names), caller_complex=caller_complex)
        per_region_promoted.append(promoted)

        if fd.name == graph.root:
            # Region lives in the entry point: promote in place (no new symbol).
            _promote_in_place(tree, fd, ls, le, reads, writes, scalar_type,
                              two_limb, caller_type, shim_include, ckw)
            in_place_regions += 1
            touched.add(fd.file)
            depth = 0
        else:
            paths, tr = graph.enumerate_paths(fd.name, max_paths=max_paths)
            if not paths:
                raise FanoutError(
                    f"chain {manifest.chain_id}: function {fd.name!r} (containing region "
                    f"{file}:{ls}-{le}) is unreachable from entry point {graph.root!r}")
            truncated = truncated or tr
            paths_enumerated += len(paths)
            _accumulate_region_specs(
                paths=paths, fd=fd, line_start=ls, line_end=le,
                reads=reads, writes=writes, integral=manifest.integral, graph=graph,
                scalar_type=scalar_type, two_limb=two_limb, shim_include=shim_include,
                caller_type=caller_type, ckw=ckw,
                new_specs=new_specs, root_reroutes=root_reroutes, name_maps=name_maps)
            depth = min(len(p) for p in paths) - 1     # hops from root (shallower = outer)

        region_meta.append(dict(depth=depth, region_text=region_text,
                                reads=reads, writes=writes, promoted=promoted))

    assert_no_collisions(name_maps)

    # --- one merge per touched file (the whole chain's specs for that file) ----
    declared: list[str] = []
    for file_key, specs in new_specs.items():
        _merge_into_file(Path(file_key), specs)
        touched.add(file_key)
        declared.extend(specs.keys())

    # --- reroute the entry-point body once (union of every region's root reroute) --
    root_edited = in_place_regions > 0
    if root_reroutes:
        root_file = _resolve_root_file(graph)
        root_fd = _pick_def(graph, graph.root)
        if _reroute_in_function(Path(root_file), root_fd, root_reroutes):
            touched.add(root_file)
        root_edited = True

    # --- chain-scope gates ----------------------------------------------------
    # 2d-B fires only on INTERIOR regions (an intra-chain truncation breaks the chain);
    # the outermost region's truncation is the chain's designed exit boundary (Fix 1).
    promotion_applied = any(per_region_promoted)
    write_truncation = False
    if promotion_applied and region_meta:
        write_truncation = chain_write_truncation(
            region_meta, two_limb=two_limb, caller_type=caller_type,
            complex_tokens=frozenset(complex_tokens), caller_complex=caller_complex)

    return ChainFanoutResult(
        declared_variants=sorted(set(declared)),
        files_touched=sorted(touched), root_edited=root_edited,
        in_place_regions=in_place_regions, paths_enumerated=paths_enumerated,
        truncated=truncated, promotion_applied=promotion_applied,
        write_truncation=write_truncation, reroutes=dict(root_reroutes),
        reads_used=reads_used)
