"""Unit tests for the deterministic regional boundary-patch synthesizer."""

from __future__ import annotations

from agents.integrator_base import boundary

_SCALAR = "quad::ffun::ffloat"


def _apply(file_text: str, diff: str) -> str:
    """Apply a unified diff (as produced by the synthesizer) to ``file_text``.

    A tiny, self-contained hunk applier — enough to check the patched result
    without shelling out to ``git apply`` in a unit test.
    """
    import difflib  # noqa: F401  (kept for parity/readability)

    src = file_text.split("\n")
    out: list[str] = []
    src_idx = 0
    lines = diff.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("@@"):
            # @@ -l,s +l,s @@
            hdr = line.split("@@")[1].strip()
            old = hdr.split(" ")[0]          # -l,s
            old_start = int(old[1:].split(",")[0])
            # copy unchanged prefix up to the hunk
            while src_idx < old_start - 1:
                out.append(src[src_idx]); src_idx += 1
            i += 1
            while i < len(lines) and not lines[i].startswith("@@"):
                hl = lines[i]
                if hl.startswith(" "):
                    out.append(src[src_idx]); src_idx += 1
                elif hl.startswith("-"):
                    src_idx += 1
                elif hl.startswith("+"):
                    out.append(hl[1:])
                i += 1
            continue
        i += 1
    while src_idx < len(src):
        out.append(src[src_idx]); src_idx += 1
    return "\n".join(out)


def test_promote_rename_demote_single_write():
    file_text = (
        "#pragma once\n"
        "\n"
        "void f() {\n"
        "    double r = a + b;\n"
        "    res = r;\n"
        "}\n"
    )
    diff = boundary.synthesize_boundary_patch(
        rel_file="box/B.h", file_text=file_text,
        line_start=4, line_end=4, reads=["a", "b"], writes=[],
        scalar_type=_SCALAR, caller_type="double", shim_include="region_ff.h",
    )
    assert diff is not None
    patched = _apply(file_text, diff)

    # reads promoted at entry
    assert f"{_SCALAR} a__ff = {_SCALAR}(a);" in patched
    assert f"{_SCALAR} b__ff = {_SCALAR}(b);" in patched
    # region declaration retyped + reads renamed
    assert f"    {_SCALAR} r__ext = a__ff + b__ff;" in patched
    # write demoted back under its original name for downstream
    assert ("    double r = static_cast<double>(r__ext.hi) + "
            "static_cast<double>(r__ext.lo);") in patched
    # downstream use of r is untouched (outside the region)
    assert "    res = r;" in patched
    # shim included once, after pragma once
    assert patched.count('#include "region_ff.h"') == 1
    lines = patched.split("\n")
    assert lines[0] == "#pragma once"
    assert lines[1] == '#include "region_ff.h"'


def test_native_float_demote_uses_plain_cast_not_two_limb():
    # Wave 2: a native single-limb `float` target (two_limb=False) has no .hi/.lo,
    # so writes are widened with a plain static_cast — the two-limb reconstruction
    # would reference nonexistent members and never compile.
    file_text = (
        "#pragma once\n"
        "template <class T> T f(T a, T b) {\n"
        "    T r = a * b;\n"
        "    return r;\n"
        "}\n"
    )
    diff = boundary.synthesize_boundary_patch(
        rel_file="k.h", file_text=file_text,
        line_start=3, line_end=3, reads=["a", "b"], writes=[],
        scalar_type="float", caller_type="double", shim_include="k_float.h",
        two_limb=False,
    )
    assert diff is not None
    patched = _apply(file_text, diff)
    assert "float a__ff = float(a);" in patched
    assert "float r__ext = a__ff * b__ff;" in patched
    # plain cast, NOT two-limb reconstruction
    assert "T r = static_cast<T>(r__ext);" in patched
    assert ".hi" not in patched and ".lo" not in patched


def test_precheck_style_no_edit_returns_none():
    file_text = "#pragma once\nint x = 1;\n"
    # no reads, no writes, no include → nothing to do
    assert boundary.synthesize_boundary_patch(
        rel_file="a.h", file_text=file_text, line_start=2, line_end=2,
        reads=[], writes=[], scalar_type=_SCALAR, caller_type="double",
    ) is None


def test_multi_read_multi_write_source_order():
    file_text = (
        "#pragma once\n"
        "void g() {\n"
        "    double x = a * b;\n"
        "    double y = x + c;\n"
        "    sink(x, y);\n"
        "}\n"
    )
    diff = boundary.synthesize_boundary_patch(
        rel_file="g.h", file_text=file_text, line_start=3, line_end=4,
        reads=["a", "b", "c"], writes=[], scalar_type=_SCALAR,
        caller_type="double", shim_include="g_ff.h",
    )
    patched = _apply(file_text, diff)
    # x is a region-local write used later in the region as an operand: it is
    # renamed (not promoted as a read) throughout the region.
    assert f"    {_SCALAR} x__ext = a__ff * b__ff;" in patched
    assert f"    {_SCALAR} y__ext = x__ext + c__ff;" in patched
    # both writes demoted after the region
    assert "    double x = static_cast<double>(x__ext.hi)" in patched
    assert "    double y = static_cast<double>(y__ext.hi)" in patched
    # x is NOT promoted as a read (it is a write)
    assert "x__ff" not in patched


def test_caseB_predeclared_write_is_seeded_and_assigned():
    # ``acc`` is declared before the region and re-assigned inside it; Fix-C would
    # report it in ``writes`` (Case B): seed at entry, assign back at exit.
    file_text = (
        "#pragma once\n"
        "void h() {\n"
        "    double acc = 0.0;\n"
        "    acc = acc + a;\n"
        "    use(acc);\n"
        "}\n"
    )
    diff = boundary.synthesize_boundary_patch(
        rel_file="h.h", file_text=file_text, line_start=4, line_end=4,
        reads=["a"], writes=["acc"], scalar_type=_SCALAR, caller_type="double",
        shim_include="h_ff.h",
    )
    patched = _apply(file_text, diff)
    assert f"{_SCALAR} acc__ext = {_SCALAR}(acc);" in patched      # seeded at entry
    assert "    acc__ext = acc__ext + a__ff;" in patched          # renamed in region
    assert ("    acc = static_cast<double>(acc__ext.hi) + "
            "static_cast<double>(acc__ext.lo);") in patched       # assigned back
    # the pre-region declaration is untouched
    assert "    double acc = 0.0;" in patched


def test_whole_word_and_comment_string_safety():
    file_text = (
        "#pragma once\n"
        "void s() {\n"
        '    double r = a + abc + 1; // a is a read, abc is not\n'
        '    const char* msg = "a a a";\n'
        "}\n"
    )
    diff = boundary.synthesize_boundary_patch(
        rel_file="s.h", file_text=file_text, line_start=3, line_end=3,
        reads=["a"], writes=[], scalar_type=_SCALAR, caller_type="double",
        shim_include="s_ff.h",
    )
    patched = _apply(file_text, diff)
    # ``a`` renamed, ``abc`` left alone (substring safety)
    assert "a__ff + abc + 1" in patched
    assert "abc__ff" not in patched
    # the comment text ``a is a read`` is not rewritten
    assert "// a is a read, abc is not" in patched


def test_body_local_promoted_signature_untouched():
    # Realistic region: a statement inside a method body (line 4), NOT the
    # signature.  The parameter ``a`` in the signature (line 3) is outside the
    # region, so it is not renamed; the body-local ``double r`` is promoted because
    # its RHS consumes the promoted read ``a``.
    file_text = (
        "#pragma once\n"
        "struct T {\n"
        "    double compute(double a) {\n"
        "        double r = a + 1.0;\n"
        "        return r;\n"
        "    }\n"
        "};\n"
    )
    diff = boundary.synthesize_boundary_patch(
        rel_file="t.h", file_text=file_text, line_start=4, line_end=4,
        reads=["a"], writes=[], scalar_type=_SCALAR, caller_type="double",
    )
    patched = _apply(file_text, diff)
    assert "    double compute(double a) {" in patched     # signature untouched
    assert f"{_SCALAR} r__ext = a__ff + 1.0;" in patched   # body local promoted
    assert "double r = static_cast<double>(r__ext.hi) + static_cast<double>(r__ext.lo);" in patched


def test_template_alias_local_uses_original_type_on_demote():
    # Real HPC kernels declare locals through template aliases (e.g. qcdloop's
    # TMass), not the literal caller_type the Patcher passes.  Dataflow detection
    # promotes the local anyway and demotes to its OWN declared type.
    file_text = (
        "#pragma once\n"
        "TOutput f(TMass const& x1, TMass const& x2) {\n"
        "    TMass arg = x1 * x2;\n"
        "    return g(arg);\n"
        "}\n"
    )
    diff = boundary.synthesize_boundary_patch(
        rel_file="k.h", file_text=file_text, line_start=3, line_end=3,
        reads=["x1", "x2"], writes=[], scalar_type=_SCALAR, caller_type="double",
    )
    patched = _apply(file_text, diff)
    assert f"{_SCALAR} arg__ext = x1__ff * x2__ff;" in patched
    # demote target is the local's own declared type (TMass), not caller_type
    assert "TMass arg = static_cast<TMass>(arg__ext.hi) + static_cast<TMass>(arg__ext.lo);" in patched


def test_shim_include_after_app_includes():
    # (a) The shim specializes templates the app includes declare, so it must be
    # spliced AFTER every #include in the preamble, before the first code line.
    file_text = (
        "#pragma once\n"
        '#include "constants.h"\n'
        '#include "maths.h"\n'
        "\n"
        "TOutput f(TMass const& x) {\n"
        "    TMass r = x + 1.0;\n"
        "    return g(r);\n"
        "}\n"
    )
    diff = boundary.synthesize_boundary_patch(
        rel_file="k.h", file_text=file_text, line_start=6, line_end=6,
        reads=["x"], writes=[], scalar_type=_SCALAR, caller_type="double",
        shim_include="k_ff.h",
    )
    patched = _apply(file_text, diff)
    lines = patched.split("\n")
    assert patched.count('#include "k_ff.h"') == 1
    shim_i = lines.index('#include "k_ff.h"')
    # after both app includes …
    assert shim_i > lines.index('#include "maths.h"')
    assert shim_i > lines.index('#include "constants.h"')
    # … and before the first code/decl line.
    assert shim_i < lines.index("TOutput f(TMass const& x) {")


def test_shim_include_pragma_once_only_fallback():
    # (b) No includes: fall back to the top, right after #pragma once.
    file_text = (
        "#pragma once\n"
        "void f() {\n"
        "    double r = a + 1.0;\n"
        "}\n"
    )
    diff = boundary.synthesize_boundary_patch(
        rel_file="b.h", file_text=file_text, line_start=3, line_end=3,
        reads=["a"], writes=[], scalar_type=_SCALAR, caller_type="double",
        shim_include="b_ff.h",
    )
    patched = _apply(file_text, diff)
    lines = patched.split("\n")
    assert lines[0] == "#pragma once"
    assert lines[1] == '#include "b_ff.h"'


def test_shim_include_classic_include_guard_fallback():
    # (c) Classic #ifndef/#define guard, no includes: insert after the #define so
    # the shim stays *inside* the guard (never before #ifndef).
    file_text = (
        "#ifndef BOX_C_H\n"
        "#define BOX_C_H\n"
        "\n"
        "void f() {\n"
        "    double r = a + 1.0;\n"
        "}\n"
        "#endif\n"
    )
    diff = boundary.synthesize_boundary_patch(
        rel_file="c.h", file_text=file_text, line_start=5, line_end=5,
        reads=["a"], writes=[], scalar_type=_SCALAR, caller_type="double",
        shim_include="c_ff.h",
    )
    patched = _apply(file_text, diff)
    lines = patched.split("\n")
    assert lines[0] == "#ifndef BOX_C_H"
    assert lines[1] == "#define BOX_C_H"
    assert lines[2] == '#include "c_ff.h"'
    # guard still closes the file
    assert lines[-2] == "#endif" or lines[-1] == "#endif"


def test_shim_include_mixed_system_and_app_includes():
    # (d) Mixed <system> + "app" includes (guard + license banner): shim lands
    # after ALL includes (so trivially after the app include that declares the
    # specialized templates) and before code.
    file_text = (
        "/* Copyright banner\n"
        " * spanning several lines\n"
        " */\n"
        "#ifndef BOX_D_H\n"
        "#define BOX_D_H\n"
        "#include <vector>\n"
        '#include "constants.h"\n'
        "#include <cmath>\n"
        "\n"
        "struct T {\n"
        "    double compute(double a) {\n"
        "        double r = a + 1.0;\n"
        "        return r;\n"
        "    }\n"
        "};\n"
        "#endif\n"
    )
    diff = boundary.synthesize_boundary_patch(
        rel_file="d.h", file_text=file_text, line_start=12, line_end=12,
        reads=["a"], writes=[], scalar_type=_SCALAR, caller_type="double",
        shim_include="d_ff.h",
    )
    patched = _apply(file_text, diff)
    lines = patched.split("\n")
    shim_i = lines.index('#include "d_ff.h"')
    # after every include (system and app) …
    assert shim_i > lines.index('#include "constants.h"')
    assert shim_i > lines.index("#include <cmath>")
    assert shim_i > lines.index("#include <vector>")
    # … and before the struct decl; the license banner did not truncate the scan.
    assert shim_i < lines.index("struct T {")


def test_shim_include_idempotent():
    # Re-inserting an already-present shim is a no-op (no duplicate include line).
    file_text = (
        "#pragma once\n"
        '#include "k_ff.h"\n'
        '#include "constants.h"\n'
        "void f() {\n"
        "    double r = a + 1.0;\n"
        "}\n"
    )
    diff = boundary.synthesize_boundary_patch(
        rel_file="k.h", file_text=file_text, line_start=5, line_end=5,
        reads=["a"], writes=[], scalar_type=_SCALAR, caller_type="double",
        shim_include="k_ff.h",
    )
    patched = _apply(file_text, diff)
    assert patched.count('#include "k_ff.h"') == 1


def test_integer_local_not_promoted():
    # An int index derived from a promoted read stays int (Rule 1).
    file_text = (
        "#pragma once\n"
        "void h(double a) {\n"
        "    int n = 2;\n"
        "    double r = a * 2.0;\n"
        "}\n"
    )
    diff = boundary.synthesize_boundary_patch(
        rel_file="h.h", file_text=file_text, line_start=3, line_end=4,
        reads=["a"], writes=[], scalar_type=_SCALAR, caller_type="double",
    )
    patched = _apply(file_text, diff)
    assert "    int n = 2;" in patched                       # int untouched
    assert f"{_SCALAR} r__ext = a__ff * 2.0;" in patched     # double promoted


# --------------------------------------------------------------------------- #
# Blocker A §8 — closure_names awareness in the boundary transform.
# A carrier is a chain variable declared OUTSIDE the region whose decl the emission
# layer widens to the extended type.  The boundary transform must treat it as
# already-promoted: excluded from pure_reads/caseB/decl_writes, no r__/w__ alias,
# and a carrier write counts as a landing for the no-op guard.
# --------------------------------------------------------------------------- #

_DD = "quad::ddfun::ddouble"


def test_carrier_write_only_region_is_not_a_no_op():
    # The ONLY landing in this region is the carrier write ``Y`` (declared
    # elsewhere, widened by emission).  Excluding Y from caseB (as a carrier must be)
    # empties decl_writes AND caseB — so WITHOUT the carrier-write landing the no-op
    # guard would fire and report promoted=False.  Counting the carrier write as a
    # landing keeps promoted=True.
    region = "    Y = a + b;"
    block, promoted = boundary.promote_region_block(
        region, reads=["a", "b"], writes=["Y"], scalar_type=_DD,
        caller_type="double", two_limb=True, closure_names={"Y"},
    )
    assert promoted is True
    body = "\n".join(block)
    # reads still promoted at entry …
    assert f"{_DD} a__ff = {_DD}(a);" in body
    assert f"{_DD} b__ff = {_DD}(b);" in body
    # … but the carrier itself is NOT renamed and NOT seeded/demoted as w__ext.
    assert "Y__ext" not in body
    assert "Y__ff" not in body
    assert "Y = a__ff + b__ff;" in body          # carrier write, un-aliased


def test_carrier_write_excluded_from_caseB_caller_write_gate_unchanged():
    # A region with a genuine caller-precision Case-B write (``acc``) AND a carrier
    # write (``Y``).  ``Y`` must be excluded from caseB (no w__ext seed/demote);
    # ``acc``'s boundary treatment is exactly what it would be without any carrier.
    region = (
        "    acc = acc + a;\n"
        "    Y = acc * a;"
    )
    block_carrier, _ = boundary.promote_region_block(
        region, reads=["a"], writes=["acc", "Y"], scalar_type=_DD,
        caller_type="double", two_limb=True, closure_names={"Y"},
    )
    body_c = "\n".join(block_carrier)
    # acc treated as Case-B exactly as usual: seeded + renamed + demoted.
    assert f"{_DD} acc__ext = {_DD}(acc);" in body_c
    assert "acc__ext = acc__ext + a__ff;" in body_c
    assert ("acc = static_cast<double>(acc__ext.hi) + "
            "static_cast<double>(acc__ext.lo);") in body_c
    # Y is a carrier: never aliased, never demoted; written under its own name.
    assert "Y__ext" not in body_c
    assert "Y = acc__ext * a__ff;" in body_c

    # The acc reasoning is identical when Y is NOT a carrier's difference — compare
    # against the same region with Y absent from closure_names but present as a plain
    # Case-B write: acc's three boundary lines are unchanged either way.
    region_acc_only = "    acc = acc + a;"
    block_plain, _ = boundary.promote_region_block(
        region_acc_only, reads=["a"], writes=["acc"], scalar_type=_DD,
        caller_type="double", two_limb=True,
    )
    body_p = "\n".join(block_plain)
    for line in (f"{_DD} acc__ext = {_DD}(acc);",
                 "acc__ext = acc__ext + a__ff;",
                 "acc = static_cast<double>(acc__ext.hi) + "
                 "static_cast<double>(acc__ext.lo);"):
        assert line in body_p and line in body_c


def test_carrier_seeds_dataflow_for_dependent_local():
    # A region-local decl whose RHS consumes a carrier promotes (Rule R2), even
    # though the carrier itself is never aliased.
    region = "    double h = Y + Y - one;"
    block, promoted = boundary.promote_region_block(
        region, reads=["one"], writes=[], scalar_type=_DD,
        caller_type="double", two_limb=True, closure_names={"Y"},
    )
    assert promoted is True
    body = "\n".join(block)
    # h promoted because Y (carrier, seeded into the promoted set) flows into it.
    assert f"{_DD} h__ext = Y + Y - one__ff;" in body
    assert "Y__ff" not in body and "Y__ext" not in body   # carrier un-aliased


def test_write_truncation_inert_ignores_carrier_writes():
    # A region whose only "truncating" write is a carrier (widened by emission) is
    # NOT inert: the widened value survives in the widened carrier decl.
    region = "    Y = a + b;"
    # Without carrier awareness: Y is a Case-B write demoted to double → a provable
    # truncating landing → the 2d-B gate would flag it inert.
    assert boundary.write_truncation_inert(
        region, reads=["a", "b"], writes=["Y"], two_limb=True,
        caller_type="double") is True
    # With Y as a carrier: excluded from caseB; no other landing → not inert.
    assert boundary.write_truncation_inert(
        region, reads=["a", "b"], writes=["Y"], two_limb=True,
        caller_type="double", closure_names={"Y"}) is False


def test_write_truncation_inert_noncarrier_reasoning_unchanged():
    # A genuine Case-B truncating write alongside a carrier still flags inert on the
    # strength of the non-carrier write — the gate reasoning for non-carriers is
    # unchanged (strictly additive).
    region = (
        "    acc = a + b;\n"
        "    Y = acc + a;"
    )
    assert boundary.write_truncation_inert(
        region, reads=["a", "b"], writes=["acc", "Y"], two_limb=True,
        caller_type="double", closure_names={"Y"}) is True


def test_scan_bare_decls_multi_declarator():
    # The worked example from the design doc: ``TMass Y, S, A;`` — a bare
    # multi-declarator the init-only _scan_decls misses.
    toks = boundary._tokenize("    TMass Y, S, A;\n")
    decls = boundary._scan_bare_decls(toks)
    assert len(decls) == 1
    d = decls[0]
    assert d.type_text == "TMass"
    assert d.names == ["Y", "S", "A"]


def test_scan_bare_decls_forms_and_rejections():
    src = (
        "void f() {\n"
        "    const TMass H = Y + one;\n"   # init single (with qualifier)
        "    TScale s2;\n"                 # bare single
        "    TMass Y, S, A;\n"             # bare multi-declarator
        "    Kokkos::complex<double> z, w;\n"  # qualified + template, multi
        "    foo(x, y);\n"                 # call — not a decl
        "    res(i, 0) = H;\n"            # subscript store — not a decl
        "    obj.member = 1;\n"           # member access — not a decl
        "}\n"
    )
    decls = {n: d for d in boundary._scan_bare_decls(boundary._tokenize(src))
             for n in d.names}
    assert decls["H"].type_text == "TMass"
    assert decls["s2"].type_text == "TScale"
    assert decls["Y"].names == ["Y", "S", "A"]
    assert decls["z"].type_text == "complex" and decls["z"].names == ["z", "w"]
    # calls / stores / member accesses are not declarations
    assert "foo" not in decls and "res" not in decls
    assert "obj" not in decls and "member" not in decls


def test_closure_names_default_empty_no_regression():
    # With closure_names defaulting to empty, the transform is byte-identical to the
    # pre-carrier behavior (the whole existing suite already covers this; this pins
    # the default explicitly).
    region = "    double r = a + b;"
    b1, p1 = boundary.promote_region_block(
        region, reads=["a", "b"], writes=[], scalar_type=_DD, caller_type="double")
    b2, p2 = boundary.promote_region_block(
        region, reads=["a", "b"], writes=[], scalar_type=_DD, caller_type="double",
        closure_names=frozenset())
    assert b1 == b2 and p1 == p2
