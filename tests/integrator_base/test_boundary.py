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
