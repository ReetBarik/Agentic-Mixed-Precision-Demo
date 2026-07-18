"""Plain-type-edit (P3a) — keyword-token rewriter corruption safety."""

from pathlib import Path

import pytest

from agents.patcher import edits


def _write(tmp_path, text):
    p = tmp_path / "f.h"
    p.write_text(text)
    return p


def test_swaps_double_to_float_in_range(tmp_path):
    p = _write(tmp_path, "double a;\ndouble b;\ndouble c;\n")
    n = edits.rewrite_types(p, 2, 2, "double", "float")
    assert n == 1
    assert p.read_text() == "double a;\nfloat b;\ndouble c;\n"


def test_leaves_comments_strings_identifiers_untouched(tmp_path):
    src = (
        'double x = 1;                 // this double is in a comment\n'
        'const char* s = "double y";   // and this one in a string\n'
        'struct double_wrap { int z; }; // wait, not a keyword: double_wrap\n'
        'double real = x;\n'
    )
    p = _write(tmp_path, src)
    # rewrite the whole span
    n = edits.rewrite_types(p, 1, 4, "double", "float")
    out = p.read_text()
    # only the two real keyword tokens (line 1 decl, line 4 decl) change
    assert n == 2
    assert out.startswith("float x = 1;")
    assert "// this double is in a comment" in out       # comment untouched
    assert '"double y"' in out                            # string untouched
    assert "double_wrap" in out                           # identifier untouched
    assert "float real = x;" in out


def test_float_to_double_roundtrip(tmp_path):
    p = _write(tmp_path, "inline float f(float q){ return q; }\n")
    edits.rewrite_types(p, 1, 1, "float", "double")
    assert p.read_text() == "inline double f(double q){ return q; }\n"


def test_no_occurrence_raises(tmp_path):
    p = _write(tmp_path, "int a;\nint b;\n")
    with pytest.raises(edits.EditError):
        edits.rewrite_types(p, 1, 2, "double", "float")


def test_out_of_range_line_not_touched(tmp_path):
    p = _write(tmp_path, "double a;\ndouble b;\n")
    edits.rewrite_types(p, 1, 1, "double", "float")
    assert p.read_text() == "float a;\ndouble b;\n"
