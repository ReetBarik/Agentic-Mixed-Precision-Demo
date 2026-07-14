"""Unit tests for deterministic C8 library-patch synthesis (no LLM, no build).

C8 detects int<->tracked crossings from the compiler's own diagnostics and maps
them to source annotations.  These tests feed synthetic gcc-style stderr to
``derive_c8_patch`` and verify: the three crossing patterns (a/b/c) map to the
correct rewrites, the synthesized diff ``git apply``s cleanly, an unrecognized
int<->tracked error hard-fails (C8_UNCLASSIFIED_ERROR), and a non-C8 build error
yields no patch.
"""

import subprocess

import pytest

from agents.tracked_integrator import agent as ti


def _init_lib(root):
    """A git repo with a header carrying one of each crossing pattern."""
    lib = root / "lib"
    (lib / "box").mkdir(parents=True)
    header = lib / "box" / "Widget.h"
    header.write_text(
        "        int ir12 = 0;\n"
        "        if (cond) ir12 = ql::Constants<TScale>::_ten() * ql::Sign(rr);\n"
        "        res = ql::xspence<TOutput, TMass, TScale>(x4, ix4, r14, ir12) +\n"
        "        if (ql::Imag(r13) == 0) { done(); }\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "i"],
        cwd=root, check=True,
    )
    return lib, header


def _diag(h, line_no, col, msg, src, target, notes=""):
    """One gcc diagnostic block with a caret ruler aligned under ``target``.

    Mirrors gcc's layout: ``NN | <src>`` then ``   | <carets>``, where the
    content after ``|`` aligns.  The caret ruler is generated so the first/last
    ``~``/``^`` fall exactly under ``target`` within ``src`` (no hand-counting).
    """
    src_content = " " + src            # the single space gcc prints after '|'
    start = src_content.index(target)
    ruler = [" "] * len(src_content)
    for k in range(start, start + len(target)):
        ruler[k] = "~"
    ruler[start] = "^"
    caret_content = "".join(ruler)
    out = f"{h}:{line_no}:{col}: error: {msg}\n"
    out += f"  {line_no} |{src_content}\n"
    out += f"     |{caret_content}\n"
    if notes:
        out += notes
    return out


def _stderr(lib):
    """Synthetic gcc diagnostics (curly quotes) for the three crossings."""
    h = lib / "box" / "Widget.h"
    a = _diag(
        h, 2, 35,
        "cannot convert ‘tracked::Tracked<double>’ to ‘int’ in assignment",
        "        if (cond) ir12 = ql::Constants<TScale>::_ten() * ql::Sign(rr);",
        "ir12 = ql::Constants<TScale>::_ten() * ql::Sign(rr)",
    )
    # (b) uses the file line + the "in passing argument N" note (not the caret).
    b = _diag(
        h, 3, 63,
        "invalid initialization of reference of type ‘const tracked::Tracked<double>&’ from expression of type ‘int’",
        "        res = ql::xspence<TOutput, TMass, TScale>(x4, ix4, r14, ir12) +",
        "ir12",
        notes="other.h:606:1: note: in passing argument 4 of ‘TOutput ql::xspence(const X&, const Y&, const Z&, const tracked::Tracked<double>&)’\n  606 |   sig\n     |   ^\n",
    )
    c = _diag(
        h, 4, 27,
        "no match for ‘operator==’ (operand types are ‘tracked::Tracked<double>’ and ‘int’)",
        "        if (ql::Imag(r13) == 0) { done(); }",
        "ql::Imag(r13) == 0",
    )
    return a + b + c


def test_derive_maps_three_patterns_and_applies(tmp_path):
    lib, header = _init_lib(tmp_path)
    patch = ti.derive_c8_patch(_stderr(lib), lib, tmp_path)
    assert patch is not None
    # (a) tracked->int assignment
    assert "ir12 = static_cast<int>((ql::Constants<TScale>::_ten() * ql::Sign(rr)).value())" in patch
    # (b) int->tracked ref bind, leading ', ' preserved
    assert "r14, tracked::Tracked<double>(ir12))" in patch
    # (c) tracked==int -> .value() on the tracked side
    assert "(ql::Imag(r13)).value() == 0" in patch

    patch_file = tmp_path / "app.patch"
    patch_file.write_text(patch, encoding="utf-8")
    subprocess.run(["git", "apply", "--check", str(patch_file)], cwd=tmp_path, check=True)
    subprocess.run(["git", "apply", str(patch_file)], cwd=tmp_path, check=True)
    patched = header.read_text(encoding="utf-8")
    assert "static_cast<int>(" in patched
    assert "tracked::Tracked<double>(ir12)" in patched
    assert "(ql::Imag(r13)).value() == 0" in patched


def test_no_c8_errors_returns_none(tmp_path):
    lib, _ = _init_lib(tmp_path)
    genuine = (
        f"{lib}/box/Widget.h:9:1: error: 'foo' was not declared in this scope\n"
        "    9 | foo();\n      | ^~~\n"
    )
    assert ti.derive_c8_patch(genuine, lib, tmp_path) is None


def test_unclassified_int_tracked_error_hard_fails(tmp_path):
    lib, _ = _init_lib(tmp_path)
    # int<->tracked flavored (mentions tracked::Tracked + int) but not a/b/c.
    weird = (
        f"{lib}/box/Widget.h:2:1: error: no known conversion from "
        "‘int’ to ‘tracked::Tracked<double>’ for some novel reason\n"
        "    2 | x;\n      | ^\n"
    )
    with pytest.raises(RuntimeError, match="C8_UNCLASSIFIED_ERROR"):
        ti.derive_c8_patch(weird, lib, tmp_path)


def test_errors_outside_headers_dir_ignored(tmp_path):
    lib, _ = _init_lib(tmp_path)
    # An int<->tracked error in the DRIVER (outside headers_dir) is not C8's job.
    outside = (
        f"{tmp_path}/driver.cpp:5:1: error: cannot convert "
        "‘tracked::Tracked<double>’ to ‘int’ in assignment\n"
        "    5 | q = z;\n      | ~~^~~\n"
    )
    assert ti.derive_c8_patch(outside, lib, tmp_path) is None


def test_split_top_level_respects_templates():
    # commas inside <...> and (...) must not split the argument list
    parts = ti._split_top_level("a, Kokkos::Array<T, 2>, f(x, y), -z")
    assert [p[0].strip() for p in parts] == ["a", "Kokkos::Array<T, 2>", "f(x, y)", "-z"]


def test_extract_call_arg_negated():
    line = "   ql::xspence<TOutput, TMass, TScale>(x4, ix4, a / r14, -ir14) +"
    text, start, end = ti._extract_call_arg(line, "ql::xspence", 4)
    assert text.strip() == "-ir14"
    assert line[start:end] == text
