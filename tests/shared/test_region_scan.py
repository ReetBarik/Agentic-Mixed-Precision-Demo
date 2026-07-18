"""Tests for ``agents.shared.region_scan.extract_region_writes`` (Fix C).

Every behavioural test runs under BOTH backends via the ``backend`` fixture:

* ``libclang`` — the preferred AST path (skipped if the bindings are absent, so a
  bindings-less CI never gives false confidence);
* ``fallback`` — the keyword-token lexer, forced by monkeypatching the lazy
  ``_import_clang`` to raise ``ImportError``.

Fixtures are self-contained C++ (they carry a minimal ``Tracked`` definition) so
the libclang path resolves the tracked type off the in-file AST rather than a
missing header.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agents.shared import region_scan
from agents.shared.region_scan import extract_region_writes

# A minimal, self-contained tracked type on a single line so body line numbers
# below are stable and easy to reason about.
PREAMBLE = ("template<class T> struct Tracked { T v; Tracked(){} "
            "Tracked(T x):v(x){} Tracked& operator=(const Tracked&){return *this;} "
            "T value() const { return v; } };")


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args],
                   capture_output=True, text=True, check=True)


def _init_repo(tmp_path, name, content):
    """Init a repo with ``name`` -> ``content``; return (root, sha, abs_path)."""
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    f = root / name
    f.write_text(content, encoding="utf-8")
    if not (root / ".git").exists():
        _git(root, "init", "-q")
        _git(root, "config", "user.email", "t@t.t")
        _git(root, "config", "user.name", "t")
        _git(root, "config", "commit.gpgsign", "false")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "snapshot")
    sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return root, sha, f


def _recommit(root, f, content):
    f.write_text(content, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "change")
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def _raise_import_error():
    raise ImportError("forced: libclang bindings unavailable")


@pytest.fixture(params=["libclang", "fallback"])
def backend(request, monkeypatch):
    """Parameterize each test over both extraction backends."""
    if request.param == "libclang":
        try:
            region_scan._import_clang()
        except ImportError:
            pytest.skip("libclang bindings not installed")
    else:
        monkeypatch.setattr(region_scan, "_import_clang", _raise_import_error)
    return request.param


# --------------------------------------------------------------------------- #
# core behaviour (both backends)
# --------------------------------------------------------------------------- #

def test_three_tracked_locals_in_source_order(tmp_path, backend):
    content = PREAMBLE + "\n" + (
        "inline void f() {\n"                                 # 2
        "    Tracked<double> a = Tracked<double>(1.0);\n"     # 3
        "    Tracked<double> b = Tracked<double>(2.0);\n"     # 4
        "    Tracked<double> c = Tracked<double>(3.0);\n"     # 5
        "}\n")                                                # 6
    root, sha, f = _init_repo(tmp_path, "k.h", content)
    assert extract_region_writes(str(f), 3, 5, sha) == ["a", "b", "c"]


def test_reads_only_returns_empty(tmp_path, backend):
    content = PREAMBLE + "\n" + (
        "inline double g(const Tracked<double>& a) {\n"       # 2
        "    double s = a.value();\n"                         # 3
        "    double t = s * 2.0;\n"                           # 4
        "    return t;\n"                                     # 5
        "}\n")                                                # 6
    root, sha, f = _init_repo(tmp_path, "k.h", content)
    assert extract_region_writes(str(f), 3, 5, sha) == []


def test_excludes_non_tracked_declaration(tmp_path, backend):
    content = PREAMBLE + "\n" + (
        "inline void f() {\n"                                 # 2
        "    int counter = 0;\n"                              # 3
        "    Tracked<double> x = Tracked<double>(1.0);\n"     # 4
        "}\n")                                                # 5
    root, sha, f = _init_repo(tmp_path, "k.h", content)
    assert extract_region_writes(str(f), 3, 4, sha) == ["x"]


def test_only_counts_writes_within_range(tmp_path, backend):
    content = PREAMBLE + "\n" + (
        "inline void f() {\n"                                 # 2
        "    Tracked<double> a = Tracked<double>(1.0);\n"     # 3
        "}\n"                                                 # 4
        "inline void h() {\n"                                 # 5
        "    Tracked<double> b = Tracked<double>(2.0);\n"     # 6
        "}\n")                                                # 7
    root, sha, f = _init_repo(tmp_path, "k.h", content)
    # Range stops at line 4 -> b (line 6, another function) is excluded.
    assert extract_region_writes(str(f), 3, 4, sha) == ["a"]
    # Widening the range picks up b too.
    assert extract_region_writes(str(f), 3, 6, sha) == ["a", "b"]


def test_reassignment_is_set_semantics(tmp_path, backend):
    # Documented choice: a write SET -> each name once, in first-write order.
    content = PREAMBLE + "\n" + (
        "inline void f() {\n"                                 # 2
        "    Tracked<double> a = Tracked<double>(1.0);\n"     # 3
        "    a = Tracked<double>(2.0);\n"                     # 4
        "}\n")                                                # 5
    root, sha, f = _init_repo(tmp_path, "k.h", content)
    assert extract_region_writes(str(f), 3, 4, sha) == ["a"]


def test_reassignment_of_var_declared_above_region(tmp_path, backend):
    # A tracked local declared before the region but written inside it must be
    # captured (both backends build the tracked-name universe file-wide).
    content = PREAMBLE + "\n" + (
        "inline void f() {\n"                                 # 2
        "    Tracked<double> a = Tracked<double>(1.0);\n"     # 3
        "    a = Tracked<double>(9.0);\n"                     # 4
        "}\n")                                                # 5
    root, sha, f = _init_repo(tmp_path, "k.h", content)
    # Region covers ONLY the re-assignment line, not the declaration.
    assert extract_region_writes(str(f), 4, 4, sha) == ["a"]


def test_sha_resolution_two_commits(tmp_path, backend):
    v1 = PREAMBLE + "\n" + (
        "inline void f() {\n"                                 # 2
        "    Tracked<double> a = Tracked<double>(1.0);\n"     # 3
        "}\n")                                                # 4
    root, sha1, f = _init_repo(tmp_path, "k.h", v1)
    v2 = PREAMBLE + "\n" + (
        "inline void f() {\n"                                 # 2
        "    Tracked<double> a = Tracked<double>(1.0);\n"     # 3
        "    Tracked<double> b = Tracked<double>(2.0);\n"     # 4
        "}\n")                                                # 5
    sha2 = _recommit(root, f, v2)
    # Same call, different SHA -> different write set (source-only, per-commit).
    assert extract_region_writes(str(f), 3, 4, sha1) == ["a"]
    assert extract_region_writes(str(f), 3, 4, sha2) == ["a", "b"]


def test_comment_and_string_are_not_written_vars(tmp_path, backend):
    # A `Tracked<double>` mentioned only in a comment / string must not register.
    content = PREAMBLE + "\n" + (
        "inline void f() {\n"                                 # 2
        "    // Tracked<double> ghost = Tracked<double>(0);\n"  # 3
        '    const char* s = "Tracked<double> nope = x;";\n'    # 4
        "    Tracked<double> real = Tracked<double>(1.0);\n"    # 5
        "}\n")                                                  # 6
    root, sha, f = _init_repo(tmp_path, "k.h", content)
    assert extract_region_writes(str(f), 3, 5, sha) == ["real"]


def test_parameterized_tracked_type(tmp_path, backend):
    # The tracked type name is parameterizable for future scalar types.
    content = ("template<class T> struct FF { T v; FF(){} FF(T x):v(x){} };\n"  # 1
               "inline void f() {\n"                                  # 2
               "    FF<double> a = FF<double>(1.0);\n"                 # 3
               "    Tracked<double> b = a;\n"                          # 4  (not FF)
               "}\n")                                                  # 5
    root, sha, f = _init_repo(tmp_path, "k.h", content)
    assert extract_region_writes(str(f), 3, 4, sha, tracked_type="FF") == ["a"]


# --------------------------------------------------------------------------- #
# backend-specific coverage
# --------------------------------------------------------------------------- #

def test_fallback_matches_libclang_on_same_fixture(tmp_path, monkeypatch):
    """Explicit equivalence: identical result with and without libclang."""
    content = PREAMBLE + "\n" + (
        "inline void f() {\n"
        "    int counter = 0;\n"
        "    Tracked<double> a = Tracked<double>(1.0);\n"
        "    Tracked<double> b = Tracked<double>(2.0);\n"
        "    a = b;\n"
        "}\n")
    root, sha, f = _init_repo(tmp_path, "k.h", content)

    try:
        region_scan._import_clang()
        have_libclang = True
    except ImportError:
        have_libclang = False

    if have_libclang:
        libclang_result = extract_region_writes(str(f), 3, 6, sha)
        assert libclang_result == ["a", "b"]

    monkeypatch.setattr(region_scan, "_import_clang", _raise_import_error)
    fallback_result = extract_region_writes(str(f), 3, 6, sha)
    assert fallback_result == ["a", "b"]


def test_libclang_empty_over_unresolved_type_falls_back(tmp_path, monkeypatch):
    """libclang present but the tracked header is unresolved -> lexer rescue.

    Without an in-file ``Tracked`` definition and no include context, clang
    mis-recovers the decl; the empty-result-with-tracked-text guard must hand the
    region to the include-free lexer, which still finds the writes textually.
    """
    try:
        region_scan._import_clang()
    except ImportError:
        pytest.skip("libclang bindings not installed")

    content = ('#include "tracked_missing.hpp"\n'                      # 1
               "inline void f() {\n"                                   # 2
               "    Tracked<double> a = make(1.0);\n"                  # 3
               "    Tracked<double> b = make(2.0);\n"                  # 4
               "}\n")                                                  # 5
    root, sha, f = _init_repo(tmp_path, "k.h", content)
    # The guard should route to the lexer and still recover both writes.
    assert extract_region_writes(str(f), 3, 4, sha) == ["a", "b"]


def test_absolute_and_relative_paths(tmp_path, backend, monkeypatch):
    content = PREAMBLE + "\n" + (
        "inline void f() {\n"
        "    Tracked<double> a = Tracked<double>(1.0);\n"
        "}\n")
    root, sha, f = _init_repo(tmp_path, "k.h", content)
    # Absolute path.
    assert extract_region_writes(str(f), 3, 3, sha) == ["a"]
    # Relative path, resolved from a cwd inside the repo.
    monkeypatch.chdir(root)
    assert extract_region_writes("k.h", 3, 3, sha) == ["a"]


def test_bad_sha_raises(tmp_path, backend):
    content = PREAMBLE + "\ninline void f() {}\n"
    root, sha, f = _init_repo(tmp_path, "k.h", content)
    with pytest.raises(region_scan.RegionScanError):
        extract_region_writes(str(f), 2, 2, "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
