"""Fixtures for the Phase-2a fan-out tests.

A synthetic, self-contained C++ header with a template call graph that exercises
multi-caller (fan-in) enumeration:

    entry -> h  -> g  -> f
    entry -> h  -> g2 -> f
    entry -> h2 -> g  -> f

So ``g`` has two callers (``h``, ``h2``) and ``f`` has two callers (``g``, ``g2``):
three distinct caller-paths from ``entry`` to ``f``, two to ``g``.  The bodies are
template functions whose calls are dependent (the real qcdloop shape), so they also
prove the token-scan edge extraction works where the AST drops dependent calls.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

APP_H = """\
#pragma once
namespace app {

template<class T>
T f(T x) {
    T a = x + T(1);
    T b = a * T(2);
    return b;
}

template<class T>
T g(T x) {
    return f<T>(x) * T(2);
}

template<class T>
T g2(T x) {
    return f<T>(x) - T(3);
}

template<class T>
T h(T x) {
    return g<T>(x) + g2<T>(x);
}

template<class T>
T h2(T x) {
    return g<T>(x);
}

template<class T>
T entry(T x) {
    return h<T>(x) + h2<T>(x);
}

}  // namespace app
"""


def _libclang_ok() -> bool:
    try:
        import clang.cindex as C  # noqa: PLC0415
        C.Index.create()
        return True
    except Exception:
        return False


LIBCLANG = _libclang_ok()
requires_libclang = pytest.mark.skipif(not LIBCLANG, reason="libclang bindings unavailable")

_GXX = shutil.which("g++")
requires_gxx = pytest.mark.skipif(_GXX is None, reason="g++ not on PATH")


@pytest.fixture
def synth_tree(tmp_path) -> Path:
    """A tree containing the synthetic ``app.h`` header; returns the tree root."""
    (tmp_path / "app.h").write_text(APP_H)
    return tmp_path


@pytest.fixture
def synth_graph(synth_tree):
    """Call graph rooted at ``entry`` over the synthetic tree (skips w/o libclang)."""
    if not LIBCLANG:
        pytest.skip("libclang bindings unavailable")
    from agents.patcher.call_graph import build_call_graph
    from agents.patcher import fanout
    fanout.clear_graph_cache()
    return build_call_graph("entry", synth_tree, tu_file=synth_tree / "app.h")


def gxx_compile(src: Path, out: Path) -> subprocess.CompletedProcess:
    """Compile ``src`` to object ``out`` with g++ (C++17), returning the result."""
    return subprocess.run([_GXX, "-std=c++17", "-c", str(src), "-o", str(out)],
                          capture_output=True, text=True, timeout=120)
