"""Unit tests for the C8 library-patch synthesis (no LLM, no build).

Exercises the deterministic Python half of C8: splitting the LLM response on the
``===C8PATCH===`` sentinel, turning edit records into a git-apply-able unified
diff, the exactly-once hard-fail guard, and the PATCH_HASH cache validation.
"""

import subprocess

import pytest

from agents.tracked_integrator import agent as ti


def _init_git_tree(root):
    """A minimal git repo with one library header, for git-apply round-trips."""
    lib = root / "lib"
    (lib / "box").mkdir(parents=True)
    header = lib / "box" / "Widget.h"
    header.write_text(
        "int flag = 0;\n"
        "if (cond) flag = ten() * Sign(x);\n"
        "call(a, flag);\n"
        "if (Imag(z) == 0) { done(); }\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=root, check=True,
    )
    return lib, header


def test_split_response_no_sentinel_yields_no_records():
    shim, records = ti._split_llm_response("#pragma once\nint foo();\n")
    assert "#pragma once" in shim
    assert records == []


def test_split_response_empty_array():
    resp = "#pragma once\n===C8PATCH===\n[]\n"
    shim, records = ti._split_llm_response(resp)
    assert "#pragma once" in shim
    assert "===C8PATCH===" not in shim  # sentinel is split off, never in the shim
    assert records == []


def test_synthesized_diff_applies_with_git(tmp_path):
    lib, header = _init_git_tree(tmp_path)
    records = [
        {"file": "box/Widget.h", "pattern": "a",
         "original": "flag = ten() * Sign(x);",
         "replacement": "flag = static_cast<int>((ten() * Sign(x)).value());",
         "rule": "C8(a)"},
        {"file": "box/Widget.h", "pattern": "b",
         "original": "call(a, flag);",
         "replacement": "call(a, TScale(flag));",
         "rule": "C8(b)"},
        {"file": "box/Widget.h", "pattern": "c",
         "original": "if (Imag(z) == 0) { done(); }",
         "replacement": "if (Imag(z).value() == 0.0) { done(); }",
         "rule": "C8(c)"},
    ]
    diff = ti._synthesize_patch(records, lib, tmp_path)
    assert diff is not None
    assert "a/lib/box/Widget.h" in diff and "b/lib/box/Widget.h" in diff

    patch_file = tmp_path / "app.patch"
    patch_file.write_text(diff, encoding="utf-8")

    # git apply --check must accept it, and applying yields the annotated source.
    subprocess.run(["git", "apply", "--check", str(patch_file)], cwd=tmp_path, check=True)
    subprocess.run(["git", "apply", str(patch_file)], cwd=tmp_path, check=True)
    patched = header.read_text(encoding="utf-8")
    assert "static_cast<int>((ten() * Sign(x)).value())" in patched
    assert "call(a, TScale(flag));" in patched
    assert "Imag(z).value() == 0.0" in patched


def test_original_must_be_unique(tmp_path):
    lib, _ = _init_git_tree(tmp_path)
    # "flag" appears many times → not uniqueness-sufficient → hard fail.
    records = [{"file": "box/Widget.h", "pattern": "a",
                "original": "flag", "replacement": "FLAG", "rule": "bad"}]
    with pytest.raises(RuntimeError, match="exactly once"):
        ti._synthesize_patch(records, lib, tmp_path)


def test_missing_original_hard_fails(tmp_path):
    lib, _ = _init_git_tree(tmp_path)
    records = [{"file": "box/Widget.h", "pattern": "a",
                "original": "no_such_text_here();", "replacement": "x", "rule": "bad"}]
    with pytest.raises(RuntimeError, match="exactly once"):
        ti._synthesize_patch(records, lib, tmp_path)


def test_empty_records_returns_none(tmp_path):
    lib, _ = _init_git_tree(tmp_path)
    assert ti._synthesize_patch([], lib, tmp_path) is None


def test_patch_cache_valid_logic(tmp_path):
    shim = tmp_path / "app_interop.hpp"
    patch = tmp_path / "app.patch"

    # No PATCH_HASH line (pre-C8 shim) → always valid.
    assert ti._patch_cache_valid("// SOURCE_HASH: abc\n", shim, "app") is True

    # Declares NONE and no patch on disk → valid; a stray patch → invalid.
    none_shim = "// SOURCE_HASH: abc\n// PATCH_HASH: NONE\n"
    assert ti._patch_cache_valid(none_shim, shim, "app") is True
    patch.write_text("stray\n", encoding="utf-8")
    assert ti._patch_cache_valid(none_shim, shim, "app") is False
    patch.unlink()

    # Declares a hash → patch must exist and match.
    body = b"diff body\n"
    import hashlib
    h = hashlib.sha256(body).hexdigest()
    hash_shim = f"// SOURCE_HASH: abc\n// PATCH_HASH: {h}\n"
    assert ti._patch_cache_valid(hash_shim, shim, "app") is False  # no file yet
    patch.write_bytes(body)
    assert ti._patch_cache_valid(hash_shim, shim, "app") is True
    patch.write_bytes(b"tampered\n")
    assert ti._patch_cache_valid(hash_shim, shim, "app") is False
