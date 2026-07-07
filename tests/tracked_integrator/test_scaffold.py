"""Structural smoke test for the tracked_integrator scaffold.

No LLM, no build — just verifies the shared-service shape: integrate() writes a
placeholder ``<app>_interop.hpp`` carrying a real ``// SOURCE_HASH:`` line, reuses
it when the headers are unchanged, and regenerates when they change.
"""

from pathlib import Path

import pytest

from agents.tracked_integrator import agent as tracked_integrator


def _make_target(tmp_path: Path, header_body: str = "int foo();\n"):
    """Create a minimal (headers dir, driver source) pair under tmp_path."""
    headers = tmp_path / "libfoo_headers"
    headers.mkdir()
    (headers / "foo.h").write_text(header_body, encoding="utf-8")

    src = tmp_path / "src"
    src.mkdir()
    driver = src / "micro_driver.cpp"
    driver.write_text("int main() { return 0; }\n", encoding="utf-8")
    return headers, driver


def _source_hash_line(shim: Path) -> str:
    for line in shim.read_text(encoding="utf-8").splitlines():
        if "SOURCE_HASH:" in line:
            return line.split("SOURCE_HASH:", 1)[1].strip()
    raise AssertionError(f"no SOURCE_HASH line in {shim}")


def test_integrate_writes_placeholder_shim(tmp_path):
    headers, driver = _make_target(tmp_path)

    shim = tracked_integrator.integrate(
        target_library_headers=headers,
        driver_source_path=driver,
    )

    # Written next to the driver, named from the (suffix-stripped) headers dir.
    assert shim.exists()
    assert shim.name == "libfoo_interop.hpp"
    assert shim.parent == driver.parent

    text = shim.read_text(encoding="utf-8")
    assert "#pragma once" in text
    assert "SCAFFOLD PLACEHOLDER" in text
    # A real hash was embedded, not the PENDING placeholder.
    assert _source_hash_line(shim) != "PENDING"
    assert len(_source_hash_line(shim)) == 64  # sha256 hexdigest


def test_integrate_is_cached_when_headers_unchanged(tmp_path):
    headers, driver = _make_target(tmp_path)

    first = tracked_integrator.integrate(headers, driver)
    first_mtime = first.stat().st_mtime_ns

    # Second call, headers unchanged, pointed at the existing shim → cache hit:
    # same path returned and the file is left untouched (not rewritten).
    second = tracked_integrator.integrate(headers, driver, existing_shim=first)
    assert second == first
    assert second.stat().st_mtime_ns == first_mtime


def test_integrate_regenerates_when_headers_change(tmp_path):
    headers, driver = _make_target(tmp_path)

    first = tracked_integrator.integrate(headers, driver)
    hash_before = _source_hash_line(first)

    # Mutate a header → hash changes → shim is regenerated with the new hash.
    (headers / "foo.h").write_text("int foo(); double bar();\n", encoding="utf-8")
    second = tracked_integrator.integrate(headers, driver, existing_shim=first)

    assert second == first  # same output path
    assert _source_hash_line(second) != hash_before


def test_integrate_rejects_non_directory_headers(tmp_path):
    _, driver = _make_target(tmp_path)
    not_a_dir = tmp_path / "nope.h"
    not_a_dir.write_text("x\n", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        tracked_integrator.integrate(not_a_dir, driver)
