"""Thin AMP wrapper over ``tracked_tools.inject`` (the librarized line injector).

The libclang per-statement ``line=`` scope injection moved into the Tracked
library (``third_party/tracked/tools``, console script ``tracked-line-inject``)
with its qcdloop specifics turned into seams.  This wrapper bakes AMP's
historical parameterization back in, so regenerated patches are byte-identical
to the committed ones:

* the ``_ql_line_scope`` RAII variable spelling (committed
  ``ql_tracked_lines.patch`` + ``.hash`` sidecars depend on it),
* the qcdloop ``box/`` include subdir and the ``KOKKOS_ENABLE_SERIAL`` define
  for the libclang parse,
* the historical ``kokkos_include=`` keyword.

See the library module for the full transform documentation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tracked_tools.inject as _inject
from tracked_tools.inject import (  # noqa: F401
    TRANSFORM_VERSION,
    _HEADER_SUFFIXES,
    _Site,
    _cindex,
    _collect_sites,
    _control_kinds,
    _decl_kinds,
    _has_op,
    _kinds,
    _op_kinds,
    _stmt_span,
    cache_key,
    gcc_search_dirs,
)

# AMP's historical RAII scope variable spelling (cache-affecting output).
SCOPE_VAR = "_ql_line_scope"
_DEFINES = ["KOKKOS_ENABLE_SERIAL"]


def _extra_dirs(headers_dir, kokkos_include) -> list:
    dirs = [Path(headers_dir) / "box"]
    if kokkos_include is not None:
        dirs.append(Path(kokkos_include))
    return dirs


def _instrument_text(src: bytes, sites: list) -> bytes:
    """Splice line= wraps into one file's bytes (AMP scope-var spelling)."""
    return _inject._instrument_text(src, sites, SCOPE_VAR)


def build_line_patch(
    driver_source: Path,
    headers_dir: Path,
    tracked_include: Path,
    repo_root: Path,
    kokkos_include: Path | None = None,
    target_basenames: set[str] | None = None,
    cxx_standard: int = 17,
    system_include_dirs: list[str] | None = None,
) -> tuple[str | None, dict]:
    """Historical signature; see tracked_tools.inject.build_line_patch."""
    return _inject.build_line_patch(
        driver_source=driver_source, headers_dir=headers_dir,
        tracked_include=tracked_include, repo_root=repo_root,
        target_basenames=target_basenames, cxx_standard=cxx_standard,
        system_include_dirs=system_include_dirs,
        extra_include_dirs=_extra_dirs(headers_dir, kokkos_include),
        defines=_DEFINES, scope_var_name=SCOPE_VAR,
    )


def generate(
    driver_source: Path,
    headers_dir: Path,
    tracked_include: Path,
    repo_root: Path,
    out_patch: Path,
    kokkos_include: Path | None = None,
    c8_patch: Path | None = None,
    target_basenames: set[str] | None = None,
    cxx_standard: int = 17,
    force: bool = False,
) -> dict:
    """Historical signature; see tracked_tools.inject.generate."""
    return _inject.generate(
        driver_source=driver_source, headers_dir=headers_dir,
        tracked_include=tracked_include, repo_root=repo_root,
        out_patch=out_patch, c8_patch=c8_patch,
        target_basenames=target_basenames, cxx_standard=cxx_standard,
        extra_include_dirs=_extra_dirs(headers_dir, kokkos_include),
        defines=_DEFINES, scope_var_name=SCOPE_VAR, force=force,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate a per-statement line= scope injection patch.")
    ap.add_argument("--driver", required=True, help="driver .cpp translation unit")
    ap.add_argument("--headers", required=True, help="vendored header tree (patch-only)")
    ap.add_argument("--tracked-include", required=True, help="tracked/include dir")
    ap.add_argument("--repo-root", default=".", help="repo root for a/ b/ patch labels")
    ap.add_argument("--out", required=True, help="output .patch path")
    ap.add_argument("--kokkos-include", default=None)
    ap.add_argument("--c8-patch", default=None,
                    help="int<->tracked patch applied before parsing (and reset after)")
    ap.add_argument("--cxx-standard", type=int, default=17)
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    args = ap.parse_args(argv)

    res = generate(
        driver_source=Path(args.driver), headers_dir=Path(args.headers),
        tracked_include=Path(args.tracked_include), repo_root=Path(args.repo_root),
        out_patch=Path(args.out),
        kokkos_include=Path(args.kokkos_include) if args.kokkos_include else None,
        c8_patch=Path(args.c8_patch) if args.c8_patch else None,
        cxx_standard=args.cxx_standard, force=args.force,
    )
    if res["cached"]:
        print(f"line_injector: up to date (cache hit): {args.out}")
    else:
        total = sum(res["stats"].values())
        print(f"line_injector: wrote {args.out} ({total} sites): {res['stats']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
