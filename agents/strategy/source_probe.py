"""Source-level probes Strategy consults to gate mechanically-doomed rungs.

The only probe today is :func:`region_has_bare_double` — the predicate that
decides whether the plain-edit ``-to-float`` speedup rung can possibly apply to a
region.  The Patcher's plain-type-edit rewrites a bare ``double`` *keyword token*
to ``float`` on the region line; a template-typed HPC kernel writes the region in
terms of a template parameter ``T`` (``tracked::Tracked<T>`` etc.), so there is no
literal ``double`` token to rewrite and the rung is inapplicable *to this code*
(it stays valid for non-templated code — see CALIBRATION.md §Bug 4).

Rather than infer templated-ness from the characterization report (which carries
only variable *names*, not their types), we read the ground-truth predicate
straight from source: does a bare ``double`` token appear on the region's lines?
If yes, the rung is attempted; if no, Strategy skips it (the Patcher's
``patch_inapplicable`` is the belt-and-suspenders net for anything this misses).

The probe reads the *working tree* on disk (not a git ref): a speedup region is,
by construction, not dd-promoted, so its lines are still the pristine ``double``
baseline the plain edit would operate on after reverting the ff install.  Any
read/resolution failure returns ``True`` (do not gate) so a probe hiccup never
silently suppresses a legitimate float demotion.
"""

from __future__ import annotations

import re
from pathlib import Path

# A bare ``double`` keyword token: not part of a longer identifier on either side.
_BARE_DOUBLE_RE = re.compile(r"(?<![A-Za-z0-9_])double(?![A-Za-z0-9_])")


def _resolve_in_tree(repo_path: Path, file: str) -> Path | None:
    """Resolve ``file`` (repo-relative or bare basename) to a path in the tree.

    Mirrors :func:`agents.patcher.intent.resolve_in_tree` (kept local so Strategy
    does not import the Patcher): try the path as written, then a unique basename
    match anywhere under ``repo_path`` (skipping ``.git``).
    """
    direct = repo_path / file
    if direct.is_file():
        return direct.resolve()
    base = Path(file).name
    root = repo_path.resolve()
    matches = sorted(
        p for p in repo_path.rglob(base)
        if p.is_file() and ".git" not in p.resolve().relative_to(root).parts
    ) if root.is_dir() else []
    return matches[0].resolve() if matches else None


def region_has_bare_double(repo_path, file: str, line_start: int, line_end: int,
                           *, cache: dict | None = None) -> bool:
    """True if a bare ``double`` token appears on lines ``[line_start, line_end]``.

    Returns ``True`` (do not gate the plain-edit float rung) when the repo/file
    cannot be resolved or read — the Patcher's ``patch_inapplicable`` catches any
    genuinely-inapplicable rung this misses.  ``cache`` (path -> lines) is an
    optional per-run memo so a file is read once across its many regions.
    """
    if not repo_path:
        return True
    try:
        path = _resolve_in_tree(Path(repo_path), file)
        if path is None:
            return True
        if cache is not None and str(path) in cache:
            lines = cache[str(path)]
        else:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            if cache is not None:
                cache[str(path)] = lines
    except OSError:
        return True
    region = "\n".join(lines[max(0, line_start - 1): line_end])
    return bool(_BARE_DOUBLE_RE.search(region))
