"""P1 intent parsing + P4 cheap pre-checks.

Strategy hands the Patcher the wire dict locked in design §P1/§P3::

    {"target": {"file","line_start","line_end","variables"},
     "kind": <one of the 11>, "intent": "correctness"|"speedup",
     "current_precision": <ladder rung>, "rationale_id": "iter_23",
     "identity": <catalog id>   # only for kind == "reformulate-identity"}

We reuse the shared data model in :mod:`agents.strategy.models` verbatim — the
region-record shape and the kind vocabulary are the *contract* module, not
Strategy-loop logic, so there is one definition of "the 11 kinds" across the
producer (Strategy) and this consumer (Patcher).

The P4 pre-checks are the cheap "can this intent even be attempted?" gate the
design specifies (file exists, line range valid, variables appear in the region);
a miss is reported by the caller as ``patch_apply_failed`` (malformed intent is
the same category as "diff doesn't apply").
"""

from __future__ import annotations

import re
from pathlib import Path

from agents.strategy.models import ALL_KINDS, RegionTarget, RemediationIntent


class IntentError(ValueError):
    """A malformed intent — surfaced by the Patcher as ``patch_apply_failed``."""


def parse_intent(wire: dict) -> RemediationIntent:
    """Build a typed :class:`RemediationIntent` from the P1 wire dict.

    Raises :class:`IntentError` on any structural problem (missing field, bad
    kind, identity mismatch) so the Patcher can map it to ``patch_apply_failed``
    rather than crashing.
    """
    if not isinstance(wire, dict):
        raise IntentError(f"intent must be a dict, got {type(wire).__name__}")
    tgt = wire.get("target")
    if not isinstance(tgt, dict):
        raise IntentError("intent.target missing or not a dict")
    try:
        file = str(tgt["file"])
        line_start = int(tgt["line_start"])
        line_end = int(tgt["line_end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise IntentError(f"intent.target malformed: {exc}") from exc
    if line_start < 1 or line_end < line_start:
        raise IntentError(f"invalid line range [{line_start}, {line_end}]")
    variables = list(tgt.get("variables", []) or [])

    kind = wire.get("kind")
    if kind not in ALL_KINDS:
        raise IntentError(f"unknown kind {kind!r} (expected one of the 11)")

    intent_flavor = wire.get("intent")
    identity = wire.get("identity")
    try:
        return RemediationIntent(
            target=RegionTarget(file=file, line_start=line_start,
                                line_end=line_end, variables=variables),
            kind=kind,
            intent=intent_flavor,
            current_precision=str(wire.get("current_precision", "")),
            rationale_id=str(wire.get("rationale_id", "")),
            identity=identity,
        )
    except ValueError as exc:      # RemediationIntent.__post_init__ guards
        raise IntentError(str(exc)) from exc


# ---------------------------------------------------------------------------
# P4 cheap pre-checks (file existence, line range, variable presence)
# ---------------------------------------------------------------------------

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def resolve_in_tree(repo_root: Path, file: str) -> Path | None:
    """Resolve ``file`` (repo-relative or bare basename) to a path in the tree.

    Characterization region keys are bare basenames (``B2m.h``) while the file
    may live in a subdir (``box/B2m.h``); test reports use repo-relative paths
    (``headers/A.h``).  Try the path as written first, then a unique basename
    match anywhere under ``repo_root``.
    """
    direct = repo_root / file
    if direct.is_file():
        return direct.resolve()
    base = Path(file).name
    matches = sorted(p for p in repo_root.rglob(base)
                     if p.is_file() and _under_tree(p, repo_root))
    return matches[0].resolve() if matches else None


def _under_tree(p: Path, repo_root: Path) -> bool:
    # Skip the .git dir so a basename match never resolves into git internals.
    try:
        rel = p.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    return ".git" not in rel.parts


def precheck(intent: RemediationIntent, repo_root: Path) -> str | None:
    """Return an error string if the intent fails a pre-check, else ``None``.

    Checks, in order (design §P4 "cheap pre-check"):

    * the target file exists in the working tree;
    * the file has at least ``line_end`` lines;
    * every name in ``variables`` appears as an identifier somewhere in
      ``[line_start, line_end]``.
    """
    path = resolve_in_tree(repo_root, intent.target.file)
    if path is None:
        return f"target file not found in working tree: {intent.target.file}"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"cannot read target file {intent.target.file}: {exc}"
    if len(lines) < intent.target.line_end:
        return (f"line range [{intent.target.line_start}, {intent.target.line_end}] "
                f"exceeds file length {len(lines)} in {intent.target.file}")

    region_text = "\n".join(lines[intent.target.line_start - 1: intent.target.line_end])
    present = set(_IDENT_RE.findall(region_text))
    missing = [v for v in intent.target.variables if v not in present]
    if missing:
        return (f"variables {missing} not found in region "
                f"{intent.target.location} of {intent.target.file}")
    return None
