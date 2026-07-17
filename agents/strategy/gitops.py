"""Git operations Strategy owns directly (Q3 / P2).

Strategy creates the ``strategy/<run_id>`` branch from the caller-supplied
starting SHA, resets the branch tip back to ``parent_sha`` when a Patcher-``ok``
candidate is later rejected by the Validator (option (a): rejected commits become
dangling, reflog-reachable), and writes the cumulative ``final.diff`` at the end.

Patcher owns the per-patch commits themselves (P2) — Strategy never commits code;
it only branches, resets, and diffs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(RuntimeError):
    pass


def _git(repo: str | Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed in {repo}:\n{r.stderr.strip()}")
    return r.stdout.strip()


class GitRepo:
    """Thin wrapper over a working-tree git repo Strategy branches on."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def rev_parse(self, ref: str) -> str:
        return _git(self.path, "rev-parse", ref)

    def create_branch(self, branch: str, start_sha: str) -> None:
        """Create + checkout ``branch`` at ``start_sha`` (fresh run branch)."""
        _git(self.path, "checkout", "-B", branch, start_sha)

    def head(self) -> str:
        return _git(self.path, "rev-parse", "HEAD")

    def reset_hard(self, sha: str) -> None:
        """Reset the current branch tip to ``sha`` (revert a rejected candidate)."""
        _git(self.path, "reset", "--hard", sha)

    def diff(self, from_sha: str, to_ref: str = "HEAD") -> str:
        """Unified diff ``from_sha..to_ref`` (the cumulative run diff)."""
        return _git(self.path, "diff", f"{from_sha}..{to_ref}")

    def write_cumulative_diff(self, start_sha: str, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(self.diff(start_sha, "HEAD") + "\n")
        return out_path
