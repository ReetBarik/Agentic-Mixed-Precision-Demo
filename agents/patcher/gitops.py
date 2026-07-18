"""Git operations the Patcher owns (P2): apply, commit, reset, revert-lookup.

The Patcher works on the live ``strategy/<run_id>`` checkout at ``ctx.repo_path``
(Strategy created + checked out the branch).  Per P2 the Patcher — not Strategy —
makes the per-patch commit (parent = current HEAD); Strategy only branches,
resets rejected tips, and diffs.  On any non-``ok`` outcome the Patcher resets the
tree back to ``parent_sha`` so the next intent starts from a clean tree (Strategy
only resets on a Validator *reject* of a Patcher-``ok`` candidate).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


class GitError(RuntimeError):
    pass


def git(repo: str | Path, *args: str, check: bool = True,
        input_text: str | None = None) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=True, text=True, input=input_text)
    if check and r.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed in {repo}:\n{r.stderr.strip()}")
    return r


def head(repo: str | Path) -> str:
    return git(repo, "rev-parse", "HEAD").stdout.strip()


def apply_patch(repo: str | Path, patch_text: str) -> None:
    """Apply a unified diff (``a/… b/…``, ``-p1``) to the working tree.

    Raises :class:`GitError` if the patch does not apply (the caller maps this to
    ``patch_apply_failed``).
    """
    r = git(repo, "apply", "-p1", "--whitespace=nowarn", "-",
            check=False, input_text=patch_text)
    if r.returncode != 0:
        raise GitError(f"git apply failed:\n{r.stderr.strip()}")


def reset_hard(repo: str | Path, sha: str) -> None:
    """Discard all working-tree + index changes back to ``sha`` and clean."""
    git(repo, "reset", "--hard", sha)
    git(repo, "clean", "-fdq", check=False)


def commit_all(repo: str | Path, message: str) -> str:
    """Stage everything and commit; return the new commit SHA.

    Raises :class:`GitError` on commit failure (caller → ``commit_failed``).
    Commits are made with a fixed identity + ``--no-gpg-sign`` so the run never
    stalls on a missing user config or a signing prompt.
    """
    git(repo, "add", "-A")
    r = git(repo, "-c", "user.name=patcher", "-c", "user.email=patcher@local",
            "commit", "--no-gpg-sign", "-q", "-m", message, check=False)
    if r.returncode != 0:
        raise GitError(f"git commit failed:\n{r.stderr.strip()}\n{r.stdout.strip()}")
    return head(repo)


def revert_no_commit(repo: str | Path, sha: str) -> None:
    """Apply the inverse of commit ``sha`` to the working tree without committing.

    ``git revert --no-commit`` stages the inverse diff; the Patcher then builds +
    smoke-tests before making its own commit (so the revert follows the same
    build-then-commit gate as every other path).  Raises on conflict.
    """
    r = git(repo, "revert", "--no-commit", "--no-edit", sha, check=False)
    if r.returncode != 0:
        git(repo, "revert", "--abort", check=False)
        raise GitError(f"git revert --no-commit {sha[:8]} failed:\n{r.stderr.strip()}")


_SUBJECT_RE = re.compile(r"^\[[^\]]*\]\s+(\S+)\s+(\S+)")   # "[iter_N] <kind> <loc>"


def find_introducing_commit(repo: str | Path, branch_range: str,
                            file: str, line_start: int,
                            target_suffix: str) -> str | None:
    """Most-recent commit on ``branch_range`` that installed ``target_suffix``.

    Scans commit subjects (schema ``[iter_N] <kind> <file>:<lines>``, design Q3)
    newest-first for one whose ``kind`` ends with ``target_suffix`` (e.g.
    ``-to-ff`` / ``-to-dd``) and whose location matches ``file:line_start``.  This
    is the git-history equivalent of "look up the introducing commit from
    Strategy's iteration log" — the commits *are* that log, keyed by the same
    schema, so no separate log ingestion is needed.  Returns the SHA or ``None``.
    """
    r = git(repo, "log", "--format=%H%x00%s", branch_range, check=False)
    if r.returncode != 0:
        return None
    base = Path(file).name
    for row in r.stdout.splitlines():
        if "\x00" not in row:
            continue
        sha, subject = row.split("\x00", 1)
        m = _SUBJECT_RE.match(subject)
        if not m:
            continue
        kind, loc = m.group(1), m.group(2)
        if not kind.endswith(target_suffix):
            continue
        loc_file, _, loc_lines = loc.rpartition(":")
        if Path(loc_file).name != base:
            continue
        first_line = loc_lines.split("-", 1)[0]
        if first_line.isdigit() and int(first_line) == line_start:
            return sha
    return None
