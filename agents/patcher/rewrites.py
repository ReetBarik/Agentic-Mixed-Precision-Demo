"""LLM-rewrite dispatch path (P3 path 4) — ``reformulate-kahan`` / ``-identity``.

Strategy reaches this path at the DD ceiling (or for a local-cancellation region)
and has already chosen *which* rewrite to try: ``kahan`` for a cancellation
cascade, or a specific algebraic ``identity`` (``log1p``, ``expm1``, …; design
§P3b) for a local cancellation.  The Patcher's job is mechanical: build a narrow
prompt (region source + variable list + the exact identity Strategy picked), ask
the LLM for a drop-in replacement of just those lines, and splice it back in.

The LLM call itself is injected (``llm_call(system, user, attempt) -> str``) so
tests drive canned rewrites and the live path uses
:func:`agents.integrator_base.llm.stream_llm`.  ``attempt`` lets the caller vary
the seed across the bounded retry (P4).
"""

from __future__ import annotations

from pathlib import Path

from agents.strategy.models import RemediationIntent

_SYSTEM = (
    "You are a numerical-analysis code surgeon. You rewrite a small span of C++ "
    "to be numerically stable WITHOUT changing its interface or the surrounding "
    "code. Output ONLY the replacement C++ for the given line span — no prose, no "
    "code fences, no extra lines. Preserve indentation and any trailing braces."
)

_IDENTITY_HINT = {
    "log1p": "Use std::log1p(x) in place of log(1+x).",
    "expm1": "Use std::expm1(x) in place of exp(x)-1.",
    "hypot": "Use std::hypot(x, y) in place of sqrt(x*x + y*y).",
    "1-cos->2sin2": "Replace 1 - cos(x) with 2*sin(x/2)*sin(x/2).",
}


def build_prompt(intent: RemediationIntent, region_source: str) -> tuple[str, str]:
    """Return ``(system, user)`` for the rewrite the intent asks for."""
    if intent.kind == "reformulate-kahan":
        strategy_line = (
            "Apply Kahan / compensated summation to the accumulation(s) in this "
            "span so catastrophic cancellation is mitigated. Keep the same result "
            "variable(s).")
    else:  # reformulate-identity
        hint = _IDENTITY_HINT.get(intent.identity or "",
                                  f"Apply the algebraic identity '{intent.identity}'.")
        strategy_line = f"Apply this algebraic identity: {hint}"

    user = (
        f"{strategy_line}\n"
        f"Variables in play: {', '.join(intent.target.variables) or '(none named)'}\n"
        f"Replace exactly these lines "
        f"({intent.target.line_start}-{intent.target.line_end}) and output only "
        f"the replacement:\n\n```cpp\n{region_source}\n```\n"
    )
    return _SYSTEM, user


def apply_rewrite(path: Path, line_start: int, line_end: int,
                  new_source: str) -> None:
    """Splice ``new_source`` in place of the inclusive line span in ``path``.

    The replacement's own newline structure is used verbatim; a single trailing
    newline is normalized so line counts stay sane for later edits.
    """
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if line_end > len(lines):
        raise ValueError(f"line_end {line_end} exceeds {path.name} length {len(lines)}")
    replacement = new_source.rstrip("\n") + "\n"
    new_lines = lines[: line_start - 1] + [replacement] + lines[line_end:]
    path.write_text("".join(new_lines), encoding="utf-8")


def region_source(path: Path, line_start: int, line_end: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[line_start - 1: line_end])
