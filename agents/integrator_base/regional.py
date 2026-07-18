"""Shared engine for the *regional* integrators (ff / dd).

``ff_integrator.integrate_region`` and ``dd_integrator.integrate_region`` are
structural twins: given a region descriptor ``(file, line_start, line_end,
variables)`` and a working-tree SHA, both (1) read the region source at that SHA,
(2) recover its write set, (3) ask an LLM for an extended-precision **shim** (the
types / operators / named constants the region needs), and (4) synthesize a
deterministic **boundary patch** that promotes the region's reads to the extended
scalar on entry and demotes its writes back on exit.  They differ only in their
ruleset (system prompt), their concrete C++ scalar/complex spellings
(``quad::ffun::ffloat`` vs ``quad::ddfun::ddouble``), and a per-type note in the
user turn (DD needs hex-encoded constant tables).  That common flow lives here;
the two ``agent.py`` modules are thin wrappers that supply a :class:`RegionalSpec`.

The design (§P4) keeps the boundary patch deterministic and out of the LLM's hands
— the LLM produces only the shim; :mod:`agents.integrator_base.boundary` turns
``(reads, writes, scalar_type, caller_type)`` into the diff.  A re-roll (Patcher's
N=3 retry, ``attempt`` varying the user turn) regenerates only the shim.

Modeled on :func:`agents.tracked_integrator.agent.integrate` (SOURCE_HASH cache,
streaming generation, rule-justified output); the regional differences are the
region-scoped inputs and the boundary patch as a first-class output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from agents.integrator_base import boundary, cache, llm
from agents.integrator_base.region import RegionIntegrationResult
from agents.shared.region_scan import RegionScanError, extract_region_writes

# Max output tokens for a regional shim.  A regional shim is far smaller than the
# whole-app tracked shim (a single region's types/constants), but keep generous
# headroom so a shim is never truncated mid-file.
_MAX_OUTPUT_TOKENS = 8000

# A regional shim's include set is CLOSED (system-prompt rule C1): the vendored
# extended-precision headers, and nothing app-source.  The dominant Patcher
# failure in the 2026-07-18 shakedown (run 20260718_194556_67dbcf37) was DD shims
# hallucinating app-source includes — ``#include "ql/constants.h"``,
# ``<qcdloop/types.h>``, ``<Kokkos_Macros.hpp>`` — that are not on the shim's
# include path, so every such build died with ``fatal error: <path>: No such file
# or directory`` before the shim was ever honestly tested (P6a "dd_untested").
# The prompt is the primary fix; this lint is the deterministic safety net.
#
# Standard-library headers are harmless (always on the include path) and a minimal
# shim should not need them, but allowing them avoids rejecting an otherwise
# buildable shim — a false reject would burn a retry.  App-source headers are the
# actual failure mode and are always rejected.
_STDLIB_HEADERS = frozenset({
    "cstdint", "cstddef", "cstdlib", "cmath", "cfloat", "climits", "cassert",
    "complex", "limits", "type_traits", "utility", "array", "tuple", "algorithm",
})

# ``#include <foo>`` / ``#include "foo"`` as a real preprocessor directive (first
# non-whitespace token is ``#include``, so a ``//``-commented line never matches).
_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]')


@dataclass
class RegionalSpec:
    """The per-integrator surface the shared engine needs (ff vs dd)."""

    system_prompt: str            # the LLM ruleset (bytes feed SOURCE_HASH)
    cpp_scalar: str               # e.g. "quad::ffun::ffloat"
    cpp_complex: str              # e.g. "quad::ffun::ffcomplex"
    vendored_headers: list[str]   # e.g. ["ff_math.hpp", "ff_complex.hpp"]
    shim_prefix: str              # "ff" | "dd" — filename tag
    constant_note: str = ""       # extra user-turn guidance (DD hex constants)
    # Include-set allowlist for the C1 lint.  ``None`` -> derived from
    # ``vendored_headers`` ∪ the standard-library set; set explicitly to override.
    allowed_includes: list[str] | None = None


def run_integrate_region(
    spec: RegionalSpec,
    *,
    file: str,
    line_start: int,
    line_end: int,
    variables: list[str],
    working_tree: str,
    scalar_type: str,
    caller_type: str = "double",
    direction: str = "in",
    out_dir: Path,
    attempt: int = 0,
    repo_path: str | None = None,
    cfg=None,
    llm_fn=None,
) -> RegionIntegrationResult:
    """Generate a regional ff/dd shim + boundary patch for one region.

    ``scalar_type`` is the short tag the Patcher passes (``"ffloat"`` /
    ``"ddouble"``); the concrete C++ spelling used in the shim and boundary patch
    comes from ``spec.cpp_scalar``.  ``llm_fn(system, user, attempt) -> str`` is a
    test seam — when ``None`` the real streaming call is used (built from ``cfg``,
    defaulting to :class:`~agents.config.PipelineConfig`).  Never raises past the
    seam: any failure is returned as an ``llm_failed`` result so the Patcher's
    bounded retry (P4) can re-roll.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve the region source at the pinned SHA.
    try:
        src = _git_show(repo_path, working_tree, file)
    except Exception as exc:  # noqa: BLE001
        return RegionIntegrationResult.failed(f"could not read {file}@{working_tree}: {exc}")

    lines = src.split("\n")
    if line_start < 1 or line_end > len(lines) or line_start > line_end:
        return RegionIntegrationResult.failed(
            f"region {file}:{line_start}-{line_end} out of range (file has {len(lines)} lines)")
    region_src = "\n".join(lines[line_start - 1:line_end])

    # 2. Region writes (Fix C).  Keyed on the caller type; for a vanilla region the
    #    template-keyed scan finds nothing and the boundary module recovers the
    #    write set from its own decl scan.  A scan error is non-fatal.
    try:
        writes = extract_region_writes(file, line_start, line_end, working_tree,
                                       tracked_type=caller_type)
    except RegionScanError:
        writes = []

    # 3. SOURCE_HASH cache key (region ⊕ ruleset ⊕ scalar ⊕ writes).
    cache_key = cache.compute_region_hash(region_src, spec.system_prompt,
                                          spec.cpp_scalar, writes)
    shim_name = f"{Path(file).stem}_{spec.shim_prefix}_{cache_key[:8]}.h"
    shim_out = out_dir / shim_name

    # 4. Cache hit (only on the first attempt — a retry must re-roll the shim).
    if attempt == 0 and _is_cache_hit(shim_out, cache_key):
        shim_text = shim_out.read_text(encoding="utf-8")
        _install_in_tree(repo_path, shim_name, shim_text)
        patch = _boundary(spec, file, src, line_start, line_end, variables, writes,
                          caller_type, shim_name, repo_path)
        return RegionIntegrationResult(status="ok", shim_paths=[str(shim_out)],
                                       boundary_patch=patch, llm_tokens=0)

    # 5. Generate the shim (LLM).
    user_msg = _build_user_message(spec, file, region_src, variables, writes,
                                   line_start, line_end, caller_type)
    if attempt > 0:
        user_msg += f"\n// regeneration attempt {attempt}\n"

    try:
        if llm_fn is not None:
            shim_body = llm_fn(spec.system_prompt, user_msg, attempt)
            tokens = 0
        else:
            from agents.config import PipelineConfig
            shim_body, tokens = llm.stream_shim(
                spec.system_prompt, user_msg, cfg or PipelineConfig(), _MAX_OUTPUT_TOKENS)
    except Exception as exc:  # noqa: BLE001
        return RegionIntegrationResult.failed(f"LLM generation failed: {exc}")

    if not shim_body or not shim_body.strip():
        return RegionIntegrationResult.failed("LLM returned empty shim")

    # 5a. Deterministic include-set lint (C1 safety net).  A shim that pulls an
    #     app-source header cannot build — reject it as a misgen so the Patcher's
    #     N=3 retry re-rolls, exactly as for any other bad generation.  This does
    #     NOT count against the Strategy transition budget (a failed gen never
    #     produces an accepted transition).
    bad_include = _lint_include_set(shim_body, _allowed_include_set(spec))
    if bad_include is not None:
        return RegionIntegrationResult.failed(bad_include, llm_tokens=tokens)

    # 6. Stamp the SOURCE_HASH and persist (artifact copy + tree copy for the build).
    shim_text = cache.apply_source_hash(shim_body, cache_key)
    shim_out.write_text(shim_text, encoding="utf-8")
    _install_in_tree(repo_path, shim_name, shim_text)

    # 7. Deterministic boundary patch.
    patch = _boundary(spec, file, src, line_start, line_end, variables, writes,
                      caller_type, shim_name, repo_path)

    return RegionIntegrationResult(status="ok", shim_paths=[str(shim_out)],
                                   boundary_patch=patch, llm_tokens=tokens)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _allowed_include_set(spec: RegionalSpec) -> frozenset[str]:
    """Header names a regional shim may `#include` (C1): vendored ∪ stdlib.

    ``spec.allowed_includes`` overrides the vendored set when given; the
    standard-library headers are always permitted (harmless, always on path).
    """
    base = spec.allowed_includes if spec.allowed_includes is not None else spec.vendored_headers
    return frozenset(base) | _STDLIB_HEADERS


def _lint_include_set(shim_body: str, allowed: frozenset[str]) -> str | None:
    """Return an error message if the shim `#include`s anything outside ``allowed``.

    The C1 rule keeps the shim's include set closed to the vendored extended-
    precision headers (plus harmless stdlib headers).  Any other `#include` — an
    app-source path like ``ql/constants.h`` / ``qcdloop/types.h`` / ``Kokkos_*.hpp``
    — is not on the shim's include path and guarantees a hard build failure, so we
    treat it as a misgeneration.  Returns ``None`` when the include set is clean.
    """
    forbidden: list[str] = []
    for line in shim_body.splitlines():
        m = _INCLUDE_RE.match(line)
        if not m:
            continue
        header = m.group(1).strip()
        if header not in allowed:
            forbidden.append(header)
    if not forbidden:
        return None
    return (
        "C1 include lint: shim includes forbidden app-source header(s) "
        f"{forbidden} — a regional shim's include set is closed to the vendored "
        f"headers {sorted(allowed - _STDLIB_HEADERS)} (+ stdlib). App-source "
        "headers are not on the shim include path and break the build; the boundary "
        "patch owns caller-side wiring. Treating as a retryable misgeneration."
    )


def _boundary(spec, file, src, line_start, line_end, variables, writes,
              caller_type, shim_name, repo_path) -> str | None:
    return boundary.synthesize_boundary_patch(
        rel_file=_rel_file(file, repo_path),
        file_text=src,
        line_start=line_start, line_end=line_end,
        reads=list(variables), writes=list(writes),
        scalar_type=spec.cpp_scalar, caller_type=caller_type,
        shim_include=shim_name,
    )


def _is_cache_hit(shim_out: Path, cache_key: str) -> bool:
    if not shim_out.exists():
        return False
    return cache.extract_source_hash(shim_out.read_text(encoding="utf-8")) == cache_key


def _install_in_tree(repo_path: str | None, shim_name: str, shim_text: str) -> None:
    """Write the shim into the candidate tree so the vanilla build can find it.

    In Patcher usage ``repo_path`` is the candidate repo root, which equals
    ``QL_HEADERS`` (on the compiler's include path), so a shim at the tree root is
    resolved by a basename ``#include``.  The Patcher cleans the tree per attempt
    and ``git add -A`` picks the file up at commit time.
    """
    if not repo_path:
        return
    (Path(repo_path) / shim_name).write_text(shim_text, encoding="utf-8")


def _rel_file(file: str, repo_path: str | None) -> str:
    p = Path(file)
    if not p.is_absolute():
        return p.as_posix()
    if repo_path:
        try:
            return p.resolve().relative_to(Path(repo_path).resolve()).as_posix()
        except ValueError:
            pass
    return p.name


def _git_show(repo_path: str | None, sha: str, file: str) -> str:
    """Return ``file`` content at ``sha`` via ``git show`` (mirrors region_scan)."""
    import subprocess

    p = Path(file)
    anchor = Path(repo_path) if repo_path else (p.parent if p.is_absolute() else Path.cwd())
    root = subprocess.run(["git", "-C", str(anchor), "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if p.is_absolute():
        rel = p.resolve().relative_to(Path(root).resolve()).as_posix()
    else:
        rel = p.as_posix()
    r = subprocess.run(["git", "-C", root, "show", f"{sha}:{rel}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    return r.stdout


def _build_user_message(spec: RegionalSpec, file: str, region_src: str,
                        variables: list[str], writes: list[str],
                        line_start: int, line_end: int, caller_type: str) -> str:
    """Assemble the user turn: region source + read/write sets + scalar + API."""
    parts: list[str] = []
    parts.append(
        f"Promote the code region below to the extended scalar type "
        f"`{spec.cpp_scalar}` (and `{spec.cpp_complex}` for complex values). Emit a "
        f"C++ shim header that supplies the named-constant wrappers and any missing "
        f"operators/overloads this region needs so it compiles once its reads are "
        f"promoted and its locals are the extended type. The caller precision is "
        f"`{caller_type}`.\n"
    )
    parts.append(f"## Region — `{file}:{line_start}-{line_end}`\n```cpp\n{region_src}\n```\n")

    reads = ", ".join(variables) if variables else "(none)"
    written = ", ".join(writes) if writes else "(none reported; boundary handles region-local decls)"
    parts.append(
        f"## Boundary variables\n"
        f"- Reads (promoted to `{spec.cpp_scalar}` on entry — handled by the "
        f"deterministic boundary patch, NOT by your shim): {reads}\n"
        f"- Writes (demoted back to `{caller_type}` on exit — also boundary-handled): "
        f"{written}\n"
        f"Do NOT emit the promote/demote casts yourself; the boundary patch inserts "
        f"them. Your shim provides only the types/operators/constants the promoted "
        f"region references.\n"
    )

    parts.append(
        f"## Extended-precision API (vendored; call these, do NOT redefine the type)\n"
        f"The scalar `{spec.cpp_scalar}` and complex `{spec.cpp_complex}` are already "
        f"defined in: {', '.join(spec.vendored_headers)}. Your shim must `#include` "
        f"them (angle-bracket, they are on the include path).\n"
    )
    if spec.constant_note:
        parts.append(spec.constant_note + "\n")

    parts.append(
        f"## Output\n"
        f"Emit ONLY the complete shim header contents — no prose, no markdown fences. "
        f"Start with `#pragma once`, then a `// SOURCE_HASH: PENDING` line (the caller "
        f"stamps the real hash). `#include` the vendored headers above. Every "
        f"generated overload, specialization, or constant wrapper must carry a "
        f"comment naming the rule (Rule N / C-N / R-N) that justified it. If you "
        f"cannot classify something, emit the Rule R4 `#error` escape hatch rather "
        f"than guessing."
    )
    return "\n".join(parts)
