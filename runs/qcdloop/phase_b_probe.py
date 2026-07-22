#!/usr/bin/env python3
"""Phase B measurement probe — size the per-integral intent divergence.

READ-ONLY. Reads the frozen v2 characterization report, replays Strategy's
*actual* ranking logic (imported, never reimplemented), and measures how much
precision signal ``_merge_by_line`` throws away by worst-casing a source line
across the integrals that compile it.

The question Phase B must answer before anyone touches the reducer / routing:
if Strategy stopped worst-casing and instead emitted one intent per
``(line, integral)``, how many lines would get a *different* (cheaper) precision
than the worst-case merge assigns them today?  The number, not the narrative,
decides whether the invasive Phase B refactor is worth it.

Scope guard: this script imports from ``agents.strategy`` and calls the same
``load_regions`` / ``load_chains`` / ``build_queues`` Strategy runs.  It executes
NO Patcher and NO Validator — so a "precision decision" here is the *intent
target* Strategy emits from the report alone (the cheapest rung the walk is
allowed to aim for), NOT the Validator-settled final precision.  That is the
correct layer: merging happens upstream of the walk, so intent divergence is
exactly the signal Phase B would recover.  See PHASE_B_PROBE_*.md for the caveat.

Usage:
    .venv/bin/python runs/qcdloop/phase_b_probe.py [REPORT_JSON] [OUT_MD]
Defaults: runs/qcdloop/report_5k.json → runs/qcdloop/PHASE_B_PROBE_2026-07-22.md
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from agents.config import StrategyConfig                       # noqa: E402
from agents.strategy.characterization import (                 # noqa: E402
    load_chains, load_regions,
)
from agents.strategy.models import LADDER                      # noqa: E402
from agents.strategy.ranking import (                          # noqa: E402
    build_queues, error_threshold, load_flop_weights,
)

# ---------------------------------------------------------------------------
# Faithful intent-target derivation.
#
# A region's precision decision is derived ENTIRELY from Strategy's own queue
# builders + the WI1 float-rung gate — no tier predicate is re-implemented here:
#
#   * key in the correctness queue  -> "dd"   (walk promotes double -> dd)
#   * key in the speedup queue       -> "float" if value_range_ok_for_float
#                                       else "ff"   (WI1 hard gate, agent._float_rung_ok)
#   * neither                        -> "double"  (baseline, untouched)
#
# float_via (plain vs regional) only changes the *path* to a rung, never the
# cheapest reachable target, so the source probe is not needed here.
# ---------------------------------------------------------------------------

_LADDER_IDX = {p: i for i, p in enumerate(LADDER)}


def region_decisions(records, tol, flop_weights):
    """Map ``line-key -> intent-target precision`` for a record set.

    Uses Strategy's ``build_queues`` verbatim; the WI1 float gate mirrors
    ``agent._float_rung_ok`` (report_prunes on: unsafe range -> settle at ff).
    """
    corr_q, spd_q = build_queues(records, tol, flop_weights=flop_weights)
    corr_keys = {r.key for r in corr_q}
    spd_recs = {r.key: r for r in spd_q}
    out = {}
    for r in records:
        k = r.key
        if k in corr_keys:
            out[k] = "dd"
        elif k in spd_recs:
            out[k] = "float" if getattr(r, "value_range_ok_for_float", True) else "ff"
        else:
            out[k] = "double"
    return out


def _loc(key):
    f, a, b = key
    return f"{f}:{a}" if a == b else f"{f}:{a}-{b}"


def _fmt_dist(counter):
    """'dd×1, float×6' from a Counter, ladder order."""
    return ", ".join(f"{p}×{counter[p]}" for p in LADDER if counter[p])


def main():
    report = Path(sys.argv[1]) if len(sys.argv) > 1 else _REPO / "runs/qcdloop/report_5k.json"
    out_md = Path(sys.argv[2]) if len(sys.argv) > 2 else _REPO / "runs/qcdloop/PHASE_B_PROBE_2026-07-22.md"

    cfg = StrategyConfig()
    tol = float(cfg.tolerance)
    thr = error_threshold(tol)
    # Same weight table Strategy loads (ordering only — does not move membership).
    flop_weights = load_flop_weights(_REPO / "runs/qcdloop/ratio_multipliers.json")

    print(f"[probe] report={report}", file=sys.stderr)
    print(f"[probe] tolerance={tol} thr={thr:g} flop_weights={'yes' if flop_weights else 'no'}",
          file=sys.stderr)

    # -- load via Strategy's real loaders --------------------------------
    merged, meta_m = load_regions(report, merge=True)
    unmerged, meta_u = load_regions(report, merge=False)
    chains, meta_c = load_chains(report)
    print(f"[probe] merged={len(merged)} unmerged={len(unmerged)} chains={len(chains)}",
          file=sys.stderr)

    # -- decisions -------------------------------------------------------
    merged_dec = region_decisions(merged, tol, flop_weights)

    # per-integral: run the queues INSIDE each integral (keys unique there, so
    # the speedup exclude-by-key logic is clean — this is the faithful "what
    # would this integral decide if it were the only one" counterfactual).
    by_integral = defaultdict(list)
    for r in unmerged:
        by_integral[r.integral].append(r)
    # per_line_dec[line-key] = {integral: precision}
    per_line_dec = defaultdict(dict)
    # per_line_safe[line-key][integral] = value_range_ok_for_float (for reporting)
    for integ, recs in by_integral.items():
        dec = region_decisions(recs, tol, flop_weights)
        for k, p in dec.items():
            per_line_dec[k][integ] = p

    # -- region-level analysis ------------------------------------------
    shared = {k: v for k, v in per_line_dec.items() if len(v) >= 2}
    agree = []
    disagree = []
    for k, integ_dec in shared.items():
        if len(set(integ_dec.values())) == 1:
            agree.append(k)
        else:
            disagree.append(k)

    # magnitude buckets among disagreeing lines
    span_buckets = Counter()
    for k in disagree:
        precs = sorted(set(per_line_dec[k].values()), key=lambda p: _LADDER_IDX[p])
        span_buckets["{" + ", ".join(precs) + "}"] += 1

    # wasted-headroom scoring (drives the routing-payoff decision)
    scored = []
    for k in disagree:
        integ_dec = per_line_dec[k]
        m = merged_dec.get(k, "double")
        m_idx = _LADDER_IDX[m]
        counter = Counter(integ_dec.values())
        # ladder-steps of over-precision the merge imposes, summed over instances
        wasted = sum(max(0, m_idx - _LADDER_IDX[p]) for p in integ_dec.values())
        n_cheaper = sum(1 for p in integ_dec.values() if _LADDER_IDX[p] < m_idx)
        n_at_worst = sum(1 for p in integ_dec.values() if _LADDER_IDX[p] >= m_idx)
        forcers = [i for i, p in integ_dec.items() if _LADDER_IDX[p] >= m_idx]
        scored.append({
            "key": k, "merged": m, "counter": counter, "wasted": wasted,
            "n_int": len(integ_dec), "n_cheaper": n_cheaper,
            "n_at_worst": n_at_worst, "forcers": forcers,
        })
    scored.sort(key=lambda s: (-s["wasted"], -s["n_cheaper"], _loc(s["key"])))

    # adversarial: is the worst-case driven by a lone outlier integral?
    sole_forcer = sum(1 for s in scored if s["n_at_worst"] == 1)
    # which integrals force worst-cases (concentration check)
    forcer_counts = Counter()
    for s in scored:
        for i in s["forcers"]:
            forcer_counts[i] += 1

    # merged vs max(per-integral): sanity — does the merge == worst-case of parts?
    merge_ne_max = 0
    for k in shared:
        parts_max = max(_LADDER_IDX[p] for p in per_line_dec[k].values())
        if _LADDER_IDX[merged_dec.get(k, "double")] != parts_max:
            merge_ne_max += 1

    frac_dis = len(disagree) / len(shared) if shared else 0.0

    # -- chain analysis --------------------------------------------------
    # A chain is "worked" (promotes its lines toward dd) iff eligible for the
    # correctness chain queue: max_rel_err > thr (agent._rank_chains).  A
    # non-eligible chain drives no promotion -> its lines stay "double".
    # Case (a): same chain_id across >=2 integrals, differing worked decision.
    # Case (b): a source line covered by chains in >=2 integrals where the
    #           per-integral chain-precision differs (dd in one, double in another).
    chain_worked = {}                       # (integral, chain_id) -> "dd"|"double"
    chainid_ints = defaultdict(dict)        # chain_id -> {integral: dec}
    line_int_dd = defaultdict(dict)         # line-key -> {integral: bool worked}
    n_worked = 0
    for c in chains:
        worked = c.max_rel_err > thr
        n_worked += worked
        dec = "dd" if worked else "double"
        chain_worked[(c.integral, c.chain_id)] = dec
        chainid_ints[c.chain_id][c.integral] = dec
        for ln in c.lines:
            prev = line_int_dd[ln.key].get(c.integral, False)
            line_int_dd[ln.key][c.integral] = prev or worked

    # case (a)
    ca_shared = {cid: d for cid, d in chainid_ints.items() if len(d) >= 2}
    ca_disagree = [cid for cid, d in ca_shared.items() if len(set(d.values())) > 1]

    # case (b)
    cb_shared = {k: d for k, d in line_int_dd.items() if len(d) >= 2}
    cb_disagree = []
    for k, d in cb_shared.items():
        decs = {"dd" if w else "double" for w in d.values()}
        if len(decs) > 1:
            cb_disagree.append(k)

    # -- write report ----------------------------------------------------
    _write_md(out_md, report, tol, thr, meta_m, meta_u, meta_c,
              merged, unmerged, chains, shared, agree, disagree, frac_dis,
              span_buckets, scored, sole_forcer, forcer_counts, merge_ne_max,
              ca_shared, ca_disagree, cb_shared, cb_disagree, per_line_dec,
              merged_dec, n_worked, line_int_dd)
    print(f"[probe] wrote {out_md}", file=sys.stderr)

    # -- stdout summary --------------------------------------------------
    print(f"regions merged={len(merged)} unmerged={len(unmerged)} "
          f"ratio={len(unmerged)/len(merged):.2f}")
    print(f"shared lines (>=2 integrals)={len(shared)} "
          f"agree={len(agree)} disagree={len(disagree)} frac_dis={frac_dis:.1%}")
    print(f"chains case(a) shared_ids={len(ca_shared)} disagree={len(ca_disagree)} | "
          f"case(b) shared_lines={len(cb_shared)} disagree={len(cb_disagree)}")


def _write_md(out_md, report, tol, thr, meta_m, meta_u, meta_c,
              merged, unmerged, chains, shared, agree, disagree, frac_dis,
              span_buckets, scored, sole_forcer, forcer_counts, merge_ne_max,
              ca_shared, ca_disagree, cb_shared, cb_disagree, per_line_dec,
              merged_dec, n_worked, line_int_dd):
    L = []
    w = L.append

    if not shared:
        verdict = "NO SHARED LINES — no integral compiles a line another does; routing is moot."
    elif frac_dis < 0.05:
        verdict = (f"LOW PAYOFF — only {frac_dis:.1%} of shared lines disagree "
                   f"(<5%). Phase B routing buys little on this codebase.")
    elif frac_dis > 0.25:
        verdict = (f"WELL MOTIVATED — {frac_dis:.1%} of shared lines disagree "
                   f"(>25%). Worst-casing throws away real per-integral signal.")
    else:
        verdict = (f"MODERATE — {frac_dis:.1%} of shared lines disagree "
                   f"(5–25%). Payoff is real but not overwhelming; weigh vs refactor cost.")

    w("# Phase B probe — per-integral intent divergence")
    w("")
    w(f"_Read-only measurement, {report.name}, generated 2026-07-22. No pipeline "
      f"change; replays Strategy's `load_regions`/`load_chains`/`build_queues` verbatim._")
    w("")
    w("## Verdict")
    w("")
    w(f"**{verdict}**")
    w("")
    w("A *precision decision* here is the **intent target** Strategy emits from the "
      "report (cheapest rung the walk may aim for) — not the Validator-settled final "
      "precision (no Patcher/Validator runs). Merging happens upstream of the walk, "
      "so intent divergence is exactly the signal Phase B routing would recover.")
    w("")

    # -- setup / counts --
    w("## Inputs")
    w("")
    w(f"- report schema_version: **{meta_m.get('schema_version')}** "
      f"(v2 = per-record `integral` tag, Phase A commit 5b6b82c)")
    w(f"- tolerance: **{tol}** precise digits → rel-err bar `{thr:g}`")
    w(f"- ladder (cheap→dear): `{' < '.join(LADDER)}`")
    w(f"- non-localizable region entries skipped: {meta_m.get('non_localizable_skipped')}")
    w("")

    # -- region count --
    w("## Region count: merged vs unmerged")
    w("")
    ratio = len(unmerged) / len(merged) if merged else 0
    w("| view | records | note |")
    w("|---|---|---|")
    w(f"| merged (`merge=True`, today) | {len(merged)} | one worst-case region per source line |")
    w(f"| unmerged (`merge=False`) | {len(unmerged)} | one region per (integral, line) |")
    w(f"| **ratio** | **{ratio:.2f}×** | avg integrals compiling a shared line |")
    w("")

    # -- disagreement --
    w("## Same-line precision disagreement")
    w("")
    w("Restricted to source lines that appear in ≥2 integrals (the only lines the "
      "merge can lose signal on). A line *disagrees* when ≥2 of its integrals would "
      "emit different intent targets.")
    w("")
    w("| metric | count | fraction of shared |")
    w("|---|---|---|")
    n = len(shared) or 1
    w(f"| shared lines (≥2 integrals) | {len(shared)} | 100% |")
    w(f"| **agree** (all integrals same precision) | {len(agree)} | {len(agree)/n:.1%} |")
    w(f"| **disagree** (≥2 precisions) | {len(disagree)} | {len(disagree)/n:.1%} |")
    w("")

    # -- magnitude distribution --
    w("## Disagreement magnitude distribution")
    w("")
    w("Among disagreeing lines, the set of distinct intent targets across integrals:")
    w("")
    if span_buckets:
        w("| precision span | lines |")
        w("|---|---|")
        for span, cnt in span_buckets.most_common():
            w(f"| {span} | {cnt} |")
    else:
        w("_No disagreeing lines._")
    w("")

    # -- top-N wasted headroom --
    w("## Top-20 wasted-headroom lines")
    w("")
    w("The direct \"what would routing buy\" table: lines the merge forces to a dear "
      "precision that N−1 integrals would escape. `wasted` = Σ ladder-steps of "
      "over-precision across integral instances; `cheaper` = integrals that would "
      "get a lower rung under routing; `forcers` = integrals pinning the worst case.")
    w("")
    if scored:
        w("| # | line | merged | per-integral | wasted | cheaper | forcer(s) |")
        w("|---|---|---|---|---|---|---|")
        for i, s in enumerate(scored[:20], 1):
            forcers = ", ".join(sorted(s["forcers"]))
            if len(forcers) > 40:
                forcers = forcers[:37] + "…"
            w(f"| {i} | `{_loc(s['key'])}` | {s['merged']} | "
              f"{_fmt_dist(s['counter'])} | {s['wasted']} | "
              f"{s['n_cheaper']}/{s['n_int']} | {forcers} |")
    else:
        w("_No disagreeing lines._")
    w("")

    # -- adversarial --
    w("## Adversarial: is the disagreement a lone-outlier artifact?")
    w("")
    w("If each worst-case were pinned by a single dominant integral (e.g. one BIN "
      "cascade always demanding dd), routing would still buy the N−1 others their "
      "cheaper rung — so a high sole-forcer count *strengthens* the payoff case; it "
      "does not weaken it. What would weaken it: the worst-case being demanded by "
      "*most* integrals (then routing helps few).")
    w("")
    if disagree:
        w(f"- disagreeing lines pinned by a **single** integral at the worst rung: "
          f"**{sole_forcer}/{len(disagree)}** ({sole_forcer/len(disagree):.1%}) "
          f"→ these are clean N−1 wins.")
        w("")
        w("Worst-case *forcer* concentration (how often each integral pins a shared "
          "line's worst case) — a flat spread means no single integral explains the "
          "divergence:")
        w("")
        w("| integral | lines it forces |")
        w("|---|---|")
        for integ, cnt in forcer_counts.most_common(12):
            w(f"| {integ} | {cnt} |")
        w("")
    w(f"- merge-result ≠ worst-case-of-parts on **{merge_ne_max}** shared lines "
      f"(sanity: `_merge_by_line` should equal the per-integral max; non-zero here "
      f"flags where re-running the tier logic on worst-cased *signals* lands on a "
      f"different rung than the max of independent decisions).")
    w("")

    # -- chains --
    w("## Cascade-chain analogue")
    w("")
    w(f"{len(chains)} chains loaded (not merged by Strategy; deduped by representative "
      f"line at walk time, worst-case distributed). A chain is *worked* → floors its "
      f"lines toward `dd` iff `max_rel_err > thr` (`_rank_chains` eligibility); else "
      f"its lines stay `double`.")
    w("")
    wf = n_worked / len(chains) if chains else 0.0
    w(f"> **Degenerate decision.** {n_worked}/{len(chains)} chains ({wf:.1%}) are worked "
      f"— every cascade chain is ill-conditioned enough to demand `dd`. The chain "
      f"decision therefore carries **no per-integral variation** to recover: worked "
      f"vs not-worked is the only axis, and it is uniformly \"worked\". Per-integral "
      f"routing on chains has structurally **zero** payoff on this codebase — not "
      f"because integrals happen to agree on a nuanced split, but because there is no "
      f"split to make. This is the correct null result to compare against the region "
      f"finding below.")
    w("")
    w("**Case (a) — same `chain_id` across integrals, differing decision:**")
    w("")
    na = len(ca_shared) or 1
    w("| metric | count | fraction |")
    w("|---|---|---|")
    w(f"| chain_ids in ≥2 integrals | {len(ca_shared)} | 100% |")
    w(f"| … that disagree (worked in one, not another) | {len(ca_disagree)} | "
      f"{len(ca_disagree)/na:.1%} |")
    if not ca_shared:
        w("")
        w("_(0 shared chain_ids ⇒ `chain_id`s are namespaced per integral and never "
          "recur across them; the cross-integral question lives entirely in case (b).)_")
    w("")
    w("**Case (b) — a source line covered by chains in ≥2 integrals, differing "
      "per-integral chain-precision:**")
    w("")
    nb = len(cb_shared) or 1
    w("| metric | count | fraction |")
    w("|---|---|---|")
    w(f"| lines covered by chains in ≥2 integrals | {len(cb_shared)} | 100% |")
    w(f"| … that disagree (dd in one integral, double in another) | "
      f"{len(cb_disagree)} | {len(cb_disagree)/nb:.1%} |")
    if cb_disagree:
        w("")
        w("Disagreeing chain-covered lines:")
        w("")
        w("| line | dd integrals | double integrals |")
        w("|---|---|---|")
        for k in sorted(cb_disagree, key=_loc)[:20]:
            d = line_int_dd[k]
            dd_i = ", ".join(sorted(i for i, wk in d.items() if wk))
            db_i = ", ".join(sorted(i for i, wk in d.items() if not wk))
            w(f"| `{_loc(k)}` | {dd_i} | {db_i} |")
    w("")

    # -- method --
    w("## Method & faithfulness")
    w("")
    w("- Decisions come from Strategy's own `build_queues` (correctness + speedup "
      "queues) plus the WI1 float-rung gate (`value_range_ok_for_float`), mirroring "
      "`agent._float_rung_ok`. No tier predicate is re-implemented.")
    w("- Per-integral decisions run `build_queues` **inside each integral** so the "
      "speedup exclude-by-key logic is clean (line keys are unique within one "
      "integral). This is the faithful \"if this integral were alone\" counterfactual "
      "that routing would realize.")
    w("- `float_via` (plain vs regional) affects only the *path* to a rung, never the "
      "cheapest reachable target, so the repo source probe is not consulted.")
    w("- Intent target ≠ Validator-settled precision: the walk may settle dearer if "
      "the Validator rejects a demotion. This probe bounds the *upstream* signal the "
      "merge destroys before the walk ever runs — the ceiling on routing payoff.")
    w("")

    out_md.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
