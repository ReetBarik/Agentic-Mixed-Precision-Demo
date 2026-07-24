# -*- coding: utf-8 -*-
"""Phase 2c e2e analysis: correctness-queue emptiness (B1) + rung discrimination (B12).

Reads the per-integral manifest + scorer manifest and reports, per integral:
  * decision counts split correctness / speedup,
  * fan-out failure_modes histogram (promotion_no_op should now appear where an
    empty payload would previously have been a silent inert `measured` cell),
  * scorer cell tally: measured vs inert (delta==baseline) vs DISCRIMINATING
    (delta != baseline -- the reframe finally moving on a real problem).

Robust to partial runs: falls back to scored_<I>.jsonl when the scorer manifest
is absent, and prints what artifacts exist when the manifest is missing.
"""

import json
from collections import Counter
from pathlib import Path


def _load_cells(base, integral):
    for name in (f"manifest_scorer_{integral}.jsonl", f"scored_{integral}.jsonl"):
        p = base / name
        if p.is_file():
            cells = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            return name, cells
    return None, None


def analyze(outdir, integral):
    base = Path(outdir) / integral
    print(f"\n===== {integral} =====")
    if not base.is_dir():
        print(f"  no output dir at {base}")
        return
    mj = base / f"manifest_{integral}.json"
    if mj.is_file():
        m = json.loads(mj.read_text())
        print("status:", m.get("status"))
        print("counts:", m.get("counts"))
        print("failure_modes:", m.get("failure_modes"))
        print("precision_distribution:", m.get("precision_distribution"))
        decs = m.get("decisions", [])
        corr = [d for d in decs if d.get("phase") == "correctness"]
        spd = [d for d in decs if d.get("phase") == "speedup"]
        print(f"decisions: total={len(decs)} correctness={len(corr)} speedup={len(spd)}")
    else:
        print(f"  no manifest ({mj.name}); artifacts present:",
              sorted(p.name for p in base.iterdir() if p.is_file()))

    name, cells = _load_cells(base, integral)
    if cells is None:
        print("  no scorer cells")
        return
    meas = [c for c in cells if c.get("status") == "measured"]
    inert = [c for c in meas
             if c.get("delta_effective") == c.get("baseline_delta_effective")]
    disc = [c for c in meas
            if c.get("delta_effective") != c.get("baseline_delta_effective")]
    print(f"scorer cells ({name}): {len(cells)}  "
          f"status={dict(Counter(c.get('status') for c in cells))}")
    print(f"  measured={len(meas)}  inert(delta==baseline)={len(inert)}  "
          f"DISCRIMINATING={len(disc)}")
    for c in disc[:16]:
        de, be = c.get("delta_effective"), c.get("baseline_delta_effective")
        print(f"    * {c['region_id']} rung={c['rung']} "
              f"delta={de:.3e} baseline={be:.3e}")


if __name__ == "__main__":
    analyze("runs/qcdloop/per_integral_out_b1_2c", "B1")
    analyze("runs/qcdloop/per_integral_out_b12_2c", "B12")
