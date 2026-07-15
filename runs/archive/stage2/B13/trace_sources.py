#!/usr/bin/env python3
"""Attribution walk for the B13 Tracked v0.3 journal.

Answers "which source variables ultimately fed the hottest log record?" by
walking the value DAG backward over the journal's `in` edges — the on-disk
equivalent of tracked::journal::trace_sources(id).

v0.3 model (see third_party/tracked/docs/PROVENANCE.md):
  * every record has a stable `id`; `in` lists its direct-operand ids verbatim.
  * source variables (track()) seed `prov_vars`; their id is the bare name and
    they are leaves (no record is produced *for* them).
  * named constants (constant()) seed `prov_consts` and are excluded from
    attribution.

Usage:
    python3 trace_sources.py [journal.jsonl|journal.jsonl.gz]
"""
from __future__ import annotations

import collections
import gzip
import json
import sys


def load(path: str) -> list[dict]:
    opn = gzip.open if path.endswith(".gz") else open
    with opn(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "journal.jsonl.gz"
    records = load(path)

    id_to_rec = {r["id"]: r for r in records}
    source_names = {n for r in records for n in r.get("prov_vars", [])}

    # Hottest log record = the aliasing-bug canary.
    logs = [r for r in records if r["op"] == "log"]
    top = max(logs, key=lambda r: r["cond"])
    print(f"top log record:")
    print(f"  id          = {top['id']}")
    print(f"  in          = {top['in']}")
    print(f"  cond        = {top['cond']:.6g}")
    print(f"  val         = {top['val']:.6g}")
    print(f"  rel_err     = {top['rel_err']:.6g}")
    print(f"  prov_vars   = {sorted(top['prov_vars'])}")
    print(f"  prov_consts = {sorted(top['prov_consts'])}")

    # BFS backward via `in` edges, collecting every visited id that is a source
    # variable. Leaf source-variable ids appear in `in` but own no record.
    seen: set[str] = set()
    found: set[str] = set()
    q = collections.deque([top["id"]])
    while q:
        cur = q.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        if cur in source_names:
            found.add(cur)
        rec = id_to_rec.get(cur)
        if rec:
            q.extend(rec.get("in", []))

    print(f"\ntrace_sources({top['id']}):")
    print(f"  {sorted(found)}")

    # Sanity: the walk should reproduce the record's own accumulated prov_vars.
    if found == set(top["prov_vars"]):
        print("  OK: matches the record's prov_vars (attribution consistent).")
    else:
        print(f"  NOTE: differs from prov_vars {sorted(top['prov_vars'])}")


if __name__ == "__main__":
    main()
