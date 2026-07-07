#!/usr/bin/env bash
# B13 journal analysis one-liners, updated for the Tracked v0.3 schema.
#
# v0.3 journal record fields (see third_party/tracked/docs/PROVENANCE.md):
#   op, at, id, in, val, cond, rel_err, prov_vars, prov_consts
#
#   id          stable per-value tag (name for track/constant; <op>@<loc>#<n>[@<scope>] for derived)
#   in          direct-operand ids, verbatim (no more primary_id heuristic)
#   prov_vars   source-variable roots (attribution)
#   prov_consts named constants (audit only)
#
# Usage: ./analysis_queries.sh [journal.jsonl|journal.jsonl.gz]
set -euo pipefail
J="${1:-journal.jsonl.gz}"
cat() { case "$J" in *.gz) zcat "$J";; *) command cat "$J";; esac; }

echo "== total records =="
cat | wc -l

echo "== records with cond > 1e15 (exact-cancel fix; expect 0) =="
cat | jq -c 'select(.cond > 1e15)' | wc -l

echo "== top 5 log records by cond =="
cat | jq -r 'select(.op=="log")
  | [.cond, .val, .rel_err, .id, (.in|tostring), (.prov_vars|tostring), (.prov_consts|tostring)]
  | @tsv' | sort -t$'\t' -k1 -rn | head -5

# Aliasing-bug canary. In v0.2 a constant's *name* was wrongly picked into `in`
# by primary_id (alphabetical), so a hot log showed in=["_four"]. Under v0.3 the
# honest invariants are:
#   (a) NO underscore-prefixed name appears in `in` (constants renamed; ids real)
#   (b) NO log/hot op has a bare constant as a direct operand (log-of-a-constant)
# Note: a *bare* constant name (e.g. "half") CAN legitimately appear in `in` when
# an op genuinely operates on it (half * x) — that is correct, not the bug.
echo "== (a) underscore names in 'in' (expect 0) =="
cat | jq -c 'select(.in | any(startswith("_")))' | wc -l

echo "== (b) log records operating directly on a constant (expect 0) =="
cat | jq -c 'select(.op=="log")
  | select(.in | any(. == "four" or . == "half" or . == "two" or . == "one"
                     or . == "pi" or . == "zero" or . == "three"))' | wc -l

echo "== attribution walk on the hottest log record =="
python3 trace_sources.py "$J" 2>/dev/null || ../../.venv/bin/python trace_sources.py "$J"
