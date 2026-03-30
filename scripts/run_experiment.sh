#!/usr/bin/env bash
# Incremental build (optional), run a catalog driver, compare against baseline CSV.
#
# Usage:
#   ./scripts/run_experiment.sh [--driver ID] [options]
#
# Environment:
#   AGENTIC_MIXED_PRECISION_DEMO_ROOT — repo root (default: parent of scripts/)

set -euo pipefail

ROOT="${AGENTIC_MIXED_PRECISION_DEMO_ROOT:-}"
if [[ -z "$ROOT" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export ROOT

DRIVER_ID="ddilog"
SKIP_BUILD=0
BATCH=10
SEED=123
MIN_DIGITS=10
OUTPUT=""
BASELINE=""

usage() {
  cat <<EOF
run_experiment.sh — build (optional), run driver from targets.json, compare to baseline.

Options:
  --driver ID           targets.json drivers[].id (default: $DRIVER_ID)
  -o, --output PATH     Write driver CSV here (default: experiments/<driver>/generated/<driver>_run_<batch>_<seed>_<timestamp>.csv)
  --baseline PATH       Baseline CSV (default: baselines/<driver>/<driver>_baseline_<batch>_<seed>.csv)
  --batch N             batch_size (default: $BATCH)
  --seed N              RNG seed (default: $SEED)
  --min-digits X        compare_results.py threshold (default: $MIN_DIGITS)
  --no-build            Skip source scripts/compile.sh (use when binary is already fresh)
  -h, --help            This help

Exit code: same as scripts/compare_results.py (0 pass, 1 fail, 2 error).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build) SKIP_BUILD=1 ;;
    --driver)
      DRIVER_ID="$2"
      shift
      ;;
    --batch)
      BATCH="$2"
      shift
      ;;
    --seed)
      SEED="$2"
      shift
      ;;
    --min-digits)
      MIN_DIGITS="$2"
      shift
      ;;
    --baseline)
      BASELINE="$2"
      shift
      ;;
    -o | --output)
      OUTPUT="$2"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

EXEC_REL="$(DRIVER_ID="$DRIVER_ID" python3 - <<'PY'
import json, os
root = os.environ["ROOT"]
did = os.environ["DRIVER_ID"]
with open(os.path.join(root, "targets.json"), encoding="utf-8") as f:
    data = json.load(f)
for d in data.get("drivers", []):
    if d.get("id") == did:
        er = d.get("executable_relative") or ""
        print(er)
        raise SystemExit(0 if er else 2)
raise SystemExit(2)
PY
)" || {
  echo "error: unknown driver id: $DRIVER_ID (check targets.json)" >&2
  exit 2
}

if [[ -z "$EXEC_REL" ]]; then
  echo "error: driver $DRIVER_ID has no executable_relative in targets.json" >&2
  exit 2
fi

if [[ -z "$BASELINE" ]]; then
  BASELINE="${ROOT}/baselines/${DRIVER_ID}/${DRIVER_ID}_baseline_${BATCH}_${SEED}.csv"
fi

if [[ -z "$OUTPUT" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUTPUT="${ROOT}/experiments/${DRIVER_ID}/generated/${DRIVER_ID}_run_${BATCH}_${SEED}_${TS}.csv"
fi

mkdir -p "$(dirname "$OUTPUT")"

DRIVER="${ROOT}/${EXEC_REL}"
COMPARE="${ROOT}/scripts/compare_results.py"

cd "$ROOT"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/scripts/compile.sh" "${ROOT}"
fi

if [[ ! -f "$DRIVER" ]]; then
  echo "error: driver not found: $DRIVER (build the project first)" >&2
  exit 2
fi

"$DRIVER" "${BATCH}" "${OUTPUT}" "${SEED}"
python3 "${COMPARE}" "${BASELINE}" "${OUTPUT}" --min-digits "${MIN_DIGITS}"
