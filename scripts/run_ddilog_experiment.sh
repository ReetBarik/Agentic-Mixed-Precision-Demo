#!/usr/bin/env bash
# Incremental build (optional), run ddilog_driver, compare against baseline CSV.
#
# Usage:
#   ./scripts/run_ddilog_experiment.sh [options]
#   # or from repo root:
#   bash scripts/run_ddilog_experiment.sh [options]
#
# Environment:
#   AGENTIC_MIXED_PRECISION_DEMO_ROOT — repo root (default: parent of scripts/)
#
# Examples:
#   ./scripts/run_ddilog_experiment.sh
#   ./scripts/run_ddilog_experiment.sh -o experiments/ddilog/generated/my_run.csv
#   ./scripts/run_ddilog_experiment.sh --no-build --batch 10 --seed 123

set -euo pipefail

ROOT="${AGENTIC_MIXED_PRECISION_DEMO_ROOT:-}"
if [[ -z "$ROOT" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

SKIP_BUILD=0
BATCH=10
SEED=123
MIN_DIGITS=10
OUTPUT=""
BASELINE=""

usage() {
  cat <<EOF
run_ddilog_experiment.sh — build (optional), run ddilog_driver, compare to baseline.

Options:
  -o, --output PATH     Write driver CSV here (default: experiments/ddilog/generated/ddilog_run_<batch>_<seed>_<timestamp>.csv)
  --baseline PATH       Baseline CSV (default: baselines/ddilog/ddilog_baseline_<batch>_<seed>.csv)
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

if [[ -z "$BASELINE" ]]; then
  BASELINE="${ROOT}/baselines/ddilog/ddilog_baseline_${BATCH}_${SEED}.csv"
fi

if [[ -z "$OUTPUT" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUTPUT="${ROOT}/experiments/ddilog/generated/ddilog_run_${BATCH}_${SEED}_${TS}.csv"
fi

mkdir -p "$(dirname "$OUTPUT")"

DRIVER="${ROOT}/build/ddilog_driver"
COMPARE="${ROOT}/scripts/compare_results.py"

if [[ ! -f "$DRIVER" ]]; then
  echo "error: driver not found: $DRIVER (build the project first)" >&2
  exit 2
fi

cd "$ROOT"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/scripts/compile.sh" "${ROOT}"
fi

"$DRIVER" "${BATCH}" "${OUTPUT}" "${SEED}"
python3 "${COMPARE}" "${BASELINE}" "${OUTPUT}" --min-digits "${MIN_DIGITS}"
