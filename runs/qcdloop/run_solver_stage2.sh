#!/usr/bin/env bash
# Phase 2e Stage 2 — greedy solver across all 21 integrals (per-integral trees).
# Consumes the measurement pass output (per-integral scorer manifests) and fans the
# single-integral solver across every integral, then aggregates SOLVER_STAGE2.md.
#
# Run detached so it survives the Claude session:
#     tmux new-session -d -s solver2 runs/qcdloop/run_solver_stage2.sh
#     tmux attach -t solver2
set -euo pipefail
cd "$(dirname "$0")/../.."

source /etc/profile.d/modules.sh 2>/dev/null || true
module use /soft/modulefiles
module load gcc/13.3.0 cmake/3.28.3

exec .venv/bin/python runs/qcdloop/run_solver_stage2.py \
  --report       runs/qcdloop/report_5k.json \
  --manifest-dir runs/qcdloop/per_integral_out_stage2 \
  --out-dir      runs/qcdloop/solver_stage2 \
  --workers "${WORKERS:-4}" \
  "$@"
