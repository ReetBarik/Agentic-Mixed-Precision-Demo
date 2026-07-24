#!/usr/bin/env bash
# Phase 2e Stage 1 — greedy solver on B12 (single integral).  Detached tmux:
#     tmux new-session -d -s solver1 runs/qcdloop/run_solver_stage1.sh
# The argo proxy env must already be exported in the launching shell.
set -euo pipefail
cd "$(dirname "$0")/../.."
source /etc/profile.d/modules.sh 2>/dev/null || true
module use /soft/modulefiles
module load gcc/13.3.0 cmake/3.28.3
exec .venv/bin/python runs/qcdloop/run_solver_stage1.py \
  --integral B12 \
  --manifest runs/qcdloop/per_integral_out_2e_measure/B12/manifest_scorer_B12.jsonl \
  --report   runs/qcdloop/report_5k.json \
  --out-dir  runs/qcdloop/solver_stage1_B12 \
  --clean
