#!/usr/bin/env bash
# Phase 2e prereq + Stage-1 input measurement — per-integral fan-out e2e on
# B1 + B12 at 5k, mirroring run_2d_measure.sh exactly except the out-dir.
# Purpose:
#   (1) Prereq 1: confirm the 2d-A downcast-guard fix (HEAD includes @57651ac)
#       restores B1 boxGPU.h:140/141/142 float from promotion_no_op -> measured
#       DISCRIM.  This run uses the POST-fix boundary.py (unlike per_integral_out_2d,
#       which measured the pre-fix guard).
#   (2) Produce the canonical post-fix B12 scorer manifest the Stage-1 solver
#       consumes (per_integral_out_2e_measure/B12/manifest_scorer_B12.jsonl).
# Also uses the WI1-backfilled report_5k.json (prereq 2), so no float-range warning.
#
# Run detached so it survives the Claude session:
#     tmux new-session -d -s measure2e runs/qcdloop/run_2e_measure.sh
#     tmux attach -t measure2e
set -euo pipefail
cd "$(dirname "$0")/../.."

source /etc/profile.d/modules.sh 2>/dev/null || true
module use /soft/modulefiles
module load gcc/13.3.0 cmake/3.28.3

exec .venv/bin/python runs/qcdloop/run_all_integrals.py \
  --integrals B1 B12 \
  --sample-count 5000 \
  --tolerance 10 \
  --fanout \
  --out-dir runs/qcdloop/per_integral_out_2e_measure
