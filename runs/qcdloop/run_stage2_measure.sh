#!/usr/bin/env bash
# Phase 2e Stage 2 — measurement pass for ALL 21 integrals (produces the per-integral
# scorer manifests the Stage-2 solver consumes).  Mirrors run_2e_measure.sh but across
# every integral and with parallel workers.  Includes the Phase-2e signal_class filter
# (cancellation-cascade / local-cancellation regions -> awaiting_algorithmic_rewrite,
# no LLM/build).
#
# Run detached so it survives the Claude session:
#     tmux new-session -d -s measure2 runs/qcdloop/run_stage2_measure.sh
#     tmux attach -t measure2
#
# Env knobs: WORKERS (default 4), INTEGRALS (space-separated; default all in report).
set -euo pipefail
cd "$(dirname "$0")/../.."

source /etc/profile.d/modules.sh 2>/dev/null || true
module use /soft/modulefiles
module load gcc/13.3.0 cmake/3.28.3

INTEGRALS_ARG=()
if [[ -n "${INTEGRALS:-}" ]]; then
  # shellcheck disable=SC2206
  INTEGRALS_ARG=(--integrals ${INTEGRALS})
fi

exec .venv/bin/python runs/qcdloop/run_all_integrals.py \
  "${INTEGRALS_ARG[@]}" \
  --sample-count 5000 \
  --tolerance 10 \
  --fanout \
  --workers "${WORKERS:-4}" \
  --out-dir runs/qcdloop/per_integral_out_stage2
