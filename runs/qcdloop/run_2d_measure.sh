#!/usr/bin/env bash
# Phase 2d-A interim measurement — per-integral fan-out e2e on B1 + B12 at 5k,
# mirroring the Phase 2c run (runs/qcdloop/PHASE_2C_2026-07-24.md) exactly except
# for the out-dir, so results compare directly.  Goal: confirm the complex-container
# promotion fix (boundary.py + type_resolve.py) lets the previously llm_gen_failed
# complex regions (B2m.h:188/193, B0m.h:405, boxGPU.h:140) build + measure.
#
# Run inside a detached tmux session so it survives the Claude session dying:
#     tmux new-session -d -s measure2d runs/qcdloop/run_2d_measure.sh
#     tmux attach -t measure2d          # watch live
#     tmux kill-session -t measure2d    # stop early
#
# The argo proxy env (ANTHROPIC_BASE_URL=…:8084, token) must already be exported
# in the shell that starts tmux; tmux inherits them.
set -euo pipefail
cd "$(dirname "$0")/../.."

# Build module chain (required for the C++ compile/build inside the Patcher/Validator).
source /etc/profile.d/modules.sh 2>/dev/null || true
module use /soft/modulefiles
module load gcc/13.3.0 cmake/3.28.3

exec .venv/bin/python runs/qcdloop/run_all_integrals.py \
  --integrals B1 B12 \
  --sample-count 5000 \
  --tolerance 10 \
  --fanout \
  --out-dir runs/qcdloop/per_integral_out_2d
