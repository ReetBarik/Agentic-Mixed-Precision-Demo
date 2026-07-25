#!/usr/bin/env bash
# Phase 2f Tier-B Stage-1 — chain-scoped dd promotion on B10/B12/B13/B14.  Detached:
#     tmux new-session -d -s tierb runs/qcdloop/tier_b_stage1.sh
# The Argo proxy env (ANTHROPIC_AUTH_TOKEN etc.) must already be exported in the shell.
set -euo pipefail
cd "$(dirname "$0")/../.."
source /etc/profile.d/modules.sh 2>/dev/null || true
module use /soft/modulefiles
module load gcc/13.3.0 cmake/3.28.3
exec .venv/bin/python runs/qcdloop/tier_b_stage1.py \
  --integrals B10,B12,B13,B14 \
  --report   runs/qcdloop/report_5k.json \
  --out-dir  runs/qcdloop/tier_b_stage1 \
  --clean
