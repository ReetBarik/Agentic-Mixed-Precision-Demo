#!/usr/bin/env bash
# Closure Subtask 2b — Tier-B Stage-2 e2e re-run, B10/B13/B14 (design §6.3 Stage 2).
# B10 is the headline (rule c cross-frame return propagation); B13 piggybacks; B14 is
# a non-regression check (1b clearance must not regress).  B12 skipped (Subtask 3).
# Non-destructive: dedicated out-dir, no --clean, so canonical Stage-1 artifacts are
# preserved.  Detached:
#     tmux new-session -d -s tierb2b runs/qcdloop/tier_b_stage2_subtask2b.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source /etc/profile.d/modules.sh 2>/dev/null || true
module use /soft/modulefiles
module load gcc/13.3.0 cmake/3.28.3
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:8084/argoapi/}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-$USER}"
export CLAUDE_CODE_SKIP_ANTHROPIC_AUTH=1
exec .venv/bin/python runs/qcdloop/tier_b_stage1.py \
  --integrals B10,B13,B14 \
  --report   runs/qcdloop/report_5k.json \
  --out-dir  runs/qcdloop/tier_b_stage2_subtask2b \
  --seed 12345 \
  --sample-count 5000 \
  --entry-point BO \
  --margin 0.5
