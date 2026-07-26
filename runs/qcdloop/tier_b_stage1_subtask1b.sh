#!/usr/bin/env bash
# Closure Subtask 1b — Tier-B Stage-1 e2e re-run, B13/B14 ONLY (design §D).
# B10 skipped (needs rule c, out of scope); B12 skipped (out of scope).
# Non-destructive: dedicated out-dir, no --clean, so canonical B10/B12 artifacts
# under tier_b_stage1/ are preserved.  Detached:
#     tmux new-session -d -s tierb1b runs/qcdloop/tier_b_stage1_subtask1b.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source /etc/profile.d/modules.sh 2>/dev/null || true
module use /soft/modulefiles
module load gcc/13.3.0 cmake/3.28.3
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:8084/argoapi/}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-$USER}"
export CLAUDE_CODE_SKIP_ANTHROPIC_AUTH=1
exec .venv/bin/python runs/qcdloop/tier_b_stage1.py \
  --integrals B13,B14 \
  --report   runs/qcdloop/report_5k.json \
  --out-dir  runs/qcdloop/tier_b_stage1_subtask1b \
  --seed 12345 \
  --sample-count 5000 \
  --entry-point BO \
  --margin 0.5
