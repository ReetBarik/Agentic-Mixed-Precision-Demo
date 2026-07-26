#!/usr/bin/env bash
# Subtask 3 Steps 4-5: B12 e2e + B13/B14 non-regression sweep (same catalog+normaliser).
set -euo pipefail
cd "$(dirname "$0")/../.."
source /etc/profile.d/modules.sh 2>/dev/null || true
module use /soft/modulefiles
module load gcc/13.3.0 cmake/3.28.3
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:8084/argoapi/}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-$USER}"
export CLAUDE_CODE_SKIP_ANTHROPIC_AUTH=1
exec .venv/bin/python runs/qcdloop/tier_b_stage1.py \
  --integrals B12,B13,B14 \
  --report   runs/qcdloop/report_5k.json \
  --out-dir  runs/qcdloop/tier_b_stage2_subtask3 \
  --seed 12345 --sample-count 5000 --entry-point BO --margin 0.5
