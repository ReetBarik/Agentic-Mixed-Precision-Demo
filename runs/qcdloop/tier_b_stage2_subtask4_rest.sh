#!/usr/bin/env bash
# Closure Subtask 4 — Step 4: B12/B13/B14 non-regression sweep.
# Same config as Subtask 3 (only MAX_INTEGRATOR_RETRIES 3->6 differs). Verifies the
# retry-budget bump does not regress the other three integrals:
#   B12 — still builds cleanly (chain_no_lift ~3.6906 unchanged; not a Subtask-4 blocker)
#   B13 — stays write_truncation byte-identical
#   B14 — stays chain_no_lift ~13.1855 byte-identical
# STOP #I fires if any previously-succeeding path fails under budget=6 (should be
# impossible — more attempts can only help). Detached:
#     tmux new-session -d -s tierb4rest runs/qcdloop/tier_b_stage2_subtask4_rest.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source /etc/profile.d/modules.sh 2>/dev/null || true
module use /soft/modulefiles
module load gcc/13.3.0 cmake/3.28.3
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:8084/argoapi/}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-$USER}"
export CLAUDE_CODE_SKIP_ANTHROPIC_AUTH=1
exec .venv/bin/python runs/qcdloop/tier_b_stage1.py \
  --integrals B12 B13 B14 \
  --report   runs/qcdloop/report_5k.json \
  --out-dir  runs/qcdloop/tier_b_stage2_subtask4 \
  --seed 12345 \
  --sample-count 5000 \
  --entry-point BO \
  --margin 0.5
