#!/usr/bin/env bash
# Closure Subtask 3 — Step 3: B10 measurement (headline: first Group A measured lift).
# Re-runs 2b's B10 config with the NEW π-family catalog + deterministic shim normaliser
# live, so the _pi2o6 #error and T__ff redeclaration that blocked B10's shim BUILD in 2b
# are cleared.  Success gate: kernel_measured_lift >= +8 digits on B10's own kernel.
# Non-destructive: dedicated out-dir, no --clean.  Detached:
#     tmux new-session -d -s tierb3b10 runs/qcdloop/tier_b_stage2_subtask3_B10.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source /etc/profile.d/modules.sh 2>/dev/null || true
module use /soft/modulefiles
module load gcc/13.3.0 cmake/3.28.3
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:8084/argoapi/}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-$USER}"
export CLAUDE_CODE_SKIP_ANTHROPIC_AUTH=1
exec .venv/bin/python runs/qcdloop/tier_b_stage1.py \
  --integrals B10 \
  --report   runs/qcdloop/report_5k.json \
  --out-dir  runs/qcdloop/tier_b_stage2_subtask3 \
  --seed 12345 \
  --sample-count 5000 \
  --entry-point BO \
  --margin 0.5
