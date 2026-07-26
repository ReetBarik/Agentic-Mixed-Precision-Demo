#!/usr/bin/env bash
# Closure Subtask 4 — Step 3: B10 re-measurement (headline: first Group A measured lift).
# Re-runs Subtask 3's B10 config UNCHANGED except for the single knob this Subtask turns:
# MAX_INTEGRATOR_RETRIES 3->6 (agents/patcher/agent.py). Subtask 3 STOP #F showed the
# 3-attempt budget lost B10 to LLM non-determinism on the ql::Lnrat<ddouble> R4 #error
# escape hatch (a symbol B12 recovered on attempt 2 in the same silo). More attempts can
# only help a retryable misgen; success gate: kernel_measured_lift >= +8 digits on B10.
# Non-destructive: dedicated out-dir, no --clean. Detached:
#     tmux new-session -d -s tierb4b10 runs/qcdloop/tier_b_stage2_subtask4_B10.sh
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
  --out-dir  runs/qcdloop/tier_b_stage2_subtask4 \
  --seed 12345 \
  --sample-count 5000 \
  --entry-point BO \
  --margin 0.5
