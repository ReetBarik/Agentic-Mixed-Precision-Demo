#!/usr/bin/env bash
# Launch the 10k two-phase Strategy calibration walk.
#
# Run inside a detached tmux session so it survives the controlling
# terminal / Claude session dying:
#     tmux new-session -d -s strategy10k runs/qcdloop/run_strategy_10k.sh
#     tmux attach -t strategy10k          # watch live
#     tmux kill-session -t strategy10k    # stop early
#
# The venv python + the argo proxy env (ANTHROPIC_BASE_URL=…:8084, token)
# must already be exported in the shell that starts tmux; tmux inherits them.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"

# Budget caps + dr_k per CALIBRATION.md §50k recommendation (the same faithful
# config validated by the Wave-1 dedup/fuse re-run): correctness 200
# (~126 distinct dd promotions expected + margin), speedup 250 (~180 ff demotions
# + margin to drain).  dr_k is left at StrategyConfig's default (now 60, up from
# 20) — see CALIBRATION.md §Bugs 2/3: chain-phase llm_gen_failed streaks bump the
# DR counter without consuming budget, so 20 tripped `partial` before the
# correctness cap could bind.  The chain-representative dedup (Wave 1) removes the
# redundant re-drives that made 300 necessary in the pre-dedup Run A.
exec .venv/bin/python runs/qcdloop/run_strategy_e2e.py \
  --report runs/qcdloop/report_10k.json \
  --sample-count 1000 --seed 12345 \
  --tolerance 7.0 \
  --max-iters-correctness 200 --max-iters-speedup 250 \
  --max-wall-hours 12 \
  --dd-repo "$HOME/qcdloop" --dd-ref ddfun_enabled \
  --kokkos-root "$HOME/kokkos-install"
