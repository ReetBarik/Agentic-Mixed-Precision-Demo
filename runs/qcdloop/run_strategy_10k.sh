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

exec .venv/bin/python runs/qcdloop/run_strategy_e2e.py \
  --report runs/qcdloop/report_10k.json \
  --sample-count 1000 --seed 12345 \
  --tolerance 7.0 \
  --max-iters-correctness 300 --max-iters-speedup 200 \
  --max-wall-hours 12 \
  --dd-repo "$HOME/qcdloop" --dd-ref ddfun_enabled \
  --kokkos-root "$HOME/kokkos-install"
