#!/usr/bin/env bash
# Supplementary 10k probe to capture SPEEDUP-phase first-data.
#
# The faithful 300/200 run (run_strategy_10k.sh) stops `partial` in the
# correctness phase: the cascade-chain phase re-drives double-to-dd on the same
# representative lines across many chains, each failing llm_gen_failed (which does
# NOT consume budget but DOES increment the diminishing-returns streak), so it
# trips dr_k=20 at ~iter 128 with only ~80 of the 300 correctness budget-iters
# used — before speedup ever runs.
#
# To reach speedup we bind the correctness BUDGET cap low (40) so the soft
# hand-off fires during the productive accept phase (~iter 60; run 1 achieved 80
# accepts), and raise dr_k to 40 so a transient chain-failure streak doesn't
# hard-stop the run first. Speedup then runs on the 113-region stable queue.
#
# tmux (durable):
#     tmux new-session -d -s strategy10k_spd runs/qcdloop/run_strategy_10k_speedup.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

exec .venv/bin/python runs/qcdloop/run_strategy_e2e.py \
  --report runs/qcdloop/report_10k.json \
  --sample-count 1000 --seed 12345 \
  --tolerance 7.0 \
  --max-iters-correctness 40 --max-iters-speedup 200 \
  --dr-k 40 \
  --max-wall-hours 12 \
  --dd-repo "$HOME/qcdloop" --dd-ref ddfun_enabled \
  --kokkos-root "$HOME/kokkos-install"
