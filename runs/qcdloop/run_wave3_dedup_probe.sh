#!/usr/bin/env bash
# WAVE3 DEDUP pre-flight probe launcher — land two C-COLL regions sequentially so
# the second must merge into the first's Constants<DoubleDouble>, then build the TU.
#
# Detached tmux (survives the controlling terminal dying):
#     tmux new-session -d -s wave3-dedup \
#       'bash runs/qcdloop/run_wave3_dedup_probe.sh 2>&1 | tee runs/qcdloop/wave3_dedup_probe.log'
#     tmux attach -t wave3-dedup          # watch live
set -euo pipefail
cd "$(dirname "$0")/../.."

# Build toolchain (memory: gcc/13.3.0 + cmake/3.28.3 required for any C++ build).
source /etc/profile.d/modules.sh 2>/dev/null || true
module use /soft/modulefiles
module load gcc/13.3.0 cmake/3.28.3

exec .venv/bin/python runs/qcdloop/wave3_dedup_probe.py
