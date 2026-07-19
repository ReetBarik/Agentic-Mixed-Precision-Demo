#!/usr/bin/env bash
# Float retro probe launcher — re-validate the Wave-1+2 10k run's 86
# float-accepted regions at tighter tolerances (8/9/10/11).
#
# Validator-only replay: no walk, no LLM, no fresh characterization. See
# retro_probe.py for the method (one build per region; tolerance is a pure
# post-build threshold, so all four tolerances share the single build).
#
# Detached tmux (survives the controlling terminal dying):
#     tmux new-session -d -s float-retro \
#       'bash runs/qcdloop/run_retro_probe.sh 2>&1 | tee runs/qcdloop/retro_probe.log'
#     tmux attach -t float-retro          # watch live
set -euo pipefail
cd "$(dirname "$0")/../.."

# Build toolchain (memory: gcc/13.3.0 + cmake/3.28.3 required for any C++ build).
source /etc/profile.d/modules.sh 2>/dev/null || true
module use /soft/modulefiles
module load gcc/13.3.0 cmake/3.28.3

exec .venv/bin/python runs/qcdloop/retro_probe.py
