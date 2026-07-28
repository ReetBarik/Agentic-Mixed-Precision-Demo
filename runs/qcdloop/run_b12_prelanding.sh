#!/bin/bash
# Pre-landing (5 prod files reverted to HEAD) B12 no-leaf baseline for STOP #II attribution.
set -euo pipefail
cd "$(dirname "$0")/../.."
module use /soft/modulefiles >/dev/null 2>&1 || true
module load gcc/13.3.0 cmake/3.28.3 >/dev/null 2>&1 || true
source .venv/bin/activate
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:8084/argoapi/}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-rbarik}"
python runs/qcdloop/tier_b_stage1.py \
    --integrals B12 \
    --report runs/qcdloop/report_5k.json \
    --sample-count 5000 --seed 12345 \
    --entry-point BO --margin 0.5 --tolerance 6 \
    --out-dir runs/qcdloop/tier_b_stage2_leaf_promotion/region_core_b12_prelanding \
    --clean
echo "B12_PRELANDING_DONE_EXIT_$?"
