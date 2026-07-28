#!/bin/bash
# Region-core element promotion — measured-4 regression sweep + L-measure (step 4 + 5).
# B14 honest dd baseline (first ever clean build), B10 expected still-blocked (out-of-scope
# View region-core + 71 leaf clones), B12/B13 regression check (none-touch → unchanged).
# B15/B16 clean already evidenced by the B14 co-variant build (shared B2m/B3m chain).
set -euo pipefail
cd "$(dirname "$0")/../.."
module use /soft/modulefiles >/dev/null 2>&1 || true
module load gcc/13.3.0 cmake/3.28.3 >/dev/null 2>&1 || true
source .venv/bin/activate
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:8084/argoapi/}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-rbarik}"
python runs/qcdloop/tier_b_stage1.py \
    --integrals B10,B12,B13,B14 \
    --report runs/qcdloop/report_5k.json \
    --sample-count 5000 --seed 12345 \
    --entry-point BO --margin 0.5 --tolerance 6 \
    --leaf-promotion \
    --out-dir runs/qcdloop/tier_b_stage2_leaf_promotion/region_core_sweep \
    --clean
echo "REGION_CORE_SWEEP_DONE_EXIT_$?"
