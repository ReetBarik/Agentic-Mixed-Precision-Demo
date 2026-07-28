#!/bin/bash
# Region-core element promotion — B14 go/no-go build + L-measure (step 3 + step 5).
# B14 is dd-sufficient (no clonable leaf), so NO --leaf-promotion.  This is the first
# honest dd build after element-level promotion (STOP #CC fix): the instantiation gate
# is the sole arbiter.  Fresh out-dir so the prior lmeasure_run is preserved.
set -euo pipefail
cd "$(dirname "$0")/../.."

module use /soft/modulefiles >/dev/null 2>&1 || true
module load gcc/13.3.0 cmake/3.28.3 >/dev/null 2>&1 || true
source .venv/bin/activate

export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:8084/argoapi/}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-rbarik}"

python runs/qcdloop/tier_b_stage1.py \
    --integrals B14 \
    --report runs/qcdloop/report_5k.json \
    --sample-count 5000 --seed 12345 \
    --entry-point BO --margin 0.5 --tolerance 6 \
    --out-dir runs/qcdloop/tier_b_stage2_leaf_promotion/region_core_b14 \
    --clean
echo "REGION_CORE_B14_DONE_EXIT_$?"
