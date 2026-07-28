#!/bin/bash
# Region-core element promotion — ISOLATED regression check (step 4, STOP #II).
# NO --leaf-promotion, so rule-d confounds (_pi2 leaf gap, write_truncation) are removed
# and any B12/B13 delta is attributable to the element landing ALONE.
# Coverage argument: B12/B13 are none-touch (no complex region-core) -> reconcile pass is a
# strict no-op -> they must match pre-landing behaviour exactly. B14 re-confirms clean build.
set -euo pipefail
cd "$(dirname "$0")/../.."
module use /soft/modulefiles >/dev/null 2>&1 || true
module load gcc/13.3.0 cmake/3.28.3 >/dev/null 2>&1 || true
source .venv/bin/activate
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:8084/argoapi/}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-rbarik}"
python runs/qcdloop/tier_b_stage1.py \
    --integrals B12,B13,B14 \
    --report runs/qcdloop/report_5k.json \
    --sample-count 5000 --seed 12345 \
    --entry-point BO --margin 0.5 --tolerance 6 \
    --out-dir runs/qcdloop/tier_b_stage2_leaf_promotion/region_core_noleaf \
    --clean
echo "REGION_CORE_NOLEAF_DONE_EXIT_$?"
