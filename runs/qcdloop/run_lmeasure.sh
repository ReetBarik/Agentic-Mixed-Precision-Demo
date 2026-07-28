#!/bin/bash
# L-measure — B10/B13/B14 kernel-scope Tier-B run WITH rule (d) leaf-callee promotion.
# The chain path (via=chain -> dispatch._gen_chain, where leaf_ctx fires) is driven by
# tier_b_stage1.py's dominant-COMPUTED-chain solver + kernel-scope positive-lift gate.
# Recipe: seed 12345, 5000 samples, tol 6 (reporting-only; gate is lift-relative),
# margin 0.5, entry BO (LEAF_CALLEE_PROMOTION_DESIGN.md §8 / L-measure spec).
# B10/B13 opt in to leaf promotion; B14 is dd-sufficient (no clonable leaf) and must be
# byte-identical to its pre-leaf-promotion Stage-2 baseline (STOP #B).
set -euo pipefail
cd "$(dirname "$0")/../.."

module use /soft/modulefiles >/dev/null 2>&1 || true
module load gcc/13.3.0 cmake/3.28.3 >/dev/null 2>&1 || true
source .venv/bin/activate

export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:8084/argoapi/}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-rbarik}"

python runs/qcdloop/tier_b_stage1.py \
    --integrals B10,B13,B14 \
    --report runs/qcdloop/report_5k.json \
    --sample-count 5000 --seed 12345 \
    --entry-point BO --margin 0.5 --tolerance 6 \
    --leaf-promotion \
    --out-dir runs/qcdloop/tier_b_stage2_leaf_promotion/lmeasure_run \
    --clean
echo "LMEASURE_DONE_EXIT_$?"
