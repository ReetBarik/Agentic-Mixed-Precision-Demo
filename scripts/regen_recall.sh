#!/usr/bin/env bash
#
# Regenerate the Phase-1 characterizer fixtures and run the recall verifier.
#
# What it does, per fixture:
#   1. clean-configure + build the existing micro-driver (CMake)
#   2. run it from the run dir so journal.jsonl lands in runs/<fixture>/
#   3. re-parse journal.jsonl -> sensitivity_profile.json
#      (relativizes paths per commit 918738e)
#   4. run the recall verifier -> stdout + runs/recall_summary.json
#
# Prerequisites (load via your environment / `module load` first):
#   - cmake >= 3.18 on PATH
#   - a Python >= 3.10 interpreter        -> set PYTHON   (default: python3)
#   - a Kokkos install (for lnrat/cln)    -> set KOKKOS_ROOT
#                                            (default: /home/rbarik/kokkos-install)
#
# Usage:
#   module load cmake/<...> python/<...>            # whatever your site provides
#   PYTHON=python3.11 KOKKOS_ROOT=/path/to/kokkos ./scripts/regen_recall.sh
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYTHON="${PYTHON:-python3}"
KOKKOS_ROOT="${KOKKOS_ROOT:-/home/rbarik/kokkos-install}"

PLAIN_FIXTURES=(cancellation kahan naive_variance log_sum_exp)
KOKKOS_FIXTURES=(lnrat cln)
ALL_FIXTURES=("${PLAIN_FIXTURES[@]}" "${KOKKOS_FIXTURES[@]}")

build_and_run() {
  local k="$1" prefix_arg="$2" d="runs/$1"
  echo "=== [$k] configure + build =================================="
  rm -rf "$d/build"
  # shellcheck disable=SC2086  # intentional word-split of prefix_arg
  cmake -S "$d" -B "$d/build" -DCMAKE_BUILD_TYPE=Release $prefix_arg
  cmake --build "$d/build" -j
  echo "=== [$k] run (journal -> $d/journal.jsonl) ================="
  ( cd "$d" && ./build/micro_driver )
  echo
}

for k in "${PLAIN_FIXTURES[@]}";  do build_and_run "$k" ""; done
for k in "${KOKKOS_FIXTURES[@]}"; do build_and_run "$k" "-DCMAKE_PREFIX_PATH=$KOKKOS_ROOT"; done

echo "=== regenerate sensitivity_profile.json (re-parse journals) ==="
"$PYTHON" -m agents.shared.regen_profile "${ALL_FIXTURES[@]/#/runs/}"
echo

echo "=== recall verifier -> runs/recall_summary.json ============="
"$PYTHON" -m agents.shared.recall_verifier runs
