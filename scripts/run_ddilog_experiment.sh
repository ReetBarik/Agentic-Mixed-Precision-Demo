#!/usr/bin/env bash
# Backward-compatible wrapper: run_experiment.sh with --driver ddilog.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_experiment.sh" --driver ddilog "$@"
