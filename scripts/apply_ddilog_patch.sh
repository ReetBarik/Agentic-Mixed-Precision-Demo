#!/usr/bin/env bash
# Backward-compatible wrapper: apply_mutation_patch.py for driver ddilog.
#
# Usage:
#   ./scripts/apply_ddilog_patch.sh apply|revert <id>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -ne 2 ]]; then
  echo "usage: $0 apply|revert <id>" >&2
  exit 2
fi
exec python3 "${SCRIPT_DIR}/apply_mutation_patch.py" "$1" ddilog "$2"
