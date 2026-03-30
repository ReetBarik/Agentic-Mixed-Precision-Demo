#!/usr/bin/env bash
# Apply or reverse a unified diff for ddilog locals under patches/ddilog/<id>.patch
#
# Usage:
#   ./scripts/apply_ddilog_patch.sh apply A
#   ./scripts/apply_ddilog_patch.sh revert A
#
# Patches use paths like a/src/kokkosUtils.h; run from repo root with -p1.

set -euo pipefail

ROOT="${AGENTIC_MIXED_PRECISION_DEMO_ROOT:-}"
if [[ -z "$ROOT" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

usage() {
  echo "usage: $0 apply|revert <id>" >&2
  echo "  id: mutation id (e.g. A); looks for patches/ddilog/<id>.patch" >&2
  exit 2
}

if [[ $# -ne 2 ]]; then
  usage
fi

ACTION="$1"
ID="$2"
PATCH="${ROOT}/patches/ddilog/${ID}.patch"

if [[ ! -f "$PATCH" ]]; then
  echo "error: patch not found: $PATCH" >&2
  exit 2
fi

cd "$ROOT"

case "$ACTION" in
  apply)
    patch -p1 --forward < "$PATCH"
    ;;
  revert)
    patch -p1 -R < "$PATCH"
    ;;
  *)
    usage
    ;;
esac
