#!/bin/bash
# run-argo.sh — Run the agentic mixed-precision optimization workflow via Argo.
#
# Handles the SSH tunnel and local proxy the same way argonne-claude.sh does,
# but reuses them if they are already running (e.g. this script is invoked
# from within a Claude Code session that is itself using argonne-claude.sh).
#
# Usage:
#   ./run-argo.sh --file src/kokkosUtils.h --function ddilog [OPTIONS...]
#
# All options after the script name are forwarded verbatim to llm_agent/run.py.
# Run `python3.12 -m llm_agent.run --help` for the full option list.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (matches argonne-claude.sh)
# ---------------------------------------------------------------------------
REMOTE_HOST="homes.cels.anl.gov"
TUNNEL_LOCAL_PORT=8082
TUNNEL_REMOTE_HOST="apps.inside.anl.gov"
TUNNEL_REMOTE_PORT=443
PROXY_PORT=8083
PROXY_SCRIPT="${HOME}/ai-agents-at-anl/claude-argo-proxy.py"
CONTROL_PATH="/tmp/ssh-argo-workflow-$$"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Track what this script started so cleanup only tears down what it owns.
STARTED_TUNNEL=false
STARTED_PROXY=false
PROXY_PID=""

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
cleanup() {
    if [ "${STARTED_PROXY}" = true ] && [ -n "${PROXY_PID}" ]; then
        echo -e "\n${YELLOW}Stopping proxy (pid ${PROXY_PID})...${NC}"
        kill "${PROXY_PID}" 2>/dev/null || true
    fi
    if [ "${STARTED_TUNNEL}" = true ]; then
        echo -e "${YELLOW}Closing SSH tunnel...${NC}"
        ssh -O exit -o ControlPath="${CONTROL_PATH}" "${REMOTE_HOST}" 2>/dev/null || true
    fi
}
trap cleanup EXIT SIGINT SIGTERM

# ---------------------------------------------------------------------------
# Step 1: SSH tunnel
# ---------------------------------------------------------------------------
if lsof -i :"${TUNNEL_LOCAL_PORT}" >/dev/null 2>&1; then
    echo -e "${GREEN}SSH tunnel already running on port ${TUNNEL_LOCAL_PORT} — reusing.${NC}"
else
    echo -e "${YELLOW}Starting SSH tunnel to ${TUNNEL_REMOTE_HOST}...${NC}"
    echo -e "${YELLOW}(You may need to complete MFA authentication)${NC}"
    ssh -f -N \
        -o ControlMaster=yes \
        -o ControlPath="${CONTROL_PATH}" \
        -L "${TUNNEL_LOCAL_PORT}:${TUNNEL_REMOTE_HOST}:${TUNNEL_REMOTE_PORT}" \
        "${REMOTE_HOST}"
    if [ $? -ne 0 ]; then
        echo -e "${RED}SSH tunnel failed. Check your credentials and MFA.${NC}"
        exit 1
    fi
    STARTED_TUNNEL=true
    echo -e "${GREEN}SSH tunnel established (port ${TUNNEL_LOCAL_PORT}).${NC}"
fi

# ---------------------------------------------------------------------------
# Step 2: Local proxy
# ---------------------------------------------------------------------------
if lsof -i :"${PROXY_PORT}" >/dev/null 2>&1; then
    echo -e "${GREEN}Proxy already running on port ${PROXY_PORT} — reusing.${NC}"
else
    if [ ! -f "${PROXY_SCRIPT}" ]; then
        echo -e "${RED}Proxy script not found: ${PROXY_SCRIPT}${NC}"
        exit 1
    fi
    echo -e "${YELLOW}Starting local Argo proxy...${NC}"
    python3 "${PROXY_SCRIPT}" &
    PROXY_PID=$!
    STARTED_PROXY=true
    sleep 2
    if ! kill -0 "${PROXY_PID}" 2>/dev/null; then
        echo -e "${RED}Proxy failed to start. Is aiohttp installed? (pip install aiohttp)${NC}"
        exit 1
    fi
    echo -e "${GREEN}Proxy running (port ${PROXY_PORT}).${NC}"
fi

# ---------------------------------------------------------------------------
# Step 3: Run the agentic workflow
# ---------------------------------------------------------------------------
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}No arguments supplied. Run with --help to see options.${NC}"
    echo ""
    ANTHROPIC_BASE_URL="http://127.0.0.1:${PROXY_PORT}/argoapi/" \
        ANTHROPIC_AUTH_TOKEN="${ARGO_USER:-$USER}" \
        python3.12 -m llm_agent.run --help
    exit 0
fi

echo -e "${GREEN}Running agentic workflow...${NC}"
ANTHROPIC_BASE_URL="http://127.0.0.1:${PROXY_PORT}/argoapi/" \
    ANTHROPIC_AUTH_TOKEN="${ARGO_USER:-$USER}" \
    python3.12 -m llm_agent.run "$@"
