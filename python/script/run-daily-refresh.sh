#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PYTHON_DIR/.." && pwd)"
ENV_FILE="$PYTHON_DIR/.env.refresh"
LOG_DIR="$PYTHON_DIR/data/derived/logs"
LOG_FILE="$LOG_DIR/daily-refresh.log"

mkdir -p "$LOG_DIR"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

cd "$PYTHON_DIR"

if [[ -x "$PYTHON_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$PYTHON_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Starting daily refresh in $REPO_ROOT"
  "$PYTHON_BIN" script/refresh-live-data.py
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Daily refresh finished"
} >> "$LOG_FILE" 2>&1
