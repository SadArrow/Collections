#!/usr/bin/env bash
# Run inference and write logs (WSL/Linux).
set -e
export PATH="$HOME/.local/bin:$PATH"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/wsl_inference_log.txt"
cd "$SCRIPT_DIR"

if command -v uv >/dev/null 2>&1; then
  uv run python "$SCRIPT_DIR/run_pi05_inference.py" 2>&1 | tee "$LOG"
else
  python3 "$SCRIPT_DIR/run_pi05_inference.py" 2>&1 | tee "$LOG"
fi

