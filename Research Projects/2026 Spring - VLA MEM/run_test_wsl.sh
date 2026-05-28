#!/usr/bin/env bash
# Run pi0.5 (pi05_droid) inference test in WSL/Linux (PyTorch-only).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v uv >/dev/null 2>&1; then
  exec uv run python "$SCRIPT_DIR/run_pi05_inference.py"
else
  exec python3 "$SCRIPT_DIR/run_pi05_inference.py"
fi

