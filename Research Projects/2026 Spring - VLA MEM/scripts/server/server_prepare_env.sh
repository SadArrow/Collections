#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv_server"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools
python -m pip install -r "${ROOT_DIR}/requirements.txt"

if python -c "import torch" >/dev/null 2>&1; then
  echo "[env] torch already available"
else
  echo "[env] torch is not installed in ${VENV_DIR}; install the correct CUDA build manually if needed"
fi

python -c "import transformers, safetensors, sentencepiece, numpy, einops; print('[env] python deps ok')"
