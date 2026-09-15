#!/bin/bash
set -euo pipefail

PY="/runpod-volume/uv-python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12"
SITE="/runpod-volume/venvs/moncusoai/lib/python3.12/site-packages"

if [[ ! -x "$PY" ]]; then
  echo "FATAL: volume Python not found at $PY" >&2
  ls -la /runpod-volume/uv-python 2>&1 || true
  ls -la /runpod-volume/venvs 2>&1 || true
  exit 127
fi

if [[ ! -d "$SITE" ]]; then
  echo "FATAL: site-packages not found at $SITE" >&2
  exit 127
fi

export VIRTUAL_ENV="/runpod-volume/venvs/moncusoai"
export PYTHONPATH="$SITE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

echo "Starting with: $PY"
"$PY" -c "import sys,torch,torchvision; print(sys.executable); print('torch', torch.__version__, 'tv', torchvision.__version__, 'cuda', torch.cuda.is_available())"

exec "$PY" -u /rp_handler.py
