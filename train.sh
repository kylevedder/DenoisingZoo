#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

# Ensure uv is available
if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' is not installed. Install via Homebrew: brew install uv" >&2
  exit 1
fi

# Create venv if missing
if [[ ! -d .venv ]]; then
  uv venv --python 3.12 .venv
fi

# Activate venv
# shellcheck disable=SC1091
source .venv/bin/activate

# Sync dependencies without installing the project
uv sync --no-install-project

# Detect MPS availability
DEVICE_ARG="device=mps"
echo "[train.sh] torch version: $(python - <<'PY'
import torch
print(torch.__version__)
PY
)"
echo "[train.sh] mps available: $(python - <<'PY'
import torch
print(bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()))
PY
)"
MPS_OR_CPU=$(python - <<'PY'
import torch
print('mps' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 'cpu')
PY
)
if [[ "${MPS_OR_CPU}" != "mps" ]]; then
  DEVICE_ARG="device=cpu"
fi
echo "[train.sh] selected device: ${DEVICE_ARG#device=}"

# Run training (pass through any extra CLI args after this script)
python train.py "${DEVICE_ARG}" "$@"
