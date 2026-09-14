#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${MAIZEMASK_PYTHON:-python3}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

ACTUAL_PYTHON=$($PYTHON_BIN -c 'import platform; print(platform.python_version())')
if [ "$ACTUAL_PYTHON" != "3.12.3" ]; then
  echo "Warning: the paper environment used Python 3.12.3; found $ACTUAL_PYTHON." >&2
fi

cd "$ROOT"

# Some RunPod runtime images ship CUDA 12.8 under /usr/local/cuda-12.8 but
# deliberately omit it from PATH.  Detectron2 and Mask2Former compile custom
# CUDA extensions, so resolve the compiler explicitly rather than assuming the
# /usr/local/cuda symlink exists.
if ! command -v nvcc >/dev/null 2>&1; then
  for CUDA_CANDIDATE in /usr/local/cuda-12.8 /usr/local/cuda; do
    if [ -x "$CUDA_CANDIDATE/bin/nvcc" ]; then
      export CUDA_HOME="$CUDA_CANDIDATE"
      export PATH="$CUDA_HOME/bin:$PATH"
      break
    fi
  done
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is required to compile Detectron2 and Mask2Former CUDA extensions." >&2
  echo "Use a CUDA development image or install a CUDA toolkit matching Torch CUDA 12.8." >&2
  exit 4
fi
nvcc --version | tail -1

"$PYTHON_BIN" -m pip install --upgrade "pip==25.2" "setuptools==80.9.0" "wheel==0.45.1"

TORCH_OK=$($PYTHON_BIN - <<'PY'
try:
    import torch, torchvision
    print(int(torch.__version__ == "2.8.0+cu128" and torchvision.__version__ == "0.23.0+cu128"))
except Exception:
    print(0)
PY
)
if [ "$TORCH_OK" != "1" ]; then
  if [ "${MAIZEMASK_SKIP_TORCH_INSTALL:-0}" = "1" ]; then
    echo "This image does not provide torch 2.8.0+cu128 and torchvision 0.23.0+cu128." >&2
    echo "Unset MAIZEMASK_SKIP_TORCH_INSTALL to install the pinned cu128 wheels." >&2
    exit 3
  fi
  "$PYTHON_BIN" -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.8.0" "torchvision==0.23.0"
fi

"$PYTHON_BIN" -m pip install -r requirements.txt

# Compile the vendored, patched source against this exact Torch/CUDA pair.
export FORCE_CUDA=1
# Do not use `pip install -e` here.  With pip 25/setuptools 80 it invokes a
# nested PEP-517 build environment, which cannot see the image-provided Torch
# needed by Detectron2's setup.py.  Training and verification add this vendored
# source directory to sys.path, so an in-place extension build is both enough
# and deterministic for this bundle.
(
  cd third_party/detectron2
  "$PYTHON_BIN" setup.py build_ext --inplace
)
(
  cd third_party/Mask2Former/mask2former/modeling/pixel_decoder/ops
  "$PYTHON_BIN" setup.py build install
)

"$PYTHON_BIN" -m pip check
echo "Environment installation complete."
echo "Next: python scripts/prepare_pretrained_weights.py --cache-root .cache"
