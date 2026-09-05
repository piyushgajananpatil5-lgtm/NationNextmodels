#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3-pip build-essential git \
  libopencv-dev libopenblas-dev liblapack-dev libjpeg-dev libpng-dev \
  libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
  libnvinfer-bin libnvinfer-dev python3-libnvinfer

python3.10 -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
# PyPI's torch wheels are not CUDA-enabled for Jetson ARM64. Set this to the
# wheel matching the board's JetPack version before running this script.
: "${JETSON_TORCH_WHEEL_URL:?Set JETSON_TORCH_WHEEL_URL to NVIDIA's Jetson PyTorch wheel URL}"
python -m pip install --no-cache-dir "${JETSON_TORCH_WHEEL_URL}"
if [[ -n "${JETSON_TORCHVISION_WHEEL_URL:-}" ]]; then
  python -m pip install --no-cache-dir "${JETSON_TORCHVISION_WHEEL_URL}"
fi
python -m pip install -r "${PROJECT_DIR}/requirements.txt"
python -c "import tensorrt, torch; assert torch.cuda.is_available(), 'Jetson CUDA is unavailable in this environment'"
printf 'Environment ready. Activate with: source %s/bin/activate\n' "${VENV_DIR}"
