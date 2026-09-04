#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3-pip build-essential git \
  libopencv-dev libopenblas-dev liblapack-dev libjpeg-dev libpng-dev \
  libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
  libnvinfer-bin libnvinfer-dev python3-libnvinfer

python3.10 -m venv "${VENV_DIR}"
# Jetson's NVIDIA-provided torch/torchvision wheels must be installed before this file's requirements.
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${PROJECT_DIR}/requirements.txt"
printf 'Environment ready. Activate with: source %s/bin/activate\n' "${VENV_DIR}"
