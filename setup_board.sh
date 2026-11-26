#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt 'setuptools<81'
.venv/bin/python -m pip install https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
printf '%s\n' 'Start with: BASKETBALL_PYTHON="$PWD/.venv/bin/python" bash run_board.sh'
