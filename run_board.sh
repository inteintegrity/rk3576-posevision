#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/.deps${PYTHONPATH:+:$PYTHONPATH}"
exec "${BASKETBALL_PYTHON:-/home/seeed/rockchip-yolo11-quant-benchmark/.venv/bin/python}" -m uvicorn server:app --host 0.0.0.0 --port "${BASKETBALL_PORT:-8093}" --workers 1
