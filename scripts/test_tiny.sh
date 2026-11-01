#!/usr/bin/env bash
# Plumbing test of every stage in minutes on a laptop GPU (Apple MPS or CUDA): a small random Qwen3-VL with the real
# processor, the real FLUX.2 [klein] 4B, 6 COCO training photos, 4 test photos, no CC3M, 3 + 2 training steps.
# Usage: scripts/test_tiny.sh [WORK]   (set VLENS_COCO_DIR to a folder with the two coco-karpathy parquet files to skip
# the Hub download)
set -e
cd "$(dirname "$0")/.."
export VLENS_TINY=1 VLENS_NTE=4 VLENS_NEXTRA=3 VLENS_B=2 VLENS_BC=1
W=${1:-$PWD/tiny_run}
PY=${PYTHON:-python3}
[ -f "$W/data.pt" ] || $PY -m vlens data "$W" 6 0
$PY -m vlens train "$W" "$W/run" 3 5
$PY -m vlens cycle "$W" "$W/run" 2 5
$PY -m vlens eval "$W" "$W/run"
echo TINY_OK
