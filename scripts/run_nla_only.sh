#!/usr/bin/env bash
# The NLA-only arm on one 80 GB GPU: the adapter starts blank and learns from the reconstruction loss alone (no photo
# loss anywhere), then the same eval. Training reads only activations; photos are still downloaded once for the source
# activations and the eval.
# Usage: scripts/run_nla_only.sh WORK RUN [N_COCO N_CC3M STEPS MINUTES]
set -uo pipefail
cd "$(dirname "$0")/.."
WORK=$1; RUN=$2; NC=${3:-10000}; NW=${4:-10000}; S=${5:-3000}; MIN=${6:-100}
PY=${PYTHON:-python3}
mkdir -p "$WORK" "$RUN"
[ -f "$WORK/data.pt" ] || $PY -m vlens data "$WORK" "$NC" "$NW" 2>&1 | tee -a "$RUN/data.log"
[ -f "$WORK/data.pt" ] || { echo "VLENS data stage failed"; exit 1; }
[ -f "$RUN/con_N_fp16.pt" ] || VLENS_PURE=1 $PY -m vlens cycle "$WORK" "$RUN" "$S" "$MIN" 2>&1 | tee -a "$RUN/train_N.log"
[ -f "$WORK/con_N.pt" ] || { echo "VLENS NLA-only stage failed"; exit 1; }
$PY -m vlens eval "$WORK" "$RUN" 2>&1 | tee -a "$RUN/eval.log" && echo "VLENS done"
