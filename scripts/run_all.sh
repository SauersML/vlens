#!/usr/bin/env bash
# The whole pipeline on one 80 GB GPU: data, stage B (flow matching), stage C (+ direct readback), eval, robustness.
# A stage whose output already exists is skipped, so a rerun resumes. If stage C fails, B is still evaluated.
# Usage: scripts/run_all.sh WORK RUN [N_COCO N_CC3M STEPS_B MINUTES_B STEPS_C MINUTES_C]
#   WORK  large files (data.pt with ~90k photo latents is ~18 GB, full-precision checkpoints)
#   RUN   small results (previews, fp16 adapters, logs, pictures, eval.json)
set -uo pipefail
cd "$(dirname "$0")/.."
WORK=$1; RUN=$2; NC=${3:-30000}; NW=${4:-60000}; SB=${5:-3000}; MB=${6:-75}; SC=${7:-1000}; MC=${8:-55}
PY=${PYTHON:-python3}
mkdir -p "$WORK" "$RUN"
[ -f "$WORK/data.pt" ] || $PY -m vlens data "$WORK" "$NC" "$NW" 2>&1 | tee -a "$RUN/data.log"
[ -f "$WORK/data.pt" ] || { echo "VLENS data stage failed"; exit 1; }
[ -f "$RUN/con_B_fp16.pt" ] || $PY -m vlens train "$WORK" "$RUN" "$SB" "$MB" 2>&1 | tee -a "$RUN/train_B.log"
[ -f "$WORK/con_B.pt" ] || { echo "VLENS stage B failed"; exit 1; }
[ -f "$RUN/con_C_fp16.pt" ] || $PY -m vlens cycle "$WORK" "$RUN" "$SC" "$MC" 2>&1 | tee -a "$RUN/train_C.log" || echo "VLENS stage C failed; evaluating B only"
$PY -m vlens eval "$WORK" "$RUN" 2>&1 | tee -a "$RUN/eval.log"
$PY -m vlens robust "$RUN" 1000 2>&1 | tee -a "$RUN/robust.log" && echo "VLENS done"
