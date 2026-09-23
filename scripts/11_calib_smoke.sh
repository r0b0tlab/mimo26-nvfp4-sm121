#!/usr/bin/env bash
set -uo pipefail
C=~/projects/mimo26-nvfp4-sm121
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $C
~/.venvs/modelopt-main/bin/python scripts/10_calibrate_mimo.py \
  --src ~/models/XiaomiMiMo/MiMo-V2.6-Flash-RL --out $C/calib/smoke16 \
  --calib_size 16 --calib_seq 512 --batch 16
echo "EXIT=$?"
