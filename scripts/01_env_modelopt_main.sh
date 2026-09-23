#!/usr/bin/env bash
# Fresh ModelOpt-main venv for the MiMo-V2.6 NVFP4 campaign (does not touch ~/.venvs/modelopt).
set -euo pipefail
C=~/projects/mimo26-nvfp4-sm121; V=~/.venvs/modelopt-main; S=~/src/Model-Optimizer-main
exec > >(tee -a $C/logs/01_env_modelopt_main.log) 2>&1
echo "[start] $(date -u +%FT%TZ)"
MAIN_SHA=$(git ls-remote https://github.com/NVIDIA/Model-Optimizer refs/heads/main | cut -f1)
echo "modelopt main at execute: $MAIN_SHA"
if [ ! -d $S/.git ]; then git clone https://github.com/NVIDIA/Model-Optimizer $S; fi
git -C $S fetch origin main && git -C $S checkout -q --detach "$MAIN_SHA"
git -C $S log -1 --format='checked out %H %cI %s'
python3 -m venv $V
$V/bin/pip install -q -U pip wheel "setuptools>=80"
$V/bin/pip install -q torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
$V/bin/pip install -q -e "$S[hf]"
$V/bin/python - <<'PY'
import torch, transformers, modelopt, accelerate, datasets, safetensors, sys
print("python", sys.version.split()[0]); print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("transformers", transformers.__version__, "modelopt", modelopt.__version__, "accelerate", accelerate.__version__, "datasets", datasets.__version__, "safetensors", safetensors.__version__)
from modelopt.torch.quantization.calib.nvfp4_act_headroom import NVFP4ActHeadroomCalibrator
from modelopt.torch.quantization.utils.numeric_utils import mxfp4_to_nvfp4_global_amax
print("headroom calibrator + mxfp4 cast helpers import OK")
PY
$V/bin/pip freeze > $C/evidence/pip_freeze_modelopt_main.txt
echo "$MAIN_SHA" > $C/evidence/modelopt_main_sha.txt
echo "[done] $(date -u +%FT%TZ)"
