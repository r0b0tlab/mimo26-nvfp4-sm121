#!/usr/bin/env bash
# Stop the comparison serve and boot the vendor MXFP4 checkpoint
# with the same EAGLE MTP flags. KV stays bf16: this checkpoint has no scales.
set -euo pipefail
C="${HOME}/projects/mimo26-nvfp4-sm121"
bash "$C/scripts/44_stop_serve.sh"
ssh -i "$HOME/.ssh/id_ed25519_shared" -o IdentitiesOnly=yes -o BatchMode=yes \
  r0b0tdgx@192.168.68.78 'bash -s' << 'EOF'
set -euo pipefail
C=$HOME/projects/mimo26-nvfp4-sm121
mkdir -p "$C/logs/serve"
tmux kill-session -t mimo26-cmp 2>/dev/null || true
tmux new -d -s mimo26-cmp "cd $C && \
  TAG=cmp-vendor-mtp \
  MODEL=\$HOME/models/XiaomiMiMo/MiMo-V2.6-Flash-RL \
  SPEC=eagle DRAFT_WINDOW=4096 PATCH_SWIZZLE=1 SWA_RATIO=0.02 \
  MEMFRAC=0.90 MAXRUN=8 CTX=524288 KVDT=bf16 \
  IMG=r0b0tlab/sglang-mimo26-env:20260922-582389ce \
  EXTRA='--tool-call-parser mimo --reasoning-parser mimo --enable-multimodal --mm-attention-backend triton_attn --image-processor-backend torchvision' \
  bash scripts/43_serve_profile.sh > logs/serve/cmp-vendor-mtp.launcher.log 2>&1"
sleep 2
docker ps --filter name=mimo26 --format '{{.Names}} {{.Status}}'
EOF
