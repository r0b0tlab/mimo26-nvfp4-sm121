#!/usr/bin/env bash
# Mirror the NVFP4 export from node 3 to node 4 over the fabric, then verify.
# r0b0tlab mimo26.  Runs ON NODE 3 (invoked inside tmux).
set -euo pipefail
C="$HOME/projects/mimo26-nvfp4-sm121"
DST="$HOME/models/r0b0tlab/MiMo-V2.6-Flash-RL-NVFP4"
N4="r0b0tdgx@192.168.100.4"

n_src=$(ls "$DST"/*.safetensors | wc -l)
echo "[start] $(date -u +%T) files=$n_src"

rsync -a --info=progress2 -e "ssh -o StrictHostKeyChecking=yes" \
      "$DST/" "$N4:$DST/"

n_dst=$(ssh -n "$N4" "ls '$DST'/*.safetensors | wc -l")
[ "$n_src" = "$n_dst" ] || { echo "[MIRROR_FAIL] count $n_src != $n_dst"; exit 1; }

ssh -n "$N4" "cd '$DST' && sha256sum config.json model.safetensors.index.json kv_scales.json model-nvfp4-input-scales.safetensors" > /tmp/n4_hashes.txt
(cd "$DST" && sha256sum config.json model.safetensors.index.json kv_scales.json model-nvfp4-input-scales.safetensors) > /tmp/n3_hashes.txt
diff /tmp/n3_hashes.txt /tmp/n4_hashes.txt || { echo "[MIRROR_FAIL] hash mismatch"; exit 1; }

echo "[MIRROR_OK] $(date -u +%T) files=$n_src"
