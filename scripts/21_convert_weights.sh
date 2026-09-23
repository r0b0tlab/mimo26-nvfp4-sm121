#!/usr/bin/env bash
# r0b0tlab mimo26-nvfp4-sm121: 3 parallel ModelOpt MXFP4->NVFP4 cast workers, then finalize
# (index / config / hf_quant_config manifest / aux hard links).  Resume-safe: finished shards are
# renamed into place atomically and skipped on rerun.
set -uo pipefail
C=$HOME/projects/mimo26-nvfp4-sm121
DST=$HOME/models/r0b0tlab/MiMo-V2.6-Flash-RL-NVFP4
SRC=$HOME/models/XiaomiMiMo/MiMo-V2.6-Flash-RL
PY=$HOME/.venvs/modelopt-main/bin/python
cd "$C"
mkdir -p "$DST"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[start] $(date -u +%FT%TZ) dst=$DST"
pids=()
for w in 0 1 2; do
  "$PY" scripts/20_convert_mimo_nvfp4.py --phase weights --src "$SRC" --dst "$DST" \
      --num_workers 3 --worker_id "$w" --verify_per_shard 2 > "logs/21_convert_w$w.log" 2>&1 &
  pids+=("$!")
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
echo "[workers] rc=$rc $(date -u +%FT%TZ)"
grep -h "lossless" logs/21_convert_w*.log | tail -3
[ "$rc" = 0 ] || { echo "[FAIL] worker failure"; exit "$rc"; }
"$PY" scripts/20_convert_mimo_nvfp4.py --phase weights --src "$SRC" --dst "$DST" 2>&1 | grep -v Warning | tail -3
echo "[done] $(date -u +%FT%TZ)"
