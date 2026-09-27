#!/usr/bin/env bash
# Serve-profile launcher: TP=2 across node 3 (rank 0) + node 4 (rank 1),
# NVFP4 MiMo-V2.6-Flash-RL. Live profile: profiles/MTP-500k-mm.env.
# SPEC=eagle is that MTP lane. profiles/DFLASH-500k.env is the DFlash lane.
# Profile knobs (env):
#   TAG           log tag (required)
#   MODEL         checkpoint (default: r0b0tlab NVFP4 export)
#   SPEC          dflash|eagle|none                (default eagle, the preferred profile)
#   DFLASH_BLOCK  DFlash verify block (draft trained with 8)  (default 8)
#   DRAFT_WINDOW  --speculative-draft-window-size   (unset = full)
#   SWA_RATIO     --swa-full-tokens-ratio           (unset = sglang 0.8)
#   CTX           --context-length                  (unset = model 1048576)
#   MEMFRAC       --mem-fraction-static             (default 0.90)
#   MAXRUN        --max-running-requests            (unset = sglang 48 w/ spec)
#   GRAPH_BS      cuda-graph max bs decode/prefill  (default 8)
#   CHUNK         --chunked-prefill-size            (unset = 8192)
#   MAXPREFILL    --max-prefill-tokens              (unset = 16384)
#   MOE_RUNNER    --moe-runner-backend              (default marlin)
#   KVDT          --kv-cache-dtype                  (default fp8_e4m3)
#   EXTRA         extra launch_server flags (word-split)
#   DENV          extra docker -e VAR=VAL entries (space separated)
#   MEMDIAG=1     mount scripts/diag sitecustomize (diagnostic boots only)
#   DRYRUN=1      print both rank command lines and exit
# Run ON node 3 (rank 0 runs in the foreground of this script); rank 1 is
# started in tmux mimo26-r1 on node 4.  Stop with 44_stop_serve.sh.
# r0b0tlab mimo26.
set -euo pipefail
C="$HOME/projects/mimo26-nvfp4-sm121"
TAG="${TAG:?TAG}"
MODEL="${MODEL:-$HOME/models/r0b0tlab/MiMo-V2.6-Flash-RL-NVFP4}"
SPEC="${SPEC:-eagle}"; DFLASH_BLOCK="${DFLASH_BLOCK:-8}"
MEMFRAC="${MEMFRAC:-0.90}"; GRAPH_BS="${GRAPH_BS:-8}"
MOE_RUNNER="${MOE_RUNNER:-marlin}"; KVDT="${KVDT:-fp8_e4m3}"
IMG="${IMG:-r0b0tlab/sglang-mimo26-env:20260922-582389ce}"
N3IP="192.168.68.78"; N4IP="192.168.68.56"
mkdir -p "$C/logs/serve"

ARGS=(--model-path "$MODEL" --served-model-name mimo26
      --tp-size 2 --ep-size 2 --dist-init-addr "$N3IP:20000"
      --host 0.0.0.0 --port 30000 --mem-fraction-static "$MEMFRAC"
      --moe-runner-backend "$MOE_RUNNER" --kv-cache-dtype "$KVDT"
      --trust-remote-code
      --cuda-graph-max-bs-decode "$GRAPH_BS" --cuda-graph-max-bs-prefill "$GRAPH_BS")
case "$SPEC" in
  dflash) ARGS+=(--speculative-algorithm DFLASH
                 --speculative-draft-model-path "$MODEL/dflash"
                 --speculative-dflash-block-size "$DFLASH_BLOCK")
          [ -n "${DRAFT_WINDOW:-}" ] && ARGS+=(--speculative-draft-window-size "$DRAFT_WINDOW") ;;
  eagle)  # In-checkpoint MTP. Draft path is the target tree; the loader
          # rewrites the draft arch to MiMoV2MTP. Steps match
          # num_nextn_predict_layers=3. Pin the draft window or the draft
          # KV pool is sized to the full pool.
          ARGS+=(--speculative-algorithm EAGLE
                 --speculative-draft-model-path "$MODEL"
                 --speculative-num-steps 3
                 --speculative-eagle-topk 1
                 --speculative-num-draft-tokens 4
                 --enable-multi-layer-eagle)
          [ -n "${DRAFT_WINDOW:-}" ] && ARGS+=(--speculative-draft-window-size "$DRAFT_WINDOW") ;;
  none)   ;;
  *)      echo "bad SPEC=$SPEC (dflash|eagle|none)" >&2; exit 2 ;;
esac
[ -n "${SWA_RATIO:-}" ]  && ARGS+=(--swa-full-tokens-ratio "$SWA_RATIO")
[ -n "${CTX:-}" ]        && ARGS+=(--context-length "$CTX")
[ -n "${MAXRUN:-}" ]     && ARGS+=(--max-running-requests "$MAXRUN")
[ -n "${CHUNK:-}" ]      && ARGS+=(--chunked-prefill-size "$CHUNK")
[ -n "${MAXPREFILL:-}" ] && ARGS+=(--max-prefill-tokens "$MAXPREFILL")
# shellcheck disable=SC2206
[ -n "${EXTRA:-}" ]      && ARGS+=($EXTRA)

DOCKER_ENV=(-e NCCL_IB_HCA='=rocep1s0f0:1,roceP2p1s0f0:1' -e NCCL_IB_GID_INDEX=3
            -e NCCL_SOCKET_IFNAME=enP7s7 -e GLOO_SOCKET_IFNAME=enP7s7)
for kv in ${DENV:-}; do DOCKER_ENV+=(-e "$kv"); done
if [ "${MEMDIAG:-0}" = "1" ]; then
  DOCKER_ENV+=(-e MEMDIAG=1 -e "MEMDIAG_OUTDIR=$C/logs/serve" -e "PYTHONPATH=$C/scripts/diag")
fi
MOUNTS=(-v "$HOME/models:$HOME/models:ro" -v "$C:$C"
        -v "$C/cache/root-cache:/root/.cache" -v "$C/cache/root-triton:/root/.triton")
# The JIT caches (sglang-jit / flashinfer / triton / tvm-ffi) must persist across
# --rm boots: a cold compile during CUDA-graph capture spikes host RAM by >7 GiB
# on GB10 unified memory (two guard trips on node 3 at mem-frac 0.90).
# PATCH_SWIZZLE=1: one-file fix for sglang 582389ce - the NVFP4 MoE method
# allocates *_blockscale_swizzled placeholders even on the marlin path, which
# never reads them (~9 GiB/rank here).  See patches/sglang-582389ce/*.diff.
if [ "${PATCH_SWIZZLE:-0}" = "1" ]; then
  MOUNTS+=(-v "$C/patches/sglang-582389ce/modelopt_quant.py:/sgl-workspace/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py:ro")
fi
# MTP draft QKV scales need the same deferred dequant/requant path as the target.
# The draft attention module is SWA-shaped; the stock loader chunks the scale and dies.
if [ "$SPEC" = "eagle" ]; then
  MOUNTS+=(-v "$C/patches/sglang-582389ce/mimo_v2.py:/sgl-workspace/sglang/python/sglang/srt/models/mimo_v2.py:ro" \
           -v "$C/patches/sglang-582389ce/mimo_v2_nextn.py:/sgl-workspace/sglang/python/sglang/srt/models/mimo_v2_nextn.py:ro" \
           -v "$C/patches/sglang-582389ce/triton_backend.py:/sgl-workspace/sglang/python/sglang/srt/layers/attention/triton_backend.py:ro")
fi

rank_args() {  # rank_args <node-rank> <tp-rank> -> docker argv on ONE line (shell-quoted)
  local nr="$1" tr="$2"
  printf '%q ' docker run --rm --name "mimo26-r$tr" --ipc host --network host --gpus all \
    "${MOUNTS[@]}" "${DOCKER_ENV[@]}" \
    "$IMG" python3 -m sglang.launch_server "${ARGS[@]}" --nnodes 2 --node-rank "$nr"
  echo
}
RANK1=$(rank_args 1 1)
RANK0=$(rank_args 0 0)
if [ "${DRYRUN:-0}" = "1" ]; then echo "rank1: $RANK1"; echo "rank0: $RANK0"; exit 0; fi

printf '%s\n' "# TAG=$TAG $(date -u +%FT%TZ)" "$RANK0" > "$C/logs/serve/cmd.$TAG.txt"
printf '%s' "$RANK1" | base64 -w0 > /tmp/mimo26_r1_cmd.b64
scp -q -o StrictHostKeyChecking=yes /tmp/mimo26_r1_cmd.b64 "r0b0tdgx@$N4IP:/tmp/"
ssh -n -o StrictHostKeyChecking=yes "r0b0tdgx@$N4IP" \
  "base64 -d /tmp/mimo26_r1_cmd.b64 > /tmp/mimo26_r1_cmd.sh && chmod +x /tmp/mimo26_r1_cmd.sh && tmux new -d -s mimo26-r1 'mkdir -p $C/logs/serve && bash /tmp/mimo26_r1_cmd.sh > $C/logs/serve/r1.$TAG.log 2>&1'"
eval "$RANK0" > "$C/logs/serve/r0.$TAG.log" 2>&1 &
R0PID=$!
for i in $(seq 1 240); do
  sleep 15
  if curl -sf "http://$N3IP:30000/health" >/dev/null; then
    echo "[ok] $TAG healthy after $((i*15))s"
    wait "$R0PID"; exit 0
  fi
  if ! kill -0 "$R0PID" 2>/dev/null; then
    echo "[fail] rank0 died; r0 tail:"; tail -30 "$C/logs/serve/r0.$TAG.log"; exit 1
  fi
  if ! ssh -n "r0b0tdgx@$N4IP" "tmux has-session -t mimo26-r1 2>/dev/null"; then
    echo "[fail] rank1 tmux died; r1 tail:"; ssh -n "r0b0tdgx@$N4IP" "tail -30 $C/logs/serve/r1.$TAG.log"; exit 1
  fi
done
echo "[fail] timeout 3600s"; tail -30 "$C/logs/serve/r0.$TAG.log"; exit 1
