#!/usr/bin/env bash
# TP=2 across nodes 3 (rank 0) + 4 (rank 1) over mgmt enP7s7, RDMA on the two
# RoCE rails.  Env selects the lane:
#   TAG=<name> MODEL=<ckpt-path> SPEC=none|dflash  KV=default|dtype-only|fp8scales|fp8plain
#   DRYRUN=1 prints the two rank command lines and exits.
# Older DFlash helper. Not the preferred launcher.
# Preferred serve is scripts/43_serve_profile.sh with profiles/MTP-500k-mm.env
# (EAGLE MTP, 3 steps, 4 draft tokens). This script does not launch that profile.
set -euo pipefail
C="$HOME/projects/mimo26-nvfp4-sm121"
TAG="${TAG:?TAG}"; MODEL="${MODEL:?MODEL}"; SPEC="${SPEC:-none}"; KV="${KV:-default}"
MOE_RUNNER="${MOE_RUNNER:-auto}"
IMG="${IMG:-r0b0tlab/sglang-mimo26-env:20260922-582389ce}"
N3IP="192.168.68.78"; N4IP="192.168.68.56"
mkdir -p "$C/logs/serve"

SPEC_FLAGS=()
case "$SPEC" in
  dflash) SPEC_FLAGS=(--speculative-algorithm DFLASH
                      --speculative-draft-model-path "$MODEL/dflash"
                      --speculative-dflash-block-size 2
                      --speculative-num-steps 1 --speculative-eagle-topk 1
                      --speculative-num-draft-tokens 2) ;;
  none)   ;;
  eagle)  echo "SPEC=eagle is not this script. Use scripts/43_serve_profile.sh with profiles/MTP-500k-mm.env" >&2; exit 2 ;;
  *)      echo "bad SPEC=$SPEC" >&2; exit 2 ;;
esac

KV_FLAGS=()
case "$KV" in
  dtype-only) KV_FLAGS=(--kv-cache-dtype nvfp4) ;;       # P5.5 evidence run
  fp8scales)  KV_FLAGS=(--kv-cache-dtype fp8_e4m3) ;;    # scales via config.json pointer
  fp8plain)   KV_FLAGS=(--kv-cache-dtype fp8_e4m3) ;;    # after 30_kv_scales.py off
  default)    ;;
  *)          echo "bad KV=$KV" >&2; exit 2 ;;
esac

COMMON=(--model-path "$MODEL" --served-model-name mimo26
        --tp-size 2 --ep-size 2 --dist-init-addr "$N3IP:20000"
        --host 0.0.0.0 --port 30000 --mem-fraction-static 0.90
        --moe-runner-backend "$MOE_RUNNER"
        --trust-remote-code --cuda-graph-max-bs-decode 8 --cuda-graph-max-bs-prefill 8)

rank_args() {  # rank_args <node-rank> <tp-rank> -> prints docker argv on ONE line
  local nr="$1" tr="$2"
  printf '%s ' docker run --rm --name "mimo26-r$tr" --ipc host --network host --gpus all \
    -v "$HOME/models:$HOME/models:ro" -v "$C:$C" \
    -e NCCL_IB_HCA='=rocep1s0f0:1,roceP2p1s0f0:1' -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_SOCKET_IFNAME=enP7s7 -e GLOO_SOCKET_IFNAME=enP7s7 \
    "$IMG" python3 -m sglang.launch_server "${COMMON[@]}" "${KV_FLAGS[@]}" "${SPEC_FLAGS[@]}" \
    --nnodes 2 --node-rank "$nr"
  echo
}

RANK1=$(rank_args 1 1)
RANK0=$(rank_args 0 0)

# The nightly image lacks torchcodec, so the MiMo-V2 mm processor import is
# swallowed and MiMoV2ForCausalLM never registers a processor (boot dies with
# "No processor registered"). Install it before launch (pip caches on first
# node only, near-instant afterwards).
ensure_torchcodec() {
  docker run --rm --entrypoint bash "$IMG" -c \
    'python3 -c "import torchcodec" 2>/dev/null || pip install -q torchcodec'
}

if [ "${DRYRUN:-0}" = "1" ]; then
  echo "rank1: $RANK1"
  echo "rank0: $RANK0"
  exit 0
fi

# rank 1 on node 4 first (it connects back to init on node 3):
# (torchcodec preflight runs on both nodes from the orchestrator before the
# lane starts — see scripts/40_preflight_env.sh; kept out of this script to
# avoid nested-quoting issues in the rank-1 ssh line.)
printf '%s' "$RANK1" | base64 -w0 > /tmp/mimo26_r1_cmd.b64
scp -q -o StrictHostKeyChecking=yes /tmp/mimo26_r1_cmd.b64 "r0b0tdgx@$N4IP:/tmp/"
ssh -n -o StrictHostKeyChecking=yes "r0b0tdgx@$N4IP" \
  "base64 -d /tmp/mimo26_r1_cmd.b64 > /tmp/mimo26_r1_cmd.sh && chmod +x /tmp/mimo26_r1_cmd.sh && tmux new -d -s mimo26-r1 'mkdir -p $C/logs/serve && bash /tmp/mimo26_r1_cmd.sh > $C/logs/serve/r1.$TAG.log 2>&1'"
# rank 0 on node 3 (this node), foreground inside the caller's tmux:
$RANK0 > "$C/logs/serve/r0.$TAG.log" 2>&1 &
R0PID=$!
for i in $(seq 1 240); do
  sleep 15
  if curl -sf "http://$N3IP:30000/health" >/dev/null; then
    echo "[ok] healthy after $((i*15))s"
    wait "$R0PID"
    exit 0
  fi
  if ! kill -0 "$R0PID" 2>/dev/null; then
    echo "[fail] rank0 died; r0 tail:"
    tail -30 "$C/logs/serve/r0.$TAG.log"
    exit 1
  fi
  if ! ssh -n "r0b0tdgx@$N4IP" "tmux has-session -t mimo26-r1 2>/dev/null"; then
    echo "[fail] rank1 tmux died; r1 tail:"
    ssh -n "r0b0tdgx@$N4IP" "tail -30 $C/logs/serve/r1.$TAG.log"
    exit 1
  fi
done
echo "[fail] timeout 3600s; r0 tail:"
tail -30 "$C/logs/serve/r0.$TAG.log"
exit 1
