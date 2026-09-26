#!/bin/bash
# One full systems invocation on the winning MTP profile. No --only. No seed files.
set -u
export R0B0BENCH_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false,"thinking":false}'
export R0B0BENCH_BFCL_PYTHON="$HOME/.venvs/r0b0bench/bin/python"
export R0B0BENCH_BFCL_SCRIPTS="/home/r0b0tdgx/projects/r0b0bench/scripts/bfcl"
export BFCL_NUM_THREADS=4
export BFCL_HTTP_TIMEOUT=3600
export BFCL_MAX_RETRIES=1
python3 -c 'import json,os; assert json.loads(os.environ["R0B0BENCH_CHAT_TEMPLATE_KWARGS"])=={"enable_thinking":False,"thinking":False}'
test -x "$R0B0BENCH_BFCL_PYTHON"
curl -sf -m 5 http://192.168.68.78:30000/v1/models | python3 -c 'import sys,json; m=json.load(sys.stdin)["data"][0]; assert m["id"]=="mimo26" and m["max_model_len"]==524288'
OUT=/home/r0b0tdgx/projects/mimo26-nvfp4-sm121/results/r0b0bench/MTP-500k-mm-systems
RUN=MTP-500k-mm-systems
mkdir -p "$OUT"
exec "$HOME/.venvs/r0b0bench/bin/r0b0bench" run \
  --profile systems \
  --base-url http://192.168.68.78:30000/v1 \
  --model mimo26 \
  --tokenizer "$HOME/models/r0b0tlab/MiMo-V2.6-Flash-RL-NVFP4" \
  --output "$OUT" \
  --run-id "$RUN" \
  --timeout 14400
