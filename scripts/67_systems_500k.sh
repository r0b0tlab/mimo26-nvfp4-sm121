#!/bin/bash
# Full systems publication run. No --only. Think-off. Client off the serve node.
set -u
export R0B0BENCH_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false,"thinking":false}'
export R0B0BENCH_BFCL_PYTHON="$HOME/.venvs/r0b0bench/bin/python"
export R0B0BENCH_BFCL_SCRIPTS="/home/r0b0tdgx/projects/r0b0bench/scripts/bfcl"
export BFCL_NUM_THREADS=4
# Adapter setdefault is 600s with 1 retry, so a slow multi-turn step dies at 1200s
# and the kit counts that as an inference error. Long-context turns already took 1190s.
export BFCL_HTTP_TIMEOUT=3600
export BFCL_MAX_RETRIES=1
python3 -c 'import json,os; assert json.loads(os.environ["R0B0BENCH_CHAT_TEMPLATE_KWARGS"])=={"enable_thinking":False,"thinking":False}'
test -x "$R0B0BENCH_BFCL_PYTHON"
test -f "$R0B0BENCH_BFCL_SCRIPTS/bfcl_run.py"
test -f "$R0B0BENCH_BFCL_SCRIPTS/bfcl_ast_run.py"
OUT=/home/r0b0tdgx/projects/mimo26-nvfp4-sm121/results/r0b0bench/FINAL3-500k-systems-b
mkdir -p "$OUT"
python3 /home/r0b0tdgx/projects/mimo26-nvfp4-sm121/scripts/64_telemetry.py "$OUT/telemetry.jsonl" 2 &
echo "telemetry_pid=$!"
exec "$HOME/.venvs/r0b0bench/bin/r0b0bench" run \
  --profile systems \
  --base-url http://192.168.68.78:30000/v1 \
  --model mimo26 \
  --tokenizer "$HOME/models/r0b0tlab/MiMo-V2.6-Flash-RL-NVFP4" \
  --output "$OUT" \
  --run-id FINAL3-500k-systems-b \
  --timeout 14400
