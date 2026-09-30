#!/bin/bash
# Regenerate only BFCL-MT rows whose result is "Error during inference".
# Does not touch the 193 good rows. Official adapter modes: strip-errors, then resume.
set -u
export R0B0BENCH_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false,"thinking":false}'
export BFCL_PROJECT_ROOT=/home/r0b0tdgx/projects/mimo26-nvfp4-sm121/results/r0b0bench/FINAL3-500k-systems/FINAL3-500k-systems/lanes/bfcl_mt/bfcl-project
export OPENAI_BASE_URL=http://192.168.68.78:30000/v1
export OPENAI_API_KEY=EMPTY
export R0B0BENCH_SERVED_MODEL=mimo26
export BFCL_NUM_THREADS=4
export BFCL_HTTP_TIMEOUT=3600
export BFCL_MAX_RETRIES=1
PY="$HOME/.venvs/r0b0bench/bin/python"
SCRIPT=/home/r0b0tdgx/projects/r0b0bench/scripts/bfcl/bfcl_run.py
cp -a "$BFCL_PROJECT_ROOT/result/r0b0bench-openai-FC/multi_turn/BFCL_v4_multi_turn_base_result.json" \
  "$BFCL_PROJECT_ROOT/result/r0b0bench-openai-FC/multi_turn/BFCL_v4_multi_turn_base_result.json.bak-7timeouts"
echo "=== strip-errors ==="
"$PY" "$SCRIPT" strip-errors
echo "=== resume ==="
"$PY" "$SCRIPT" resume
echo "=== status ==="
"$PY" "$SCRIPT" status
