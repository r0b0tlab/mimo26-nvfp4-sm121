#!/usr/bin/env bash
# Launch the Q200v2-protocol qualification on the live serve:
#   phase both = concurrency ladder c1/c2/c4/c8 (2 reps, 1024 tok, think-off)
#             + Q200 (GSM8K-200 think-off, 2048 tok cap, retry 4096)
# with the serve-host telemetry sampler running alongside.
# r0b0tlab mimo26.
set -u
C="$HOME/projects/mimo26-nvfp4-sm121"
cd "$C"
mkdir -p results/e2e results/q200
TEL_OUT="results/e2e/telemetry-$(date +%H%M).jsonl"
python3 scripts/64_telemetry.py "$TEL_OUT" 10 > results/e2e/telemetry.log 2>&1 &
TEL=$!
echo "telemetry_pid=$TEL out=$TEL_OUT"
python3 scripts/62_qualify.py --phase both --repeats 2 --max-tokens 1024 \
  --q-max-tokens 2048 --n 200 > results/e2e/QUALIFY.log 2>&1
echo "QUALIFY_EXIT=$?"
kill "$TEL" 2>/dev/null
wait "$TEL" 2>/dev/null
echo "Q200PROTO_DONE"
