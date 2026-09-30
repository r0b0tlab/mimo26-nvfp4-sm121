#!/bin/bash
# Publication client. Off the serve node. Think-off pin is explicit.
set -u
C=/home/r0b0tdgx/projects/mimo26-nvfp4-sm121
OUT=$C/results/r0b0bench/FINAL3-20260925
RB=~/.venvs/r0b0bench/bin/r0b0bench
PY=$C/scripts/q200_py
KIT=/home/r0b0tdgx/projects/r0b0bench/subsets/q200v2
mkdir -p "$OUT/systems" "$OUT/q200v2"
export R0B0BENCH_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false,"thinking":false}'
python3 -c 'import json,os; json.loads(os.environ["R0B0BENCH_CHAT_TEMPLATE_KWARGS"]); print("kwargs-ok")'

python3 "$C/scripts/64_telemetry.py" "$OUT/systems/telemetry.jsonl" 2 &
TEL=$!
echo "telemetry_pid=$TEL"
set +e
"$RB" run \
  --profile systems \
  --only canary,latency,concurrency,throughput \
  --base-url http://192.168.68.78:30000/v1 \
  --model mimo26 \
  --output "$OUT/systems" \
  --run-id FINAL3-systems-20260925 \
  --timeout 900
echo "systems_rc=$?"
kill "$TEL" 2>/dev/null
wait "$TEL" 2>/dev/null
set -e

curl -sf -m 10 -o /dev/null http://192.168.68.78:30000/health || { echo "health-failed-before-q200"; exit 3; }

python3 "$C/scripts/64_telemetry.py" "$OUT/q200v2/telemetry.jsonl" 2 &
TEL=$!
echo "q200_telemetry_pid=$TEL"
cd "$OUT/q200v2"
set +e
PYTHONPATH="$PY" python3 "$KIT/scripts/run_quality_set.py" \
  --base-url http://192.168.68.78:30000 \
  --run-id FINAL3-q200v2 \
  --set "$KIT/artifacts/quality-text-180-v2.jsonl" \
  --model mimo26 \
  --max-tokens 8192 \
  --timeout 900 \
  --workers 1 \
  --image-id sha256:9ab175696a136534026ebcc7f94d7f3ab5b4bb1970706424054788a85ad26eb0 \
  --profile-id ccd6a7e972ae5a05001c3316f2c2ceb21e81400597dc158902ca380131ff5799 \
  --candidate-id mimo26-nvfp4-final3 \
  --admission-config "$OUT/admission.json" \
  --chat-template-kwargs '{"enable_thinking":false,"thinking":false}'
echo "q200_rc=$?"
kill "$TEL" 2>/dev/null
wait "$TEL" 2>/dev/null
echo "publication-client-done"
