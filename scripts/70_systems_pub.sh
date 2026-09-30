#!/bin/bash
# One full systems invocation. No --only.
# BFCL-MT result file is the repaired 200-row file (7 timeouts regenerated).
# Generation skips those ids (allow_overwrite=false). This process still validates,
# scores, and runs every other lane. Timing sidecar drops the 7 timeout rows;
# the retried answers are in the result file, not invented timing rows.
set -u
export R0B0BENCH_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false,"thinking":false}'
export R0B0BENCH_BFCL_PYTHON="$HOME/.venvs/r0b0bench/bin/python"
export R0B0BENCH_BFCL_SCRIPTS="/home/r0b0tdgx/projects/r0b0bench/scripts/bfcl"
export BFCL_NUM_THREADS=4
export BFCL_HTTP_TIMEOUT=3600
export BFCL_MAX_RETRIES=1
python3 -c 'import json,os; assert json.loads(os.environ["R0B0BENCH_CHAT_TEMPLATE_KWARGS"])=={"enable_thinking":False,"thinking":False}'
test -x "$R0B0BENCH_BFCL_PYTHON"
test -f "$R0B0BENCH_BFCL_SCRIPTS/bfcl_run.py"
test -f "$R0B0BENCH_BFCL_SCRIPTS/bfcl_ast_run.py"

OUT=/home/r0b0tdgx/projects/mimo26-nvfp4-sm121/results/r0b0bench/FINAL3-500k-systems-pub
RUN=FINAL3-500k-systems-pub
SRC=/home/r0b0tdgx/projects/mimo26-nvfp4-sm121/results/r0b0bench/FINAL3-500k-systems/FINAL3-500k-systems/lanes/bfcl_mt/bfcl-project
PROJ="$OUT/$RUN/lanes/bfcl_mt/bfcl-project"
mkdir -p "$PROJ/result/r0b0bench-openai-FC/multi_turn"
cp -a "$SRC/result/r0b0bench-openai-FC/multi_turn/BFCL_v4_multi_turn_base_result.json" \
  "$PROJ/result/r0b0bench-openai-FC/multi_turn/BFCL_v4_multi_turn_base_result.json"
python3 - <<'PY'
import json
from pathlib import Path
src=Path("/home/r0b0tdgx/projects/mimo26-nvfp4-sm121/results/r0b0bench/FINAL3-500k-systems/FINAL3-500k-systems/lanes/bfcl_mt/bfcl-project/e2e-requests.jsonl")
dst=Path("/home/r0b0tdgx/projects/mimo26-nvfp4-sm121/results/r0b0bench/FINAL3-500k-systems-pub/FINAL3-500k-systems-pub/lanes/bfcl_mt/bfcl-project/e2e-requests.jsonl")
kept=dropped=0
with src.open() as inp, dst.open("w") as out:
    for line in inp:
        if not line.strip():
            continue
        row=json.loads(line)
        if int(row.get("http_status") or 0) != 200 or row.get("error"):
            dropped += 1
            continue
        out.write(line if line.endswith("\n") else line+"\n")
        kept += 1
res=Path("/home/r0b0tdgx/projects/mimo26-nvfp4-sm121/results/r0b0bench/FINAL3-500k-systems-pub/FINAL3-500k-systems-pub/lanes/bfcl_mt/bfcl-project/result/r0b0bench-openai-FC/multi_turn/BFCL_v4_multi_turn_base_result.json")
rows=[json.loads(l) for l in res.read_text().splitlines() if l.strip()]
errs=sum(1 for r in rows if isinstance(r.get("result"), str) and str(r["result"]).startswith("Error during inference:"))
print(f"seed_rows={len(rows)} seed_errors={errs} timing_kept={kept} timing_dropped={dropped}")
if len(rows) != 200 or errs or kept == 0:
    raise SystemExit(2)
PY

mkdir -p "$OUT"
python3 /home/r0b0tdgx/projects/mimo26-nvfp4-sm121/scripts/64_telemetry.py "$OUT/telemetry.jsonl" 2 &
echo "telemetry_pid=$!"
exec "$HOME/.venvs/r0b0bench/bin/r0b0bench" run \
  --profile systems \
  --base-url http://192.168.68.78:30000/v1 \
  --model mimo26 \
  --tokenizer "$HOME/models/r0b0tlab/MiMo-V2.6-Flash-RL-NVFP4" \
  --output "$OUT" \
  --run-id "$RUN" \
  --timeout 14400
