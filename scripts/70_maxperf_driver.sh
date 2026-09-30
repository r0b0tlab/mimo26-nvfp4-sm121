#!/usr/bin/env bash
# usage: 70_maxperf_driver.sh <cell-name>
# Overnight max-perf cell driver for the mimo26 TP=2 serve.
# Each cell: stop serve, boot profile with cell env, wait healthy, run
# 60_profile_bench.py (short/medium/prose/prefill), record JSON+log, move on.
# Guards (41b) are armed on both nodes and stay armed throughout.
# r0b0tlab mimo26.
set -uo pipefail
C="$HOME/projects/mimo26-nvfp4-sm121"
cd "$C"
mkdir -p results/maxperf logs/serve
SSHO="-i $HOME/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=$HOME/.ssh/known_hosts_crs812"
N4="r0b0tdgx@192.168.68.56"

run_cell() {  # run_cell <tag> <env-file>
  local tag="$1" envf="$2"
  echo "=== [$(date -u +%FT%TZ)] CELL $tag START ==="
  bash scripts/44_stop_serve.sh > "results/maxperf/stop.$tag.log" 2>&1
  sleep 5
  tmux kill-session -t mimo26-r0 2>/dev/null
  ssh $SSHO "$N4" "tmux kill-session -t mimo26-r1 2>/dev/null; true"
  tmux new -d -s mimo26-r0 "cd $C && set -a && . profiles/$envf && set +a && bash scripts/43_serve_profile.sh > logs/serve/launcher.$tag.log 2>&1"
  # wait for health up to 25 min
  local ok=0
  for i in $(seq 1 100); do
    sleep 15
    if curl -sf http://192.168.68.78:30000/health >/dev/null 2>&1; then ok=1; break; fi
    if ! tmux has-session -t mimo26-r0 2>/dev/null; then echo "=== CELL $tag RANK0 TMUX DIED"; break; fi
  done
  if [ "$ok" = 1 ]; then
    echo "=== [$(date -u +%FT%TZ)] CELL $tag HEALTHY after ~$((i*15))s"
    sleep 20   # settle + freeze_gc equivalent
    python3 scripts/60_profile_bench.py --url http://192.168.68.78:30000 \
      --tag "$tag" --out "results/maxperf/$tag.json" \
      > "results/maxperf/$tag.log" 2>&1
    echo "=== [$(date -u +%FT%TZ)] CELL $tag BENCH DONE rc=$?"
    tail -4 "results/maxperf/$tag.log"
  else
    echo "=== [$(date -u +%FT%TZ)] CELL $tag NEVER HEALTHY"
    tail -20 "$C/logs/serve/launcher.$tag.log" > "results/maxperf/$tag.FAIL.txt" 2>&1
  fi
}

run_cell "$@"

# restore the live winner profile at the end if RESTORE=1
if [ "${RESTORE:-0}" = "1" ]; then
  echo "=== restoring live profile MTP-500k-mm-graph"
  bash scripts/44_stop_serve.sh > results/maxperf/stop.RESTORE.log 2>&1
  sleep 5
  tmux new -d -s mimo26-r0 "cd $C && set -a && . profiles/MTP-500k-mm-graph.env && set +a && bash scripts/43_serve_profile.sh > logs/serve/launcher.RESTORE.log 2>&1"
fi
echo "=== [$(date -u +%FT%TZ)] DRIVER DONE"
