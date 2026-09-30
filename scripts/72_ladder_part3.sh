#!/usr/bin/env bash
# Ladder part 3: waits for ladder2 (which ends with a RESTORE boot), then
# runs the remaining single-variable cells and restores the live profile.
# r0b0tlab mimo26.
set -uo pipefail
C="$HOME/projects/mimo26-nvfp4-sm121"
cd "$C"
for i in $(seq 1 400); do
  grep -q "LADDER DONE" results/maxperf/LADDER2.log 2>/dev/null && break
  sleep 20
done
echo "=== [$(date -u +%FT%TZ)] ladder2 complete; starting ladder part 3"
bash scripts/70_maxperf_driver.sh CELL-KV16 CELL-KV16.env    > results/maxperf/DRIVER-KV16.log 2>&1
bash scripts/70_maxperf_driver.sh CELL-CHUNK16 CELL-CHUNK16.env > results/maxperf/DRIVER-CHUNK16.log 2>&1
bash scripts/70_maxperf_driver.sh CELL-CONS CELL-CONS.env    > results/maxperf/DRIVER-CONS.log 2>&1
echo "=== [$(date -u +%FT%TZ)] part 3 cells done; final restore"
RESTORE=1 bash scripts/70_maxperf_driver.sh RESTORE MTP-500k-mm-graph.env > results/maxperf/DRIVER-RESTORE3.log 2>&1
echo "=== [$(date -u +%FT%TZ)] LADDER3 DONE"
