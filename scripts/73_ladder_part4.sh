#!/usr/bin/env bash
# Ladder part 4: waits for ladder3 (ends with RESTORE3 DONE), runs the last
# three cells, restores the live profile, and writes the final tally.
# r0b0tlab mimo26.
set -uo pipefail
C="$HOME/projects/mimo26-nvfp4-sm121"
cd "$C"
for i in $(seq 1 600); do
  grep -q "LADDER3 DONE" results/maxperf/LADDER3.log 2>/dev/null && break
  sleep 20
done
echo "=== [$(date -u +%FT%TZ)] ladder3 complete; starting ladder part 4"
bash scripts/70_maxperf_driver.sh CELL-TC CELL-TC.env      > results/maxperf/DRIVER-TC.log 2>&1
bash scripts/70_maxperf_driver.sh CELL-SI8 CELL-SI8.env    > results/maxperf/DRIVER-SI8.log 2>&1
bash scripts/70_maxperf_driver.sh CELL-NCCLCH CELL-NCCLCH.env > results/maxperf/DRIVER-NCCLCH.log 2>&1
echo "=== [$(date -u +%FT%TZ)] part 4 cells done; final restore"
RESTORE=1 bash scripts/70_maxperf_driver.sh RESTORE MTP-500k-mm-graph.env > results/maxperf/DRIVER-RESTORE4.log 2>&1
echo "=== [$(date -u +%FT%TZ)] LADDER4 DONE"
