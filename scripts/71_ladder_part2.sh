#!/usr/bin/env bash
# Overnight max-perf ladder part 2: waits for the S2 driver, then runs
# CELL-K, CELL-MOE, CELL-NCCL, and finally a COMBO of any winners is decided
# manually in the morning; part 2 restores the live winner profile at the end.
# r0b0tlab mimo26.
set -uo pipefail
C="$HOME/projects/mimo26-nvfp4-sm121"
cd "$C"
# wait for part-1 driver to finish (DRIVER_EXIT line appears)
for i in $(seq 1 200); do
  grep -q "DRIVER_EXIT" results/maxperf/DRIVER-S2.log 2>/dev/null && break
  sleep 15
done
echo "=== [$(date -u +%FT%TZ)] S2 driver finished; starting ladder part 2"
bash scripts/70_maxperf_driver.sh CELL-K CELL-K.env   > results/maxperf/DRIVER-K.log 2>&1
bash scripts/70_maxperf_driver.sh CELL-MOE CELL-MOE.env > results/maxperf/DRIVER-MOE.log 2>&1
bash scripts/70_maxperf_driver.sh CELL-NCCL CELL-NCCL.env > results/maxperf/DRIVER-NCCL.log 2>&1
echo "=== [$(date -u +%FT%TZ)] ladder part 2 complete; restoring live profile"
RESTORE=1 bash scripts/70_maxperf_driver.sh RESTORE MTP-500k-mm-graph.env > results/maxperf/DRIVER-RESTORE.log 2>&1
echo "=== [$(date -u +%FT%TZ)] LADDER DONE"
