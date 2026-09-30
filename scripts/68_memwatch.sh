#!/bin/bash
# Stop the systems client if either serve node drops below 4 GiB. Do not kill the serve.
set -u
source /home/r0b0tdgx/projects/mimo26-nvfp4-sm121/env.sh
MARK=/home/r0b0tdgx/projects/mimo26-nvfp4-sm121/logs/systems_500k_memwatch.log
while tmux has-session -t mimo26-systems 2>/dev/null; do
  a3=$(n3 "awk '/MemAvailable/ {printf \"%d\", \$2/1048576}' /proc/meminfo" || echo 0)
  a4=$(n4 "awk '/MemAvailable/ {printf \"%d\", \$2/1048576}' /proc/meminfo" || echo 0)
  echo "$(date -u +%H:%M:%S) n3=${a3}GiB n4=${a4}GiB" >> "$MARK"
  if [ "${a3:-0}" -lt 4 ] || [ "${a4:-0}" -lt 4 ]; then
    echo "STOP below 4 GiB n3=$a3 n4=$a4" >> "$MARK"
    tmux kill-session -t mimo26-systems || true
    exit 2
  fi
  sleep 30
done
echo "suite-session-ended" >> "$MARK"
