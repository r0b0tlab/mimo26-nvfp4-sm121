#!/usr/bin/env bash
# usage: 41b_mem_guard_docker.sh <min_avail_gb> [poll_s]
# Fast, container-aware OOM guard for GB10 unified memory.  Every poll_s
# (default 2 s) it reads MemAvailable; below the floor it SIGKILLs every
# running mimo26-r* serve container on THIS node (frees memory in seconds)
# and kills the rank tmux wrappers.  The v1 guard (41_mem_guard.sh) polled
# every 30 s and only killed tmux sessions, which misses a detached
# `docker run` and is too slow for a ~1 GB/s weight load.
# r0b0tlab mimo26.
MIN_GB="${1:?min_avail_gb}"; POLL="${2:-2}"
MIN_KB=$((MIN_GB * 1024 * 1024))
echo "$(date -u +%FT%TZ) guard up: floor ${MIN_GB} GiB, poll ${POLL}s on $(hostname)"
while true; do
  AVAIL_KB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  if [ "$AVAIL_KB" -lt "$MIN_KB" ]; then
    CTRS=$(docker ps --filter name=mimo26-r --format '{{.Names}}' 2>/dev/null)
    if [ -n "$CTRS" ]; then
      echo "$(date -u +%FT%TZ) LOW MemAvailable $((AVAIL_KB / 1024)) MiB < ${MIN_GB} GiB: killing $CTRS"
      # shellcheck disable=SC2086
      docker kill $CTRS >/dev/null 2>&1 || true
      tmux kill-session -t mimo26-r1 2>/dev/null || true
      tmux kill-session -t mimo26-r0 2>/dev/null || true
      sleep 10
    fi
  fi
  sleep "$POLL"
done
