#!/usr/bin/env bash
# usage: 41_mem_guard.sh <tmux-serve-session> <min_free_gb>
# Kills the serve session (and its rank-1 twin) if free memory drops below MIN.
# r0b0tlab mimo26 — guards the GB10 OOM wedge pattern.
SESSION="${1:?session}"; MIN="${2:?min_gb}"
while true; do
  FREE=$(free -g | awk 'NR==2{print $7}')
  if [ "$FREE" -lt "$MIN" ] && tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) low mem ${FREE}GB < ${MIN}GB: stopping $SESSION" >&2
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux kill-session -t mimo26-r1 2>/dev/null || true
  fi
  sleep 30
done
