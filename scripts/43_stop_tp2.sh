#!/usr/bin/env bash
# Graceful full-lane stop. Never SIGKILL a live long-context client (plan §1).
# r0b0tlab mimo26.  Runs on node 3.
set -u
N4IP="192.168.68.56"
tmux kill-session -t mimo26-serve 2>/dev/null || true
ssh -n "r0b0tdgx@$N4IP" "tmux kill-session -t mimo26-r1 2>/dev/null" || true
docker stop -t 60 mimo26-r0 2>/dev/null || true
ssh -n "r0b0tdgx@$N4IP" "docker stop -t 60 mimo26-r1 2>/dev/null" || true
sleep 5
LEFT=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
LEFT4=$(ssh -n "r0b0tdgx@$N4IP" "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l")
if [ "$LEFT" = 0 ] && [ "$LEFT4" = 0 ]; then
  echo "[stopped] gpus free on both nodes"
else
  echo "[warn] compute procs remain: n3=$LEFT n4=$LEFT4"
fi
