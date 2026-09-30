#!/usr/bin/env bash
# Stop the TP2 serve on both nodes: graceful docker stop (SIGTERM, 30 s grace,
# then SIGKILL by docker), then drop the rank tmux wrappers.  Run from the
# orchestrator.  Waits until MemAvailable on both nodes is back above 90 GiB
# so the next boot never starts on top of a half-freed pool.
# r0b0tlab mimo26.
set -uo pipefail
N3IP="192.168.68.78"; N4IP="192.168.68.56"
SSHO=(-i "$HOME/.ssh/id_ed25519_shared" -o IdentitiesOnly=yes -o BatchMode=yes
      -o ConnectTimeout=10 -o StrictHostKeyChecking=yes -o LogLevel=ERROR)
for ip in "$N3IP" "$N4IP"; do
  ssh "${SSHO[@]}" "r0b0tdgx@$ip" 'c=$(docker ps --filter name=mimo26-r --format "{{.Names}}"); [ -n "$c" ] && docker stop -t 30 $c >/dev/null; tmux kill-session -t mimo26-r1 2>/dev/null; tmux kill-session -t mimo26-r0 2>/dev/null; true' &
done
wait
for i in $(seq 1 60); do
  ok=1
  for ip in "$N3IP" "$N4IP"; do
    a=$(ssh "${SSHO[@]}" "r0b0tdgx@$ip" "awk '/^MemAvailable:/ {print int(\$2/1048576)}' /proc/meminfo")
    [ "${a:-0}" -lt 90 ] && ok=0
    printf '%s avail=%sGiB  ' "$ip" "$a"
  done
  echo
  [ "$ok" = 1 ] && { echo "[ok] both nodes >= 90 GiB available"; exit 0; }
  sleep 5
done
echo "[warn] memory did not recover above 90 GiB within 300 s"; exit 1
