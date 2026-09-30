#!/usr/bin/env bash
# usage (from orchestrator): 45_launch_profile.sh profiles/<name>.env
# Syncs scripts+profiles to both nodes, then starts 43_serve_profile.sh on
# node 3 inside tmux mimo26-r0 with the profile's variables exported.
# r0b0tlab mimo26.
set -euo pipefail
cd "$(dirname "$0")/.."
source ./env.sh
P="${1:?profile env file}"
NAME=$(basename "$P" .env)
for ip in "$N3IP" "$N4IP"; do
  rsync -a -e "ssh $SSHO" scripts/ "r0b0tdgx@$ip:$C/scripts/"
  rsync -a -e "ssh $SSHO" profiles/ "r0b0tdgx@$ip:$C/profiles/"
  rsync -a -e "ssh $SSHO" patches/ "r0b0tdgx@$ip:$C/patches/"
done
n3 "tmux has-session -t mimo26-r0 2>/dev/null && { echo 'mimo26-r0 already running'; exit 1; }; tmux new -d -s mimo26-r0 'cd $C && set -a && . profiles/$NAME.env && set +a && bash scripts/43_serve_profile.sh > logs/serve/launcher.$NAME.log 2>&1'"
echo "[launched] $NAME"
