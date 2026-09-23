#!/usr/bin/env bash
set -uo pipefail
SRC=~/models/XiaomiMiMo/MiMo-V2.6-Flash-RL
echo "[start] $(date -u +%FT%TZ)"
rsync -a --info=progress2 --exclude .cache -e "ssh -o BatchMode=yes -c aes128-gcm@openssh.com" "$SRC/" r0b0tdgx@192.168.100.4:models/XiaomiMiMo/MiMo-V2.6-Flash-RL/ < /dev/null
echo "rsync rc=$?"
L=$(cd $SRC && find . -type f -not -path "./.cache/*" -printf "%s %p\n" | sort -k2 | md5sum)
R=$(ssh -n -o BatchMode=yes r0b0tdgx@192.168.100.4 'cd ~/models/XiaomiMiMo/MiMo-V2.6-Flash-RL && find . -type f -not -path "./.cache/*" -printf "%s %p\n" | sort -k2 | md5sum')
echo "local=$L remote=$R"; [ "$L" = "$R" ] && echo "MIRROR_OK" || echo "MIRROR_MISMATCH"
echo "[done] $(date -u +%FT%TZ)"
