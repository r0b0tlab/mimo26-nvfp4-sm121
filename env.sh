# r0b0tlab mimo26-nvfp4-sm121 — source this before anything
C="$HOME/projects/mimo26-nvfp4-sm121"
SRC="$HOME/models/XiaomiMiMo/MiMo-V2.6-Flash-RL"
DST="$HOME/models/r0b0tlab/MiMo-V2.6-Flash-RL-NVFP4"
PY="$HOME/.venvs/modelopt-main/bin/python"
IMG_BASE="lmsysorg/sglang:nightly-dev-cu13-20260922-582389ce"
IMG="r0b0tlab/sglang-mimo26-env:20260922-582389ce"   # base + torchcodec (env fix)
N3IP="192.168.68.78"; N4IP="192.168.68.56"; N4FAB="192.168.100.4"
SSHO="-i $HOME/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes -o LogLevel=ERROR"
n3() { ssh -n $SSHO "r0b0tdgx@$N3IP" "$@"; }
n4() { ssh -n $SSHO "r0b0tdgx@$N4IP" "$@"; }
push() {
  rsync -a --delete -e "ssh $SSHO" "$C/scripts/" "$N3IP:$C/scripts/" &&
  rsync -a --delete -e "ssh $SSHO" "$C/scripts/" "$N4IP:$C/scripts/" &&
  rsync -a -e "ssh $SSHO" "$C/docker/"  "$N3IP:$C/docker/"  &&
  rsync -a -e "ssh $SSHO" "$C/docker/"  "$N4IP:$C/docker/"
}
export C SRC DST PY IMG_BASE IMG N3IP N4IP N4FAB SSHO
