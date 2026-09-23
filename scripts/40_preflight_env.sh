#!/usr/bin/env bash
# One-time per-image environment preflight, run from the orchestrator on both
# nodes before the first lane boot. Idempotent.
# Root cause it fixes: the nightly image lacks torchcodec, so sglang swallows
# the MiMo-V2 mm-processor import error and boot dies with
# "No processor registered for architecture: ['MiMoV2ForCausalLM']".
# r0b0tlab mimo26.
set -euo pipefail
SSHO="-i $HOME/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes -o LogLevel=ERROR"
for h in "$@"; do
  echo "== preflight $h"
  ssh -n $SSHO "r0b0tdgx@$h" "
    docker run --rm --entrypoint bash ${IMG_BASE:?IMG_BASE} -c \
      'python3 -c \"import torchcodec\" 2>/dev/null || pip install -q torchcodec' \
    && echo torchcodec_ok
    ssh -n -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 r0b0tdgx@192.168.68.56 hostname >/dev/null 2>&1 || true
    echo hostkey_ok"
done
echo "PREFLIGHT OK"
