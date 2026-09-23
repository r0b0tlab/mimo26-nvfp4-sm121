#!/usr/bin/env bash
# Durable source download for MiMo-V2.6-Flash-RL (pinned revision), N3.
set -euo pipefail
C=~/projects/mimo26-nvfp4-sm121
DST=~/models/XiaomiMiMo/MiMo-V2.6-Flash-RL
PY=~/venvs/hf/bin/python
HF=~/venvs/hf/bin/hf
REPO=XiaomiMiMo/MiMo-V2.6-Flash-RL
REV=$($PY - <<'P'
from huggingface_hub import HfApi
i=HfApi().model_info("XiaomiMiMo/MiMo-V2.6-Flash-RL", files_metadata=True)
import json,os
man={"repo":i.id,"sha":i.sha,"last_modified":str(i.last_modified),
     "files":[{"name":s.rfilename,"size":s.size,"sha256":(s.lfs.sha256 if s.lfs else None)} for s in i.siblings]}
man["total_bytes"]=sum(f["size"] or 0 for f in man["files"])
json.dump(man,open(os.path.expanduser("~/projects/mimo26-nvfp4-sm121/evidence/source_manifest.json"),"w"),indent=1)
print(i.sha)
P
)
echo "$(date -u +%FT%TZ) START repo=$REPO rev=$REV dst=$DST"
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60
for attempt in 1 2 3 4 5; do
  if $HF download "$REPO" --revision "$REV" --local-dir "$DST" --max-workers 16; then
    echo "$(date -u +%FT%TZ) DOWNLOAD_OK attempt=$attempt"; break
  fi
  echo "$(date -u +%FT%TZ) DOWNLOAD_RETRY attempt=$attempt"; sleep 30
done
# Verify every file size (and LFS sha256) against the pinned manifest
$PY - <<'P'
import json,os,hashlib,sys,concurrent.futures as cf
C=os.path.expanduser("~/projects/mimo26-nvfp4-sm121"); D=os.path.expanduser("~/models/XiaomiMiMo/MiMo-V2.6-Flash-RL")
man=json.load(open(f"{C}/evidence/source_manifest.json"))
def chk(f):
    p=os.path.join(D,f["name"])
    if not os.path.exists(p): return (f["name"],"MISSING")
    sz=os.path.getsize(p)
    if f["size"] is not None and sz!=f["size"]: return (f["name"],f"SIZE {sz}!={f['size']}")
    if f["sha256"]:
        h=hashlib.sha256()
        with open(p,"rb") as fh:
            for b in iter(lambda: fh.read(1<<24), b""): h.update(b)
        if h.hexdigest()!=f["sha256"]: return (f["name"],"SHA256_MISMATCH")
    return (f["name"],"OK")
with cf.ThreadPoolExecutor(8) as ex: res=list(ex.map(chk,man["files"]))
bad=[r for r in res if r[1]!="OK"]
json.dump({"sha":man["sha"],"n_files":len(res),"bad":bad},open(f"{C}/evidence/source_verify.json","w"),indent=1)
print("VERIFY n=%d bad=%d"%(len(res),len(bad)))
sys.exit(1 if bad else 0)
P
echo "$(date -u +%FT%TZ) SOURCE_VERIFIED rev=$REV" | tee $C/logs/SOURCE_DONE
