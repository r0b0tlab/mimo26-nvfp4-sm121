#!/usr/bin/env bash
# Read-only probes of an sglang image's mimo sources. r0b0tlab mimo26.
set -euo pipefail
IMG="${1:?usage: 44_image_probe.sh IMAGE}"
docker run --rm --entrypoint bash "$IMG" -c '
S=/sgl-workspace/sglang/python/sglang/srt
echo "== hybrid detect:"; grep -n "Auto-detected hybrid FP8+NVFP4 checkpoint" $S/configs/model_config.py | head -2
echo "== nextn deferred (expect 0; P8 out of scope):"; grep -n "deferred" $S/models/mimo_v2_nextn.py | wc -l
echo "== qkv sizes:"; grep -n -A3 "def _get_ckpt_qkv_shard_sizes" $S/models/mimo_v2.py | head -5
echo "== mha pool:"; grep -n "mha_pool_class = " $S/mem_cache/kv_cache_configurator.py | head -2
echo "== fp8 kv method:"; grep -n "Fp8KVCacheMethod" $S/layers/quantization/fp8.py | head -2
echo "== uint8 view:"; grep -n "view(torch.int8)" $S/layers/quantization/modelopt_quant.py | head -2
echo "== fa4 gate:"; grep -rn "SM100\|sm_100" $S/layers/attention/flashinfer_backend.py | head -2
echo "== ours:"; grep -rn "r0b0tlab-mimo26" $S/models/mimo_v2.py $S/models/mimo_v2_nextn.py | wc -l
sha256sum $S/models/mimo_v2.py $S/models/mimo_v2_nextn.py'
