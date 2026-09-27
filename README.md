# mimo26-nvfp4-sm121

NVFP4 weight quantization of [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) (MIT) with calibrated FP8 KV cache, served on **two NVIDIA DGX Spark GB10 nodes (SM121, TP=2)** via SGLang nightly `20260922-582389ce` + DFlash speculative decoding.

**Headline:** our NVFP4 export beats the vendor's own MXFP4 checkpoint on the same hardware — **+21% decode throughput (median 69.1 vs 56.9 tok/s at bs=8)** at statistically identical quality (NLL Δ 0.0027; GSM8K-200 no-think 0.965 vs 0.970).

- Full evidence ledger: [LEDGER.md](LEDGER.md)
- Quantized model: https://huggingface.co/r0b0tlab/MiMo-V2.6-Flash-RL-NVFP4
- Runtime container: `ghcr.io/r0b0tlab/sglang-mimo26-env:20260922-582389ce-marlin-skip` (digest `sha256:42737e9dfd3731072c8fd3d65d479ba03381e0e0cb5e171cbab632a8bddeb507`). Parent tag `20260922-582389ce` is torchcodec only.

## Repo map

- `scripts/10_calibrate_mimo.py` — 512×512 calibration, runtime-style QDQ scoring, `peer_headroom` policy
- `scripts/20_convert_mimo_nvfp4.py` + `21_convert_weights.sh` — MXFP4→NVFP4 hybrid export (ModelOpt APIs)
- `scripts/22_verify_checkpoint.py` — shard/tensor verification (0 mismatches)
- `scripts/23_mirror_nvfp4_to_n4.sh` — fabric mirror across nodes (hashes verified)
- `scripts/40–44_*.sh` — preflight, mem guard, TP=2 lane launch/stop, image probe
- `scripts/50_eval.py` — NLL + GSM8K (thinking on/off)
- `results/` — eval JSONs, calibration stats, conversion manifests, KV-verdict, logs
- `docker/Dockerfile` — env-only image definition
- `publish/MODEL_CARD.md` — HF model card

## Quick start (2× GB10)

```bash
export TAG=nv4-df MODEL=~/models/r0b0tlab/MiMo-V2.6-Flash-RL-NVFP4 SPEC=dflash KV=fp8scales MOE_RUNNER=marlin
docker pull ghcr.io/r0b0tlab/sglang-mimo26-env:20260922-582389ce-marlin-skip
bash scripts/40_preflight_env.sh && bash scripts/42_serve_tp2.sh
curl -s http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"mimo26","messages":[{"role":"user","content":"hello"}]}'
```

Flags: the measured systems lane for this card is EAGLE MTP, 3 steps and 4 draft tokens. The earlier DFlash block-8 row is a different serve. `KV=fp8scales` loads `kv_scales.json`.

Systems row, think-off, advertised context 524288: https://github.com/r0b0tlab/r0b0bench/blob/b35bb28ca058e74eca5642a732c0d791481a7f36/results/entries/mimo26-nvfp4-mtp-500k-mm-systems-20260927.json

Q200v2 on the earlier FINAL3 serve, not that systems row: 168/180. GSM8K 78/80, HumanEval 38/40, IFEval 35/40, hard reasoning 17/20. Serial client rate 23.06 tok/s. Full-token pool 1,044,581. Details are on the model card.

## License

Base model MIT (© XiaomiMiMo). Quantization scripts and this repo MIT (© r0b0tlab).
