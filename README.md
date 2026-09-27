# mimo26-nvfp4-sm121

NVFP4 weight quantization of [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) (MIT) with calibrated FP8 KV cache, served on **two NVIDIA DGX Spark GB10 nodes (SM121, TP=2)** via SGLang nightly `20260922-582389ce`.

**Preferred serve profile: EAGLE MTP.** 3 steps, 4 draft tokens, top-k 1, advertised context 524288, multimodal on. Profile file: `profiles/MTP-500k-mm.env`. The earlier DFlash block-8 row is a different serve and is not the preferred profile.

**Headline:** our NVFP4 export beats the vendor's own MXFP4 checkpoint on the same hardware — **+21% decode throughput (median 69.1 vs 56.9 tok/s at bs=8)** at statistically identical quality (NLL Δ 0.0027; GSM8K-200 no-think 0.965 vs 0.970).

- Full evidence ledger: [LEDGER.md](LEDGER.md)
- Quantized model: https://huggingface.co/r0b0tlab/MiMo-V2.6-Flash-RL-NVFP4
- Preferred runtime image, public: `ghcr.io/r0b0tlab/sglang-mimo26-env:20260922-582389ce-mtp-mm` (digest `sha256:84857252a1a9b4196702154ae38eb3cfafdf4cd832795eba18ac2772f8a83f1e`). Parent: `20260922-582389ce-marlin-skip` (`sha256:42737e9dfd3731072c8fd3d65d479ba03381e0e0cb5e171cbab632a8bddeb507`). The MTP systems suite ran on that parent plus bind-mounted loader and window-index files. The tag copies those same three files.

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
docker pull ghcr.io/r0b0tlab/sglang-mimo26-env:20260922-582389ce-mtp-mm
set -a && source profiles/MTP-500k-mm.env && set +a
bash scripts/40_preflight_env.sh && bash scripts/43_serve_profile.sh
```

`scripts/42_serve_tp2.sh` is the older DFlash helper. It is not the preferred launcher. `scripts/43_serve_profile.sh` with `profiles/MTP-500k-mm.env` is. That profile is EAGLE, 3 steps, 4 draft tokens, draft window 4096, mem-fraction 0.90, context 524288, mimo tool and reasoning parsers, multimodal on.

Systems row, think-off, advertised context 524288: https://github.com/r0b0tlab/r0b0bench/blob/b35bb28ca058e74eca5642a732c0d791481a7f36/results/entries/mimo26-nvfp4-mtp-500k-mm-systems-20260927.json

Q200v2 on the earlier FINAL3 serve, not that systems row: 168/180. GSM8K 78/80, HumanEval 38/40, IFEval 35/40, hard reasoning 17/20. Serial client rate 23.06 tok/s. Full-token pool 1,044,581. Details are on the model card.

## Credits

- Base model: [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL), MIT, commit `5711b268`. XiaomiMiMo is the base-model author.
- Quantization: r0b0tlab, using NVIDIA Model Optimizer public APIs at `7159c01d`. No third-party quantization code or configs were copied.
- Serving: [SGLang](https://github.com/sgl-project/sglang), Apache-2.0, nightly `20260922-582389ce`, with FlashInfer 0.6.18 and PyTorch 2.13.0+cu130. MoE runner is Marlin, as shipped in that nightly.
- Preferred speculation: SGLang's EAGLE implementation, in-checkpoint MTP. DFlash weights ship with the base checkpoint; that lane is recorded and is not preferred.
- Patches under `patches/sglang-582389ce/` keep their upstream licenses. `modelopt_quant.py` is adapted from [vLLM](https://github.com/vllm-project/vllm), Apache-2.0.
- Systems harness: [r0b0bench](https://github.com/r0b0tlab/r0b0bench), MIT. BFCL scores use the Berkeley Function-Calling Leaderboard tasks.

## License

MIT for r0b0tlab original work. Copyright (c) 2026 r0b0tlab. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). Upstream patch files remain Apache-2.0. Base-model weights remain XiaomiMiMo MIT.
