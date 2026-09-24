---
license: mit
language:
- en
- zh
tags:
- text-generation
- multimodal
- vision-language
- audio
- long-context
- nvfp4
- fp8
- quantization
- mimo_v2
- sglang
- dflash
library_name: transformers
base_model:
- XiaomiMiMo/MiMo-V2.6-Flash-RL
pipeline_tag: text-generation
---

# MiMo-V2.6-Flash-RL-NVFP4

NVFP4 weight-quantized build of [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) (commit `5711b268`) for NVIDIA DGX Spark / GB10 (SM121) serving with [SGLang](https://github.com/sgl-project/sglang) (nightly `582389ce`), TP=2 across two GB10 nodes.

Quantized by **r0b0tlab**. Independent implementation from upstream APIs only (NVIDIA ModelOpt `7159c01d`, SGLang `8ab21c8a`); no third-party quantization code or configs were used.

## What's inside

| Component | Choice |
|---|---|
| Routed experts (MoE) | NVFP4 (E2M1, block 16), converted from vendor MXFP4 |
| Attention / dense linears | NVFP4 with per-tensor FP8 scales where beneficial, BF16 elsewhere |
| KV cache | FP8 (`e4m3`) with **calibrated per-layer scales** (`kv_scales.json`, SGLang QuantParamSchema; identical across TP ranks) |
| Activations | per-tensor scales, `peer_headroom` policy (calibration-gated, see below) |
| Draft model | DFlash (fused 5-layer KV) — unchanged from base |

Non-quantized aux weights (audio tokenizer, vision tower, nextn) remain BF16: 555 tensors.

`hf_quant_config.json` is included for engine auto-detection (hybrid FP8+NVFP4 mode).

## Quality vs vendor checkpoint (measured, same harness/hardware)

| Check | Vendor MXFP4 | This NVFP4 | Verdict |
|---|---|---|---|
| NLL, 32,768 calibration tokens | 1.3360 | 1.3387 | Δ 0.0027 (< 0.05 gate) PASS |
| GSM8K-200, thinking off, temp 0 | 0.970 | 0.965 | Δ 0.5 pt PASS |
| Decode throughput, bs=8, 2×GB10 | 52.0–60.8 tok/s (median 56.9) | 63.7–73.5 tok/s (median 69.1) | **+21%** |
| Checkpoint integrity | — | 0 mismatched tensors across 24 shards (36,096 quantized linears) | PASS |

GSM8K harness: chat endpoint, `enable_thinking: false`, temperature 0, 512 max tokens, `#### <answer>` extraction.

## Provenance / method (summary)

1. **Calibration**: 512×512 rows from internal corpus (NLL 1.5317 post-cal). Per-layer activation-scale policy `peer_headroom` selected by runtime-style QDQ scoring (`nvfp4_act_relmse`); gate: chosen ≤1.10× best candidate per layer (worst observed 1.056).
2. **Conversion**: MXFP4 routed experts → NVFP4 (E2M1/block-16, FP8 block scales), adapted from ModelOpt DeepSeek example APIs. Output verified bit-exact where expected: 0 non-exact blocks among 9.46B/9.46B compared.
3. **KV scales**: per-layer min/max calibration → `kv_cache.scaling_factor[tp_rank][layer_idx]`; both ranks identical by construction.
4. **Serving**: SGLang nightly `582389ce` + `torchcodec` (env-only image, no source patches), `--tp-size 2 --ep-size 2 --moe-runner-backend marlin --mem-fraction-static 0.90`, DFlash speculative decoding.

Runtime container: `docker.io/r0b0tlab/sglang-mimo26-env:20260922-582389ce` (base `lmsysorg/sglang:nightly-dev-cu130-20260922-582389ce` + `torchcodec`; arm64).

Campaign scripts, evidence, and logs: https://github.com/r0b0tlab/mimo26-nvfp4-sm121

## Known limits

- NVFP4 **KV cache** is not supported by SGLang `582389ce` for this model's hybrid-SWA pool (`torch.zeros(Float4_e2m1fn_x2)` → NotImplemented); FP8 KV + calibrated scales ships instead. Evidence in the campaign repo (`results/nvfp4_kv_verdict.md`).
- `moe_runner=auto` selects triton on this build, which cannot consume MiMo's packed MXFP4 experts; `marlin` is required (and is the native SM121 path).
- Context: advertised 1,048,576; the FP8-KV TP2 pool boot-tests at ~56K tokens at `mem-fraction 0.90` before tuning — see campaign repo for the KV pool measurements before attempting long-context.

## License

Base model is MIT (XiaomiMiMo). This quantization is provided under the same MIT license. XiaomiMiMo is credited as the base-model author; r0b0tlab claims only the quantization deltas described above.
