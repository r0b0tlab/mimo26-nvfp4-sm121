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
- base_model:XiaomiMiMo/MiMo-V2.6-Flash-RL
- base_model:quantized:XiaomiMiMo/MiMo-V2.6-Flash-RL
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
4. **Serving**: SGLang nightly `582389ce` + `torchcodec`, `--tp-size 2 --ep-size 2 --moe-runner-backend marlin --mem-fraction-static 0.90`, DFlash block 8. The measured FINAL3-500k profile also sets `--context-length 524288`, `--tool-call-parser mimo`, and SWA ratio 0.02. It skips allocating unused NVFP4 `*_blockscale_swizzled` tensors on the Marlin path.

Runtime container that matches that serve: `ghcr.io/r0b0tlab/sglang-mimo26-env:20260922-582389ce-marlin-skip` (digest `sha256:42737e9dfd3731072c8fd3d65d479ba03381e0e0cb5e171cbab632a8bddeb507`). It is the 2026-09-24 env image plus that one-file skip. The parent tag `20260922-582389ce` is torchcodec only and does not contain the skip. The package is still private. It is not on Docker Hub.

Campaign scripts, evidence, and logs: https://github.com/r0b0tlab/mimo26-nvfp4-sm121

## r0b0bench systems, FINAL3-500k (2026-09-26)

One `r0b0bench run --profile systems` on the 2×GB10 TP=2 serve. Advertised `max_model_len` 524288. Think-off (`enable_thinking=false`, `thinking=false`). Harness 1.0.0rc2. All seven lanes passed. Infrastructure errors 0. `invalid_for_publish` false.

Ledger: [`mimo26-nvfp4-final3-500k-systems-20260925`](https://github.com/r0b0tlab/r0b0bench/blob/2fe1f6da92c626860e77f955bd978eca565547fd/results/entries/mimo26-nvfp4-final3-500k-systems-20260925.json)

| Lane | Result |
|---|---|
| Canary | PASS, 5/5, including the tool-call check |
| BFCL multi-turn | 114/200 (0.570) |
| BFCL AST | multiple 71/200, parallel 101/200, parallel_multiple 39/200, micro 211/600 (0.352) |
| Latency | TTFT 254 ms, inter-token 159 ms |
| Concurrency aggregate | c1 50.2, c2 66.0, c4 91.6, c6 159.6 tok/s |
| Throughput | C1 decode median 15.48 tok/s. Prefill median 16,535 tok/s at 22,771 prompt tokens |
| NIAH | 3/3 exact at depths 131008, 262016, 471628 |

The 15.48 tok/s figure is single-stream chat decode. It is not the bs=8 aggregate in the table above (median 69.1 tok/s).

Disclosures:

- Seven multi-turn rows were regenerated after 1200s client timeouts. This invocation scored the repaired 200-row file and did not regenerate the other 193.
- The package lane summary had pointed all three AST categories at the parallel_multiple file and reported micro 0.195. The numbers above are the official score-file headers from the same run.
- Q200v2 text-180 was kept from the earlier FINAL3 serve, not remeasured here. Auto-graded 151/160 (GSM8K 78/80, HumanEval 38/40, IFEval 35/40). The 20 hard_reasoning rows were answered and left ungraded. Not a core-subset quality claim.

## Systems, MTP-500k-mm, 2026-09-27

Think-off. Advertised context 524288. EAGLE, 3 steps, 4 draft tokens, top-k 1, draft window 4096, `--mem-fraction-static 0.90`. Multimodal pins: mimo reasoning parser, mimo tool parser, Triton vision attention, torchvision. Draft-extend CUDA graphs on. One full systems profile. Not a core-subset quality claim.

| Lane | Result |
|---|---|
| Canary | 5/5 |
| BFCL-MT | 117/200 (0.585) |
| BFCL-AST multiple | 70/200 |
| BFCL-AST parallel | 103/200 |
| BFCL-AST parallel_multiple | 42/200 |
| BFCL-AST micro | 215/600 (0.358) |
| Latency, mean | TTFT 276.5 ms, ITL 139.6 ms, end-to-end 4.12 s |
| Concurrency, aggregate tok/s | 26.7 / 43.5 / 66.6 / 93.2 at 1, 2, 4, 6 |
| Throughput | decode median 17.31 tok/s; prefill median 17,941 tok/s at 22,771 prompt tokens |
| NIAH | 3/3 exact at 131008, 262016, and 471628 |

A separate 1024-token harness on the same boot, not the systems c1 number above, measured decode 24.1 / 23.6 / 15.6 tok/s on short code, medium code, and prose.

The measured process used the parent image with the loader and draft-extend window-index files bind-mounted. Tag `ghcr.io/r0b0tlab/sglang-mimo26-env:20260922-582389ce-mtp-mm` (`sha256:84857252a1a9b4196702154ae38eb3cfafdf4cd832795eba18ac2772f8a83f1e`) copies those same three files and was not the process that served this suite. The tag is still private. Q200 was not remeasured on this serve.

Ledger: https://github.com/r0b0tlab/r0b0bench/blob/b35bb28ca058e74eca5642a732c0d791481a7f36/results/entries/mimo26-nvfp4-mtp-500k-mm-systems-20260927.json

## Known limits

- NVFP4 **KV cache** is not supported by SGLang `582389ce` for this model's hybrid-SWA pool (`torch.zeros(Float4_e2m1fn_x2)` → NotImplemented); FP8 KV + calibrated scales ships instead. Evidence in the campaign repo (`results/nvfp4_kv_verdict.md`).
- `moe_runner=auto` selects triton on this build, which cannot consume MiMo's packed MXFP4 experts; `marlin` is required (and is the native SM121 path).
- The published FINAL3-500k and MTP-500k-mm serves both advertise `max_model_len` 524288, not the base model's 1,048,576. NIAH above is at 25/50/90 of that advertised length.

## License

Base model is MIT (XiaomiMiMo). This quantization is provided under the same MIT license. XiaomiMiMo is credited as the base-model author; r0b0tlab claims only the quantization deltas described above.
