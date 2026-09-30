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
- eagle
- mtp
- base_model:XiaomiMiMo/MiMo-V2.6-Flash-RL
- base_model:quantized:XiaomiMiMo/MiMo-V2.6-Flash-RL
library_name: transformers
base_model:
- XiaomiMiMo/MiMo-V2.6-Flash-RL
pipeline_tag: text-generation
---

# MiMo-V2.6-Flash-RL-NVFP4

NVFP4 weight-quantized build of [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) (commit `5711b268`) for NVIDIA DGX Spark / GB10 (SM121) serving with [SGLang](https://github.com/sgl-project/sglang) (nightly `582389ce`), TP=2 across two GB10 nodes.

**Preferred serve profile: EAGLE MTP**, 3 steps and 4 draft tokens, advertised context 524288, multimodal on. The DFlash block-8 row is an earlier serve. It is not the preferred profile.

Quantized by **r0b0tlab**. Independent implementation from upstream APIs only (NVIDIA ModelOpt `7159c01d`, SGLang `8ab21c8a`); no third-party quantization code or configs were used.

## What's inside

| Component | Choice |
|---|---|
| Routed experts (MoE) | NVFP4 (E2M1, block 16), converted from vendor MXFP4 |
| Attention / dense linears | NVFP4 with per-tensor FP8 scales where beneficial, BF16 elsewhere |
| KV cache | FP8 (`e4m3`) with **calibrated per-layer scales** (`kv_scales.json`, SGLang QuantParamSchema; identical across TP ranks) |
| Activations | per-tensor scales, `peer_headroom` policy (calibration-gated, see below) |
| Draft weights in the checkpoint | DFlash fused 5-layer KV, unchanged from base. Not the preferred serve. |
| Preferred serve | EAGLE MTP, 3 steps, 4 draft tokens, top-k 1, draft window 4096 |

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
4. **Serving**: the preferred profile is EAGLE MTP, 3 steps and 4 draft tokens, on SGLang nightly `582389ce`. The earlier FINAL3-500k serve was DFlash block 8, with `--tp-size 2 --ep-size 2 --moe-runner-backend marlin --mem-fraction-static 0.90`, `--context-length 524288`, `--tool-call-parser mimo`, and SWA ratio 0.02. It skips allocating unused NVFP4 `*_blockscale_swizzled` tensors on the Marlin path.

The package `ghcr.io/r0b0tlab/sglang-mimo26-env` is public. Preferred tag `20260922-582389ce-mtp-mm` (digest `sha256:84857252a1a9b4196702154ae38eb3cfafdf4cd832795eba18ac2772f8a83f1e`). The FINAL3-500k serve used parent `20260922-582389ce-marlin-skip` (digest `sha256:42737e9dfd3731072c8fd3d65d479ba03381e0e0cb5e171cbab632a8bddeb507`). Tag `20260922-582389ce` is torchcodec only and does not contain the Marlin skip. Not on Docker Hub.

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
- Q200v2 text-180 was kept from the earlier FINAL3 serve, not remeasured on the MTP boot. Think-off, workers 1, max tokens 8192, 180/180 stopped, longest completion 5844. GSM8K 78/80, HumanEval 38/40, IFEval 35/40. Independent review of the 20 hard_reasoning answers is 17/20. Failures: hard-04 answered 19208/715, while the 4x4 Hilbert determinant is 1/6048000; hard-12 claimed every graph of minimum degree 2 has a cycle, which is false for an infinite 2-regular graph; hard-16 did not give the uniform-lift height 1/(2π). Total 168/180. Not a core-subset quality claim.
- On that same FINAL3 boot, the serial quality run produced 51,977 completion tokens in 2,254.2 seconds, 23.06 client tok/s including prefill. A separate 1024-token harness on that boot, truncated at the token cap, measured concurrency-1 aggregate 27.25 and 26.02 tok/s (per-request decode 27.51 and 26.25) and concurrency-2 aggregate 32.39 and 34.08 tok/s. That harness is not the quality-run rate and not the MTP systems throughput.
- Memory on the Q200 boot: weights 84.436 GB, KV cache 6.575 GB, startup available 9.166 GB, full-token pool 1,044,581, SWA pool 20,891. Graph reservations were target-verify 1.495 GB and draft-decode 0.134 GB; prefill, decode, and draft-extend graphs were 0. KV is the calibrated FP8 pool. Host available at admission was 4.84 GiB and 8.71 GiB, above a 4 GiB floor.

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

The measured process used the parent image with the loader and draft-extend window-index files bind-mounted. Preferred tag `ghcr.io/r0b0tlab/sglang-mimo26-env:20260922-582389ce-mtp-mm` (`sha256:84857252a1a9b4196702154ae38eb3cfafdf4cd832795eba18ac2772f8a83f1e`) is public and copies those same three files. It was not the process that served this suite. Q200 was not remeasured on this serve.

Ledger: https://github.com/r0b0tlab/r0b0bench/blob/b35bb28ca058e74eca5642a732c0d791481a7f36/results/entries/mimo26-nvfp4-mtp-500k-mm-systems-20260927.json

## Max-performance review + Q200v2 protocol, 2026-09-30

A 12-cell one-variable-per-boot ladder tested every remaining serve-flag lever
(speculative geometry, kernel fusions, NCCL tuning, KV splits, chunk size,
scheduler conservativeness, stream interval, torch.compile) against the preferred
`MTP-500k-mm-graph` serve. **No cell won on all c1 lanes; the preferred profile is
confirmed optimal for this build.** torch.compile fails at capture on this
driver/kernel pair (triton illegal memory access) — do not retry.

Q200v2 protocol on that serve, think-off (2026-09-30):

| Measure | Result |
|---|---|
| GSM8K-200 | **192/200 (0.960)**, 0 truncations, 0 transport errors |
| Concurrency aggregate | c1 21.6 · c2 38.0 · c4 47.7 · c8 69.2 tok/s |
| Per-stream decode | 21.9 / 19.4 / 12.4 / 9.0 tok/s (TTFT 0.57–1.48 s) |
| Spec accept length | 3.38–3.47 at every level |
| Per-request e2e | p50 23.5 tok/s over 200 answers (mean 209 tokens) |
| Load telemetry | 26 W mean, 47–56 °C, GPU util ~79%, host MemAvailable ≥ 10.9 GiB |

Evidence and the full cell table live in the campaign repo:
`results/maxperf/` (FINAL-VERDICT.json, MORNING-REPORT.md, 12 CELL-*.json),
`results/e2e/FINAL3-concurrency.json`, `results/q200/FINAL3-nothink.jsonl`.

## Known limits

- NVFP4 **KV cache** is not supported by SGLang `582389ce` for this model's hybrid-SWA pool (`torch.zeros(Float4_e2m1fn_x2)` → NotImplemented); FP8 KV + calibrated scales ships instead. Evidence in the campaign repo (`results/nvfp4_kv_verdict.md`).
- `moe_runner=auto` selects triton on this build, which cannot consume MiMo's packed MXFP4 experts; `marlin` is required (and is the native SM121 path).
- The published FINAL3-500k and MTP-500k-mm serves both advertise `max_model_len` 524288, not the base model's 1,048,576. NIAH above is at 25/50/90 of that advertised length.

## Credits

- Base model: XiaomiMiMo/MiMo-V2.6-Flash-RL, MIT. XiaomiMiMo is the base-model author.
- Quantization: r0b0tlab, using NVIDIA Model Optimizer public APIs at `7159c01d`. No third-party quantization code or configs were copied.
- Serving: SGLang, Apache-2.0, nightly `582389ce`, with FlashInfer 0.6.18, PyTorch 2.13.0+cu130, and Marlin MoE kernels as shipped in that nightly.
- Preferred speculation: SGLang's EAGLE implementation. DFlash weights remain in the checkpoint for the earlier lane.
- `modelopt_quant.py` in the runtime image is adapted from the vLLM project, Apache-2.0.
- Systems harness: r0b0bench, MIT. BFCL scores use the Berkeley Function-Calling Leaderboard tasks.

## License

Base model is MIT (XiaomiMiMo). This quantization is provided under the same MIT license. XiaomiMiMo is credited as the base-model author; r0b0tlab claims only the quantization deltas described above. Campaign repo: MIT for original work, Apache-2.0 retained on upstream patch files. See https://github.com/r0b0tlab/mimo26-nvfp4-sm121
