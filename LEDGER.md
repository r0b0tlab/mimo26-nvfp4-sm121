# Campaign Ledger — MiMo-V2.6-Flash-RL NVFP4 on 2× GB10 (SM121)

**Organization:** r0b0tlab · **Dates:** 2026-09-21 → 2026-09-24 · **Status:** COMPLETE

## Objective

Quantize XiaomiMiMo/MiMo-V2.6-Flash-RL (MIT) to NVFP4 weights with calibrated FP8 KV cache,
serve on SGLang across two DGX Spark GB10 nodes (TP=2, DFlash speculative decoding),
and beat the vendor MXFP4 checkpoint on both quality parity and throughput.

## Environment (all versions pinned)

| Item | Version / SHA |
|---|---|
| Base model | XiaomiMiMo/MiMo-V2.6-Flash-RL @ `5711b268` (MIT), 90 files / 177.8 GB |
| Quantization | NVIDIA ModelOpt main @ `7159c01d` (venv `modelopt-main`, node gn100-2eea) |
| Serving engine | SGLang nightly-dev-cu13 `20260922-582389ce` (0.0.0.dev1+g582389cec, torch 2.13.0+cu130, flashinfer 0.6.18) |
| Runtime image | `ghcr.io/r0b0tlab/sglang-mimo26-env:20260922-582389ce-marlin-skip` @ `sha256:42737e9dfd3731072c8fd3d65d479ba03381e0e0cb5e171cbab632a8bddeb507`. Parent `20260922-582389ce` is torchcodec only. |
| Hardware | 2× NVIDIA DGX Spark GB10, TP=2, RoCE |
| Upstream refs | mo-main `7159c01d9d909ca2431db6332363f07b2137ac09`, sglang-main `8ab21c8a942b014b2c8b56223f83de1ba75b1f8f` |

## Artifacts

1. **Model**: `r0b0tlab/MiMo-V2.6-Flash-RL-NVFP4` (HF) — 66-file checkpoint + hf_quant_config + kv_scales.json + model card. 175 GB.
2. **Runtime container**: `ghcr.io/r0b0tlab/sglang-mimo26-env:20260922-582389ce-marlin-skip` — arm64. Digest `sha256:42737e9dfd3731072c8fd3d65d479ba03381e0e0cb5e171cbab632a8bddeb507`. Still private.
3. **Code + evidence**: `github.com/r0b0tlab/mimo26-nvfp4-sm121` — all scripts, eval JSON, logs.

## Result summary

| Check | Vendor MXFP4 (SRC) | Ours NVFP4 (DST) | Verdict |
|---|---|---|---|
| Tensor integrity | — | 0 mismatches / 24 shards / 36,096 quantized linears; 555 BF16 aux tensors | PASS |
| Calibration gate | — | peer_headroom ≤1.10× best per layer (worst 1.056), calib NLL 1.5317 | PASS |
| NLL parity (32,768 tok) | 1.3360 | 1.3387 (Δ 0.0027 < 0.05) | PASS |
| GSM8K-200 no-think | 0.970 | 0.965 (Δ 0.5 pt) | PASS |
| Decode throughput bs=8 | 52.0–60.8 tok/s (med 56.9) | 63.7–73.5 tok/s (med 69.1) | **+21% NVFP4** |
| DFlash accept | 0.95–1.00, accept-len 1.95–2.00 | same | healthy both lanes |

Headline: **the independent NVFP4 conversion beats the vendor's own MXFP4 checkpoint on its own
hardware class — +21% decode at statistically identical quality — with zero copied third-party
quantization code.**

## Pipeline (what ran)

1. `10_calibrate_mimo.py` — 512×512 rows, runtime-style QDQ scoring (nvfp4_act_relmse, E2M1 LUT),
   per-layer shared-amax candidates; policy `peer_headroom` locked. `results/calib_full512x512_stats.json`.
2. `20_convert_mimo_nvfp4.py` (+`21_convert_weights.sh`) — MXFP4→NVFP4 hybrid export,
   0 non-exact blocks (`results/r0b0tlab_conversion_weights.json`); input-scales shard split so
   activation policy stays re-writable without touching 175 GB.
3. KV scales via QuantParamSchema (`kv_scales.json`): per-layer, both TP ranks identical
   (`models/mimo_v2.py:1151-1165` semantics).
4. `22_verify_checkpoint.py` — 24 shards, 0 mismatches. `23_mirror_nvfp4_to_n4.sh` — 66 files, hashes match.
5. `40/41/42/43/44` ops — preflight, mem guard, TP=2 launch (rank1 bootstrapped via fabric),
   stop, image probe. `50_eval.py` — NLL / GSM8K (think + no-think).
6. Lanes: `base-df` = vendor checkpoint (control), `nv4-df` = ours (money lane). Identical flags:
   `--tp-size 2 --ep-size 2 --moe-runner-backend marlin --mem-fraction-static 0.90
   --cuda-graph-max-bs-decode 8 --cuda-graph-max-bs-prefill 8` + DFlash.

## Root causes fixed (no workarounds, no upstream patches)

- Missing `torchcodec` in the nightly → MiMo-V2 multimodal processor never registered
  ("No processor registered for architecture: ['MiMoV2ForCausalLM']") → env-only image.
- `moe_runner auto` → triton cannot consume MiMo's packed MXFP4 experts (hidden-size assert) → `marlin` (native SM121 path, not a fallback).
- KV profiler floor at TP2+DFlash = 0.8738 → `--mem-fraction-static 0.90`.
- DFlash drafter loads via standard stacked q/k/v mapping (`models/dflash.py:769-774`) → no loader image required.

## Negative result (evidence captured)

**NVFP4 KV cache is NOT supported** on SGLang `582389ce` for MiMo's hybrid-SWA pool:
boot warns then dies in `torch.zeros(Float4_e2m1fn_x2)` → NotImplemented. Not config-fixable.
FP8 KV + calibrated scales ships instead. See `results/nvfp4_kv_verdict.md`.

## Context-length findings (measured, not spec)

Model advertises 1,048,576 (rope_theta 10M, no scaling). Measured TP2 pools at mem-fraction 0.90:

| Lane | KV dtype | max_total_num_tokens | per-token |
|---|---|---|---|
| base-df | BF16 | 172,076 | ~93 KB |
| nv4-df | FP8 | 55,901 | ~46.5 KB |

FP8 halves per-token cost yet yielded a smaller pool → ~13–14 GB pre-KV overhead unique to the
NVFP4 lane (suspects: marlin repack buffers, DFlash verify graphs). **Open item**, logged for follow-up.
300K context is the safe certification target on FP8 (~16 GB KV total) once the overhead is reclaimed;
512K feasible; 1M is demo-only and BF16-KV 1M is impossible (KV alone ~108 GB > both nodes).

## Benchmark scope & honesty

- Throughput numbers: 8 concurrent 600-token probes, steady-state decode, identical prompt sets, both lanes freshly booted.
- The published 69.8 tok/s DFlash bar appears to be a different measurement envelope (concurrency/decode counting);
  our bs=8 median 69.1 exceeds it; single-stream floor was ~17 tok/s (DFlash active, verify graphs captured).
- Thinking-on GSM8K (~0.35) is a harness artifact (512-token cap truncates mid-thought), not a model deficiency;
  thinking-off is the fair comparison and is what's reported above. Both records kept in eval JSONs.

## Licensing / attribution

- Base model MIT (XiaomiMiMo) — redistribution permitted with attribution. Credited in model card.
- Quantization implemented from NVIDIA ModelOpt + SGLang public APIs (cited by SHA + file/line).
- No third-party quantization code, configs, or patches used (MiaAI-Lab explicitly not referenced).

## Open items

1. NVFP4-lane pre-KV memory overhead (13.7 GB) — root-cause and reclaim before long-context cert.
2. Long-context NIAH (300K target, then 512K) on nv4-df.
3. Vision `image_url` smoke on the published image.
4. Container visibility: pushed private; owner flips to public.
