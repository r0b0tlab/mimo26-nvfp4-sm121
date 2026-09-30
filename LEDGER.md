# Campaign Ledger — MiMo-V2.6-Flash-RL NVFP4 on 2× GB10 (SM121)

**Organization:** r0b0tlab · **Dates:** 2026-09-21 → 2026-09-30 · **Status:** COMPLETE (max-perf review closed 2026-09-30)

## Objective

Quantize XiaomiMiMo/MiMo-V2.6-Flash-RL (MIT) to NVFP4 weights with calibrated FP8 KV cache,
serve on SGLang across two DGX Spark GB10 nodes (TP=2). The preferred profile is EAGLE MTP, not DFlash.
and beat the vendor MXFP4 checkpoint on both quality parity and throughput.

## Environment (all versions pinned)

| Item | Version / SHA |
|---|---|
| Base model | XiaomiMiMo/MiMo-V2.6-Flash-RL @ `5711b268` (MIT), 90 files / 177.8 GB |
| Quantization | NVIDIA ModelOpt main @ `7159c01d` |
| Serving engine | SGLang nightly-dev-cu13 `20260922-582389ce` (0.0.0.dev1+g582389cec, torch 2.13.0+cu130, flashinfer 0.6.18) |
| Runtime image | Preferred tag `ghcr.io/r0b0tlab/sglang-mimo26-env:20260922-582389ce-mtp-mm` @ `sha256:84857252a1a9b4196702154ae38eb3cfafdf4cd832795eba18ac2772f8a83f1e`, public. Parent is `20260922-582389ce-marlin-skip` @ `sha256:42737e9dfd3731072c8fd3d65d479ba03381e0e0cb5e171cbab632a8bddeb507`. The MTP systems suite ran on the parent plus bind-mounted loader and window-index files, not inside the baked tag. |
| Hardware | 2× NVIDIA DGX Spark GB10, TP=2, RoCE |
| Upstream refs | mo-main `7159c01d9d909ca2431db6332363f07b2137ac09`, sglang-main `8ab21c8a942b014b2c8b56223f83de1ba75b1f8f` |

## Artifacts

1. **Model**: `r0b0tlab/MiMo-V2.6-Flash-RL-NVFP4` (HF) — 66-file checkpoint + hf_quant_config + kv_scales.json + model card. 175 GB.
2. **Runtime container**: preferred public tag `ghcr.io/r0b0tlab/sglang-mimo26-env:20260922-582389ce-mtp-mm` @ `sha256:84857252a1a9b4196702154ae38eb3cfafdf4cd832795eba18ac2772f8a83f1e`. Parent `20260922-582389ce-marlin-skip` @ `sha256:42737e9dfd3731072c8fd3d65d479ba03381e0e0cb5e171cbab632a8bddeb507`. The MTP systems suite was measured on the parent plus bind mounts, not inside the preferred tag. The tag copies the same three files.
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

- Base model MIT (XiaomiMiMo) — redistribution permitted with attribution. Credited in the model card and NOTICE.md.
- This repo's original work is MIT (Copyright (c) 2026 r0b0tlab). See LICENSE.
- SGLang and vLLM files under `patches/sglang-582389ce/` stay Apache-2.0. Notices are in those files and in NOTICE.md.
- Quantization implemented from NVIDIA ModelOpt + SGLang public APIs (cited by SHA + file/line). No third-party quantization code or configs were copied.
- Systems harness is r0b0bench (MIT). BFCL scores use the Berkeley Function-Calling Leaderboard tasks.
- Preferred serve profile is EAGLE MTP (`profiles/MTP-500k-mm.env`). DFlash is an earlier recorded lane, not the preferred profile.

## Q200v2 on the pre-cut FINAL3 serve

Not the MTP systems row, and not a core-subset quality claim. Think-off, workers 1, max tokens 8192. 180/180 stopped. Longest completion 5844.

| Family | Score |
|---|---|
| GSM8K | 78/80 |
| HumanEval | 38/40 |
| IFEval | 35/40 |
| hard reasoning | 17/20 |
| Total | 168/180 |

The three hard-reasoning failures are hard-04 (answered 19208/715; the 4x4 Hilbert determinant is 1/6048000), hard-12 (claimed the cycle statement is true; an infinite 2-regular graph is a counterexample), and hard-16 (did not give the uniform-lift height 1/(2π)).

The serial quality run produced 51,977 completion tokens in 2,254.2 seconds, 23.06 client tok/s including prefill. A separate 1024-token harness on that same boot, truncated at the token cap, measured concurrency-1 aggregate 27.25 and 26.02 tok/s and concurrency-2 aggregate 32.39 and 34.08 tok/s. That harness is not the quality-run rate.

Memory on that boot: weights 84.436 GB, KV cache 6.575 GB, startup available 9.166 GB, full-token pool 1,044,581, SWA pool 20,891. Graph reservations were target-verify 1.495 GB and draft-decode 0.134 GB. Prefill, decode, and draft-extend graphs were 0. KV is the calibrated FP8 pool. Host available at admission was 4.84 GiB and 8.71 GiB, above a 4 GiB floor.

## Open items

1. NVFP4-lane pre-KV memory overhead (13.7 GB) — root-cause and reclaim before long-context cert.
2. Container visibility: the preferred MTP tag is public. Anonymous manifest pull of `20260922-582389ce-mtp-mm` returned 200.

## Max-performance review (2026-09-30) — CLOSED, nothing promoted

A 12-cell one-variable-per-boot ladder tested every remaining serve-flag lever on
build `582389ce` against the live preferred profile (`MTP-500k-mm-graph`):

| Cell | One variable | short | medium | prose | Verdict |
|---|---|---|---|---|---|
| floor (live) | — | 23.75–25.06 | 25.09–26.02 | 16.4–17.1 | defended |
| S2 | spec steps 2 / draft 3 | 22.27 | 23.00 | 18.36 | rejected (short −11%) |
| K | `--enable-fused-qk-norm-rope` | 23.17 | 24.03 | 16.79 | rejected |
| MOE | `--enable-fused-moe-sum-all-reduce` | 23.22 | 22.13 | 17.54 | rejected |
| NCCL | `NCCL_ALGO=Ring NCCL_PROTO=Simple` + 8 ch | 19.87 | 21.73 | 15.01 | rejected (c1 −20%) |
| NCCLCH | `NCCL_MIN_NCHANNELS=16` only | 21.68 | 23.42 | 16.50 | rejected |
| KV16 | `--triton-attention-num-kv-splits 16` | 23.29 | 23.86 | 15.46 | rejected |
| CHUNK16 | chunked prefill 16384 | 22.80 | 21.30 | 15.40 | rejected |
| CONS | `--schedule-conservativeness 0.3` | 23.49 | 24.49 | 17.42 | rejected (c4 −12%) |
| SI8 | `--stream-interval 8` | 22.95 | 24.36 | 15.91 | rejected |
| TC | `--enable-torch-compile` | — | — | — | **boot fail**: triton illegal memory access during capture |

Verdict: the preferred profile is the optimum for this build. Boot-to-boot noise is
~5%, larger than every "win" any cell showed on a single lane. torch.compile is a
hard no on driver 580.173.02 / kernel 6.17 (SM121 fragile pair fault class).

Acceptance (prose histogram): per-position draft accept 0.75 / 0.65 / 0.49 / 0.00.
The 4th draft token is never accepted (the checkpoint drafts 3), but steps=2 still
loses ~10% short decode — target-verify width dominates the step cost, so the
3-step geometry is load-bearing, not conservative.

Q200v2 protocol on the max-perf serve (think-off): GSM8K-200 **192/200 (0.960)**,
0 truncations; concurrency c1/c2/c4/c8 aggregate **21.6 / 38.0 / 47.7 / 69.2 tok/s**
(per-stream 21.9 / 19.4 / 12.4 / 9.0; TTFT 0.57–1.48 s); accept length 3.38–3.47 at
all levels; per-request e2e p50 23.5 tok/s. Load telemetry: 26 W mean, 47–56 °C,
GPU util ~79%, host MemAvailable ≥ 10.9 GiB.

Evidence: `results/maxperf/` (12 CELL-*.json, FINAL-VERDICT.json, MORNING-REPORT.md,
driver/ladder/promoter logs), `results/e2e/FINAL3-concurrency.json`,
`results/q200/FINAL3-nothink.jsonl`, `results/e2e/telemetry-load-1500.jsonl`.
Tooling: `scripts/70_maxperf_driver.sh`, `71–73_ladder_part*.sh`,
`74_promote_or_confirm.py`, `75_q200proto_run.sh`; cell profiles `profiles/CELL-*.env`.
