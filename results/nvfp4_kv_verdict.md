# NVFP4 KV cache verdict — MiMo-V2.6-Flash on SM121 (GB10), SGLang 582389ce

**Verdict: NOT SUPPORTED for this model/engine/hardware combination.** FP8 KV with calibrated scales ships instead.

## Evidence (captured 2026-09-24 14:36Z, node 3 = gn100-2eea)

Boot attempt: `TAG=kv4 MODEL=<our NVFP4 export> KV=dtype-only` (`--kv-cache-dtype nvfp4`),
MoE runner marlin, TP=2/EP=2 across nodes 3+4. Weights loaded fine — the hybrid
FP8+NVFP4 checkpoint auto-detects cleanly ("Auto-detected hybrid FP8+NVFP4
checkpoint (NVFP4 MoE group_size=16)") and the engine even prints
"NVFP4 KV Cache might lead to an accuracy drop!" — but pool allocation then dies:

```
kv_cache_configurator.py:1811  _build_hybrid_swa_kv_pool -> SWAKVPool
memory_pool.py:2343  _create_buffers_normal
    torch.zeros(k_shape, dtype=self.store_dtype, device=self.device)
NotImplementedError: "fill_cuda" not implemented for 'Float4_e2m1fn_x2'
```

Root cause chain, matching the code reading in plan §6:
1. MiMo is a hybrid SWA/full model, so pool construction always routes through
   `_build_hybrid_swa_kv_pool` -> `SWAKVPool`.
2. With `--kv-cache-dtype nvfp4`, `store_dtype` becomes `Float4_e2m1fn_x2`, and
   the hybrid pool's buffer creation does `torch.zeros(..., dtype=store_dtype)`.
3. PyTorch has no `fill_cuda` kernel for Float4_e2m1fn_x2 — the allocation itself
   cannot be created. There is no FP4 pool class wired for the hybrid path in
   this build (the standalone FP4 pool `MHATokenToKVPoolFP4` is not hybrid-aware).

This is a hard engine limitation, not a config error on our side. The failure is
deterministic (pool init, before any forward).

## Decision

Ship `--kv-cache-dtype fp8_e4m3` with calibrated per-layer scales
(`kv_scales.json` via `quantization_param_path`, the KV=fp8scales lane), and
record the 1M/NIAH work against that configuration.
