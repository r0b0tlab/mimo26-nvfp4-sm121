# Notices

Original scripts, docs, and evidence in this repository are MIT. Copyright (c) 2026 r0b0tlab. See LICENSE.

That grant does not relicense the upstream files listed below, and it does not cover the model weights. The weights are not in this git tree.

## Upstream code kept in this tree

These files are modified copies. Their upstream notices stay in force. The Apache License 2.0 text is `patches/sglang-582389ce/LICENSE-APACHE-2.0`.

- `patches/sglang-582389ce/mimo_v2.py` and `mimo_v2_nextn.py`. SGLang, Apache-2.0. Copyright 2023-2024 SGLang Team, as stated in those files. https://github.com/sgl-project/sglang
- `patches/sglang-582389ce/triton_backend.py`. SGLang attention backend, Apache-2.0. The upstream nightly file has no file-level copyright header. r0b0tlab change: fill the sliding-window KV index during draft-extend CUDA-graph capture.
- `patches/sglang-582389ce/modelopt_quant.py`. Adapted from the vLLM project, Apache-2.0. SPDX header is in the file. https://github.com/vllm-project/vllm. r0b0tlab change: skip the unused NVFP4 swizzle buffers when the MoE runner is Marlin.
- `patches/sglang-582389ce/test_draft_extend_window.py`. r0b0tlab test. MIT.

## Weights

The quantized checkpoint is a derivative of XiaomiMiMo/MiMo-V2.6-Flash-RL, MIT, commit `5711b268`. XiaomiMiMo is the base-model author. r0b0tlab claims the quantization deltas only. Card: https://huggingface.co/r0b0tlab/MiMo-V2.6-Flash-RL-NVFP4

## Tools used, not copied

- NVIDIA Model Optimizer, commit `7159c01d`. Quantization used its public APIs. No ModelOpt source is in this tree.
- SGLang nightly `20260922-582389ce`, including FlashInfer 0.6.18 and PyTorch 2.13.0+cu130, as shipped in the runtime image.
- Marlin MoE kernels as selected by SGLang `--moe-runner-backend marlin`. Not vendored here.
- EAGLE speculative decoding as implemented by SGLang. That MTP path is the preferred serve profile: 3 steps, 4 draft tokens, top-k 1.
- DFlash draft weights ship with the base checkpoint. That lane is recorded. It is not the preferred profile.
- r0b0bench, MIT, is the systems harness. https://github.com/r0b0tlab/r0b0bench
- BFCL scores use the Berkeley Function-Calling Leaderboard tasks.

No third-party quantization code or configs were copied.
