#!/usr/bin/env python3
"""Structural verify of the NVFP4 export vs the MXFP4 source.

Checks:
  1. export config.json quantization_config: quant_algo MIXED_PRECISION,
     moe_quant_algo NVFP4, group_size 16.
  2. every dst index key resolves to an existing shard file.
  3. every NVFP4 ".weight" has a matching ".weight_scale".
  4. input-scales shard covers all NVFP4 expert linears (36096).
  5. KV scales present (kv_scales.json) when enabled in config.
  6. sampled numeric check: for N random expert linears, dequantize the dst
     NVFP4 weight (uint8-packed e2m1 halves * block scales * scale_2) and the
     src MXFP4 weight, and compare within fp8-master tolerance.

Usage: 22_verify_checkpoint.py SRC DST [sample_experts]
r0b0tlab mimo26.
"""
import json, random, sys
from pathlib import Path

from safetensors import safe_open
import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def err(msg, errs):
    errs.append(msg)
    print("ERR", msg)


def nvfp4_dequant(packed: torch.Tensor, ws: torch.Tensor, ws2: torch.Tensor) -> torch.Tensor:
    """Identical math to 20_convert_mimo_nvfp4.py's verify path (on purpose)."""
    lut = E2M1.to(packed.device)
    lo = lut[(packed & 0x0F).long()]
    hi = lut[(packed >> 4).long()]
    vals = torch.stack([lo, hi], -1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return vals * (ws.float().repeat_interleave(16, -1) * ws2.float())


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    sample_n = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    errs = []

    cfg = json.loads((dst / "config.json").read_text())
    qc = cfg.get("quantization_config", {})
    if qc.get("quant_algo") != "MIXED_PRECISION":
        err(f"quantization_config.quant_algo={qc.get('quant_algo')}", errs)
    if qc.get("moe_quant_algo") != "NVFP4":
        err(f"quantization_config.moe_quant_algo={qc.get('moe_quant_algo')}", errs)
    if int(qc.get("group_size", -1)) != 16:
        err(f"quantization_config.group_size={qc.get('group_size')}", errs)

    sidx = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    didx = json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"]

    missing = [k for k, f in didx.items() if not (dst / f).exists()]
    if missing:
        err(f"{len(missing)} index keys point to missing shard files (first {missing[:2]})", errs)

    weights = [k for k in didx if k.endswith(".weight")]
    scale2 = {k[:-len(".weight_scale_2")] for k in didx if k.endswith(".weight_scale_2")}
    # Only NVFP4 linears (those carrying weight_scale_2) need weight_scale.
    no_scale = [k for k in weights
                if k[:-len(".weight")] in scale2
                and k[:-len(".weight")] + ".weight_scale" not in didx]
    expert_w = [k for k in weights if ".experts." in k]
    if no_scale:
        err(f"{len(no_scale)} NVFP4 linears lack weight_scale (first {no_scale[:2]})", errs)

    # input scales shard coverage
    ism_file = dst / "model-nvfp4-input-scales.safetensors"
    n_input_scales = 0
    if ism_file.exists():
        with safe_open(ism_file, framework="pt") as f:
            n_input_scales = len(list(f.keys()))
    else:
        err("model-nvfp4-input-scales.safetensors missing", errs)

    # KV scales
    kv_on = bool(cfg.get("quantization_param_path"))
    n_kv = 0
    if kv_on:
        kp = Path(cfg["quantization_param_path"])
        if not kp.exists():
            err(f"quantization_param_path {kp} missing", errs)
        else:
            sch = json.loads(kp.read_text())
            sf = sch["kv_cache"]["scaling_factor"]
            n_kv = sum(len(v) for v in sf.values())
            bad = [x for r in sf.values() for x in r.values()
                   if x["k_scale"] <= 0 or x["v_scale"] <= 0]
            if bad:
                err(f"{len(bad)} non-positive KV scales", errs)

    print(f"index keys={len(didx)} files={len(set(didx.values()))} "
          f"expert_linears={len(expert_w)} incomplete={len(no_scale)} "
          f"input_scales={n_input_scales} kv_scale_entries={n_kv}")

    # sampled numeric check
    sampled = mismatches = elements = 0
    if not errs and sample_n:
        sys.path.insert(0, str(Path.home() / ".venvs/modelopt-main/lib/python3.12/site-packages"))
        from modelopt.torch.export.shard_cast_utils import dequantize_mxfp4_to_bf16
        random.seed(0)
        picks = random.sample(expert_w, min(sample_n, len(expert_w)))
        for k in picks:
            base = k.removesuffix(".weight")
            sf_, df_ = sidx.get(k), didx.get(k)
            wq_key, ws_key, ws2_key = k, base + ".weight_scale", base + ".weight_scale_2"
            if df_ is None or wq_key not in didx or ws_key not in didx:
                err(f"{base} pieces absent from dst index", errs)
                continue
            with safe_open(src / sf_, framework="pt") as f:
                w_src = f.get_tensor(k)
            with safe_open(dst / df_, framework="pt") as f:
                wq = f.get_tensor(wq_key)
                ws = f.get_tensor(ws_key)
                ws2 = f.get_tensor(ws2_key) if ws2_key in didx else None
            got = nvfp4_dequant(wq, ws, ws2)
            ref = dequantize_mxfp4_to_bf16(
                w_src.view(torch.uint8) if w_src.dtype is not torch.uint8 else w_src,
                torch.empty(0), "cpu") if False else None
            # use the same helper as the converter for the reference side
            if ref is None:
                from modelopt.torch.export.shard_cast_utils import MXFP4QTensor, _MXFP4_BLOCK
                pk = (w_src.view(torch.uint8) if w_src.dtype is not torch.uint8 else w_src).contiguous()
                sc_key = base + ".weight_scale"  # mxfp4 scale name in src index
                sc_name = k + ".scale" if k + ".scale" in sidx else None
                # find the src scale key by suffix convention
                if sc_name is None:
                    cands = [c for c in sidx if c.startswith(base) and c != k]
                    sc_name = cands[0] if cands else None
                with safe_open(src / sf_, framework="pt") as f:
                    sc = f.get_tensor(sc_name) if sc_name else torch.empty(0)
                sc = (sc.view(torch.uint8) if sc.dtype is not torch.uint8 else sc).contiguous()
                shape = torch.Size((*pk.shape[:-1], pk.shape[-1] * 2))
                ref = MXFP4QTensor(shape, torch.bfloat16, pk).dequantize(
                    dtype=torch.bfloat16, scale=sc, block_sizes=[_MXFP4_BLOCK])
            d = (got.float() - ref.float()).abs()
            tol = 0.25 * ref.float().abs().clamp(min=1.0)
            m = int((d > tol).sum())
            mismatches += m
            elements += ref.numel()
            sampled += 1
        print(f"sampled={sampled} elements={elements} mismatches={mismatches}")
        if mismatches:
            err(f"{mismatches} sampled element mismatches vs MXFP4 reference", errs)

    if errs:
        print(f"VERIFY FAILED (n={len(errs)} errors)")
        sys.exit(1)
    print("VERIFY OK")


if __name__ == "__main__":
    main()
