#!/usr/bin/env python3
"""MiMo-V2.6-Flash: MXFP4 routed experts -> NVFP4 (hybrid FP8 + NVFP4-MoE checkpoint).

r0b0tlab campaign mimo26-nvfp4-sm121.  Adapted from NVIDIA Model-Optimizer
examples/deepseek/deepseek_v4/quantize_to_nvfp4.py (Apache-2.0) and built on ModelOpt main
``modelopt.torch.export.shard_cast_utils``:

* routed experts ``model.layers.L.mlp.experts.E.{gate,up,down}_proj``: closed-form MXFP4 -> NVFP4
  cast (``quantize_mxfp4_to_nvfp4_lossless``); gate/up share k_max so the fused GEMM1 sees one
  ``weight_scale_2`` per expert.  Blocks whose E8M0 exponent is inside the 17-binade E4M3 window
  below k_max are bit-exact; the non-exact block count is reported and sampled experts are
  re-dequantised and compared element-wise against the MXFP4 source.
* ``input_scale`` = calibrated activation amax / (6 * 448), one value per layer and projection
  shared by all experts (ModelOpt MoE peer-max sync).  The amax comes from 10_calibrate_mimo.py
  (ModelOpt ``NVFP4ActHeadroomCalibrator`` per expert, native top-k routing).  Scales live in a
  separate small shard (``--phase scales``) so calibration policies can be swapped without
  rewriting the 170 GB of expert weights.
* everything else passes through byte-identical: 128x128 block-FP8 attention / dense MLP / MTP,
  BF16 o_proj, router, embeddings, lm_head, vision and audio towers, DFlash draft, tokenizer.
* config.json: quant_method=fp8 + quant_algo=MIXED_PRECISION + moe_quant_algo=NVFP4 + group_size=16
  (SGLang ``HybridFp8NvFp4Config``); ``store_dtype``/``mxfp4_block_size`` dropped because no MXFP4
  tensors remain.  ``hf_quant_config.json`` manifest as in the ModelOpt DSV4 export.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale)$")
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
EXCLUDE = [
    "lm_head",
    "model.embed_tokens",
    "*.self_attn.*",
    "model.layers.0.mlp.*",
    "*.mlp.gate",
    "model.mtp*",
    "visual*",
    "audio_encoder*",
    "speech_embeddings*",
]
SCALES_FILE = "model-nvfp4-input-scales.safetensors"
SKIP_TOP = {"model.safetensors.index.json", "config.json", ".cache", "hf_quant_config.json", SCALES_FILE}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def nvfp4_dequant(packed: torch.Tensor, ws: torch.Tensor, ws2: torch.Tensor) -> torch.Tensor:
    lut = E2M1.to(packed.device)
    lo = lut[(packed & 0x0F).long()]
    hi = lut[(packed >> 4).long()]
    vals = torch.stack([lo, hi], -1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return vals * (ws.float().repeat_interleave(16, -1) * ws2.float())


def moe_layers(cfg) -> list[int]:
    return [l for l, f in enumerate(cfg["moe_layer_freq"]) if f]


def phase_weights(args):
    from modelopt import __version__ as mo_version
    from modelopt.torch.export.shard_cast_utils import (
        dequantize_mxfp4_to_bf16,
        link_or_copy,
        mxfp4_kmax,
        quantize_mxfp4_to_nvfp4_lossless,
    )

    dev = "cuda"
    t0 = time.time()
    src, dst = args.src, args.dst
    dst.mkdir(parents=True, exist_ok=True)
    index = json.loads((src / "model.safetensors.index.json").read_text())
    cfg = json.loads((src / "config.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    if args.num_workers > 1:
        todo = shards[args.worker_id :: args.num_workers]
    else:
        todo = [s for s in shards if not args.shards or s in args.shards.split(",")]
    todo = [s for s in todo if not (dst / s).exists()]  # resume: shards are renamed into place atomically
    stats_dir = dst / ".conv_stats"
    stats_dir.mkdir(exist_ok=True)
    tot_blocks = tot_lossless = 0
    verify = {"experts": 0, "mismatch_elems": 0, "checked_elems": 0}
    for si, shard in enumerate(todo):
        ts = time.time()
        f = safe_open(str(src / shard), framework="pt", device="cpu")
        groups, passthrough = defaultdict(dict), []
        for k in f.keys():
            m = EXPERT_RE.match(k)
            if m:
                groups[(int(m.group(1)), int(m.group(2)))][(m.group(3), m.group(4))] = k
            else:
                passthrough.append(k)
        if not groups:
            if not (dst / shard).exists():
                link_or_copy(src / shard, dst / shard)
            log(f"[{si + 1}/{len(todo)}] {shard}: no routed experts, hard-linked")
            continue
        out = {k: f.get_tensor(k) for k in passthrough}
        nverify = 0
        for (l, e), parts in sorted(groups.items()):
            assert len(parts) == 6, (shard, l, e, sorted(parts))
            t = {pk: f.get_tensor(kk) for pk, kk in parts.items()}
            k13 = max(mxfp4_kmax(t[("gate_proj", "weight_scale")], dev), mxfp4_kmax(t[("up_proj", "weight_scale")], dev))
            kdn = mxfp4_kmax(t[("down_proj", "weight_scale")], dev)
            do_verify = nverify < args.verify_per_shard
            for proj, kmax in (("gate_proj", k13), ("up_proj", k13), ("down_proj", kdn)):
                w, s = t[(proj, "weight")], t[(proj, "weight_scale")]
                q, ws, ws2, nb, nl = quantize_mxfp4_to_nvfp4_lossless(w, s, kmax, dev)
                tot_blocks += nb
                tot_lossless += nl
                base = f"model.layers.{l}.mlp.experts.{e}.{proj}"
                if do_verify:
                    ref = dequantize_mxfp4_to_bf16(w, s, dev).float()
                    got = nvfp4_dequant(q, ws, ws2)
                    verify["mismatch_elems"] += int((ref != got).sum())
                    verify["checked_elems"] += ref.numel()
                out[base + ".weight"] = q.cpu()
                out[base + ".weight_scale"] = ws.cpu()
                out[base + ".weight_scale_2"] = ws2.float().reshape(()).cpu()
            if do_verify:
                nverify += 1
                verify["experts"] += 1
        tmp = dst / (shard + ".tmp")
        save_file(out, str(tmp), metadata={"format": "pt"})
        tmp.rename(dst / shard)
        del out
        log(f"[{si + 1}/{len(todo)}] {shard}: {len(groups)} experts in {time.time() - ts:.1f}s "
            f"(lossless {tot_lossless}/{tot_blocks} blocks; verify mismatches {verify['mismatch_elems']}/{verify['checked_elems']})")
    tag = f"worker{args.worker_id}of{args.num_workers}" if args.num_workers > 1 else f"main{int(t0)}"
    (stats_dir / f"{tag}.json").write_text(json.dumps(
        {"shards": todo, "blocks": tot_blocks, "lossless_blocks": tot_lossless, "verify": verify, "wall_s": round(time.time() - t0, 1)}))

    if args.shards or args.num_workers > 1:
        log("partial run (debug subset or worker): skipping index/config/aux finalize")
        return
    missing = [s for s in shards if not (dst / s).exists()]
    assert not missing, f"shards missing in dst: {missing}"
    # Keys added per shard + quantized-linear manifest come from the source index (no tensor reads).
    wm = dict(index["weight_map"])
    quantized = set()
    for k, shard in index["weight_map"].items():
        m = EXPERT_RE.match(k)
        if m and m.group(4) == "weight":
            base = k[: -len(".weight")]
            quantized.add(base)
            wm[base + ".weight_scale_2"] = shard
    agg = {"blocks": 0, "lossless_blocks": 0, "verify": {"experts": 0, "mismatch_elems": 0, "checked_elems": 0}, "runs": []}
    for p in sorted(stats_dir.glob("*.json")):
        r = json.loads(p.read_text())
        agg["blocks"] += r["blocks"]
        agg["lossless_blocks"] += r["lossless_blocks"]
        for k in agg["verify"]:
            agg["verify"][k] += r["verify"][k]
        agg["runs"].append({"file": p.name, "n_shards": len(r["shards"]), "wall_s": r["wall_s"]})
    meta = dict(index.get("metadata", {}))
    meta["r0b0tlab_quantization"] = "routed experts NVFP4 (ModelOpt closed-form MXFP4 cast; calibrated input_scale in " + SCALES_FILE + ")"
    (dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": meta, "weight_map": wm}, indent=2))
    qlayers = sorted(quantized)
    manifest = {
        "producer": {"name": "modelopt", "version": mo_version, "recipe": "mimo26-nvfp4-experts"},
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "group_size": 16,
            "quantized_layers": {n: {"quant_algo": "NVFP4", "group_size": 16} for n in qlayers},
            "exclude_modules": EXCLUDE,
        },
    }
    (dst / "hf_quant_config.json").write_text(json.dumps(manifest, indent=2))
    qc = dict(cfg.get("quantization_config") or {})
    dropped = {k: qc.pop(k) for k in ("store_dtype", "mxfp4_block_size") if k in qc}
    qc.setdefault("activation_scheme", "dynamic")
    qc["quant_method"] = "fp8"
    qc.setdefault("weight_block_size", [128, 128])
    qc.update(
        moe_quant_algo="NVFP4", quant_algo="MIXED_PRECISION", kv_cache_quant_algo=None, group_size=16,
        producer=manifest["producer"], quantized_layers=manifest["quantization"]["quantized_layers"], ignore=EXCLUDE,
        config_groups={"group_0": {
            "input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            "targets": ["Linear"]}},
    )
    cfg["quantization_config"] = qc
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    nlink = 0
    shard_set = set(shards)
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if rel.parts[0] in SKIP_TOP or "__pycache__" in rel.parts or not p.is_file():
            continue
        if len(rel.parts) == 1 and p.name in shard_set:
            continue
        q = dst / rel
        if q.exists():
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        link_or_copy(p, q)
        nlink += 1
    rep = {
        "src": str(src), "dst": str(dst), "modelopt": mo_version, "blocks": agg["blocks"],
        "lossless_blocks": agg["lossless_blocks"], "non_exact_blocks": agg["blocks"] - agg["lossless_blocks"],
        "verify": agg["verify"], "runs": agg["runs"], "quantized_linears": len(qlayers),
        "config_dropped": dropped, "aux_linked": nlink, "finalize_wall_s": round(time.time() - t0, 1),
    }
    (dst / "r0b0tlab_conversion_weights.json").write_text(json.dumps(rep, indent=2))
    log(f"WEIGHTS DONE {json.dumps(rep)}")


def phase_scales(args):
    dst = args.dst
    calib = json.loads(args.calib.read_text())
    cfg = json.loads((dst / "config.json").read_text())
    E = cfg["n_routed_experts"]
    tensors, table = {}, {}
    for l in moe_layers(cfg):
        rec = calib["layers"][str(l)]
        a13, a2 = float(rec["cand_w13"][args.act_policy]), float(rec["cand_w2"][args.act_policy])
        assert a13 > 0 and a2 > 0, (l, a13, a2)
        table[l] = {"w13_amax": a13, "w2_amax": a2,
                    "relmse_w13": rec.get("qdq_relmse_w13", {}).get(args.act_policy),
                    "relmse_w2": rec.get("qdq_relmse_w2", {}).get(args.act_policy)}
        for e in range(E):
            for proj, a in (("gate_proj", a13), ("up_proj", a13), ("down_proj", a2)):
                tensors[f"model.layers.{l}.mlp.experts.{e}.{proj}.input_scale"] = torch.tensor(a / (6.0 * 448.0), dtype=torch.float32)
    save_file(tensors, str(dst / SCALES_FILE), metadata={"format": "pt", "act_policy": args.act_policy})
    idx_p = dst / "model.safetensors.index.json"
    index = json.loads(idx_p.read_text())
    for k in tensors:
        index["weight_map"][k] = SCALES_FILE
    idx_p.write_text(json.dumps(index, indent=2))
    cfg["quantization_config"]["producer"]["recipe"] = f"mimo26-nvfp4-experts-{args.act_policy}"
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    hq = json.loads((dst / "hf_quant_config.json").read_text())
    hq["producer"]["recipe"] = cfg["quantization_config"]["producer"]["recipe"]
    (dst / "hf_quant_config.json").write_text(json.dumps(hq, indent=2))
    rep = {"act_policy": args.act_policy, "calib": str(args.calib), "calib_nll": calib.get("nll"),
           "calib_meta": calib.get("meta"), "n_input_scales": len(tensors), "layers": table}
    (dst / "r0b0tlab_conversion_scales.json").write_text(json.dumps(rep, indent=2))
    log(f"SCALES DONE policy={args.act_policy} n={len(tensors)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--phase", choices=["weights", "scales"], required=True)
    ap.add_argument("--calib", type=Path, help="calib_stats.json from 10_calibrate_mimo.py (phase scales)")
    ap.add_argument("--act_policy", default="peer_headroom", choices=["peer_headroom", "union_headroom", "max", "unit_scale"])
    ap.add_argument("--verify_per_shard", type=int, default=2)
    ap.add_argument("--shards", default="", help="comma list (debug)")
    ap.add_argument("--worker_id", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=1)
    args = ap.parse_args()
    phase_weights(args) if args.phase == "weights" else phase_scales(args)


if __name__ == "__main__":
    main()
