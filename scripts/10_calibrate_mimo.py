#!/usr/bin/env python3
"""Layer-streamed ModelOpt calibration for XiaomiMiMo/MiMo-V2.6-Flash-RL on one GB10.

r0b0tlab campaign mimo26-nvfp4-sm121.

The source checkpoint stores routed experts as MXFP4 (U8 E2M1 nibbles + U8 E8M0 scale per
32) and attention / dense MLP as 128x128 block FP8.  It does not fit in one GB10, and the
vendor HF modeling code has no MXFP4 loader, so we stream one decoder layer at a time:

  * build the vendor ``MiMoV2DecoderLayer`` (trust_remote_code module shipped with the
    checkpoint) on the meta device and materialise it with dequantised BF16 weights:
      - routed experts: ``modelopt.torch.export.shard_cast_utils.dequantize_mxfp4_to_bf16``
      - FP8 block tensors: ``w_fp8 * weight_scale_inv`` per 128x128 block
      - fused ``qkv_proj``: the release is TP=4-interleaved with per-shard block scales
        (SGLang ``load_mimo_v2_qkv_proj_weight``), so each of the 4 shards is dequantised
        with its own scale rows and de-interleaved to canonical [Q; K; V]
  * run the calibration tokens through the layer and feed ModelOpt calibrators:
      - routed-expert NVFP4 activation amax with ModelOpt's ``NVFP4ActHeadroomCalibrator``
        (per expert, native top-k routing; export applies ModelOpt's MoE peer-max sync)
      - KV cache amax (post-RoPE K, post-``attention_value_scale`` V) per layer
  * propagate the *unquantised* layer output to the next layer (ModelOpt layerwise
    ``get_qdq_activations_from_prev_layer: false``)

The final norm + lm_head next-token NLL on the calibration tokens is an end-to-end check that
the streamed forward (dequant, de-interleave, routing, attention) is correct.
Calibration data: ModelOpt default ``cnn_nemotron_v2_mix`` (hf_ptq default when --dataset is
unset), packed rows (``pack=True``, as ModelOpt recommends for calibration callers).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Ckpt:
    def __init__(self, src: Path):
        self.src = Path(src)
        self.wm = json.loads((self.src / "model.safetensors.index.json").read_text())["weight_map"]
        self._h = {}

    def get(self, key: str, device="cpu") -> torch.Tensor:
        fn = self.wm[key]
        h = self._h.get(fn)
        if h is None:
            h = self._h[fn] = safe_open(str(self.src / fn), framework="pt", device="cpu")
        t = h.get_tensor(key)
        return t.to(device, non_blocking=True) if str(device) != "cpu" else t

    def has(self, key: str) -> bool:
        return key in self.wm


def dequant_fp8_block(w: torch.Tensor, s: torch.Tensor, block: int = 128) -> torch.Tensor:
    n, k = w.shape
    se = s.float().repeat_interleave(block, 0)[:n].repeat_interleave(block, 1)[:, :k]
    return (w.float() * se).to(torch.bfloat16)


def attn_dims(cfg, layer_idx: int):
    is_swa = cfg.hybrid_layer_pattern[layer_idx] == 1
    if is_swa:
        nh = getattr(cfg, "swa_num_attention_heads", cfg.num_attention_heads)
        nkv = getattr(cfg, "swa_num_key_value_heads", cfg.num_key_value_heads)
        hd = getattr(cfg, "swa_head_dim", cfg.head_dim)
        vhd = getattr(cfg, "swa_v_head_dim", getattr(cfg, "v_head_dim", hd))
    else:
        nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        vhd = getattr(cfg, "v_head_dim", hd)
    return is_swa, nh, nkv, hd, vhd


def load_qkv(ck: Ckpt, cfg, layer_idx: int, prefix: str, device, layout: str) -> torch.Tensor:
    w = ck.get(prefix + "self_attn.qkv_proj.weight", device)
    s = ck.get(prefix + "self_attn.qkv_proj.weight_scale_inv", device)
    _, nh, nkv, hd, vhd = attn_dims(cfg, layer_idx)
    if layout == "canonical":
        return dequant_fp8_block(w, s)
    ckpt_tp = cfg.num_key_value_heads  # == SGLang get_mimo_v2_fused_qkv_expected_tp_size
    assert w.shape[0] % ckpt_tp == 0 and s.shape[0] % ckpt_tp == 0, (w.shape, s.shape)
    q = (nh // ckpt_tp) * hd
    k = max(1, nkv // ckpt_tp) * hd
    v = max(1, nkv // ckpt_tp) * vhd
    shards = [dequant_fp8_block(a, b) for a, b in zip(w.chunk(ckpt_tp, 0), s.chunk(ckpt_tp, 0))]
    assert shards[0].shape[0] == q + k + v, (shards[0].shape, q, k, v)
    return torch.cat([x[:q] for x in shards] + [x[q : q + k] for x in shards] + [x[q + k :] for x in shards], 0)


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MID = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def nvfp4_act_relmse(x: torch.Tensor, amax: float) -> float:
    """Relative MSE of runtime-style NVFP4 activation QDQ (static global scale, dynamic E4M3 block-16 scales)."""
    if not (amax and math.isfinite(amax) and amax > 0):
        return float("nan")
    xb = x.float().reshape(-1, 16)
    gs = float(amax) / (6.0 * 448.0)
    bamax = xb.abs().amax(-1, keepdim=True)
    bs = (bamax / 6.0 / gs).clamp(max=448.0).to(torch.float8_e4m3fn).float() * gs
    safe = torch.where(bs > 0, bs, torch.ones_like(bs))
    v = (xb.abs() / safe).clamp(max=6.0)
    grid = torch.tensor(_E2M1, device=x.device)
    mid = torch.tensor(_E2M1_MID, device=x.device)
    q = grid[torch.bucketize(v, mid)] * xb.sign() * bs
    q = torch.where(bs > 0, q, torch.zeros_like(q))
    return float((q - xb).pow(2).sum() / xb.pow(2).sum().clamp(min=1e-30))


def build_calib_rows(src: Path, n: int, seq: int, dataset: str) -> torch.Tensor:
    from transformers import AutoTokenizer

    from modelopt.torch.utils.dataset_utils import get_dataset_dataloader

    tok = AutoTokenizer.from_pretrained(str(src), trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dl = get_dataset_dataloader(
        dataset_name=dataset, tokenizer=tok, batch_size=16, num_samples=n, max_sample_length=seq, pack=True
    )
    rows = []
    for b in dl:
        ids, am = b["input_ids"], b.get("attention_mask")
        for i in range(ids.shape[0]):
            if ids.shape[1] == seq and (am is None or bool(am[i].all())):
                rows.append(ids[i])
    assert rows, "no full calibration rows"
    return torch.stack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset", default="cnn_nemotron_v2_mix")
    ap.add_argument("--calib_size", type=int, default=512)
    ap.add_argument("--calib_seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_layers", type=int, default=0, help="debug: stop after N layers (no NLL)")
    ap.add_argument("--qkv_layout", choices=["tp4_interleaved", "canonical"], default="tp4_interleaved")
    ap.add_argument("--qdq_sample_rows", type=int, default=32768, help="routed rows sampled per layer for QDQ scoring")
    args = ap.parse_args()

    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from modelopt.torch.export.shard_cast_utils import dequantize_mxfp4_to_bf16
    from modelopt.torch.quantization.calib.nvfp4_act_headroom import NVFP4ActHeadroomCalibrator

    t_start = time.time()
    dev = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    ck = Ckpt(args.src)
    cfg = AutoConfig.from_pretrained(str(args.src), trust_remote_code=True)
    cfg._attn_implementation = "eager"  # sinks need eager (the vendor SDPA path falls back too)
    DL = get_class_from_dynamic_module("modeling_mimo_v2.MiMoV2DecoderLayer", str(args.src))
    Rot = get_class_from_dynamic_module("modeling_mimo_v2.MiMoV2RotaryEmbedding", str(args.src))
    Norm = get_class_from_dynamic_module("modeling_mimo_v2.MiMoV2RMSNorm", str(args.src))
    MoE = get_class_from_dynamic_module("modeling_mimo_v2.MiMoV2MoE", str(args.src))
    M = sys.modules[DL.__module__]
    L = cfg.num_hidden_layers if not args.max_layers else args.max_layers
    E = cfg.n_routed_experts

    ids = build_calib_rows(args.src, args.calib_size, args.calib_seq, args.dataset)
    N, S = ids.shape
    log(f"calib rows={N} seq={S} tokens={N * S} dataset={args.dataset} qkv_layout={args.qkv_layout}")
    ids = ids.to(dev)

    emb = ck.get("model.embed_tokens.weight", dev)
    H = F.embedding(ids, emb)
    del emb
    H2 = torch.empty_like(H)
    pos = torch.arange(S, device=dev)[None]
    pe_full = Rot(cfg, is_swa=False, device=dev)(H[:1], pos)
    pe_swa = Rot(cfg, is_swa=True, device=dev)(H[:1], pos)
    neg = torch.finfo(torch.bfloat16).min
    i_idx = torch.arange(S, device=dev)[:, None]
    j_idx = torch.arange(S, device=dev)[None, :]
    causal = j_idx <= i_idx
    swa_ok = causal & (j_idx > i_idx - cfg.sliding_window)
    mask_full = torch.where(causal, 0.0, neg).to(torch.bfloat16)[None, None]
    mask_swa = torch.where(swa_ok, 0.0, neg).to(torch.bfloat16)[None, None]

    # KV-cache statistics through the vendor eager attention function.
    kv = {}
    orig_eager = M.eager_attention_forward

    def eager_wrapper(module, query, key, value, attention_mask, scaling, dropout=0.0, sinks=None, **kw):
        d = kv.setdefault(
            module.layer_idx,
            {"k": NVFP4ActHeadroomCalibrator(), "v": NVFP4ActHeadroomCalibrator(), "kh": None, "vh": None},
        )
        d["k"].collect(key)
        d["v"].collect(value)
        kh = key.detach().abs().amax(dim=(0, 2, 3)).float()
        vh = value.detach().abs().amax(dim=(0, 2, 3)).float()
        d["kh"] = kh if d["kh"] is None else torch.maximum(d["kh"], kh)
        d["vh"] = vh if d["vh"] is None else torch.maximum(d["vh"], vh)
        return orig_eager(module, query, key, value, attention_mask, scaling, dropout=dropout, sinks=sinks, **kw)

    M.eager_attention_forward = eager_wrapper

    stats = {"layers": {}, "kv": {}, "meta": vars(args) | {"src": str(args.src), "out": str(args.out)}}
    amax_pt = {}
    for l in range(L):
        t0 = time.time()
        p = f"model.layers.{l}."
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            layer = DL(cfg, l, attention_projection_layout="fused_qkv")
        torch.set_default_dtype(torch.float32)
        sd = {
            "input_layernorm.weight": ck.get(p + "input_layernorm.weight", dev),
            "post_attention_layernorm.weight": ck.get(p + "post_attention_layernorm.weight", dev),
            "self_attn.qkv_proj.weight": load_qkv(ck, cfg, l, p, dev, args.qkv_layout),
            "self_attn.o_proj.weight": ck.get(p + "self_attn.o_proj.weight", dev),
        }
        if layer.self_attn.attention_sink_bias is not None:
            sd["self_attn.attention_sink_bias"] = ck.get(p + "self_attn.attention_sink_bias", dev)
        is_moe = isinstance(layer.mlp, MoE)
        if is_moe:
            sd["mlp.gate.weight"] = ck.get(p + "mlp.gate.weight", dev).float()
            sd["mlp.gate.e_score_correction_bias"] = ck.get(p + "mlp.gate.e_score_correction_bias", dev).float()
            for proj in ("gate_proj", "up_proj", "down_proj"):
                # One stacked dequant per projection (the helper accepts leading dims).
                packed = torch.stack([ck.get(f"{p}mlp.experts.{e}.{proj}.weight") for e in range(E)])
                scale = torch.stack([ck.get(f"{p}mlp.experts.{e}.{proj}.weight_scale") for e in range(E)])
                deq = dequantize_mxfp4_to_bf16(packed, scale, dev)
                del packed, scale
                for e in range(E):
                    sd[f"mlp.experts.{e}.{proj}.weight"] = deq[e]
        else:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                base = f"{p}mlp.{proj}"
                sd[f"mlp.{proj}.weight"] = dequant_fp8_block(
                    ck.get(base + ".weight", dev), ck.get(base + ".weight_scale_inv", dev)
                )
        layer.load_state_dict(sd, strict=True, assign=True)
        del sd
        layer.eval().requires_grad_(False)
        t_load = time.time() - t0

        handles = []
        if is_moe:
            c13 = [NVFP4ActHeadroomCalibrator() for _ in range(E)]
            c2 = [NVFP4ActHeadroomCalibrator() for _ in range(E)]
            tok = torch.zeros(E, dtype=torch.long)
            keep_p = min(1.0, args.qdq_sample_rows / float(N * S * cfg.num_experts_per_tok))
            s13, s2 = [], []
            for e, ex in enumerate(layer.mlp.experts):
                def h13(mod, inp, e=e):
                    x = inp[0]
                    c13[e].collect(x)
                    tok[e] += x.shape[0]
                    s13.append(x[torch.rand(x.shape[0], device=x.device) < keep_p])

                def h2(mod, inp, e=e):
                    x = inp[0]
                    c2[e].collect(x)
                    s2.append(x[torch.rand(x.shape[0], device=x.device) < keep_p])

                handles.append(ex.gate_proj.register_forward_pre_hook(h13))
                handles.append(ex.down_proj.register_forward_pre_hook(h2))

        is_swa = cfg.hybrid_layer_pattern[l] == 1
        pe, mask = (pe_swa, mask_swa) if is_swa else (pe_full, mask_full)
        with torch.no_grad():
            for b0 in range(0, N, args.batch):
                H2[b0 : b0 + args.batch] = layer(
                    H[b0 : b0 + args.batch], attention_mask=mask, position_embeddings=pe, position_ids=pos
                )
        torch.cuda.synchronize()
        H, H2 = H2, H
        for h in handles:
            h.remove()

        rec = {"swa": is_swa, "moe": is_moe, "load_s": round(t_load, 1), "resid_rms": float(H.float().pow(2).mean().sqrt())}
        if is_moe:
            def fin(cals):
                hr, mx = [], []
                for c in cals:
                    a = c.compute_amax()
                    hr.append(float(a) if a is not None else float("nan"))
                    mx.append(float(c._running_max) if c._running_max is not None else float("nan"))
                return hr, mx

            def union(cals):
                u = NVFP4ActHeadroomCalibrator()
                hs = [c._hist for c in cals if c._hist is not None]
                u._hist = torch.stack(hs).sum(0)
                u._running_max = torch.stack([c._running_max for c in cals if c._running_max is not None]).max()
                return float(u.compute_amax()), float(u._running_max)

            w13_hr, w13_mx = fin(c13)
            w2_hr, w2_mx = fin(c2)
            u13, u13m = union(c13)
            u2, u2m = union(c2)
            # Candidate shared (per-layer) activation amax values, scored by runtime-style
            # NVFP4 QDQ relative MSE on a uniform sample of routed rows.
            x13 = torch.cat(s13) if s13 else None
            x2 = torch.cat(s2) if s2 else None
            peer13 = max(v for v in w13_hr if math.isfinite(v))
            peer2 = max(v for v in w2_hr if math.isfinite(v))
            cand13 = {"union_headroom": u13, "peer_headroom": peer13, "max": u13m, "unit_scale": 2688.0}
            cand2 = {"union_headroom": u2, "peer_headroom": peer2, "max": u2m, "unit_scale": 2688.0}
            q13 = {k: nvfp4_act_relmse(x13, v) for k, v in cand13.items()} if x13 is not None else {}
            q2 = {k: nvfp4_act_relmse(x2, v) for k, v in cand2.items()} if x2 is not None else {}
            del s13, s2, x13, x2
            rec.update(
                tokens=tok.tolist(), experts_hit=int((tok > 0).sum()), w13_headroom=w13_hr, w13_max=w13_mx,
                w2_headroom=w2_hr, w2_max=w2_mx, union_w13_headroom=u13, union_w13_max=u13m,
                union_w2_headroom=u2, union_w2_max=u2m, cand_w13=cand13, cand_w2=cand2,
                qdq_relmse_w13=q13, qdq_relmse_w2=q2,
            )
            for e in range(E):
                if tok[e] > 0:
                    for proj, val in (("gate_proj", w13_hr[e]), ("up_proj", w13_hr[e]), ("down_proj", w2_hr[e])):
                        amax_pt[f"{p}mlp.experts.{e}.{proj}_input_quantizer._amax"] = torch.tensor(val)
        d = kv.get(l)
        if d is not None:
            kvrec = {
                "k_max": float(d["k"]._running_max), "v_max": float(d["v"]._running_max),
                "k_headroom": float(d["k"].compute_amax()), "v_headroom": float(d["v"].compute_amax()),
                "k_head_max": d["kh"].tolist(), "v_head_max": d["vh"].tolist(),
            }
            stats["kv"][l] = kvrec
            amax_pt[f"{p}self_attn.k_bmm_quantizer._amax"] = torch.tensor(kvrec["k_max"])
            amax_pt[f"{p}self_attn.v_bmm_quantizer._amax"] = torch.tensor(kvrec["v_max"])
        rec["total_s"] = round(time.time() - t0, 1)
        stats["layers"][l] = rec
        msg = f"layer {l:2d} {'SWA' if is_swa else 'GA '} {'MoE' if is_moe else 'MLP'} load={t_load:.1f}s total={rec['total_s']}s rms={rec['resid_rms']:.3f}"
        if is_moe:
            f = lambda q: " ".join(f"{k[:5]}={v:.2e}" for k, v in q.items())
            msg += (
                f" hit={rec['experts_hit']}/{E} w13[u_hr={u13:.3g} peer_hr={peer13:.3g} max={u13m:.3g} | relmse {f(q13)}]"
                f" w2[u_hr={u2:.3g} peer_hr={peer2:.3g} max={u2m:.3g} | relmse {f(q2)}]"
            )
        if d is not None:
            msg += f" K={kvrec['k_max']:.3g} V={kvrec['v_max']:.3g}"
        log(msg)
        del layer
        torch.cuda.empty_cache()
        (args.out / "calib_stats.json").write_text(json.dumps(stats, indent=1))
        torch.save(amax_pt, args.out / "amax_modelopt.pt")

    if not args.max_layers:
        norm = Norm(cfg.hidden_size, eps=cfg.layernorm_epsilon).to(dev, torch.bfloat16)
        norm.weight.data.copy_(ck.get("model.norm.weight", dev))
        lm = ck.get("lm_head.weight", dev)
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for b0 in range(0, N, 4):
                h = norm(H[b0 : b0 + 4])
                logits = (h @ lm.T).float()
                tgt = ids[b0 : b0 + 4, 1:]
                tot += float(F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum"))
                cnt += tgt.numel()
        stats["nll"] = {"mean_nll": tot / cnt, "ppl": math.exp(tot / cnt), "tokens": cnt}
        log(f"end-to-end next-token NLL={tot / cnt:.4f} ppl={math.exp(tot / cnt):.3f} over {cnt} tokens")
    stats["wall_s"] = round(time.time() - t_start, 1)
    (args.out / "calib_stats.json").write_text(json.dumps(stats, indent=1))
    torch.save(amax_pt, args.out / "amax_modelopt.pt")
    log(f"done wall={stats['wall_s']}s amax entries={len(amax_pt)}")


if __name__ == "__main__":
    main()
