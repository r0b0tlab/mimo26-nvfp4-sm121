#!/usr/bin/env python3
"""Gate the calibration stats: the chosen activation-scale policy must be
within 1.10x of the best candidate on every MoE layer/projection, and no
layer may be > 1.5x.  Also re-derives the headline evidence numbers.

Stats schema (written by scripts/10_calibrate_mimo.py):
  stats["layers"][layer_id] = {
    "moe": bool, "swa": bool, "tokens": [...],
    "qdq_relmse_w13": {policy: relmse, ...}, "qdq_relmse_w2": {...},
    "w13_headroom": [per-expert ...], "w2_headroom": [...],
    "union_w13_headroom": float, "union_w2_headroom": float, ...}
  stats["kv"][layer_id] = {k_max, v_max, ...}
  stats["nll"], stats["meta"]{calib_size, calib_seq, dataset}

Usage: 12_check_calib.py <calib_stats.json> <policy>
r0b0tlab mimo26.
"""
import json, sys

stats = json.load(open(sys.argv[1]))
policy = sys.argv[2]

worst = []
n_moe = 0
experts_hit = []
for lid, rec in stats["layers"].items():
    if not rec.get("moe"):
        continue
    n_moe += 1
    eh = rec.get("experts_hit")
    if eh is not None:
        experts_hit.append(eh)
    for proj in ("w13", "w2"):
        per = rec.get(f"qdq_relmse_{proj}", {})
        if policy not in per or not per:
            print(f"ERR policy {policy!r} missing on layer {lid} {proj}; "
                  f"have {sorted(per)}")
            sys.exit(2)
        best = min(per.values())
        worst.append((per[policy] / best if best else 1.0, int(lid), proj))

worst.sort(reverse=True)
bad = [w for w in worst if w[0] > 1.5]
n_layers = len(stats["layers"])
n_kv = len(stats.get("kv", {}))
min_hit = min(experts_hit) if experts_hit else -1
nll_d = stats["nll"]
if isinstance(nll_d, dict):
    nll = nll_d["mean_nll"]
    ppl = nll_d["ppl"]
    tokens = nll_d["tokens"]
else:
    nll = nll_d
    ppl = 2.718281828459045 ** nll
    tokens = stats["meta"]["calib_size"] * stats["meta"]["calib_seq"]

print(f"layers={n_layers} moe_layers={n_moe} kv_layers={n_kv} "
      f"min_experts_hit={min_hit}")
print(f"nll={nll} ppl={ppl:.3f} tokens={tokens}")
print(f"worst {policy}/best: {[(round(r, 3), l, p) for r, l, p in worst[:3]]}")
frac = sum(r <= 1.10 for r, _, _ in worst) / max(1, len(worst))
print(f"frac(ratio<=1.10)={frac:.3f} n_bad(>1.5)={len(bad)}")
ok = frac >= 1.0 and not bad
print(f"GATE {'PASS' if ok else 'FAIL'} policy={policy}")
sys.exit(0 if ok else 1)
