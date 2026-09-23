#!/usr/bin/env python3
"""Write / toggle MiMo KV-cache scales for the NVFP4 export.

SGLang contract (model_loader/weight_utils.py kv_cache_scales_loader +
QuantParamSchema): a JSON file with
    {"kv_cache": {"scaling_factor": {"<tp_rank>": {"<layer_idx>": float}}}}
MiMo (models/mimo_v2.py load_kv_cache_scales) applies ONE value per layer to
BOTH k_scale and v_scale, per attn_tp_rank. We serve attn_tp_size=2, so we
emit ranks "0" and "1" with identical values.
Values are the calibration max |K| / 448.0 and max |V| / 448.0 (fp8_e4m3 max)
from calib_stats.json -> stats["kv"][layer].
Modes:
  write  regenerate kv_scales.json from calib stats and point config.json at it
  on     point config.json at the existing kv_scales.json
  off    clear config.json's quantization_param_path (scales default to 1.0)
r0b0tlab mimo26.
"""
import argparse, json, sys
from pathlib import Path

E4M3_MAX = 448.0


def load_kvs(calib_path: str):
    stats = json.load(open(calib_path))
    kv = stats.get("kv", {})
    assert kv, "no kv records in calib stats"
    return kv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["write", "on", "off"])
    ap.add_argument("--dst", required=True)
    ap.add_argument("--calib", help="required with 'write'")
    a = ap.parse_args()
    dst = Path(a.dst)
    cfg_p = dst / "config.json"
    cfg = json.loads(cfg_p.read_text())
    if a.mode == "write":
        assert a.calib, "--calib is required with 'write'"
        kv = load_kvs(a.calib)
        per_layer = {
            str(l): {"k_scale": rec["k_max"] / E4M3_MAX,
                     "v_scale": rec["v_max"] / E4M3_MAX}
            for l, rec in sorted(kv.items(), key=lambda x: int(x[0]))
        }
        scales = {rank: per_layer for rank in ("0", "1")}
        (dst / "kv_scales.json").write_text(json.dumps(
            {"kv_cache": {"scaling_factor": scales}}, indent=1))
        k0 = per_layer["0"]["k_scale"]
        v0 = per_layer["0"]["v_scale"]
        assert k0 > 0 and v0 > 0, "non-positive scale"
        cfg["quantization_param_path"] = str(dst / "kv_scales.json")
        cfg_p.write_text(json.dumps(cfg, indent=2))
        print(f"KV SCALES WRITTEN n={len(kv)} k_scale[{k0:.4g}] v_scale[{v0:.4g}]")
    else:
        want = a.mode == "on"
        cfg["quantization_param_path"] = str(dst / "kv_scales.json") if want else None
        cfg_p.write_text(json.dumps(cfg, indent=2))
        print(f"KV SCALES {'ON' if want else 'OFF'}")
    print("config.quantization_param_path =", cfg.get("quantization_param_path"))


if __name__ == "__main__":
    main()
