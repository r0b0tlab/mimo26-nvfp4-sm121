#!/usr/bin/env python3
"""Per-category tensor-byte inventory of a safetensors checkpoint (headers only).

Usage: 24_ckpt_inventory.py CKPT_DIR [CKPT_DIR ...]
Prints bytes per category (routed experts / attention / dense MLP / embed /
lm_head / vision / audio / mtp / other) and dtype mix, so two checkpoints can
be compared before blaming the runtime for a load-memory difference.
r0b0tlab mimo26.
"""
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

DT_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
            "I16": 2, "I8": 1, "U8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
            "F8_E8M0": 1, "BOOL": 1}


def category(name: str) -> str:
    if name.startswith(("mtp.", "model.mtp")) or ".mtp." in name:
        return "mtp"
    if name.startswith(("visual.", "model.visual", "vision")) or ".visual." in name:
        return "vision"
    if name.startswith(("audio", "model.audio")) or ".audio" in name:
        return "audio"
    if "embed_tokens" in name:
        return "embed"
    if name.startswith("lm_head"):
        return "lm_head"
    if ".mlp.experts." in name:
        kind = "scale" if re.search(r"(scale|_inv)", name) else "weight"
        return f"experts.{kind}"
    if ".self_attn." in name:
        return "attention"
    if ".mlp." in name:
        return "dense_or_shared_mlp"
    if "norm" in name:
        return "norms"
    return "other"


def inventory(ckpt: Path):
    by_cat = defaultdict(int)
    by_dtype = defaultdict(int)
    n = 0
    for f in sorted(ckpt.glob("*.safetensors")):
        with open(f, "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hlen))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            numel = 1
            for s in v["shape"]:
                numel *= s
            b = numel * DT_BYTES[v["dtype"]]
            by_cat[category(k)] += b
            by_dtype[v["dtype"]] += b
            n += 1
    return n, by_cat, by_dtype


def main():
    results = {}
    for p in sys.argv[1:]:
        results[p] = inventory(Path(p).expanduser())
    cats = sorted({c for _, bc, _ in results.values() for c in bc})
    names = list(results)
    print("category".ljust(24) + "".join(Path(n).name[:28].rjust(30) for n in names))
    for c in cats:
        row = c.ljust(24)
        for n in names:
            row += f"{results[n][1].get(c, 0) / 2**30:>28.3f}Gi"
        print(row)
    row = "TOTAL".ljust(24)
    for n in names:
        row += f"{sum(results[n][1].values()) / 2**30:>28.3f}Gi"
    print(row)
    for n in names:
        cnt, _, bd = results[n]
        print(f"{Path(n).name}: tensors={cnt} dtypes=" +
              ", ".join(f"{d}:{b / 2**30:.2f}Gi" for d, b in sorted(bd.items())))


if __name__ == "__main__":
    main()
