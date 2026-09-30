#!/usr/bin/env python3
"""mimo26 max-context NIAH: advertised window at 25/50/90% depth + multi-key 33/66.

Adapted from the r0b0tlab DSV4-Flash-Vision-Exp runner (run-niah-advertised.py).
Uses the server /tokenize endpoint (no host tokenizer).  Depth = fraction of
target tokens.  The multi-key case embeds 3 needles at 33/66/probe depth and
the answer must be the probe needle.  Think-off (enable_thinking=false).

Results are appended case-by-case (atomic JSON rewrite) so an interrupted run
resumes.  NEVER SIGKILL this client mid-request at long context: an orphaned
server-side prefill can starve both GB10 hosts.  Abort server-side instead.

usage: 61_niah.py --output results/niah/<tag>.json --target-tokens N [--cases 25%,mk66]
r0b0tlab mimo26.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

NONCE = "R0B0BENCH_NIAH_SLOT"
FILLER = "The archive records ordinary weather observations for a routine day. "
PREFIX = "BEGIN DOCUMENT\n"
SUFFIX = ("\nEND DOCUMENT\n"
          "Question: What is the exact value after NEEDLE:? Answer with that value only.")
SYSTEM = ("You are a precise retrieval system. The supplied document contains exactly one "
          "line beginning with NEEDLE:. Ignore the surrounding filler and answer the final "
          "question with only the value from that line.")
MULTIKEY_NONCES = ["ni4h7q2x9", "k7d2m8v4", "t3r9w5y1"]
SYSTEM_MK = ("You are a precise retrieval system. The supplied document contains several "
             "lines beginning with NEEDLE:. Answer the final question with only the value "
             "from the LAST such line in the document.")
SUFFIX_MK = ("\nEND DOCUMENT\n"
             "Question: What is the exact value after the LAST occurrence of NEEDLE:? "
             "Answer with that value only.")
ALL_CASES = [("25%", 0.25, False), ("50%", 0.50, False), ("90%", 0.90, False),
             ("mk33", 0.33, True), ("mk66", 0.66, True), ("mk3", None, "all3")]

# mk3: three DISTINCT keyed needles at 33/66/99% depth in one request; the
# model must return all three values (per-key retrieval recorded).
MK3_KEYS = [("NEEDLE-A", "ni4h7q2x9", 0.33), ("NEEDLE-B", "k7d2m8v4", 0.66),
            ("NEEDLE-C", "t3r9w5y1", 0.99)]
SYSTEM_MK3 = ("You are a precise retrieval system. The supplied document contains exactly "
              "three lines of the form NEEDLE-<letter>: <value>. Ignore the surrounding "
              "filler and answer the final question using only those lines.")
SUFFIX_MK3 = ("\nEND DOCUMENT\n"
              "Question: What are the exact values of NEEDLE-A, NEEDLE-B and NEEDLE-C? "
              "Answer with exactly three lines, NEEDLE-A=<value>, NEEDLE-B=<value>, "
              "NEEDLE-C=<value>, and nothing else.")


def build_all3(base, model, target, _depth):
    fw = len(tokenize(base, model, FILLER))
    lines = [f"\n{k}: {v}\n" for k, v, _ in MK3_KEYS]
    fixed = len(tokenize(base, model, PREFIX + "".join(lines) + SUFFIX_MK3))
    reps = max(8, (target - fixed) // fw)
    cuts = [int(reps * d) for _, _, d in MK3_KEYS]
    parts, prev = [PREFIX], 0
    for (k, v, _), c in zip(MK3_KEYS, cuts):
        parts.append(FILLER * (c - prev))
        parts.append(f"\n{k}: {v}\n")
        prev = c
    parts.append(FILLER * (reps - prev))
    parts.append(SUFFIX_MK3)
    return "".join(parts), SYSTEM_MK3, [v for _, v, _ in MK3_KEYS], {
        "target_prompt_tokens": target, "filler_token_width": fw, "filler_repeats": reps,
        "keys": {k: {"value": v, "depth": d} for k, v, d in MK3_KEYS}}


def request_json(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from {url}: {e.read().decode(errors='replace')[:1000]}") from e


def tokenize(base, model, text):
    data = request_json(base + "/tokenize", {"model": model, "prompt": text}, 600)
    toks = data.get("tokens")
    if not isinstance(toks, list):
        raise RuntimeError(f"unexpected /tokenize response keys: {sorted(data)}")
    return toks


def atomic_write(path, doc):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def build_single(base, model, target, depth):
    fw = len(tokenize(base, model, FILLER))
    needle = f"\nNEEDLE: {NONCE}\n"
    fixed = len(tokenize(base, model, PREFIX + needle + SUFFIX))
    reps = max(1, (target - fixed) // fw)
    pre = int(reps * depth)
    doc = PREFIX + FILLER * pre + needle + FILLER * (reps - pre) + SUFFIX
    return doc, SYSTEM, NONCE, {"target_prompt_tokens": target, "filler_token_width": fw,
                                "pre_repeats": pre, "post_repeats": reps - pre}


def build_multikey(base, model, target, probe_depth):
    fw = len(tokenize(base, model, FILLER))
    lines = [f"\nNEEDLE: {n}\n" for n in MULTIKEY_NONCES]
    fixed = len(tokenize(base, model, PREFIX + "".join(lines) + SUFFIX_MK))
    reps = max(4, (target - fixed) // fw)
    # distractors at 33% and 66%; probe (answer) placed after both, at
    # max(probe_depth, 0.66)+ so it is always the LAST needle.
    c1 = int(reps * 0.33)
    c2 = int(reps * 0.66) - c1
    probe_at = max(probe_depth, 0.66) + (0.24 if probe_depth <= 0.66 else 0.0)
    c3 = max(1, int(reps * min(probe_at, 0.97)) - c1 - c2)
    c4 = max(1, reps - c1 - c2 - c3)
    doc = (PREFIX + FILLER * c1 + lines[0] + FILLER * c2 + lines[1]
           + FILLER * c3 + lines[2] + FILLER * c4 + SUFFIX_MK)
    return doc, SYSTEM_MK, MULTIKEY_NONCES[2], {
        "target_prompt_tokens": target, "distractor_nonces": MULTIKEY_NONCES[:2],
        "probe_nonce": MULTIKEY_NONCES[2],
        "needle_depths": [0.33, 0.66, round(min(probe_at, 0.97), 3)],
        "case_key_depth": probe_depth}


def run_case(base, model, label, depth, target, mk):
    builder = build_all3 if mk == "all3" else (build_multikey if mk else build_single)
    doc, system, expected, meta = builder(base, model, target, depth)
    body = {"model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": doc}],
            "temperature": 0.0, "max_tokens": 64 if mk == "all3" else 32, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    data = request_json(base + "/v1/chat/completions", body, timeout=43200)
    el = time.perf_counter() - t0
    content = (data["choices"][0]["message"].get("content") or "")[:500]
    usage = data.get("usage") or {}
    if isinstance(expected, list):
        per_key = {v: (v in content) for v in expected}
        retrieved = all(per_key.values())
    else:
        per_key, retrieved = None, expected in content
    return {"label": label, "requested_depth": depth, "multikey": mk, **meta,
            "api_prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "elapsed_s": round(el, 3),
            "prefill_tps_est": round((usage.get("prompt_tokens") or 0) / el, 1) if el else None,
            "response": content, "needle_retrieved": retrieved, "per_key_retrieved": per_key,
            "transport_ok": True, "finished_utc": datetime.now(timezone.utc).isoformat()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://192.168.68.78:30000")
    ap.add_argument("--output", required=True)
    ap.add_argument("--target-tokens", type=int, required=True)
    ap.add_argument("--cases", default="25%,50%,90%,mk33,mk66")
    a = ap.parse_args()
    base = a.base_url.rstrip("/")
    model = "mimo26"
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(out.read_text()) if out.exists() else {
        "schema_version": 3, "method": "advertised_window_niah_plus_multikey",
        "model": model, "nonce": NONCE, "target_tokens": a.target_tokens,
        "results": {}, "started_utc": datetime.now(timezone.utc).isoformat()}
    want = set(a.cases.split(","))
    for label, depth, mk in ALL_CASES:
        if label not in want:
            continue
        if label in doc["results"] and doc["results"][label].get("transport_ok"):
            print(f"SKIP existing {label}", flush=True)
            continue
        print(f"START {label} {datetime.now(timezone.utc).isoformat()}", flush=True)
        try:
            res = run_case(base, model, label, depth, a.target_tokens, mk)
        except Exception as e:  # noqa: BLE001
            res = {"label": label, "requested_depth": depth, "transport_ok": False,
                   "needle_retrieved": False, "error": str(e),
                   "finished_utc": datetime.now(timezone.utc).isoformat()}
        doc["results"][label] = res
        atomic_write(out, doc)
        print(json.dumps(res, sort_keys=True, default=str), flush=True)
    rs = doc["results"]
    done = want.issubset(rs)
    doc["verdict"] = ("NIAH_PASS" if done and all(r.get("needle_retrieved") for r in rs.values())
                      else ("NIAH_PARTIAL" if not done else "NIAH_FAIL"))
    doc["semantic_pass_count"] = sum(bool(r.get("needle_retrieved")) for r in rs.values())
    doc["infra_error_count"] = sum(not r.get("transport_ok") for r in rs.values())
    doc["finished_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write(out, doc)
    print(doc["verdict"], flush=True)


if __name__ == "__main__":
    main()
