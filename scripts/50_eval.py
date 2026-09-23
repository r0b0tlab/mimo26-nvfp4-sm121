#!/usr/bin/env python3
"""Quality harness against a live lane at N3IP:30000.

--nll: teacher-forced NLL over calibration rows via /v1/completions with
       echo=True + logprobs (rows exported to results/calib_rows.jsonl at P6).
--gsm8k N: strict-regex exact match on the first N cached items
       (results/gsm8k_test.jsonl, fields {"q":..., "a": float}). Answers are
       parsed with a regex; model output is NEVER exec'd.
Results append to results/eval_<tag>.json.
r0b0tlab mimo26.
"""
import argparse, json, re, urllib.request
from pathlib import Path

N3IP = "192.168.68.78"
ROOT = Path(__file__).resolve().parent.parent


def post(payload, endpoint="/v1/completions"):
    req = urllib.request.Request(
        f"http://{N3IP}:30000{endpoint}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


def nll(rows_file):
    tot, n = 0.0, 0
    for line in open(rows_file):
        r = json.loads(line)
        out = post({"model": "mimo26", "prompt": r["prompt"], "max_tokens": 0,
                    "echo": True, "logprobs": 1})
        for ch in out["choices"]:
            lps = (ch.get("logprobs") or {}).get("content") or []
            tot += sum(lp["logprob"] for lp in lps)
            n += len(lps)
    val = -tot / max(1, n)
    print(f"NLL={val:.4f} tokens={n}")
    return val


def gsm8k(n_items, cache):
    pat = re.compile(r"(-?\d+(?:\.\d+)?)")
    ok = 0
    items = [json.loads(l) for l in open(cache)][:n_items]
    for it in items:
        out = post({"model": "mimo26", "prompt": it["q"], "max_tokens": 512,
                    "temperature": 0.0})
        text = out["choices"][0]["text"]
        m = pat.findall(text.split("####")[-1])
        ok += bool(m) and abs(float(m[-1]) - float(it["a"])) < 1e-4
    print(f"GSM8K {ok}/{len(items)} = {ok / len(items):.3f}")
    return ok / len(items)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    help="lane tag; appends to results/eval_<tag>.json")
    ap.add_argument("--nll", action="store_true")
    ap.add_argument("--gsm8k", type=int, metavar="N")
    ap.add_argument("--rows", default=str(ROOT / "results/calib_rows.jsonl"))
    a = ap.parse_args()
    res = {"tag": a.tag}
    if a.nll:
        res["nll"] = nll(a.rows)
    if a.gsm8k:
        res["gsm8k_acc"] = gsm8k(a.gsm8k, ROOT / "results/gsm8k_test.jsonl")
    out = ROOT / "results" / f"eval_{a.tag}.json"
    hist = json.loads(out.read_text()) if out.exists() else []
    hist.append(res)
    out.write_text(json.dumps(hist, indent=1))
    print(f"appended -> {out}")
