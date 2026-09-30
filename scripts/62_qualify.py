#!/usr/bin/env python3
"""FINAL3 qualification the short profile bench did not cover.

1. Concurrency ladder c1/c2/c4/c8 (admitted max_running_requests=8), think-off,
   streaming. Per request: TTFT, e2e, decode tok/s, prefill tok/s, finish
   reason, completion tokens. Server side: /get_server_info accept length
   before/after each level. /metrics is not served on this profile.

2. GSM8K-200, think-off (the only low setting this chat template has:
   enable_thinking=false injects an empty <think>). max_tokens starts at 2048
   so answers can finish; a length-capped item is retried once at 4096.
   Scored by regex on the text after ####. Model output is never exec'd.
   Rows append as they finish so a killed client resumes.

usage: 62_qualify.py --phase both|concurrency|q200
r0b0tlab mimo26.
"""
import argparse
import json
import re
import statistics
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "http://192.168.68.78:30000"
NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")
TOPICS = [
    "a thread-safe LRU cache with per-entry TTL, size eviction, hit/miss stats",
    "a token-bucket and sliding-window rate limiter behind one interface",
    "a trie autocomplete engine with frequency ranking and edit-distance 1",
    "an asyncio job queue with bounded concurrency, jittered retries, timeouts",
    "an A* pathfinder on a weighted grid with pluggable heuristics",
    "a append-only write-ahead log with checksums and crash recovery",
    "a consistent-hash ring with virtual nodes and bounded rebalance",
    "a small TTL DNS cache with negative caching and stale-while-revalidate",
]
INSTR = ("Write a complete Python module that implements {topic}. "
         "Include type hints, docstrings, and a pytest suite. Output only code.")


def post(payload, timeout):
    req = urllib.request.Request(
        URL + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def server_decode_lines():
    """Last server-side decode telemetry lines. /metrics is not mounted."""
    import subprocess
    cmd = ("ssh -i $HOME/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes "
           "-o ConnectTimeout=10 r0b0tdgx@192.168.68.78 "
           "\"grep -a 'Decode batch' /home/r0b0tdgx/projects/mimo26-nvfp4-sm121/logs/serve/r0.FINAL3.log "
           "| tail -3\"")
    try:
        out = subprocess.check_output(cmd, shell=True, text=True, timeout=30)
    except Exception as e:  # noqa: BLE001
        return [repr(e)]
    return [ln.strip()[:300] for ln in out.splitlines() if ln.strip()]


def server_info():
    try:
        with urllib.request.urlopen(URL + "/get_server_info", timeout=20) as r:
            d = json.loads(r.read().decode())
        st = (d.get("internal_states") or [{}])[0]
        return {"avg_spec_accept_length": st.get("avg_spec_accept_length"),
                "memory_usage": st.get("memory_usage"),
                "max_total_num_tokens": d.get("max_total_num_tokens"),
                "max_running_requests": d.get("max_running_requests")}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def chat_stream(prompt, max_tokens, timeout):
    body = json.dumps({
        "model": "mimo26",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = t_last = None
    usage = None
    finish = None
    n_chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                delta = ch.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece:
                    now = time.perf_counter()
                    t_first = now if t_first is None else t_first
                    t_last = now
                    n_chunks += 1
    t_end = time.perf_counter()
    ct = (usage or {}).get("completion_tokens")
    pt = (usage or {}).get("prompt_tokens")
    out = {"prompt_tokens": pt, "completion_tokens": ct, "chunks": n_chunks,
           "finish_reason": finish, "truncated": finish == "length",
           "ttft_s": (t_first - t0) if t_first else None, "e2e_s": t_end - t0}
    if t_first and t_last and ct and ct > 1 and t_last > t_first:
        out["decode_tps"] = (ct - 1) / (t_last - t_first)
    if pt and t_first and t_first > t0:
        out["prefill_tps"] = pt / (t_first - t0)
    return out


def run_level(conc, repeats, max_tokens, timeout):
    rows = []
    for rep in range(repeats):
        prompts = [f"Request id: c{conc}-r{rep}-i{i}\n" + INSTR.format(topic=TOPICS[i % len(TOPICS)])
                   for i in range(conc)]
        before = server_info()
        results = [None] * conc

        def worker(i, p):
            try:
                results[i] = chat_stream(p, max_tokens, timeout)
            except Exception as e:  # noqa: BLE001
                results[i] = {"error": repr(e)}

        t0 = time.perf_counter()
        ths = [threading.Thread(target=worker, args=(i, p)) for i, p in enumerate(prompts)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = time.perf_counter() - t0
        after = server_info()
        ct = sum((r or {}).get("completion_tokens") or 0 for r in results)
        dec = [r["decode_tps"] for r in results if r and "decode_tps" in r]
        ttft = [r["ttft_s"] for r in results if r and r.get("ttft_s")]
        e2e = [r["e2e_s"] for r in results if r and r.get("e2e_s")]
        row = {"concurrency": conc, "repeat": rep, "wall_s": round(wall, 3),
               "aggregate_tps": (ct / wall) if wall else None,
               "per_req_decode_median": statistics.median(dec) if dec else None,
               "ttft_median_s": statistics.median(ttft) if ttft else None,
               "e2e_median_s": statistics.median(e2e) if e2e else None,
               "errors": sum(1 for r in results if not r or r.get("error")),
               "truncated": sum(1 for r in results if r and r.get("truncated")),
               "accept_before": before.get("avg_spec_accept_length"),
               "accept_after": after.get("avg_spec_accept_length"),
               "server_decode_tail": server_decode_lines(),
               "requests": results}
        rows.append(row)
        print(f"  c{conc}[{rep}] agg={row['aggregate_tps']:.1f} "
              f"decode_med={row['per_req_decode_median']} ttft={row['ttft_median_s']:.2f} "
              f"e2e={row['e2e_median_s']:.1f} err={row['errors']} trunc={row['truncated']}",
              flush=True)
    return rows


def phase_concurrency(out, repeats, max_tokens):
    doc = {"tag": "FINAL3-e2e", "url": URL, "think": "off",
           "max_tokens": max_tokens, "repeats": repeats,
           "started_utc": time.strftime("%FT%TZ", time.gmtime()),
           "server_info_start": server_info(), "levels": {}}
    print("[e2e] warmup", flush=True)
    chat_stream("Request id: warmup\nSay hello in five words.", 32, 120)
    for conc in (1, 2, 4, 8):
        print(f"[e2e] c{conc}", flush=True)
        doc["levels"][str(conc)] = run_level(conc, repeats, max_tokens, timeout=1800)
        out.write_text(json.dumps(doc, indent=1))
    doc["server_info_end"] = server_info()
    doc["finished_utc"] = time.strftime("%FT%TZ", time.gmtime())
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[e2e] wrote {out}", flush=True)


def score(text, gold):
    tail = text.split("####")[-1]
    found = NUM_RE.findall(tail)
    if not found:
        return False, None
    got = float(found[-1])
    return abs(got - float(gold)) < 1e-4, got


def one_q(item, max_tokens):
    payload = {"model": "mimo26", "temperature": 0.0, "max_tokens": max_tokens,
               "messages": [{"role": "user", "content":
                             item["q"] + "\n\nEnd your reply with: #### <number>"}],
               "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    data = post(payload, timeout=900)
    el = time.perf_counter() - t0
    msg = data["choices"][0]["message"]
    text = msg.get("content") or ""
    usage = data.get("usage") or {}
    finish = data["choices"][0].get("finish_reason")
    ok, got = score(text, item["a"])
    return {"ok": ok, "got": got, "gold": item["a"], "finish_reason": finish,
            "truncated": finish == "length",
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "e2e_s": round(el, 3),
            "response_tail": text[-240:]}


def phase_q200(out, n, max_tokens):
    items = [json.loads(l) for l in open(ROOT / "results/gsm8k_test.jsonl")][:n]
    done = {}
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["i"]] = row
    for i, item in enumerate(items):
        if i in done and done[i].get("transport_ok") and not done[i].get("truncated"):
            continue
        print(f"[q200] {i}", flush=True)
        try:
            row = one_q(item, max_tokens)
            if row["truncated"]:
                print(f"[q200] {i} hit {max_tokens}, retry 4096", flush=True)
                row = one_q(item, 4096)
                row["retried_at"] = 4096
            row["transport_ok"] = True
        except Exception as e:  # noqa: BLE001
            row = {"transport_ok": False, "ok": False, "error": repr(e)}
        row["i"] = i
        row["think"] = "off"
        done[i] = row
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  ok={row.get('ok')} finish={row.get('finish_reason')} "
              f"ct={row.get('completion_tokens')} e2e={row.get('e2e_s')}", flush=True)
    scored = [done[i] for i in range(n) if i in done and done[i].get("transport_ok")]
    acc = sum(bool(r.get("ok")) for r in scored) / max(1, len(scored))
    trunc = sum(bool(r.get("truncated")) for r in scored)
    print(f"[q200] {sum(bool(r.get('ok')) for r in scored)}/{len(scored)} "
          f"= {acc:.3f} truncated={trunc}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("both", "concurrency", "q200"), default="both")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--q-max-tokens", type=int, default=2048)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    if a.phase in ("both", "concurrency"):
        phase_concurrency(ROOT / "results/e2e/FINAL3-concurrency.json", a.repeats, a.max_tokens)
    if a.phase in ("both", "q200"):
        phase_q200(ROOT / "results/q200/FINAL3-nothink.jsonl", a.n, a.q_max_tokens)


if __name__ == "__main__":
    main()
