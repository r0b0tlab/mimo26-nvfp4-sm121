#!/usr/bin/env python3
"""Serve-profile benchmark for the mimo26 TP2 endpoint (stdlib only).

Workloads (think-off, greedy, streaming chat completions):
  short_code   short prompt -> ~1K tokens of new code          c1 x N
  medium_code  ~4K-token code context -> ~1K tokens of tests   c1 x N
  prose        ~2K-token article context -> ~1K-token essay    c1 x N
  prefill_8k   ~8K-token prompt, max_tokens=1 (prefill tok/s)  c1 x N
  c4_short     4 concurrent short_code topics (aggregate tok/s)

Every prompt starts with a distinct request id so the radix cache cannot
serve one request's prefill from another's.  Prompt ids are deterministic,
so two profiles see identical prompts.

Decode tok/s per request = (completion_tokens - 1) / (t_last - t_first),
TTFT = first streamed content chunk.  Server-side DFlash acceptance is read
from /get_server_info (avg_spec_accept_length, cumulative since boot) before
and after each workload.

usage: 60_profile_bench.py --url http://192.168.68.78:30000 --tag P0 \
          --out results/profile_bench/P0.json [--repeats 2] [--max-tokens 1024]
r0b0tlab mimo26.
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

TOPICS = [
    "a thread-safe LRU cache with per-entry TTL expiry, size-based eviction, "
    "hit/miss statistics and a decorator interface",
    "a rate limiter offering both token-bucket and sliding-window-log "
    "strategies behind one interface, with an asyncio-friendly API",
    "a trie-based autocomplete engine with frequency ranking, prefix "
    "deletion, and fuzzy matching within edit distance 1",
    "an asyncio job queue with bounded concurrency, retries with exponential "
    "backoff and jitter, cancellation, and per-job timeouts",
    "an A* path finder on a weighted 2-D grid with diagonal moves, "
    "pluggable heuristics, and path reconstruction",
]
CODE_INSTR = ("Write a complete, production-quality Python module that "
              "implements {topic}. Include type hints and docstrings, then a "
              "pytest test suite at the end. Output only code.")


def code_context(n_chars: int) -> str:
    parts = []
    for f in sorted(HERE.glob("*.py")):
        if f.name.startswith("60_"):
            continue
        parts.append(f"# ===== file: {f.name} =====\n" + f.read_text())
    blob = "\n\n".join(parts)
    while len(blob) < n_chars:
        blob += "\n\n" + blob
    return blob[:n_chars]


def prose_context(n_chars: int) -> str:
    rows = [json.loads(l)["prompt"]
            for l in open(ROOT / "results" / "calib_rows.jsonl")]
    bad = ("\\boxed", "<think>", "def ", "```", "import ", "{", "Solve")
    prose = [r for r in rows if not any(b in r for b in bad)]
    blob = "\n\n".join(prose)
    while len(blob) < n_chars:
        blob += "\n\n" + blob
    return blob[:n_chars]


def server_info(url: str) -> dict:
    try:
        with urllib.request.urlopen(url + "/get_server_info", timeout=20) as r:
            d = json.load(r)
        st = (d.get("internal_states") or [{}])[0]
        return {"avg_spec_accept_length": st.get("avg_spec_accept_length"),
                "memory_usage": st.get("memory_usage")}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def chat_stream(url: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    body = json.dumps({
        "model": "mimo26",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = t_last = None
    usage = None
    n_chunks = 0
    text = []
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
                delta = ch.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    n_chunks += 1
                    text.append(piece)
    t_end = time.perf_counter()
    ct = (usage or {}).get("completion_tokens")
    pt = (usage or {}).get("prompt_tokens")
    out = {"prompt_tokens": pt, "completion_tokens": ct, "chunks": n_chunks,
           "ttft_s": (t_first - t0) if t_first else None,
           "e2e_s": t_end - t0, "text_head": "".join(text)[:160]}
    if t_first and t_last and ct and ct > 1 and t_last > t_first:
        out["decode_tps"] = (ct - 1) / (t_last - t_first)
    if pt and t_first:
        out["prefill_tps"] = pt / (t_first - t0)
    return out


def run_c1(url, name, prompts, max_tokens, timeout):
    before = server_info(url)
    reqs = []
    for i, p in enumerate(prompts):
        r = chat_stream(url, p, max_tokens, timeout)
        r["i"] = i
        reqs.append(r)
        print(f"  {name}[{i}] pt={r['prompt_tokens']} ct={r['completion_tokens']} "
              f"ttft={r['ttft_s']:.2f}s decode={r.get('decode_tps', 0):.1f} tok/s "
              f"prefill={r.get('prefill_tps', 0):.0f} tok/s", flush=True)
    after = server_info(url)
    dec = [r["decode_tps"] for r in reqs if "decode_tps" in r]
    pre = [r["prefill_tps"] for r in reqs if "prefill_tps" in r]
    return {"requests": reqs,
            "decode_tps_median": statistics.median(dec) if dec else None,
            "prefill_tps_median": statistics.median(pre) if pre else None,
            "ttft_median_s": statistics.median([r["ttft_s"] for r in reqs if r["ttft_s"]]),
            "accept_len_before": before.get("avg_spec_accept_length"),
            "accept_len_after": after.get("avg_spec_accept_length")}


def run_concurrent(url, name, prompts, max_tokens, timeout):
    before = server_info(url)
    results = [None] * len(prompts)

    def worker(i, p):
        try:
            results[i] = chat_stream(url, p, max_tokens, timeout)
        except Exception as e:  # noqa: BLE001
            results[i] = {"error": repr(e)}

    t0 = time.perf_counter()
    ths = [threading.Thread(target=worker, args=(i, p)) for i, p in enumerate(prompts)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    after = server_info(url)
    ct = sum((r or {}).get("completion_tokens") or 0 for r in results)
    agg = ct / wall if wall > 0 else None
    dec = [r["decode_tps"] for r in results if r and "decode_tps" in r]
    print(f"  {name}: c={len(prompts)} total_ct={ct} wall={wall:.1f}s "
          f"aggregate={agg:.1f} tok/s per-req median={statistics.median(dec) if dec else 0:.1f}",
          flush=True)
    return {"requests": results, "wall_s": wall, "aggregate_tps": agg,
            "per_req_decode_median": statistics.median(dec) if dec else None,
            "accept_len_before": before.get("avg_spec_accept_length"),
            "accept_len_after": after.get("avg_spec_accept_length")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://192.168.68.78:30000")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--skip", default="", help="comma list of workloads to skip")
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))

    res = {"tag": a.tag, "url": a.url, "started_utc": time.strftime("%FT%TZ", time.gmtime()),
           "max_tokens": a.max_tokens, "repeats": a.repeats,
           "server_info_start": server_info(a.url), "workloads": {}}
    print(f"[{a.tag}] warmup", flush=True)
    chat_stream(a.url, "Request id: warmup-0\nSay hello in five words.", 32, a.timeout)

    code4k = code_context(15000)
    prose2k = prose_context(9000)
    ctx8k = code_context(16000) + "\n\n" + prose_context(14000)

    if "short_code" not in skip:
        print(f"[{a.tag}] short_code", flush=True)
        ps = [f"Request id: short-{i}\n" + CODE_INSTR.format(topic=TOPICS[0])
              for i in range(a.repeats)]
        res["workloads"]["short_code"] = run_c1(a.url, "short_code", ps, a.max_tokens, a.timeout)
    if "medium_code" not in skip:
        print(f"[{a.tag}] medium_code", flush=True)
        ps = [f"Request id: medium-{i}\n```python\n{code4k}\n```\n\nWrite a comprehensive "
              "pytest test module for the code above, covering normal paths and edge "
              "cases. Output only code." for i in range(a.repeats)]
        res["workloads"]["medium_code"] = run_c1(a.url, "medium_code", ps, a.max_tokens, a.timeout)
    if "prose" not in skip:
        print(f"[{a.tag}] prose", flush=True)
        ps = [f"Request id: prose-{i}\n{prose2k}\n\nWrite an original, well-structured "
              "essay of at least 900 words discussing the themes raised in the articles "
              "above, with an introduction, several body sections and a conclusion. Use "
              "flowing prose, no lists or headings." for i in range(a.repeats)]
        res["workloads"]["prose"] = run_c1(a.url, "prose", ps, a.max_tokens, a.timeout)
    if "prefill_8k" not in skip:
        print(f"[{a.tag}] prefill_8k", flush=True)
        ps = [f"Request id: prefill-{i}\n{ctx8k}\n\nSummarize the material above in one "
              "sentence." for i in range(a.repeats)]
        res["workloads"]["prefill_8k"] = run_c1(a.url, "prefill_8k", ps, 1, a.timeout)
    if "c4_short" not in skip:
        print(f"[{a.tag}] c4_short", flush=True)
        ps = [f"Request id: c4-{i}\n" + CODE_INSTR.format(topic=TOPICS[i + 1]) for i in range(4)]
        res["workloads"]["c4_short"] = run_concurrent(a.url, "c4_short", ps, a.max_tokens, a.timeout)

    res["server_info_end"] = server_info(a.url)
    res["finished_utc"] = time.strftime("%FT%TZ", time.gmtime())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    w = res["workloads"]
    summ = {k: {kk: v.get(kk) for kk in ("decode_tps_median", "prefill_tps_median",
                                         "ttft_median_s", "aggregate_tps",
                                         "per_req_decode_median") if v.get(kk) is not None}
            for k, v in w.items()}
    print(json.dumps({"tag": a.tag, "summary": summ,
                      "accept_len_end": res["server_info_end"].get("avg_spec_accept_length")},
                     indent=1))


if __name__ == "__main__":
    main()
