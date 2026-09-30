#!/usr/bin/env python3
"""Matched perf block plus per-request speculative acceptance.

Official r0b0bench lanes: latency, concurrency, throughput.
The package chat path does not request spec details, so the C1 essay is
repeated with return_spec_tokens_details on the same prompt and token cap.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ESSAY = (
    "Write a detailed technical essay on NVFP4 KV cache design for MoE LLMs. "
    "Continue with dense factual prose until the token budget is exhausted."
)
THINK_OFF = {"thinking": False, "enable_thinking": False}


def post(url: str, body: dict, timeout: int) -> tuple[int, dict, float]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw.decode()), time.perf_counter() - t0
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"error": raw[:500]}
        return exc.code, parsed, time.perf_counter() - t0


def acceptance_series(base: str, model: str, out: Path, reps: int, max_tokens: int) -> dict:
    rows = []
    url = base.rstrip("/") + "/chat/completions"
    for rep in range(1, reps + 1):
        status, body, elapsed = post(
            url,
            {
                "model": model,
                "messages": [{"role": "user", "content": ESSAY}],
                "temperature": 0,
                "max_tokens": max_tokens,
                "chat_template_kwargs": THINK_OFF,
                "return_spec_tokens_details": True,
                "return_cached_tokens_details": True,
            },
            timeout=900,
        )
        choice = ((body.get("choices") or [{}])[0]) or {}
        usage = body.get("usage") or {}
        sgl = body.get("sgl_ext") or body.get("sglext") or {}
        spec = (
            sgl.get("spec_tokens_details")
            or choice.get("spec_tokens_details")
            or body.get("spec_tokens_details")
        )
        if isinstance(spec, list):
            spec = spec[0] if spec else None
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        row = {
            "rep": rep,
            "http_status": status,
            "elapsed_s": elapsed,
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": cached,
            "client_output_tok_s": (
                (usage.get("completion_tokens") or 0) / elapsed if elapsed else None
            ),
            "spec": spec,
            "response_keys": sorted(body.keys()),
        }
        rows.append(row)
        print(
            f"accept rep {rep} http={status} tok={row['completion_tokens']} "
            f"spec={spec}",
            flush=True,
        )
    ok = [r for r in rows if r["http_status"] == 200 and r.get("spec")]
    stable = ok[1:] if len(ok) > 1 else ok
    lengths = [float(r["spec"]["spec_accept_length"]) for r in stable if r.get("spec")]
    rates = [float(r["spec"]["spec_accept_rate"]) for r in stable if r.get("spec")]
    summary = {
        "prompt": ESSAY,
        "max_tokens": max_tokens,
        "reps": reps,
        "drop_first": True,
        "n_with_spec": len(ok),
        "median_accept_length": statistics.median(lengths) if lengths else None,
        "median_accept_rate": statistics.median(rates) if rates else None,
        "median_client_output_tok_s": statistics.median(
            [r["client_output_tok_s"] for r in stable if r.get("client_output_tok_s")]
        )
        if stable
        else None,
        "rows": rows,
    }
    (out / "acceptance.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="mimo26")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--r0b0bench", default="/home/r0b0tdgx/.venvs/r0b0bench/bin/r0b0bench")
    ap.add_argument("--skip-r0b0bench", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["R0B0BENCH_CHAT_TEMPLATE_KWARGS"] = json.dumps(THINK_OFF)
    bench = {"skipped": args.skip_r0b0bench}
    if not args.skip_r0b0bench:
        cmd = [
            args.r0b0bench,
            "run",
            "--profile",
            "systems",
            "--only",
            "latency,concurrency,throughput",
            "--base-url",
            args.base_url if args.base_url.endswith("/v1") else args.base_url.rstrip("/") + "/v1",
            "--model",
            args.model,
            "--output",
            str(out / "r0b0bench"),
            "--run-id",
            args.tag,
            "--timeout",
            "900",
        ]
        print("r0b0bench", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, env=env, text=True)
        bench = {"exit_code": proc.returncode, "cmd": cmd}
        (out / "r0b0bench-exit.json").write_text(json.dumps(bench, indent=2))
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)
    root = args.base_url[:-3] if args.base_url.endswith("/v1") else args.base_url.rstrip("/")
    acc = acceptance_series(root + "/v1", args.model, out, reps=5, max_tokens=2048)
    print(
        "median_accept_length",
        acc["median_accept_length"],
        "median_accept_rate",
        acc["median_accept_rate"],
        flush=True,
    )


if __name__ == "__main__":
    main()
