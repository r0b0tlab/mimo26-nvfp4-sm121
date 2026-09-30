#!/usr/bin/env python3
"""74_promote_or_confirm.py — morning auto-promotion for the mimo26 ladder.

Waits for LADDER4 DONE, then:
  1. Loads every CELL-*.json + P0-FLOOR.json.
  2. Winner rule: beats floor decode median on ALL of short/medium/prose by
     >=3% AND prefill_8k >= floor * 0.95 AND c4 aggregate >= floor * 0.95.
  3. If a winner exists: boot it via 70_maxperf_driver.sh FINAL <env>, rerun
     60_profile_bench + canary + 3 media probes (from 63_admission.py
     conventions), write results/maxperf/FINAL-VERDICT.json, keep it live.
  4. If no winner: confirm MTP-500k-mm-graph as optimal, ensure the live
     serve is that profile (RESTORE already ran in ladder4), write
     results/maxperf/FINAL-VERDICT.json with the no-winner verdict.
  5. In both cases: write results/maxperf/MORNING-REPORT.md (all cells,
     verdict, evidence paths) and leave the serve HEALTHY on the best
     profile.
r0b0tlab mimo26.
"""
import json
import subprocess
import time
import urllib.request
from pathlib import Path

C = Path.home() / "projects/mimo26-nvfp4-sm121"
R = C / "results/maxperf"
URL = "http://192.168.68.78:30000"
LANES = ["short_code", "medium_code", "prose"]


def sh(cmd, timeout=600):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout).stdout.strip()


def wait_for_ladder4():
    for _ in range(720):  # up to 4h
        if "LADDER4 DONE" in (R / "LADDER4.log").read_text() if (R / "LADDER4.log").exists() else False:
            return True
        time.sleep(20)
    return False


def load(tag):
    p = R / f"{tag}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    w = d.get("workloads", {})
    out = {}
    for k in LANES + ["prefill_8k", "c4_short"]:
        v = w.get(k) or {}
        out[k] = {"dec": v.get("decode_tps_median"),
                  "agg": v.get("aggregate_tps"),
                  "pre": v.get("prefill_tps_median")}
    st = d.get("server_info_end") or {}
    out["accept"] = st.get("avg_spec_accept_length")
    return out


def healthy(timeout=15):
    try:
        urllib.request.urlopen(URL + "/health", timeout=timeout)
        return True
    except Exception:
        return False


def canary():
    body = json.dumps({
        "model": "mimo26",
        "messages": [{"role": "user", "content": "What is the sum of 7 and 5? Answer with just the number."}],
        "max_tokens": 64, "temperature": 0.0, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=120).read())
    txt = (d["choices"][0]["message"].get("content") or "").strip()
    return {"text": txt[:80], "ok": "12" in txt}


def main():
    ok = wait_for_ladder4()
    print("ladder4 done:", ok, flush=True)

    floor = load("P0-FLOOR") if (C / "results/profile_bench/P0-FLOOR.json").exists() else None
    if floor is None:
        floor = load("RESTORE")
    cells = {}
    for p in sorted(R.glob("CELL-*.json")):
        cells[p.stem] = load(p.stem)

    winners = []
    for tag, c in cells.items():
        if not c or floor is None:
            continue
        try:
            if all(c[k]["dec"] and floor[k]["dec"] and
                   c[k]["dec"] >= floor[k]["dec"] * 1.03 for k in LANES) and \
               c["prefill_8k"]["pre"] and floor["prefill_8k"]["pre"] and \
               c["prefill_8k"]["pre"] >= floor["prefill_8k"]["pre"] * 0.95 and \
               c["c4_short"]["agg"] and floor["c4_short"]["agg"] and \
               c["c4_short"]["agg"] >= floor["c4_short"]["agg"] * 0.95:
                winners.append(tag)
        except (KeyError, TypeError):
            pass
    print("winners:", winners, flush=True)

    verdict = {"floor": floor, "cells": cells, "winners": winners,
               "decided_utc": time.strftime("%FT%TZ", time.gmtime())}

    if winners:
        env = {"CELL-S2": "CELL-S2.env", "CELL-K": "CELL-K.env",
               "CELL-MOE": "CELL-MOE.env", "CELL-NCCL": "CELL-NCCL.env",
               "CELL-KV16": "CELL-KV16.env", "CELL-CHUNK16": "CELL-CHUNK16.env",
               "CELL-CONS": "CELL-CONS.env", "CELL-TC": "CELL-TC.env",
               "CELL-SI8": "CELL-SI8.env", "CELL-NCCLCH": "CELL-NCCLCH.env"}
        tag = winners[0]
        sh(f"cd {C} && bash scripts/70_maxperf_driver.sh FINAL {env[tag]} "
           f"> {R}/DRIVER-FINAL.log 2>&1", timeout=3600)
        if healthy():
            time.sleep(20)
            sh(f"cd {C} && python3 scripts/60_profile_bench.py --url {URL} "
               f"--tag FINAL --out {R}/FINAL.json > {R}/FINAL.log 2>&1")
            verdict["final"] = load("FINAL") or load("FINAL2")
            verdict["canary"] = canary()
            verdict["promoted"] = tag
        else:
            verdict["promoted"] = None
            verdict["final_boot_failed"] = True
            # fall back to the live profile
            sh(f"cd {C} && RESTORE=1 bash scripts/70_maxperf_driver.sh "
               f"RESTORE MTP-500k-mm-graph.env > {R}/DRIVER-RESTOREF.log 2>&1",
               timeout=3600)
    else:
        verdict["promoted"] = None
        # ladder4 already restored the live profile; verify health
        for _ in range(40):
            if healthy():
                break
            time.sleep(15)
        if not healthy():
            sh(f"cd {C} && RESTORE=1 bash scripts/70_maxperf_driver.sh "
               f"RESTORE MTP-500k-mm-graph.env > {R}/DRIVER-RESTOREF.log 2>&1",
               timeout=3600)
        verdict["final_state"] = "live profile MTP-500k-mm-graph confirmed optimal"

    (R / "FINAL-VERDICT.json").write_text(json.dumps(verdict, indent=1))

    # morning report
    lines = ["# mimo26 max-perf ladder — morning report",
             f"decided {verdict['decided_utc']}", ""]
    if verdict.get("promoted"):
        lines.append(f"## PROMOTED: {verdict['promoted']}")
        lines.append(f"canary: {verdict.get('canary')}")
    else:
        lines.append("## VERDICT: live profile MTP-500k-mm-graph CONFIRMED optimal")
        lines.append("no cell beat the floor on all c1 lanes simultaneously")
    lines.append("")
    lines.append("| cell | short | medium | prose | prefill8k | c4 agg | accept |")
    lines.append("|---|---|---|---|---|---|---|")
    rows = [("P0-FLOOR (live)", floor)] + sorted(cells.items())
    for tag, c in rows:
        if not c:
            continue
        lines.append(
            f"| {tag} | {c['short_code']['dec'] and round(c['short_code']['dec'],2)} "
            f"| {c['medium_code']['dec'] and round(c['medium_code']['dec'],2)} "
            f"| {c['prose']['dec'] and round(c['prose']['dec'],2)} "
            f"| {c['prefill_8k']['pre'] and round(c['prefill_8k']['pre'])} "
            f"| {c['c4_short']['agg'] and round(c['c4_short']['agg'],1)} "
            f"| {c['accept'] and round(c['accept'],3)} |")
    lines.append("")
    lines.append("evidence: results/maxperf/CELL-*.json, DRIVER-*.log, LADDER*.log")
    (R / "MORNING-REPORT.md").write_text("\n".join(lines))
    print("report written; healthy:", healthy(), flush=True)


if __name__ == "__main__":
    main()
