# Session handoff — MTP vs vendor MXFP4 throughput

Date: 2026-09-29. Campaign checkout HEAD `683239d`. Comparison files are untracked. Do not commit unless asked.

## Done

Matched r0b0bench throughput comparison of our NVFP4 EAGLE MTP profile against the stock vendor MXFP4 checkpoint. Same client protocol. Per-request acceptance measured. Preferred NVFP4 MTP serve restored afterward.

Artifact: `results/cmp-mtp-vendor/comparison.json`
Raw runs: `results/cmp-mtp-vendor/ours/` and `results/cmp-mtp-vendor/vendor/`
Runner: `scripts/72_cmp_perf.py` (untracked)
Vendor boot helper: `scripts/73_boot_vendor_cmp.sh` (untracked)

## Live state

Preferred serve is up. tmux `mimo26-mtp` on the serve node. Container `mimo26-r0` up about 6 hours. `/health` returned 200 at handoff. Tag `MTP-500k-mm-graph`. Model is our NVFP4 checkpoint, not the vendor tree.

Serve identity for the comparison boots, not the published suite: parent image `r0b0tlab/sglang-mimo26-env:20260922-582389ce` plus bind-mounted eagle loader, nextn, triton window-index fill, and Marlin skip. Not a claim that the baked `20260922-582389ce-mtp-mm` tag served these numbers.

## What was measured

r0b0bench systems profile, `--only latency,concurrency,throughput`, think-off. All three lanes PASS on both sides. Infra errors 0. `invalid_for_publish` true because the other systems lanes were not run. Do not ledger this as a systems row. Do not stitch these numbers onto the earlier 17.31 systems C1 row. That was a different boot.

Acceptance is a second pass on the same C1 essay via chat completions with `return_spec_tokens_details`. Five reps, drop first, median. Not a lifetime `/get_server_info` average. The stock throughput lane still does not store acceptance.

Shared flags: EAGLE, 3 steps, 4 draft tokens, top-k 1, draft window 4096, mem-frac 0.90, context 524288, max running requests 8, think-off.

Labeled difference: our side used fp8 KV with `kv_scales.json`. Vendor MXFP4 has no scales file, so that side used bf16 KV. Not a pure weight comparison.

## Numbers

C1 essay, 2048 requested output tokens, median client tok/s, drop first:
- Ours 16.896785104930984
- Vendor 16.59876510706442
- Ours +1.80%

Prefill proxy, median prompt tokens 22771, wall proxy not a kernel metric:
- Ours 16238.114625548129
- Vendor 17961.489731175727
- Ours -9.59%

Short-prompt stream latency, mean, drop first:
- TTFT ms: ours 286.06826544273645, vendor 269.57590848905966, ours +6.12%
- ITL ms: ours 138.21388540288484, vendor 124.73665953342181, ours +10.80%
- E2E ms: ours 4468.771663785446, vendor 4011.855735036079, ours +11.39%

Concurrency aggregate tok/s at 1/2/4/6:
- Ours 24.819588775378946 / 45.782662691202425 / 70.20427627186481 / 89.50335409028958
- Vendor 26.185739068982585 / 47.98780460499821 / 77.15673164679302 / 100.5688983692613
- Ours -5.22% / -4.60% / -9.01% / -11.00%

Acceptance, median of stable reps, length and rate:
- Ours 2.4441412808827057 and 0.4813804269609019
- Vendor 2.5067319461444306 and 0.5022439820481436

Vendor's five acceptance reps had identical spec stats while wall time still moved between about 106 s and 123 s. That is greedy determinism on a repeated prompt, not a cached replay. The server did not return cached-token counts. A 40-token prefix hit is unmeasured.

## Do not

- Do not publish or ledger this comparison unless asked.
- Do not rewrite the published 17.31 systems row, the Q200 168/180 row, or `invalid_for_publish` on older ledgers.
- Do not treat this as a full systems suite. BFCL, NIAH, and Q200 were not rerun.
- Do not raise mem-frac above 0.90. Do not stop the Hermes gateway.
- Do not quote the old 8-wide 56.9 vs 69.1 file as this comparison. That was DFlash on both sides.

## Left

Nothing is required to close the data-gap request. Optional only if asked: commit the runner and artifact, add the table to the site, or rerun vendor with a scales file if one is produced later so KV dtype matches.
