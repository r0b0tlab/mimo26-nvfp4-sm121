Publication client started 2026-09-25T03:03Z from r0b0tdgx1 (off the serve).

Client: r0b0bench 1.0.0rc2 at df517592988ce6f2b05ca0ab01b632f111395ebf
Endpoint: http://192.168.68.78:30000/v1 model mimo26
Profile: FINAL3 sha256 ccd6a7e972ae5a05001c3316f2c2ceb21e81400597dc158902ca380131ff5799
Serve image: sha256:3eaa58037e3c6952166299939e201286fef13af22ef7ec290acc5b3f3a7a4fb2
Grading image (HumanEval sandbox only): sha256:9ab175696a136534026ebcc7f94d7f3ab5b4bb1970706424054788a85ad26eb0
Think-off: {"enable_thinking":false,"thinking":false}
Probe before the run: "What is 2+2?" -> content "4", finish_reason stop, reasoning_tokens 0.

Scope of this package run:
- systems --only canary,latency,concurrency,throughput
- then Q200v2 text180, max_tokens 8192 disclosed up front, workers 1, timeout 900
- not in this process: full BFCL-MT/AST, NIAH 25/50/90 of 1048576, BFCL hard20

Why those are out:
- A filtered --only report is invalid_for_publish. This file says so. Do not promote it to a core-subset row.
- NIAH 90% of the advertised 1,048,576 is about 943k tokens. Host free memory at start was 4.84 GiB on the server rank. That is inside the freeze envelope. The earlier mk3 at 480,655 tokens already passed and is a separate artifact.
- Canary tool_call failed: the model returned MiMo XML in content and tool_calls null. Identity, needle, structured, and zh_arithmetic passed. infra_errors 0. Not washed.
- Q200v2 admission uses a live 4 GiB floor read from /proc/meminfo on both ranks. The Qwen kit's 16 GiB floor cannot be met on this 0.90 mem-fraction profile without changing it. PROCEDURES section 4 reference ran at 5.07 GiB minimum. Both ranks admitted at preflight (4.84 and 8.71 GiB). The 3 GiB docker kill guard is still the hard stop.
- hard_reasoning was graded by independent review of the recorded responses: 17/20. The three failures are hard-04, hard-12, and hard-16. Total with the auto-graded families is 168/180. The responses were not regenerated.
- Ledger add waits for report.json. Do not commit a partial row.
