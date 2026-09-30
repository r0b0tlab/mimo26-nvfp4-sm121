# mimo26 max-perf ladder — morning report
decided 2026-09-30T06:24:51Z

## VERDICT: live profile MTP-500k-mm-graph CONFIRMED optimal
no cell beat the floor on all c1 lanes simultaneously

| cell | short | medium | prose | prefill8k | c4 agg | accept |
|---|---|---|---|---|---|---|
| P0-FLOOR (live) | 23.75 | 25.09 | 16.41 | 651 | 42.1 | 3.105 |
| CELL-CHUNK16 | 22.8 | 21.3 | 15.4 | 673 | 48.9 | 3.032 |
| CELL-CONS | 23.49 | 24.49 | 17.42 | 662 | 43.1 | 3.122 |
| CELL-K | 23.17 | 24.03 | 16.79 | 663 | 43.3 | 3.101 |
| CELL-KV16 | 23.29 | 23.86 | 15.46 | 674 | 46.9 | 3.086 |
| CELL-MOE | 23.22 | 22.13 | 17.54 | 636 | 41.5 | 3.13 |
| CELL-NCCL | 19.87 | 21.73 | 15.01 | 684 | 51.3 | 3.115 |
| CELL-NCCLCH | 21.68 | 23.42 | 16.5 | 684 | 42.9 | 3.04 |
| CELL-S2 | 22.27 | 23.0 | 18.36 | 641 | 45.9 | 2.577 |
| CELL-SI8 | 22.95 | 24.36 | 15.91 | 642 | 42.9 | 3.074 |

evidence: results/maxperf/CELL-*.json, DRIVER-*.log, LADDER*.log