# STATUS — SCH-01 fixed_delay10_v2

_Last updated 2026-10-29 01:35 EDT._ gpu-debug only; ≤5 jobs for this request. Simulation timing = deterministic table.

## Done
- Legacy 7-step-budget experiment archived ([../LEGACY_BUDGET7.md](../LEGACY_BUDGET7.md)): final progress 50,120 steps;
  pending continuation 8764189 cancelled unstarted; run dirs renamed `legacy_budget7_*` (symlinks keep old configs valid).
- v2 implementation + CPU checks (`tests/test_v2_cpu.py`): delay table, masks (KEEP iff q−1 ≥ 6), baseline timeline
  (observe t = 15, 40, 65…, install new[10:35], RTC shift 15/25, committed 10), result gating, mid-chunk snapshot
  example (p = 5: baseline observes t = 40 / hands off t = 50; REPLAN(7,5) observes t = 30, hands off t = 37, 12 old
  actions executed, 13 discarded; next RTC shift 22), step-0 parity, empty replay with new reference labels.
- GPU smoke 8764392 (7.9 min): same checks on the real simulator/pi0, prior-off supplemental eval, milestone copy,
  two interrupted segments resumed with continuous counters; recovery states 0.

## Jobs (this request)

| Job | Partition | What | State |
|---|---|---|---|
| 8764392 | gpu-debug | v2 smoke | completed |
| 8764415 | gpu-debug | Stage 1 diagnostic `diag_v2_coarse` (12 snapshots, ~456 rollouts) | running |


## Stage 1 result — diagnostic `diag_v2_coarse` (job 8764415, 29 min, 456 rollouts, 104,909 diagnostic env steps)
Both source episodes (seeds 20000100/101) succeeded under the new baseline; all 12 snapshots captured (none missing).
The fixed baseline succeeds 2/2 in every snapshot and 453/456 branch rollouts succeed: the diagnostic is
**ceiling-limited for success** (mean improvement −0.007 over 216 cells; the only deviations are three −0.5 cells,
fast-handoff N = 3 branches at seed 20000100 chunk 5, p = 0 and 5). Among fully successful cells, early-replacement
branches reach success 2.9 ticks earlier on average (SD 6.7, n = 213; by N: −3.1/−2.9/−2.7/−2.8 for N = 3/5/7/10).
Interpretation: no evidence that changing the first decision improves success in these states; a small, noisy timing
effect. Two paired replicates on baseline-solved states cannot establish a benefit.
`results/diagnostic/diag_v2_coarse/` (heatmaps.png/.svg, branch_summary.csv, branches_raw.jsonl, snapshots.json).

## Stage 2 — online seed 0 `rn_seed0_d10` (gpu-debug, 0.78 h segments, one writer)

| Training steps | Successes / 50 | Reference parity | Note |
|---|---|---|---|
| fixed baseline (N=10, 10-step delay) | **36** | — | recovery decisions 0 |
| 0 | 34 | 1.00 | identical decisions; 2-episode gap = renderer non-determinism |
| 5,334 | 35 | 1.00 | end of warm-up |
| 25,146 | 36 | 0.78 | prior strength 0.56 |
| 25,146 (prior disabled, supplemental) | 35 | 0.12 | separate file, not on the primary curve |
| 50,055 | 31 | 0.75 | prior removed |
| 75,114 | 39 | 0.65 | |
| 100,247 | 37 | 0.68 | |
| 125,116 | 36 | 0.52 | |
| **150,000** | **35** | 0.48 | **milestone saved** (`ckpt/milestone_0150000`, 38,347 replay rows) |

Segments: 8764519 (0 → 50,055), 8764520 (→ 114,806), 8764681 (→ 175,264), each exit 3 = checkpointed after ~51 min;
8764850 running; 8764852+ queued (≤ 2 jobs at a time).
