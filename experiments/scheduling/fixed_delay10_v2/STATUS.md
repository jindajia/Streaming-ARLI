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

## Next
Inspect heatmaps → Stage 2 seed-0 online run `rn_seed0_d10` in 0.78 h gpu-debug segments to 150k (milestone) and 300k.
