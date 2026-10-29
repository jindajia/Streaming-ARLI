# STATUS — SCH-01 fixed_delay10_v2

_Last updated 2026-10-29 06:15 EDT._ **Stage 1 and Stage 2 (seed 0, 300k) complete.** gpu-debug only; ≤5 jobs for this request. Simulation timing = deterministic table.

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
| 175,264 | 36 | 0.22 | |
| 200,252 | 35 | 0.48 | |
| 225,337 | 35 | 0.40 | |
| 250,150 | 36 | 0.19 | |
| 275,165 | 34 | 0.18 | |
| **300,155** | **32** | 0.07 | final (run DONE) |

Segments (all gpu-debug, A100, one writer): 8764519 (0 → 50,055), 8764520 (→ 114,806), 8764681 (→ 175,264,
incl. 150k milestone), 8764850 (→ 243,444), 8765095 (→ 300,155), each exit 3 = checkpointed after 50.7–51.7 min;
8765331 (6.6 min: final 300k evaluation, DONE). Totals: 942 training episodes (650 successful), 295,155 gradient
updates (294,899 with the actor on), 59,907 scheduler decisions, 26,081 inference requests.

### Learned behaviour (held-out evaluations, means per episode)

| | fixed baseline | learned @25k | @150k | @300k |
|---|---|---|---|---|
| successes / 50 | 36 | 36 | 35 | 32 |
| selected N | 10.0 | 7.96 | 7.68 | 7.34 |
| R (old actions after the decision) | 10.0 | 10.69 | 10.44 | 9.36 |
| launch phase p | 15.0 | 4.06 | 1.49 | 0.11 |
| old actions executed per chunk | 25.0 | 14.75 | 11.93 | 9.47 |
| observation age at handoff (steps) | 10.0 | 10.69 | 10.44 | 9.36 |
| inference calls / 1,000 steps | 39.6 | 68.4 | 86.0 | 107.4 |
| modeled inference ms / 1,000 steps | 7,914 | 12,522 | 15,206 | 18,455 |
| underrun steps | 0 | 0 | 0 | 0 |

Training-episode success (stochastic): 0.66 (0–50k, 154 episodes), 0.72 (50–150k, 319), 0.68 (150–300k, 469).

### Assessment
The scheduler learned a consistent, very different policy — request a new chunk almost immediately after each
handoff (p → 0), with fewer denoising steps (N ≈ 7–8) and R ≈ 10, so each chunk executes ~10 instead of 25 actions
and inference calls rise ~2.7× — but **held-out success did not improve**: 31–39/50 across checkpoints versus 36/50
for the fixed baseline, ending at 32/50. All differences are within the 50-episode binomial uncertainty (one training
seed); the step-0 evaluation of an identical schedule already differed by 2 episodes (34 vs 36) through renderer
non-determinism. With zero GPU-cost penalty and a tiny per-step time penalty, frequent replanning is nearly free in
the reward, so the learned preference for it is not evidence of better control. The Stage 1 diagnostic (ceiling-
limited, ~3-step earlier success for early replacement, three −0.5 cells) is consistent with "no clear success
advantage" in unperturbed ALOHA Transfer Cube.

### Limitations
One training seed; 50 evaluation episodes per point (Wilson intervals are over episodes, not seeds); simulation delay
table rather than measured latency, idealised scheduler overhead; unperturbed task with a strong baseline (72 %);
the diagnostic used 2 replicates on states both source episodes already solved. Recovery-rule states (q < 10) occur
only after departures (1,214 in training, 1–283 per evaluation); none under the fixed baseline. Measured RTC wall time is
logged per request (`wall_s`) but never advances the simulator.

### Figures
`results/online/rn_seed0_d10/eval_success.png` (+ .svg), `training_success.png`, `schedule.png`, `timing_compute.png`;
CSV/JSONL `evals.csv`, `evals_raw.jsonl`. Diagnostic: `results/diagnostic/diag_v2_coarse/`.
