# STATUS — SCH-01

_Last updated 2026-10-29 01:55 EDT._ Branch `experiment/queue-scheduler` (worktree `streaming_arli_sched`). Nothing here
claims a task-success improvement; all timing is the proxy-derived logical-time model (DECISIONS S2).

## Completed

- Repository/cluster inspection; isolated worktree and namespace `experiments/scheduling/`.
- Implementation: executor, N-step RTC sampler (vendored verified VJP port), masks/reference/prior, discrete SAC,
  empty-start replay, training loop, diagnostic, plotting.
- CPU checks (`tests/test_core_cpu.py`): all pass — timing table, masks, reference timeline (launch 18, 43, …;
  install new[7:32]), RTC shift/committed/overlap, duration-aware returns, prior parity, critic-first updates,
  exact replay/learner resume.

- Real-stack checks (job 8763771, `results/checks/gpu_checks_8763771.json`): RTC on frozen pi0 for N = 3..10 (full
  interval; committed-prefix error reduced at every N, e.g. N=10 0.042 → 0.026; RTC sample 84–137 ms on A100),
  snapshot restore (identical physics, images within ±1), reference episodes 3/3, step-0 parity 1.0.
- Smoke train/resume (jobs 8763771, 8763817, 8763859): two learner fixes (DECISIONS S12 entropy target tracks the prior;
  S14 zero-initialised critic heads); after S14 the prior decays gradually (reference probability 0.91–1.0 over 1,400
  updates) and resume continues counters/replay exactly.

- Coarse diagnostic `diag_coarse_v1` (job 8763872, 19 min; 240 rollouts, 70,064 diagnostic env steps, kept out of
  training): 6 snapshots (t = 25/100/175 of source seeds 20000000/20000001; both source episodes failed). In 5 of 6
  snapshots no branch succeeds. Snapshot 3 (seed 20000001, t = 25): 18/40 branch rollouts succeed; REPLAN(4,3),
  (5,3), (5,5) succeed 2/2, many cells 1/2, while REPLAN(7..15,10) / (15,7) and KEEP-one-tick fail 0/2. Only 2 paired
  replicates: a hypothesis (fast handoff helps in that state), not a confirmed difference.
  `results/diagnostic/diag_coarse_v1/` (heatmaps.png, branch_summary.csv, branches_raw.jsonl).
- Online seed-0 pilot `rn_seed0_v1` segment 1 (job 8763873, gpu-debug, 47 min, exit 3 = checkpointed): reached 50,120
  training ticks. Held-out evaluation (50 episodes, deterministic):

  | training steps | successes / 50 | reference parity | prior strength |
  |---|---|---|---|
  | matched fixed scheduler (N=10, K=25, RTC on) | **31** | — | — |
  | 0 | 31 | 1.00 | 1.0 |
  | 5,050 (end of warm-up) | 31 | 1.00 | 1.0 |
  | 25,126 | 36 | 0.73 | 0.56 |
  | 50,120 | 30 | 0.68 | 0.0 |

  Training: departures from the reference rule rise to ~20–40 per episode right after warm-up (≈8 % per decision
  over ~300 decisions); the policy launches more requests (56 vs 16 per episode at 50k) with many N = 3 choices.
  `results/online/rn_seed0_v1/`.

## Current jobs (SCH-01 only; ceiling 4)

| Job | Partition | What | State |
|---|---|---|---|
| 8763771, 8763817, 8763859 | gpu-debug | checks + smoke tests | completed |
| 8763872 | gpu-debug | coarse diagnostic | completed (19 min) |
| 8763873 | gpu-debug, A100 | RN seed 0, 0 → 50,120 ticks | completed segment (exit 3, checkpointed) |
| 8764189 | gpu, 5 h (budget 4.6 h) | RN seed 0, resume 50,120 → 300k (est. 3.3 h from measured throughput) | pending (Priority) |

## Next

1. When 8764189 runs: evaluations every 25k to 300k; regenerate `scripts/plot_online.py` outputs.
2. Confirm the snapshot-3 diagnostic difference with 8–10 independent replicates on a subset (small gpu-debug job).
3. Seed 1 and R-only / N-only ablations only after the seed-0 curve exists.
