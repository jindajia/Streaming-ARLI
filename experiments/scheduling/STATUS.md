# STATUS — SCH-01

_Last updated 2026-10-29 00:40 EDT._ Branch `experiment/queue-scheduler` (worktree `streaming_arli_sched`). Nothing here
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

## Current jobs (SCH-01 only; ceiling 4)

| Job | Partition | What | State |
|---|---|---|---|
| 8763771 | gpu-debug | real-stack checks + train/resume smoke | completed |
| 8763817 | gpu-debug | smoke after S12 (prior still erased) | completed |
| 8763859 | gpu-debug | smoke after S14 (prior decays gradually) | completed |
| 8763872 | gpu-debug | coarse same-state R/N diagnostic (`configs/diagnostic_coarse.yaml`) | submitted |
| 8763873 | gpu-debug | online RN seed 0 pilot segment 1 (`configs/online_rn_seed0.yaml`, 0.75 h budget) | submitted |

## Next

1. Diagnostic heatmaps (`scripts/plot_diagnostic.py`).
2. Pilot health check (evaluations at 0 / 5k / 25k, warm-up accounting, prior decay, resume) → regular `gpu` job(s)
   for the remaining ticks toward 300k, sized from measured throughput.
