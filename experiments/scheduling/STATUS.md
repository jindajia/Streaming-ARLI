# STATUS — SCH-01

_Last updated 2026-10-28 23:55 EDT._ Branch `experiment/queue-scheduler` (worktree `streaming_arli_sched`). Nothing here
claims a task-success improvement; all timing is the proxy-derived logical-time model (DECISIONS S2).

## Completed

- Repository/cluster inspection; isolated worktree and namespace `experiments/scheduling/`.
- Implementation: executor, N-step RTC sampler (vendored verified VJP port), masks/reference/prior, discrete SAC,
  empty-start replay, training loop, diagnostic, plotting.
- CPU checks (`tests/test_core_cpu.py`): all pass — timing table, masks, reference timeline (launch 18, 43, …;
  install new[7:32]), RTC shift/committed/overlap, duration-aware returns, prior parity, critic-first updates,
  exact replay/learner resume.

## Current jobs (SCH-01 only; ceiling 4)

| Job | Partition | What | State |
|---|---|---|---|
| 8763771 | gpu-debug | real-stack checks + short train/resume smoke | running |

## Next

1. Smoke results → fix anything found.
2. Coarse diagnostic (`configs/diagnostic_coarse.yaml`) on gpu-debug.
3. Online seed-0 pilot segment on gpu-debug; then regular `gpu` for the 300k run.
