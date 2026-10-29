# LEGACY — 7-step-budget SCH-01 experiment (archived 2026-10-29)

The first SCH-01 version (`src/sch01`, configs `configs/online_rn_seed0.yaml`, `configs/diagnostic_coarse.yaml`) used a
profiling-derived **proxy delay model** — L(N) = 42.5 + 8.17 N ms, synthetic jitter clip(N(0,5), ±10) ms,
budget(N) = ceil((L+10)/20) → **budget(10) = 7 control steps** — and a fixed baseline that requested at q = 7.
It is a separate experimental condition and is kept unchanged for reproducibility. The current experiment is
[`fixed_delay10_v2/`](fixed_delay10_v2/) (deterministic table, N = 10 → 10 steps).

| Item | Location / value |
|---|---|
| Code revision | branch `experiment/queue-scheduler` @ `78c4d7a` (`experiments/scheduling/src/sch01`, unchanged since) |
| Online run | `/N/scratch/jindjia/arli/runs_scheduling/online/legacy_budget7_rn_seed0_v1/` (symlink `rn_seed0_v1`): config, launch json, checkpoints at 0 / 5,050 / 25,126 / 50,120, replay parts, episodes / updates / evals logs, `LEGACY.txt` |
| Final preserved progress | **50,120 training control steps** (last segment job 8763873, gpu-debug, exit 3 = checkpointed). Its continuation 8764189 (gpu) never started and was cancelled on 2026-10-29. No progress beyond 50,120 exists. |
| Held-out results (50 episodes) | fixed 7-step baseline 31/50; learned 31 (0), 31 (5,050), 36 (25,126), 30 (50,120) |
| Diagnostic | `/N/scratch/jindjia/arli/runs_scheduling/diagnostic/legacy_budget7_diag_coarse_v1/` (240 rollouts) |
| Compact results / figures | `results/online/rn_seed0_v1/`, `results/diagnostic/diag_coarse_v1/`, `results/checks/` (legacy) |
