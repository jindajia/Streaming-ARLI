# SCH-01 fixed_delay10_v2 — adaptive R/N scheduling with a fixed 10-step N=10 delay

Question: when a chunk has already executed some actions, should we finish its planned execution or replace its
remaining actions earlier with a faster, possibly less accurate prediction? A scheduler chooses KEEP or REPLAN(R, N):
R = further old actions before handoff, N = denoising steps of the next request. The VLA (`pi0_aloha_sim`) is frozen;
only the scheduler is trained. RTC is on for every scheduled request.

Timing (single shared table, `src/sch02/core.py`): d(N) = 6, 7, 7, 8, 9, 9, 10, 10 control steps for N = 3..10; no
jitter, no margin; a simulation setting, not a GPU measurement. Fixed baseline: N = 10, KEEP while q > 10, REPLAN(10, 10)
at q = 10 (after 15 actions), handoff 10 steps later — 25 actions per chunk.

| Path | What |
|---|---|
| `src/sch02/core.py` | delay table, masks, fixed baseline / recovery rule, prior |
| `src/sch02/executor.py` | executor with in-chunk progress p, mid-chunk snapshots, request provenance |
| `src/sch02/diagnostic.py` | Stage 1 same-state heatmap diagnostic (analysis only) |
| `src/sch02/train.py`, `replay.py` | Stage 2 online discrete-SAC training (resumable segments, 150k milestone) |
| reused unchanged | `../src/sch01/learner.py` (discrete SAC), `../src/sch01/sampler.py` + `rtc_vendored.py` (N-step RTC) |
| `tests/test_v2_cpu.py`, `scripts/gpu_checks_v2.py`, `scripts/smoke_v2.sh` | validation |
| `scripts/plot_diagnostic_v2.py`, `scripts/plot_online_v2.py` | figures |
| `cluster/run_v2.sbatch` | gpu-debug launcher (account r01156) |

Reproduce (gpu-debug only, one training writer, ≤5 jobs):
```bash
WT=/N/slate/jindjia/SD/VLA/streaming_arli_sched; cd $WT/experiments/scheduling/fixed_delay10_v2
JAX_PLATFORMS=cpu PYTHONPATH=$WT/src:$WT/experiments/scheduling/src:src python tests/test_v2_cpu.py
export CMD="bash scripts/smoke_v2.sh"; sbatch -J sch2_smoke -p gpu-debug --time=00:59:00 cluster/run_v2.sbatch
export CMD="python -m sch02.diagnostic --config configs/diagnostic_v2.yaml"; sbatch -J sch2_diag -p gpu-debug --time=00:59:00 cluster/run_v2.sbatch
export CMD="python -m sch02.train --config configs/online_rn_seed0.yaml --max_wall_hours 0.78"; sbatch -J sch2_rn0_s01 -p gpu-debug --time=00:59:00 cluster/run_v2.sbatch
python scripts/plot_diagnostic_v2.py --run diag_v2_coarse; python scripts/plot_online_v2.py --run rn_seed0_d10
```
