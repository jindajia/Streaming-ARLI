# SCH-01 — queue-aware adaptive inference scheduling (RTC ON, frozen pi0, ALOHA Transfer Cube)

Isolated experiment inside Streaming-ARLI (branch `experiment/queue-scheduler`). It does not touch the ARLI /
steering code paths, jobs, checkpoints or results.

Plans: research-notes `notes/2026-10-28-01-adaptive-inference-scheduling/` (ONLINE_TRAINING, FIRST_EXPERIMENT_RTC_ON,
EXPERIMENT_PLAN). Status: [STATUS.md](STATUS.md). Design choices and deviations: [DECISIONS.md](DECISIONS.md).
Large artefacts: [MANIFEST.md](MANIFEST.md).

| Path | What |
|---|---|
| `src/sch01/core.py` | proxy timing model, action space (KEEP + REPLAN(R,N)), feasibility masks, reference rule, prior |
| `src/sch01/sampler.py`, `rtc_vendored.py` | frozen pi0 with N-step full-interval sampling and VJP RTC (vendored verified port) |
| `src/sch01/executor.py` | event-driven executor: hidden results, handoffs, RTC alignment, duration-aware decisions, snapshots |
| `src/sch01/learner.py`, `replay.py`, `train.py` | discrete SAC with reference prior; empty-start online replay; training/eval loop |
| `src/sch01/diagnostic.py` | same-state R/N branching diagnostic (understanding only, never training data) |
| `tests/test_core_cpu.py` | CPU checks (fake sim/sampler) |
| `scripts/gpu_checks.py`, `scripts/smoke.sh` | real-stack checks + short train/resume smoke |
| `scripts/plot_online.py`, `scripts/plot_diagnostic.py` | figures from run logs |
| `cluster/run_sch.sbatch` | BR200 launcher (account r01156) |

Reproduce:
```bash
WT=/N/slate/jindjia/SD/VLA/streaming_arli_sched; cd $WT/experiments/scheduling
JAX_PLATFORMS=cpu PYTHONPATH=$WT/src:src python tests/test_core_cpu.py
export CMD="bash scripts/smoke.sh"; sbatch -J sch01_smoke -p gpu-debug --time=00:59:00 cluster/run_sch.sbatch
export CMD="python -m sch01.diagnostic --config configs/diagnostic_coarse.yaml"; sbatch -J sch01_diag -p gpu-debug --time=00:59:00 cluster/run_sch.sbatch
export CMD="python -m sch01.train --config configs/online_rn_seed0.yaml --max_wall_hours 0.75"; sbatch -J sch01_rn0 -p gpu-debug --time=00:59:00 cluster/run_sch.sbatch
```
All timing is a proxy-derived logical-time simulation (A100 pi0+RTC profile), not measured native ALOHA RTC latency.
