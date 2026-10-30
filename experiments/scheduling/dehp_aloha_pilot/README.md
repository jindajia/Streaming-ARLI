# dehp_aloha_pilot — DEHP-style execution-horizon PPO on ALOHA (frozen π0)

Results: [REPORT.md](REPORT.md) · design/setup: [DESIGN.md](DESIGN.md).

| Path | What |
|---|---|
| `src/dehp/smdp.py` | semi-MDP executors: A (sync, h 1..50), B (delay 10 + RTC, h 10..40); 42-d physical state |
| `src/dehp/ppo.py` | categorical horizon actor, 201-atom two-hot critic, duration-aware GAE, PPO |
| `src/dehp/train.py` | one-hour job: baseline eval, initial eval, critic warm-up, PPO loop, periodic + final eval |
| `tests/test_dehp_cpu.py` | timing / RTC alignment / termination / GAE / two-hot / mask / PPO checks (CPU) |
| `configs/`, `cluster/run_dehp.sbatch` | run configs; gpu-debug 1-GPU 1-hour launcher |
| `scripts/plot_dehp.py`, `results/` | figures + compact results |

```bash
WT=/N/slate/jindjia/SD/VLA/streaming_arli_dehp; cd $WT/experiments/scheduling/dehp_aloha_pilot
JAX_PLATFORMS=cpu PYTHONPATH=$WT/src:$WT/experiments/scheduling/src:src python tests/test_dehp_cpu.py
export CMD="python -m dehp.train --config configs/run_A_nodelay.yaml"; sbatch -J dehp_A_nodelay cluster/run_dehp.sbatch
export CMD="python -m dehp.train --config configs/run_B_delay10.yaml"; sbatch -J dehp_B_delay10 cluster/run_dehp.sbatch
```
