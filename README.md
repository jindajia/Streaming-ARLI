# Streaming-ARLI

ALOHA Transfer Cube accuracy experiment:

1. **R0–R2** — paper-based reconstruction of ARLI ([arXiv 2608.23831v2](https://arxiv.org/html/2608.23831v2))
   Figure 6 (ALOHA panel, **no RTC**): full ARLI vs naive asynchronous DSRL vs the measured frozen π0 baseline,
   frozen `pi0_aloha_sim`, t_delay = 20, t_delay^rl = 10, chunk 50, execute 25, 10 denoising steps.
   **No author ARLI code exists**; this is built on the official DSRL code (`third_party/dsrl_pi0`, pinned).
2. **S0–S1** — our extension: one shared lightweight actor called between frozen action-expert denoising steps,
   compared from a common ARLI checkpoint (A continued ARLI, B stale per-step, C fresh per-step, D last-step).

Plan/discussion: `jindajia/research-notes` `notes/2026-10-27-01-arli-denoising-steering/`.
Status and latest numbers: [STATUS.md](STATUS.md). Every assumption: [DECISIONS.md](DECISIONS.md).
Large files: [MANIFEST.md](MANIFEST.md).

## Layout

| Path | What |
|---|---|
| `src/arli/schedule.py` | causal async clock (s_j, m_j, T_j, stale-prefix discard, committed window) |
| `src/arli/rollout.py` | episode runner; robot keeps executing committed actions while inference is in flight |
| `src/arli/pi0_policy.py` | frozen π0 (openpi fork) + step-wise sampler (begin / step / finish) |
| `src/arli/rl.py`, `controllers.py` | RL state per method, replay, DSRL PixelSAC agent with ARLI Table 1/3 settings |
| `src/arli/train.py` | R2 training (prefill → online, 25 grad steps / latent transition, eval every 25k) |
| `src/arli/steering.py`, `train_continue.py` | S0 shared steering actor, augmented-state SAC, S1 branches |
| `scripts/` | R1 checks, frozen baseline eval, checkpoint download, plotting |
| `configs/` | pinned run configs |
| `cluster/run.sbatch` | Big Red 200 launcher (account r01156) |
| `results/` | versionable CSV/JSONL and PNG/PDF curves (copied from scratch run dirs) |

## Reproduce

```bash
bash setup/create_env.sh                                  # env at /N/scratch/jindjia/envs/arli
python scripts/download_checkpoint.py                    # gs://openpi-assets/... with MD5 checks
python tests/test_schedule.py                            # timing audit (CPU)
JAX_PLATFORMS=cpu python tests/test_rl_cpu.py            # RL pieces (CPU)
export CMD="python scripts/check_r1.py"; sbatch -J arli_r1check -p gpu-debug --time=00:40:00 cluster/run.sbatch
export CMD="python -m arli.train --config configs/r2_arli_seed0.yaml"; sbatch -J r2_arli --time=2-00:00:00 cluster/run.sbatch
```
Re-submitting the same training command resumes from `<run_dir>/state_latest.pkl`.
