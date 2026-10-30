# DEHP-style execution-horizon pilot on ALOHA Transfer Cube — REPORT (2026-10-30)

A fresh DEHP-style horizon-learning experiment on ALOHA (frozen `pi0_aloha_sim`), **separate from the earlier R/N SAC
scheduler (SCH-01) and the denoising-steering experiments**: new categorical horizon actor, distributional critic,
optimizers and on-policy rollouts; nothing reused from those runs except the frozen π0 sampler and the verified RTC port.
Two independent one-GPU, one-hour `gpu-debug` jobs; no continuation. Design and verified setup: [DESIGN.md](DESIGN.md).

## Summary

1. **Matched fixed-baseline success (h = 25, same executor, 10 held-out episodes):** A (no delay) **5/10**;
   B (delay 10, RTC on) **5/10**.
2. **No demonstrated improvement.** Learned (argmax-h) evaluations: A 5/10 → 5/10 → **7/10 final**; B 6/10 → 7/10 →
   5/10 → **6/10 final**. All within 10-episode noise of the 5/10 baselines. Training-episode success (sampled h) was
   flat: A 0.64 → 0.63, B 0.63 → 0.68 (first-5 vs last-5 batch averages). Moreover the actor is still nearly uniform
   (below), so its argmax is one constant h for every state: the "learned" evaluations are effectively fixed
   horizons (A final: h = 29 for all 116 decisions; B final: h = 36 for all 100) picked by tiny logit differences —
   not a state-dependent schedule.
3. **Numerically healthy.** No non-finite values; target out-of-support fraction 0 throughout; critic cross-entropy
   decreasing (A 4.90 → 4.31, B 5.20 → 4.24), explained variance rising (A 0.01 → 0.18, B 0.01 → 0.30), mean V tracks
   targets (A 0.53 vs 0.45, B 0.57 vs 0.60); PPO KL ≤ 5e-4 (A) / 1.1e-3 (B), clip fraction ≤ 0.12 (A) / 0.41 (one B
   iteration); gradient norms ≤ 0.84 (actor) / 4.6 (critic); 0 non-terminal duration mismatches; B: 0 async timing
   violations, 0 underruns.
4. **Horizons preferred:** essentially none yet. Policy entropy fell only from 3.912 to 3.887 (A, max log 50) and
   3.434 to 3.383 (B, max log 31); the training-sample mean h stayed ≈ 25–27 (the uniform mean is 25.5 / 25). The
   argmax drifted between iterations (A: 33 at 97k steps → 29 final; B: 16 → 36/16 mix → 36 final).
5. **More training is not justified as configured; fix the learning signal / update strength first.** In ~200k env
   steps the actor barely moved: with ~256 decisions per batch, 4 full-batch epochs, LR 1e-4 and clip 0.05, each
   update is tiny and the sparse +1 advantage signal is noisy (paper: 1000 parallel envs, 2048–4096-decision
   minibatches, 5–15 epochs per iteration, up to 10^8+ env steps). Before spending more GPU time: (a) increase update
   strength per sample (more epochs / larger batches, or a larger actor LR with the same clip) and check the actor
   actually moves; (b) evaluate the stochastic policy and report its h distribution alongside argmax, since argmax of a
   near-uniform policy is a constant horizon; (c) use ≥ 50 evaluation episodes — 10 cannot resolve the expected effect;
   (d) note that h = 25 may already be near the best fixed horizon here (baseline success ≈ training success), so the
   headroom for horizon selection on this task is unknown.

## Runs

| | Run A — no delay | Run B — delay 10 |
|---|---|---|
| Job (gpu-debug, 1 GPU, `--time=01:00:00`) | 8779289, COMPLETED, 49:26 | 8779290, COMPLETED, 49:27 |
| Executor | sync, RTC off, h ∈ {1..50} | async d = 10, RTC on (committed 10), h ∈ {10..40} |
| Config | `configs/run_A_nodelay.yaml` | `configs/run_B_delay10.yaml` |
| Scratch run dir | `/N/scratch/jindjia/arli/runs_dehp/run_A_nodelay/` | `/N/scratch/jindjia/arli/runs_dehp/run_B_delay10/` |
| Checkpoints | `ckpt/learner_{it0016,final}.msgpack` | `ckpt/learner_{it0013,it0026,final}.msgpack` |
| Log | `/N/scratch/jindjia/arli/runs_dehp/logs/dehp_A_nodelay_8779289.out` | `.../dehp_B_delay10_8779290.out` |
| PPO iterations / decisions / env steps | 35 / 8,960 / 217,198 | 29 / 7,424 / 179,289 |
| Training episodes (success) | 675 (447) | 556 (365) |
| Inference calls (training) | 8,961 | 7,636 |
| Wall: load+compile / eval / collect / update | 368 / 516 / 1,974 / 14 s | 367 / 344 / 2,134 / 7 s |
| π0 inference / simulator wall (training) | 787 / 1,161 s | 1,115 / 989 s |

Held-out evaluations (seeds 10,000,000–10,000,009, fixed π0 noise per episode; learned = argmax h):

| Run | baseline h=25 | initial actor | periodic | final |
|---|---|---|---|---|
| A | 5/10 | 5/10 (argmax of uniform = h 1 for all decisions; 3,449 calls) | 5/10 @ 97k (h 33/34) | **7/10** @ 217k (h 29) |
| B | 5/10 | 6/10 (argmax of uniform = h 10) | 7/10 @ 80k (h 16/36), 5/10 @ 160k (h 36/16) | **6/10** @ 179k (h 36) |

Raw data: `results/run_*/evals.jsonl` (per-episode success, horizons, durations), `results/run_*/iters.jsonl` (per-
iteration PPO/critic diagnostics, h histograms, timing). Figures: `results/run_*/dehp_run_*.png|svg` (held-out success
vs env steps with the baseline line, success vs job time, h-distribution evolution, actor/PPO/critic diagnostics).
Reproduce: `python scripts/plot_dehp.py`.

## Caveats

- A and B differ in delay, RTC and feasible horizon range; each learner is compared with its own matched baseline.
- Ten evaluation episodes per point; counts only.
- Startup chunk is computed with the clock frozen in both runs (both A and B, learned and fixed alike).
- The critic in B sees the installed plan + timing metadata (asynchronous adaptation, not the paper's state-only V).
- Closed-loop rollouts are not bit-reproducible (EGL renderer ±1 intensity noise; physics, π0 and actor deterministic).
