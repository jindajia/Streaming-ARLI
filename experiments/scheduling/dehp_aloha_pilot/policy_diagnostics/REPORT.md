# DEHP ALOHA pilot — policy diagnostics and matched evaluation (2026-10-30)

Evaluation only (no gradient updates) of the checkpoints from the one-hour DEHP-style pilot (`../REPORT.md`, commit
867ad3b). One GPU job: **8780176**, `gpu-debug`, 1 GPU, `--time=01:00:00`, ran 22 min 43 s (15:11–15:34 EDT), exit 0.
All planned work completed; nothing skipped. No second job was needed.

## Answers

1. **Does the actor condition meaningfully on state? No — not detectably.** On a shared bank of 43 decision states per
   run, the final policies are almost uniform (normalised entropy 0.994 in A, 0.987 in B; largest single-h probability
   0.037 vs uniform 0.020 in A, 0.066 vs 0.032 in B) and almost identical across states. Of the total deviation from
   uniform, mean_s KL(p_s‖U) = KL(p̄‖U) + mean_s KL(p_s‖p̄), the state-dependent part is **1.8 % (A) and 2.6 % (B)**; the
   rest is one global, state-independent preference. Expected h varies across states by std 0.04 (A, range 25.83–25.95)
   and 0.11 (B, range 25.21–25.62). The argmax is the same horizon in every bank state (A: h = 29 in 43/43; B: h = 36 in
   43/43) and in every evaluated decision (A: 356/356 decisions, B: 292/292).
2. **Does learned stochastic sampling beat uniform sampling? No.** Paired on 20 new seeds: A — S 10/20 vs U 10/20 (both
   succeed 9, S-only 1, U-only 1, neither 9; difference 0.00, 95 % Newcombe CI −0.16 to +0.16). B — S 9/20 vs U 10/20
   (8 / 1 / 2 / 9; −0.05, CI −0.23 to +0.13). Original 10 seeds (reported separately): A 6/10 vs 5/10, B 7/10 vs 8/10.
3. **Is a global horizon preference sufficient? Yes — there is little else to explain.** The state-independent control M
   (stochastic, the frozen average vector p̄ from the bank) matches S: B — identical outcomes on all 20 paired seeds
   (9 both, 11 neither) and on all 10 original seeds; A — S 10/20 vs M 8/20 (2 S-only, 0 M-only; CI −0.06 to +0.25) and
   6/10 vs 6/10. Even the global preference is weak (TV distance of p̄ from uniform 0.09 in A, 0.12 in B), and M is not
   better than U (A 8 vs 10, B 9 vs 10 on the new seeds).
4. **What remains uncertain.** 20 paired episodes resolve only large effects (CIs ±0.15–0.25). The argmax policies D are
   exactly fixed horizons (h = 29 / 36); D vs the fixed-25 baseline is A 11/20 vs 12/20 and B 12/20 vs 10/20 (CI −0.14 to
   +0.33), so whether a longer fixed horizon helps in B is unresolved — and that would be a fixed-horizon effect, not
   learned scheduling. The bank contains no "transfer" phase states (stage 3 never coincided with a decision time in the
   16 source episodes) — marked missing, not replaced. Rollouts are not bit-reproducible (renderer ±1-intensity noise).
5. **Single most useful next experiment:** a paired **fixed-horizon sweep** in each setting (e.g. A: h ∈ {5, 10, 15, 20,
   25, 29, 35, 40, 50}; B: h ∈ {10, 15, 20, 25, 30, 36, 40}) on ≥ 50 shared seeds with the same noise coupling. It measures
   how much task success depends on h at all — the headroom any horizon scheduler can exploit — before deciding how to
   train one. (No training was started.)

Interpretation guideline matched: **C** (distributions close to uniform, S indistinguishable from U), with the
**E**-check applied: D's apparent edge in B (12/20 vs 10/20) is a fixed horizon h = 36 chosen by argmax, not state-dependent
behaviour.

## Setup verified

- Branch `experiment/dehp-aloha-pilot`; diagnostics committed on top of 867ad3b; executors (`src/dehp/smdp.py`) used
  unchanged (a read-only subclass records the per-step env reward stage for phase labels).
- Frozen π0 `pi0_aloha_sim` (`/N/scratch/jindjia/arli/checkpoints/pi0_aloha_sim`), H = 50, 10 Euler steps, RTC off (A) /
  on with committed prefix 10 (B), reward +1 on success, 400-step deadline.
- Checkpoints (sha256 prefix): A `learner_it0016` a5f127b6…, `learner_final` 08a882c3…; B `learner_it0013` 441195d3…,
  `learner_final` 9e691094… (all under `/N/scratch/jindjia/arli/runs_dehp/run_*/ckpt/`).
- **RNG finding:** `Learner.act` splits `self.key` on every call, including `deterministic=True`, so the pilot's
  evaluations advanced the checkpoint's sampling key. These diagnostics never call `act`: p(h|s) comes from a key-free
  jitted forward pass (masked softmax), and horizons are sampled explicitly.
- **RNG streams / coupling:** environment initial state = env seed (shared by all arms); π0 initial noise =
  `default_rng([2026, episode_index])`, the i-th inference call of every arm draws the i-th block (matched by call
  index — once schedules differ, the same block is used at different physical times); horizon sampling =
  `default_rng([4049, episode_index])`, decision k uses u_k with inverse-CDF sampling over valid h in ascending order
  (common random numbers for U, S, M). Stochastic arms really sample the masked categorical (e.g. A-S used 49 distinct h,
  B-S all 31); D takes the argmax. Recorded per decision: t, h, u, p(selected), policy entropy, phase.
- **Seed lists:** state bank sources 10,000,200–10,000,207 (8 per run; A 6 successes / 2 failures, B 3 / 5); primary
  evaluation 10,000,100–10,000,119 ("new20"); original pilot seeds 10,000,000–10,000,009 ("orig10"). All disjoint from
  training seeds (< 10^7).
- **State bank:** decisions 0, 3, 6, … of fixed-h=25 trajectories (time-coverage rule, not probability-based), exact
  actor inputs (physical state, aligned executable plan, masks, metadata incl. observation age). 43 states per run;
  phases by simulator contact stage at the decision time — A: approach 20, grasp 5, lift/transport 11, lost-contact 7;
  B: approach 18, grasp 6, lift/transport 4, lost-contact 15; transfer: missing in both.

## Matched evaluation (20 paired new seeds; counts)

| Run | U uniform | S trained stochastic | D trained argmax (≡ fixed h) | F25 fixed 25 | M global average |
|---|---|---|---|---|---|
| A (no delay, h 1–50) | 10/20 | 10/20 | 11/20 (h = 29 always) | 12/20 | 8/20 |
| B (delay 10, h 10–40) | 10/20 | 9/20 | 12/20 (h = 36 always) | 10/20 | 9/20 |
| A, original 10 seeds | 5/10 | 6/10 | 7/10 | 5/10 | 6/10 |
| B, original 10 seeds | 8/10 | 7/10 | 7/10 | 5/10 | 7/10 |

Mean inference calls per episode (new20): A — U 14.8, S 14.8, D 12.1, F25 14.0, M 14.9; B — U 14.9, S 14.9, D 10.1,
F25 14.8, M 14.9. Mean episode end (incl. 400 for failures): A 340–358, B 334–348. Among jointly successful pairs only
(conditioning stated): A D vs F25 282 vs 307 steps (n = 10), B D vs F25 278 vs 293 (n = 8). Timing violations and
underruns: 0 in every episode. Full per-pair tables (both / first-only / second-only / neither, Newcombe 95 % CI) for
S–U, S–M, S–F25, D–F25, M–U, D–S: `results/summary.json`.

## Files

- `diagnose.py` (job script; `--dry_run` CPU check with a fake π0; `--eval_only`), `analyze.py` (tables, statistics,
  figures). Raw outputs on scratch: `/N/scratch/jindjia/arli/runs_dehp/policy_diagnostics/` (state bank and trajectory
  distributions JSONL, `pbar_*.json` frozen before evaluation, `episodes.jsonl`, `log.jsonl`); job log
  `/N/scratch/jindjia/arli/runs_dehp/logs/dehp_diag_8780176.out`.
- `results/state_bank_{A,B}_final.csv` — per state: full p over valid h, argmax and its probability, E[h], sd, entropy,
  normalised entropy, KL to uniform and to p̄, seed, time, phase, remaining horizon, observation age.
- `results/episodes.csv` — per episode: success, end time, discounted return, inference calls, selected h and times,
  mean selected-action probability, violations, underruns.
- Figures: `fig_A_heatmap_{A,B}.png` (p(h|state) heatmaps, intermediate vs final checkpoint on the same states — figure A
  and D; these are probability distributions, not the earlier sampled-h training histograms), `fig_B_state_curves_*.png`,
  `fig_C_trajectories_*.png`, `fig_E_evaluation.png` (success with Wilson intervals and selected-h distributions per arm).
