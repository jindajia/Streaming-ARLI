# DECISIONS — facts, assumptions and deviations

Every R2 result in this repo is a **paper-based ARLI reconstruction**: no author ARLI code exists (checked
2026-10-27: the project page links only the paper; arXiv v2 source has no code link). ARLI is rebuilt on top of
the official DSRL implementation (`nakamotoo/dsrl_pi0` @ `7f48937`, openpi fork @ `a6d2400`), which the paper
says it modified ("For AlohaTransferCube, we do not make additional modifications" to dsrl_pi0 beyond ARLI).

Tags: **[paper]** stated in ARLI v2 · **[code]** taken from pinned dsrl_pi0/openpi/gym_aloha code ·
**[ours]** our assumption where neither specifies it.

## Sources checked

| Source | Version / hash | Notes |
|---|---|---|
| ARLI paper | arXiv 2608.23831v2 (HTML + e-print source, downloaded 2026-10-27) | Fig. "state_repr_aloha_mjc_swimmer_multi.pdf" = the Figure 6 panel (caption: variants of ARLI, no RTC). Text calls it Figure 7 in §5.3 (numbering inconsistency in the paper). |
| ARLI code | none released | project page links only the paper |
| DSRL | `nakamotoo/dsrl_pi0` `7f48937d4553e95244cd81c79236a3256df80597` | git submodule `third_party/dsrl_pi0` |
| openpi (DSRL fork) | `nakamotoo/openpi` `a6d2400d2534ce32e7bdf8747709b97aaef8ec04` | nested submodule; adds `noise=` to `sample_actions` |
| Simulator | `gym-aloha==0.1.1`, `dm-control==1.0.14`, `mujoco==2.3.7` | as in dsrl_pi0 requirements / run_aloha.sh |
| Checkpoint | `s3://openpi-assets/checkpoints/pi0_aloha_sim` | see MANIFEST.md for local path and hashes |

## Facts adopted

- D1 **[paper]** Task AlohaTransferCube, π0 ALOHA sim checkpoint, frozen. t_delay = 20, t_delay^rl = 10, chunk 50,
  execute 25, 10 denoising steps, no RTC (Figure 6 = no-RTC ablation).
- D2 **[paper]** SAC: batch 256, actor LR 1e-4, critic & temperature LR 3e-4, ReLU, γ = 0.999, target entropy −d/2,
  τ = 0.005, 25 gradient steps per update, action magnitude b_w = 2, 10 critics, no clipped Q (→ jaxrl2
  `critic_reduction='mean'`). Encoder: 64×64 `cam_high`, CNN (32,32,32,32), 3×128 MLP.
- D3 **[paper]** Initial collection: 1000 latent-MDP transitions, 0 offline updates, N(0, 0.3), unclipped.
- D4 **[paper]** ARLI state = (s_{t−d}, a_{t−d:t}, s_{t−d_rl}); images channel-concatenated, joint states
  concatenated, committed actions flattened and concatenated before the MLP. Naive DSRL conditions on s_{t−d} only.
- D5 **[code]** DSRL noise action = one 32-dim vector repeated across the 50-step horizon (`action_space (1,32)`),
  π0 padded action dim 32 (ALOHA uses the first 14). Entropy dimension d = 32 → target entropy −16.
  `run_aloha.sh` uses target_entropy 0.0 and 20 grad steps; we follow the ARLI table (−d/2, 25) instead.
- D6 **[code]** Reward: sparse −1 per latent step, 0 and mask 0 on the latent step that ends in success
  (`reward == 4`, gym_aloha terminates on success); per-latent-step discount γ^query_freq (here γ^25, since 25
  actions are executed per chunk). Max 400 env steps. Image = gym_aloha `top` (= π0 `cam_high`), PIL-resized to
  64×64; π0 input = `resize_with_pad(224)`; random crop + colour-jitter augmentation (jaxrl2 PixelSAC defaults).

## Assumptions (not specified by the paper)

- D7 **[ours]** "N(0, 0.3)" read as **std 0.3** (numpy/JAX `normal(0, 0.3)` convention, "scaling down the variance").
  Alternative (variance 0.3, std 0.548) recorded but not run. Prefill noise is a single 32-vector repeated over the
  horizon like the actor's action (DSRL's i==0 collection branch does this).
- D8 **[ours] Asynchronous clock.** See `src/arli/schedule.py` and `tests/test_schedule.py`. Chunk j starts executing
  at T_j = 25j; inference starts at s_j = T_j − 20 on o_{s_j}; the actor runs at m_j = T_j − 10 (ARLI reads o_{m_j});
  index i of chunk j ↔ physical time s_j + i, so the stale prefix 0..19 is discarded and indices 20..44 execute.
  Committed actions of decision j = chunk j−1's actions at times [s_j, T_j) (20 × 14). The 10-step remaining delay is
  inside the 20.
- D9 **[ours] Startup.** The robot is at rest after reset; the episode clock starts when the first chunk is ready:
  s_0 = m_0 = T_0 = 0, chunk 0 executes indices 0..24, and decision 0 sees (o_0, "hold current pose" ×20, o_0).
  Same for every method (including the frozen baseline).
- D10 **[ours] Termination.** If the episode ends while inference j is in flight (T_j ≥ t_end), decision j is
  dropped (its noise never acted). The chunk executing at success gets reward 0 / mask 0.
- D11 **[ours] Budget axis.** The paper's x-axis is "Env Steps" (total original-environment steps). DSRL's loop
  counts gradient steps. We index budget and evaluations by **online** env steps (prefill excluded), evaluate at 0
  and every 25k, and also log/plot prefill and total (prefill + online) env steps. 25 gradient steps per latent
  transition ≈ 1 gradient step per online env step.
- D12 **[ours] Evaluation.** 50 episodes on fixed held-out box-pose seeds 10,000,000–10,000,049 (training seeds are
  drawn from [0, 10^7)); stochastic actor sample (as dsrl_pi0 `perform_control_eval`) with a fixed eval key that
  does not consume the training RNG. Step-0 evaluation uses the randomly initialised actor (the method's actual
  policy at that point); the frozen π0 baseline is measured separately with i.i.d. N(0, I) (50×32) noise under the
  identical async protocol. A synchronous zero-delay frozen evaluation is a separate diagnostic only.
- D12b Unmodified simulator. A speed-only patch skipping gym_aloha's two unused camera renders was tested and
  rejected: the `top` image was not bit-identical in `scripts/check_r1.py` (2.6 → 2.1 ms/step only).
- D12c Sampler numerics. Our step-wise sampler (separately jitted VLM prefix + 10 expert Euler steps) differs from
  openpi's single `while_loop` sampler by max |Δa| = 0.003–0.005 rad for identical inputs/noise (bf16/XLA fusion;
  actions ≈ ±1.2). R2 uses openpi's `infer` (DSRL path). S1 runs **all** branches, A included, through the step-wise
  sampler; a zero residual through the steering hook is bit-identical to it.

## Steering extension (S0/S1) — our design, not ARLI

- D13 Augmented-state SAC. Per chunk: s_j —w (ARLI π_w)→ z_{j,k1} —u→ … z_{j,kL} —u→ s_{j+1}. Reward −1 (or 0 with
  mask 0 at success) only on the last steering transition of each executed chunk; internal transitions r = 0,
  mask 1; discount γ^(elapsed env steps) internally and the remainder on the last transition so every chunk
  discounts by γ^25 total, as in the baseline. No physical reward is duplicated.
- D14 Residual: X̃_k = X_k + τ_k · broadcast_50(u_k), u_k ∈ [−1, 1]^32 (tanh-Gaussian). α(τ) = τ expresses the
  correction in noise units (X_τ = τ ε + (1 − τ) a). Edit mask M = all positions: the new chunk contains no
  committed action (committed actions belong to chunk j−1 and are never modified).
- D15 Clock: expert step k of inference j runs at physical time m_j + k (10 steps share the 10-step remaining
  delay); the actor at step k reads the newest frame o_{m_j+k} (C/D) or o_{m_j} (B). Actor/encoder overhead is
  **idealised** (assumed to fit inside the same per-step slot) → S1 is an information-timing experiment, not a
  latency claim; measured actor vs expert wall-clock is logged. Startup chunk 0: clock frozen, all steps see o_0.
- D16 Warm start: π_w, Q_w, target, temperature and optimizer states from the common R2 ARLI checkpoint for all
  branches. π_u / Q_u copy ARLI's CNN encoder, bottleneck and MLP hidden layers; first-layer rows for the new
  inputs (X features, τ/δ embedding) and Q_u's action rows start at zero; π_u's output heads are fresh
  (log-std bias log 0.3); temperature_u starts at ARLI's temperature. Q_u is *not* assumed to know the value of
  the new decisions.
- D17 Replay reset for all four branches (B–D tuples are incompatible with R2's chunk tuples; A is reset too for
  identical treatment); updates start after 100 executed chunks inside the 100k budget; 25 gradient iterations
  per executed chunk; in B–D each iteration updates both modules on separate 256-batches (disclosed as a compute
  difference).

## Observed divergences from the paper (reported, not tuned away)

- V1 Frozen π0 under the async d=20 no-RTC protocol: 24/50 = 0.48 (95% Wilson 0.35–0.61) vs the paper's dashed
  line ≈ 0.37 (read off the figure). Within the interval; startup convention (D9) and eval seeds (D12) may differ.
- V2 ARLI seed 0 reaches 26/50 at 50k online env steps (75k total incl. prefill); the paper's mean ARLI curve is
  ≈ 0.05 there and ≈ 0.5 only near 400k–600k. Candidate causes, none tested yet: prefill-noise convention (D7),
  budget axis (D11), startup/termination conventions (D9/D10), one seed vs a 3-seed mean. Information leakage was
  checked: the actor reads only o_{s_j}, o_{m_j} (m_j = T_j − 10) and chunk j−1's committed actions; eval box-pose
  seeds are disjoint from training seeds.
- V3 Early instability: ARLI exploration successes appeared at 7k–20k online steps, then policy entropy collapsed
  (temperature 0.86 → 0.004) and the 25k evaluation was 0/50 (no cube contact) before recovering at 50k.
