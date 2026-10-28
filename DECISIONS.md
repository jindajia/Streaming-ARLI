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

## S1 credit-assignment correction (2026-10-28)

- D18 **v1 "chain" credit was invalid.** In S1 seed 0, branches B and C (9 steering calls per chunk) trained a
  sequential chain s → z_1 → … → z_9 → s' with γ≈1 per hop and actor maximisation at every hop. Both critics
  diverged upward (Q_w, Q_u ≈ +28 by 90k although every reward is −1 or 0, so Q ≤ 0 must hold), and success
  collapsed to ~0/50. D (2 hops) stayed bounded (Q ≈ −4.5); A and R2 (1 hop) never exceeded Q = −2.8. The B/C
  collapse is therefore an implementation failure, **not** a negative result about steering. Runs
  `s1_{B,C,D}_seed0` (v1) are kept on scratch for provenance and excluded from comparisons.
- v2 **"chunk" credit** (default, `credit: chunk`): every decision of chunk j — the initial noise w_j and each
  correction u_{j,k} — is trained toward the chunk-level target r_j + γ^{dt}·mask_j·Q_w(s_{j+1}, π_w(s_{j+1})),
  with dt = physical env steps from that decision to the next initial-noise decision (γ^25 for w_j, γ^{25−k} for
  u_{j,k} when j ≥ 1). One bootstrap hop per chunk as in ARLI; the physical reward is still counted once per chunk
  and shared (not duplicated) by that chunk's decisions; Q_u never bootstraps from itself. The W-transitions are
  exactly ARLI's. The intra-chunk steering actions inside the target come from the behaviour policy (n-step-style
  off-policy bias; accepted, as in standard n-step SAC/DQN). Tested in `tests/test_credit.py`. B, C, D are rerun
  as `s1_{B,C,D}_seed0_v2` from the same checkpoint; A is unchanged (no steering, unaffected); seed-1 S1 uses v2.

## Single-actor comparison (2026-10-28) — replaces the superseded two-actor warm-start S1

Two peer methods; neither is the "main" method. Code: `src/arli/single_actor.py`, `src/arli/train_single.py`;
configs `configs/single/`; runs `/N/scratch/jindjia/arli/runs_single/sa_<method>_seed<s>/`; results
`results/single_actor_comparison/`.

- D19 **Methods.** `shared_every_step`: one actor (one parameter set, one tanh-Gaussian output head) called before
  every denoising iteration k = 0…9 (10 calls/chunk); there is no separate initial-noise actor or head.
  `final_step_only`: one actor called only before iteration k = 9 (1 call/chunk); initial noise is π0's fixed
  convention X_0 ~ N(0, I), shape (50, 32) i.i.d.; iterations 0…8 unsteered. Same architecture for both. 10 iterations
  for both methods and the ARLI reference.
- D20 **Action semantics (fixed implementation choices, not validated).** Actor output a ∈ [−1, 1]^32.
  k = 0 (shared_every_step only): initial noise X_0 = broadcast_50(2·a) (2 = DSRL/ARLI b_w). k ≥ 1 (both methods):
  X~_k = X_k + 0.2·broadcast_50(a), constant scale 0.2 at every k (identical between methods at corresponding k).
  Rationale for 0.2 rather than the superseded α(τ)=τ with bound 1: in the superseded v2 runs, τ-scaled ±1 residuals
  at every step drove success from 19–22/50 to 0/50 within 10k steps while their critics stayed bounded (Q ≈ −7).
  Whether 0.2 is a good value is untested.
- D21 **Actor input** (both methods, at the moment of the call): cam_high 64×64 of the inference-start observation
  o_{s_j} ++ newest frame o_t (6 channels), joint states q(o_{s_j}) ++ q(o_t) (28), committed actions of [s_j, T_j)
  (20×14), current trajectory X_k (50×32, first 14 dims encoded by a 700→64 linear+LN+tanh), and
  td = (τ_k, δ = (T_j − t)/20, f) with sinusoidal features. k = 0 call: X placeholder = zeros, τ = 1, f = 1
  (first-call indicator); k ≥ 1: f = 0. Call time t = m_j + k (one expert iteration per env step inside the
  10-step remaining delay), so the call sees o_{m_j+k}; the startup chunk j = 0 is computed with the clock frozen
  (all calls see o_0). The VLM context is the inference-start observation and is never recomputed.
  Parameters: actor 1,176,214; critic ensemble (10 Q) 2,057,696 (ARLI R2: actor 1,116,886, critics 1,869,344; the
  difference is the X encoder and time features). Actor overhead is idealised into the per-iteration slot
  (information-timing experiment, not a latency claim).
- D22 **Objective.** SAC over actor calls with a chunk-level bootstrap: for call d = (j, k) of executed chunk j,
  target = r_j + γ^{dt}·mask_j·mean_i Q̄_i(z', a'), a' ~ π(z'), z' = input of the first call of chunk j+1
  (k = 0 for shared_every_step, k = 9 for final_step_only), dt = env steps from the call to z' (chunk total γ^25 as in
  ARLI). r_j = −1 per executed chunk, 0 and mask 0 for the chunk executing at success; each reward counted once per
  chunk and shared by its calls. final_step_only: exactly ARLI's chunk-level target. shared_every_step:
  n-step-style — the later corrections of the same chunk that sit between d and z' are the behaviour policy's (off-
  policy bias), and each call is improved against Q given those later corrections (no explicit intra-chunk
  sequential credit). No Q→Q hop inside a chunk (the sequential "chain" target diverged earlier, D18). Entropy target
  −16, no entropy term in the backup, 10-critic mean (no clipped Q), as in R2. **Limitations:** the intra-chunk
  off-policy bias is uncorrected; all calls share one critic with call-position inputs (τ, f); no validation yet
  that this objective optimises the 10-call composite policy well — only that it is mechanically correct
  (`tests/test_single_actor.py`).
- D23 **Training budget.** Fresh actor/critic/temperature/optimizers/replay (no R2/S1 weights). Prefill: episodes
  until > 1000 executed chunks; every actor output is drawn from N(0, (0.3/2)²) so the k = 0 initial noise is
  N(0, 0.3²) exactly like ARLI's prefill; final_step_only keeps its fixed N(0, I) initial noise in prefill and
  online (only its k = 9 correction is randomised in prefill). Online: 600k env steps, **25 gradient updates per
  executed chunk for both methods** (= ARLI; not ×10 for shared_every_step's 10 calls); batches of 256 actor-call
  transitions sampled uniformly. Logged: prefill/online/total env steps, actor calls (prefill/online/eval), gradient
  steps, wall time per phase (GPU-job time).
- D24 **Reporting.** Training jobs call `scripts/report_single.py` at each 100k milestone and at completion: dedicated
  clone `/N/scratch/jindjia/arli/report_repo`, atomic lock dir `/N/scratch/jindjia/arli/report_single.lock`,
  commits only `results/single_actor_comparison/` to main (rebase/retry). Reporting failures never stop training.

## Warm-start final-step experiment (2026-10-28) — separate from the from-scratch comparisons

Code `src/arli/warmstart_final.py`, `src/arli/train_warmstart.py`; configs `configs/warmstart/ws_{A,B}_seed{0,1}.yaml`;
runs `/N/scratch/jindjia/arli/runs_warmstart/ws_<branch>_seed<s>/`; figure `scripts/plot_warmstart.py` →
`results/warmstart_final/`; worktree `streaming_arli_ws` (branch `warmstart-final`).

- D25 **Common start.** Seed 0: `r2_arli_seed0/ckpt/agent_0600124.msgpack` (600k eval milestone, 600,124 online steps,
  eval 45/50, sha256 4e52b142…); seed 1: `r2_arli_seed1_std030/ckpt/agent_0600139.msgpack` (600,139 steps, 42/50,
  sha256 2fee1477…). Resolved from `evals.jsonl` at runtime; both branches of a seed read the same bytes; source run,
  step and sha256 written to `provenance.json` and re-verified at every resume.
- D26 **Branches.** A: restore the ARLI actor, critic, target critic, temperature and their Adam states; keep training
  them. B: ARLI actor frozen (never passed to an optimizer; its parameter hash is checked at every checkpoint and eval);
  it still samples the initial noise exactly as in ARLI. A separate residual actor (from-scratch `final_step_only`
  architecture, 1,176,214 params) + its own 10-critic ensemble (2,057,696) + temperature + Adam states; called once per
  chunk before iteration k = 9 with o_{m_j+9}, q, X_9, committed actions, (τ_9, δ, f); X~_9 = X_9 + 0.2·broadcast_50(a).
  No parameters, encoders or optimizers shared between the two actors. The old ARLI critic is not used.
- D27 **Initialisation of the residual actor.** Mean head kernel and bias exactly 0 (mean correction exactly 0); log-std
  head kernel 0, bias log 0.1 → sampled a = tanh(N(0, 0.1²)): non-zero, |0.2·a| ≈ 0.016 per dimension at start.
- D28 **Sampler.** Both branches use the step-wise sampler (prefix once on o_{s_j}, 10 frozen iterations). A therefore
  differs numerically (≤0.005 rad, D12c) from R2's while-loop sampler; a zero-correction B is bit-identical to A
  (CPU fake-sampler test and real-stack GPU check).
- D29 **Objective / budget.** B: SAC on one decision per executed chunk: target r_j + γ^25·mask_j·mean Q̄(z_{j+1,9}, a'),
  a' ~ π_res(z_{j+1,9}); γ^25 = the physical interval between consecutive k = 9 calls (25 env steps in steady state;
  startup chunk keeps ARLI's per-chunk convention). A: ARLI's chunk-level transitions (γ^25). Both: fresh replay;
  first 256 executed chunks collected without updates (inside the +300k budget); then 25 updates per executed chunk,
  batch 256, one learner per branch (B does not get twice the updates). +300k online env steps, eval at +0 and every
  +25k on the same 50 held-out seeds.
- D30 **Bug found in the from-scratch `final_step_only` runs (sa_final_step_only_seed{0,1}).** Their k = 9 decisions
  bootstrap from the next chunk's k = 9 call (25 steps later) but are stored with discount γ^16 (`single_actor.transitions`
  computes the interval to the next chunk's k = 0 point, correct only for shared_every_step). 943 of 1,016 stored
  transitions in an early replay part use γ^16. Effective per-chunk discount 0.984 instead of 0.975. Those runs were
  left untouched (still running); the warm-start code uses γ^25 and tests the interval against physical call times.
- D31 **Renderer non-determinism (affects every experiment's reproducibility).** `scripts/diag_determinism.py`
  (job 8762925): MuJoCo physics is deterministic (identical qpos for identical actions, within and across env
  instances) and π0 / the ARLI actor are deterministic for identical inputs, but the EGL-rendered `top` image differs
  by ±1 intensity between renders of the same state (even in one instance). Closed-loop episodes are therefore not
  bit-reproducible: two runs of the same policy, keys and seed diverge from t = 0. Consequences: evaluation numbers
  carry this extra randomness; equivalence checks must hold observations fixed. Check 2 (zero-correction B = A)
  therefore replays A's recorded frames to B whenever the physical state matches exactly.
