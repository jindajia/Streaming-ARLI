# DECISIONS — SCH-01

Tags: **[plan]** from the research-notes plan; **[ours]** implementation choice where the plan leaves it open.

- S1 **[plan] Setup.** AlohaTransferCube, frozen `pi0_aloha_sim`, 50 Hz (20 ms/tick), H = 50, installed cap K = 25,
  RTC ON for every arm, one worker, at most one request in flight, actions KEEP / REPLAN(R, N), R = 1..25, N = 3..10
  (201 actions before masking). No ARLI/steering actor; no VLA training.
- S2 **[plan] Timing = proxy-derived logical-time model**, not measured native ALOHA RTC latency:
  L(N) = 42.5 + 8.17 N ms; jitter = clip(N(0, 5), −10, 10) ms (synthetic); delay = ceil((L + jitter)/20);
  budget(N) = ceil((L + 10)/20) → N 3..10 budgets 4,5,5,6,6,6,7,7 ticks. The scheduler never sees the jitter draw.
  Because feasibility requires R ≥ budget(N) ≥ delay, a result is always ready before its handoff and is held until the
  handoff; jitter therefore changes the logged readiness time and inference cost but not the executed actions.
- S3 **[plan/ours] Executor.** Startup chunk: synchronous with the clock frozen, N = 10, no RTC (nothing to align to),
  installs new[0:25] **[ours]**. REPLAN(R, N) at tick t uses o_t, keeps the old queue executing, hides the result until
  t + R, installs new[R:R+25]. RTC reference = previous full normalised prediction shifted by
  (t − previous observation origin); committed full-weight prefix = R (also when ready early and held); soft overlap
  until H − shift; official exp schedule, max guidance 10. Each N integrates the whole flow interval (dt = −1/N).
- S4 **[ours] Feasibility.** REPLAN(R, N) iff budget(N) ≤ R ≤ q and R + 25 ≤ 50; KEEP iff q − 1 ≥ 4 (another tick must
  still leave a feasible REPLAN) and > 1 tick remains. Irrevocable prefix c = 1 is implied by R ≥ 4. Underrun is
  unreachable under these masks; an emergency path (hold last command, REPLAN(4, 3)) exists and is logged.
- S5 **[plan] Reward per tick:** 1·[first success] − 0.1/400 − 0.5/400·underrun; GPU-cost coefficient 0. Success and
  timeout are terminal (finite horizon, remaining time is an input). γ = 0.999 per tick **[ours]**; decision targets
  use γ^D with the actual elapsed ticks D (KEEP: 1; REPLAN: R unless the episode ends first).
- S6 **[plan] Reference rule:** q > budget(10) = 7 → KEEP; q = 7 → REPLAN(7, 10); q < 7 → REPLAN(q, largest N whose budget
  fits). Under this rule each segment executes 25 actions (KEEP for 18, then REPLAN(7, 10)).
- S7 **[plan] Initial policy:** logits = zero-initialised residual head + reference prior (reference logit
  log(p0 (M−1)/(1−p0)), p0 = 0.999, others 0, infeasible −∞), strength 1 until 5k ticks, linear to 0 at 50k.
  Deterministic step-0 argmax equals the reference rule exactly (checked at every decision).
- S8 **[plan/ours] Learner:** discrete SAC, 2 critics (+ targets, τ = 0.005), Adam lrs actor 1e-4 / critics 3e-4 /
  temperature 3e-4 **[ours: ARLI-like]**, batch 256, α₀ = 0.01, target entropy 0.1·log M(s) **[ours]**, empty replay,
  5k warm-up without updates, 256 critic-only updates, then 1 update per new post-warm-up tick. Paired random-shift
  augmentation of current/source images. Network: shared 4-layer CNN (now, source, difference) + queue MLP + runtime
  MLP → 256-256 → 201.
- S9 **[plan] Evaluation:** 50 held-out seeds 10,000,000–10,000,049, deterministic masked argmax, frozen parameters,
  checkpoint's prior strength, no replay writes; request noise/jitter streams seeded per episode
  (default_rng([seed, 1]) / ([seed, 2])), identical for every method. Matched fixed reference evaluated once per run.
- S10 **[ours] Diagnostic:** source episodes use seeds 20,000,000+ (disjoint from evaluation); snapshots at reference
  handoff ticks (q = 25, idle); one forced first decision, then the reference scheduler; paired noise/jitter per
  (snapshot, replicate). Never written to the learner's replay or read by training.
- S11 Renderer: EGL images differ by ±1 intensity for identical states (Streaming-ARLI DECISIONS D31); closed-loop runs
  are not bit-reproducible; equivalence checks compare decisions/physics, not images.
- S12 **[ours] Entropy target tracks the prior (change after smoke job 8763771).** With target entropy 0.1·log M and
  α₀ = 0.01, the entropy term erased the reference prior within ~150 actor updates while the prior was still at full
  strength (reference probability 0.98 → ~0, deterministic parity 0, evaluation switched to mean N = 5): under the
  prior, non-reference log π ≈ −12, so α·log π ≈ −0.12 dominates critic action gaps of ~1e-3, and the target (≈0.5
  nats) was far above the prior's own entropy (≈0.013), so α grew. Now the per-state target is
  min(H(prior-only policy at the current strength), 0.1·log M) (≈0.008–0.013 nats at strength 1, reaching the cap as
  the prior anneals) and α₀ = 1e-3. Fixed choices, not tuned on evaluation outcomes.
- S13 Numerics: computing the velocity inside the RTC VJP changes XLA fusion, so RTC with all-zero weights differs from
  the plain step by ≤0.011 (normalised units) for identical inputs (gpu_checks, job 8763771). All scheduled requests
  in every arm use the RTC path; only the synchronous startup chunk uses the plain step.
