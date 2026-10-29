# DECISIONS — SCH-01 fixed_delay10_v2

- V1 **Delay table (authoritative, shared):** d(N) = {3:6, 4:7, 5:7, 6:8, 7:9, 8:9, 9:10, 10:10} control steps
  (`core.DELAY`, imported by every script). Deterministic, no jitter, no margin; keeps the earlier profile's shape with
  N = 10 calibrated to 10 steps. Not a GPU measurement; measured inference wall time is logged separately
  (`wall_s` per request) and never advances the simulator. Modeled compute = d(N)·20 ms per request.
- V2 **Semantics:** REPLAN(R, N) at tick t observes o_t, runs N full-interval denoising steps with RTC, is ready at
  t + d(N), executes R old actions and installs new[R:R+25] at t + R; requires d(N) ≤ R ≤ q and R + 25 ≤ 50; no
  decisions while in flight; KEEP only if q − 1 ≥ 6 and more than one tick remains. RTC: previous full prediction
  shifted by (t − its observation tick), committed prefix R, soft overlap to H − shift. Startup chunk without RTC.
- V3 **Fixed baseline:** N = 10, KEEP while q > 10, REPLAN(10, 10) at q = 10 (p = 15), handoff at p = 25. Recovery rule
  for off-baseline states (q < 10): REPLAN(q, largest N with d(N) ≤ q); counted separately (`recovery_states`); zero in
  fixed-baseline runs (checked).
- V4 **Diagnostic (Stage 1):** snapshots from the new baseline at chunks 2 and 5 (startup = 1), p ∈ {0, 5, 10};
  branches = continue-baseline + feasible REPLAN(R, N), N ∈ {3,5,7,10}, R ∈ {6,7,9,10,15,20,25}; 2 paired replicates
  (noise indexed by call number); only the first decision changes. Source seeds 20,000,100/101 (disjoint from eval
  10,000,000–049 and training < 10,000,000). Snapshots are captured once and saved (rendering is not bit-deterministic,
  so regenerating them on resume could change them).
- V5 **Online (Stage 2):** unchanged learner/design from the legacy run (discrete SAC, zero-initialised actor residual
  and critic heads, prior-aware entropy target, α₀ = 1e-3, p0 = 0.999 held to 5k and removed by 50k, 5k warm-up,
  256 critic-only updates, 1 update per post-warm-up tick, batch 256, γ = 0.999 per tick with γ^D targets, reward
  +1 / −0.1/400 per tick / −0.5/400 per underrun tick, no GPU cost). Fresh weights and EMPTY replay (asserted).
- V6 **Evaluation:** 50 held-out seeds, deterministic masked argmax, frozen, no replay writes, per-episode partial logs
  so an evaluation interrupted by a segment's wall budget resumes where it stopped; the checkpoint's actual prior
  strength; supplemental prior-disabled evaluation at 25k (separate file); fixed baseline re-evaluated under the new
  table (`reference_eval.jsonl`).
- V7 **Execution:** gpu-debug only, internal wall budget 0.78 h per 59-min job; one training writer (run-dir owner
  guard + serial dependencies); ≤5 jobs at any time for this request; the 150k milestone is a copy of the full
  training state, not a restart.
