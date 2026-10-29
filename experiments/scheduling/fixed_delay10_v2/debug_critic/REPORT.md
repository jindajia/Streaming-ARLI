# SCH-01 fixed_delay10_v2 / rn_seed0_d10 — time-bounded critic debugging (2026-10-29)

One gpu-debug job (8766798, A100-40GB, `--time=00:20:00`): **5 min 58 s wall** including about 94 s of
initialization. All phases ran; nothing was skipped.

- Critic updates: **600** (300 per ablation branch).
- Rollouts: **21**, made up of 1 source rollout, 18 paired branch rollouts and 2 identical-action replays.
- Nothing in the run directory was written. The replay was read with `pickle`; `Replay.load` deletes part files, so it
  was never called.
- Outputs are in `/N/scratch/jindjia/arli/runs_scheduling/fixed_delay10_v2/debug_critic/job_8766798/`, with compact
  copies in `results/`.

## 1. What we confirmed

- **The logged symptoms are real** (`updates.jsonl`; each row is the mean of the last 500 updates).
  - `q_taken` first exceeds 1 at 242,374 steps.
  - The final row at 300,155 steps shows `q_taken` = 128.2, `critic_loss` = 3.84e4 and `q_max` = 1.69e4.
- **No bookkeeping bug.** All 59,907 replay transitions passed every audit check (`results/audit.json`):
  - KEEP has D = 1; non-terminal REPLAN has D = R.
  - G equals the reward model to within 2e-6 (no underruns).
  - There are 942 terminal flags, one per episode; timeouts are terminal with 0 steps left.
  - Each next state is the following decision's state.
  - Remaining time after a transition equals remaining time before it minus D.
  - Masks match the delay table and queue; every action was feasible.
  - All values are finite, and no input feature exceeds 8 standard deviations.
- **The target formula is confirmed** (`learner.py:175-176`): `y = G + γ^D (1 − term) [E_π min Q̄ + α H(π)]`. My
  explicit decomposition reproduces the learner's expression to within float tolerance.
  - The alpha update sign is correct: log α decreases when entropy exceeds its target (`learner.py:207`).
  - Inference count and compute cost are not in the reward.
- **The official dsrl_pi0 differs in three ways.**
  - It calls `update_critic(...)` without `backup_entropy`, whose default is `False`, so there is no entropy in its
    critic target.
  - It uses −1/0 rewards per chunk, discounted by `0.999**query_freq` per decision.
  - It sets `masks = 0` only at success, so timeouts are not terminal.
- **The growth comes from the bootstrapped future-Q term, concentrated in a tail.** At 300k, over all replay
  transitions:

  | quantity | median | p99 | max |
  |---|---|---|---|
  | min(Q1, Q2) | 0.62 | 1,234 | 42,640 |
  | γ^D·E[min Q̄] (future term) | 0.78 | 1,491 | 44,269 |
  | γ^D·αH (entropy term) | 0.016 | 0.031 | 0.050 |

  - The top 1% of transitions carry 97% of the squared TD error.
  - 21% of transitions have |Q| > 3. They are all non-terminal, and 41% of transitions with 300 or more steps
    remaining fall in that group, against 0.7% with fewer than 50 remaining.
  - Within the outliers, corr(Q, future term) = 0.994: Q chases its own bootstrapped target.
- **Errors run in both directions, and representations collapsed.**
  - 7.7% of Q values exceed a soft-value upper bound computed from the actual reward, remaining horizon, α and up to
    144 feasible actions. 8.9% fall below the lowest achievable task return.
  - 36.6% of all transitions receive the identical Q of 0.620; at 100k and 225k no single value covers more than 5.4%.
  - The 300k critic no longer fits terminal transitions whose target is exactly G: timeouts get a median Q of 0.69
    against a target of about 0.
- **The onset is visible at 225k.** The first |Q| > 3 cases (7 of them) are consecutive KEEP transitions (D = 1,
  reward −0.00025) in a single episode that timed out (episode 316, replay rows 28,430–28,455).
  - Along that chain Q rises 1.3 → 3.1 and tracks its own targets.
  - The episode's true discounted return-to-go is about −0.04.
  - The 100k critic gave these states 0.27–0.38.
- **Entropy in the target is not a material direct contributor.**
  - Its contribution is at most 0.005 at 100k and 225k (α ≈ 0.002), and at most 0.05 at 300k (α = 0.031, a 15× rise
    after the onset).
  - In the 300-update ablation from 100k (`results/ablation.json`), branches A and B (with and without the future
    entropy term) end with median Q 0.6522 and 0.6507 and gradient norms 0.018 in both. Neither diverged.
- **Choices were weakly distinguishable in the pilot.** At both snapshots (100k actor, held-out seed 10,000,000), all
  18 branches succeeded.
  - Early movement (t = 40): every candidate finished in 229–236 steps with discounted task return 0.738–0.745.
  - Grasp/contact (t = 130): KEEP took 146–149 steps (return 0.828–0.831). REPLAN(10,10) took 141–143 and
    REPLAN(15,5) took 141–142 (returns 0.834–0.836).
  - Restoration is exact (physics difference 0). Renderer differences are ≤ 1 intensity on 1 pixel after one step.
    Even so, the identical-action replay of the early KEEP branch diverged at request 11 and ended at 270 instead of
    272.

## 2. What remains uncertain

- Whether the grasp-snapshot difference (about 5 steps faster for REPLAN, with return Δ ≈ 0.005, consistent across 3
  paired seeds) is real. It is the same size as the identical-action replay's nondeterminism (2 steps). This pilot
  found no detectable difference in success.
- Whether removing the backed-up entropy, or treating timeouts as non-terminal, changes long-run stability. 300 updates
  cannot show long-run stability, and the divergence appeared after about 240k updates.
- What exactly triggers the onset near 225–240k, and why α rose 15× afterwards. α is probably a consequence: entropy
  fell below the target.

## 3. Leading explanation for the numerical instability (hypothesis)

Bootstrapping amplification with almost no contraction and a very high replay ratio.

- Most transitions are KEEP with D = 1, so each target is 0.999 × the next state's value. Errors shrink only 0.1% per
  backup, over chains of up to about 400 decisions.
- The replay is small: 59,907 transitions in total, 28,506 at 100k. It was trained with 295,155 updates of batch 256,
  so each transition was replayed about 1,260 times on average.
- The divergence starts on long KEEP chains far from termination and propagates through Q ≈ y.
- Once large errors exist, they dominate the squared loss. The shared network then loses state discrimination (one
  constant for 37% of transitions) and cannot fit even terminal targets.

A related design property (a confirmed mechanism; its effect on stability is uncertain):

- The critic heads are zero-initialized and trained only at the taken action (`learner.py:56-57, 65, 182`).
- So every never-taken action keeps Q exactly 0 in both critics. In the whole run only 34 of 201 actions were ever
  taken; 167 head columns are exactly zero and match the never-taken set exactly.
- A typical state has 116 feasible actions, only about 32 of which were ever tried.
- Because realistic values are about 0.6–0.8, untried actions look 0.7 worse. That is not overestimation, but it
  pins the policy to a few actions and keeps the data narrow.

## 4. Leading explanation for the lack of task improvement (hypothesis)

The scheduling choices that were explored have nearly identical task outcomes in unperturbed Transfer Cube.

- In the pilot, every candidate succeeded with returns within 0.01.
- Earlier viewer probes found that, up to 225k, Q margins between chosen and baseline actions were below the critics'
  own disagreement.

With differences this small, the learned preferences mostly reflect estimation noise, and later the diverged critic.
The 50-episode evaluations (31–39/50 vs 36/50) cannot resolve differences of this size.

## 5. Single most useful next experiment

A short controlled critic-stability comparison: re-train only the critic, fresh, on the frozen replay rows up to 100k,
with the frozen 100k actor, for about 30k updates.

- Arms:
  - (a) the current target;
  - (b) decision-level discounting with a larger effective contraction, e.g. γ per decision instead of γ^D with D = 1,
    or temporal aggregation of KEEP runs;
  - (c) timeouts treated as non-terminal, as in dsrl.
- Track the full-replay Q distribution and terminal-transition fit, using `critic_debug.py`'s pass.

This tests the bootstrapping hypothesis directly without rerunning training.

---

## Details

**Checkpoints.** All three are actual saved files; the prior strength is 0 for all.

| step | file | sha256 prefix | α | replay rows at that time |
|---|---|---|---|---|
| 100,247 | `ckpt/learner_0100247.msgpack` | c10970cdb45e | 0.00206 | 28,506 |
| 225,337 | `ckpt/learner_0225337.msgpack` | 92d3a0405051 | 0.00212 | 50,658 |
| 300,155 | `ckpt/learner_0300155.msgpack` | d439fdf12bf5 | 0.0312 | 59,907 |

**Run configuration** (`configs/online_rn_seed0.yaml` + `train.py` defaults): γ = 0.999 per control step, τ = 0.005,
batch 256, 1 update per post-warm-up step, critic lr 3e-4, actor lr 1e-4, α0 = 1e-3, target entropy
min(H_prior, 0.1 log M), prior removed at 50k, paired random-shift augmentation (pad 4).

**Reward and termination** (`executor.py:137-148`):

- Per control step: `1[first success] − 0.1/400 − (0.5/400)·underrun`.
- G accumulates `γ^k ρ_k` within a decision (`executor.py:141`).
- Horizon 400 steps; both success and timeout are terminal.
- Remaining time is in the state as `runtime[3]`.

**Decomposition** (`results/decomp_quantiles.csv`, `decomp_fixed_summary.json`; full per-transition CSV on scratch):

- Same transitions for all checkpoints, no augmentation (training used random shifts).
- Fixed sets:
  - ordinary: 2,048 uniform, plus 1,024 from rows before 100k;
  - by action: 512 KEEP, 512 REPLAN;
  - terminal: 512 terminal-success, 292 terminal-timeout (all of them);
  - by remaining steps: 512 each in the bins 0–50, 50–150, 150–300 and 300–400.
- The full-replay pass is reported separately (`full_replay_summary.json`). Its largest-error transitions are listed
  there and in `top_td_300155_fixed.csv`. They are ordinary non-terminal KEEP or REPLAN transitions with normal inputs
  (max feature z < 8), G ≈ −0.0015 to −0.0025, and future terms of 7,000–13,000.

**Soft-value bounds** used in the checks:

- Upper bound: 1 + α·ln 144·Σ_{k<rem} γ^k. This assumes at most one decision per step and that every decision reaches
  the maximum possible entropy.
- Lower task bound: −0.00025·Σ_{k<rem} γ^k (no underruns).
- "Q > 1" alone was not treated as failure.

**Pilot** (`results/pilot.json`, `pilot_branches.png`):

- Setup: frozen 100,247 actor, deterministic argmax with prior strength 0, held-out seed 10,000,000. The source
  rollout succeeded at step 280.
- Snapshots: the first KEEP-feasible decision at t ≥ 40 (t = 40, chunk 5, p = 0, q = 25) and at t ≥ 130 (t = 130,
  chunk 11, p = 1, q = 24).
- Candidates:
  - early snapshot: KEEP, the actor's preferred REPLAN(12,7), and REPLAN(10,10);
  - grasp snapshot: KEEP, the actor's preferred REPLAN(10,10), and REPLAN(15,5).
  - At both snapshots the actor's argmax was KEEP.
- Three paired noise streams per candidate: `default_rng([seed, t, rep, 101])`.
- Critic scores are **soft Q values**, not task-return estimates; the branches measure deterministic task returns. For
  example, at the grasp snapshot the 100k critic gives KEEP 0.812 and REPLAN(10,10) 0.825, while observed discounted
  task returns were 0.830 and 0.836. REPLAN(15,5) has Q = 0.000 at every checkpoint because it was never taken.

**Ablation** (`results/ablation.json`, `ablation_traces.png`):

- Setup: both branches start from the identical 100k critic, target critic and Adam state. Actor and α (0.00206) were
  frozen. Both used the same 300 batches drawn from rows before 100k and the same augmentation keys, with target
  updates on.
- Evaluation sets: rows before 100k (in-distribution) and rows after 100k (unseen).
- Results: the before/after Q, target and TD quantiles are near-identical for A and B; final batch loss is 4.5e-4 for
  A and 4.7e-4 for B; median gradient norm is 0.018 for both.

**Reproduce:**

```bash
sbatch -J sch2_critic_debug -p gpu-debug --time=00:20:00 debug_critic/run_debug.sbatch
```

```bash
python debug_critic/make_plots.py --src <job dir> --dst debug_critic/results
```

**Proposed fixes:** none applied. The zero-Q-for-untried-actions property and timeout terminality are design choices
documented above, not silent changes.
