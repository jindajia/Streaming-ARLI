# SCH-01 fixed_delay10_v2 — interactive scheduler viewer

An interactive static website for the completed run `rn_seed0_d10` (300,155 training control steps, 14 saved learner
checkpoints and the fixed baseline). It shows playback of new frozen-checkpoint rollouts, the scheduler's preferences
and critics at every decision, the execution/inference timeline, counterfactual probes of one frozen decision state
across all checkpoints, and learning diagnostics taken from the run's own logs. The viewer only inspects the run: it did
not train anything, and it does not modify the run directory (`online/rn_seed0_d10/` is read-only input).

## Open it

The site is self-contained: static files plus precomputed data, so no GPU is needed to view it. It lives at
`/N/scratch/jindjia/arli/runs_scheduling/fixed_delay10_v2/viewer/site` (about 0.7 GB, because each episode's
frames and probe arrays load lazily when you open that episode). From your laptop, run:

```bash
ssh -L 8765:127.0.0.1:8765 jindjia@bigred200.uits.iu.edu bash /N/slate/jindjia/SD/VLA/streaming_arli_viewer/experiments/scheduling/fixed_delay10_v2/viewer/serve.sh 8765
```

Then open <http://localhost:8765/> in your local browser. This single command starts a static server bound to
127.0.0.1 on whichever login node you land on and forwards the port to that same node. Ctrl-C stops it. If port
8765 is taken, use another number in both places. There is no public hosting: the repository is private, the data are
too large for GitHub Pages, and it was not tested.

## How to

- **Inspect an episode (Mode A, checkpoint rollout playback).** Pick a checkpoint on the top slider (or **Fixed
  baseline**) and a seed. Use Play (Space) and the speed selector, step with ←/→, or jump to the previous/next decision
  with Shift+←/→ ("« dec" / "dec »"). Clicking the timeline or any "Over episode time" chart pauses and seeks there.
  The video, the preferences and critic heatmaps, the readouts, the timeline cursor and all charts stay in sync.
  Switching checkpoints keeps the seed and control step, but the physical state at that step may differ because it is a
  different rollout (the page says so).
- **Lock a state (Mode B, lock state across checkpoints).** Stop on a decision tick. While a request is in flight there
  is no decision: use "Jump to nearest decision". Then press **Lock state across checkpoints**. The full scheduler input
  is frozen: current and source 64×64 images, robot state, queue and mask, plan age, source N, remaining time, runtime
  features and the feasibility mask. From then on the slider changes only the actor/critic checkpoint applied to that
  input. Both the policy *including that checkpoint's training-time prior* and the *network-only (prior disabled)*
  distribution are shown. They are identical from 50k steps on, where the prior strength is 0. The purple "Across
  training checkpoints" charts plot every checkpoint on the same input; click a point to select that checkpoint. "« dec"
  and "dec »" move the lock to another decision of the same source episode. **Unlock** returns to playback.
- **Learning diagnostics tab.** These are the run's own logs: held-out vs training-episode success kept separate;
  return, discounted return and completion time; critic loss, Q, targets, disagreement and feasible gap; entropy, target
  entropy, α, prior strength and reference probability; inference frequency, modeled compute, chunk execution length,
  N/R, launch phase and observation age. The 5k warm-up and 50k prior removal are annotated. You can view raw values or a
  moving mean. Clicking a chart selects the nearest checkpoint for playback/probes.
- The URL hash stores the view, e.g. `#pol=ck0300155&seed=10000007&t=120&lock=ck0300155_s10000007:30&probe=ck0025146`.
  Keyboard: `[` / `]` previous/next checkpoint, `L` lock/unlock.

## What is shown and how to read it

- **Actions.** KEEP, or REPLAN(R, N): observe now, run N full-interval RTC denoising steps (logically ready after d(N)
  control steps), execute R more old actions, then at t + R install the aligned suffix new[R : R + 25].
  - Feasibility: d(N) ≤ R ≤ q and R + 25 ≤ 50.
  - KEEP is feasible iff q − 1 ≥ 6 and more than one step remains.
  - The delay table is d = {3:6, 4:7, 5:7, 6:8, 7:9, 8:9, 9:10, 10:10} (simulated control steps, not GPU time).
  - R < 6 is never feasible, so those columns are not drawn.
- **Preferences.**
  - P(KEEP) is shown prominently.
  - The joint R × N heatmap shows the unconditional P(REPLAN(R, N) | state); the cells sum to 1 − P(KEEP).
  - The conditional view P(R, N | REPLAN) is labelled as conditional. It warns when the REPLAN mass is below 1e-6.
  - The colour scale is fixed: log 1e-4…1 or linear 0…1.
  - Infeasible cells are hatched.
  - Outlines: orange = selected action (Mode A) or argmax of the probed policy (Mode B); dashed = fixed-baseline action;
    pink = argmax when it differs from the outlined action.
  - Also shown: marginals over N and R, E[N | REPLAN], E[R | REPLAN], entropy, the top candidates of the full joint
    (KEEP included), and an expandable table of residual logits and prior contributions.
- **Critic.** The heatmap can show actor probability, min(Q1, Q2), |Q1 − Q2|, or Q advantage over the fixed-baseline
  action, with KEEP included.
  - Q is the online critics' learned soft return estimate (γ = 0.999 per control step, entropy-regularised target). It
    is not a success probability, and no calibration is claimed.
  - Colour ranges are fixed over the episode (Mode A) or over all checkpoints for the locked state (Mode B).
  - The "Predicted min(Q1, Q2) vs observed discounted return-to-go" chart keeps the prediction separate from what was
    observed.
- **In-flight periods.** The page says "No scheduler decision — request in flight" and shows the active request's N and
  R, elapsed steps, steps to logical readiness, and steps to handoff. The heatmaps are faded and labelled as coming from
  the last decision.
- **Timeline.** Tracks show executed chunks (IDs); actual handoffs (black lines); requests from observation to logical
  readiness (colour = N); the hatched wait from readiness to handoff; planned handoffs (▽); KEEP/REPLAN decisions; and
  underruns.
  - A request whose planned handoff falls at or after the episode end is labelled abandoned and has no actual handoff.
  - The readouts give the chunk ID, p, q, plan observation age, selected N/R, old actions committed before handoff,
    discarded actions, the aligned start index of the new chunk, reward, and cumulative return.

## Data and provenance

All inspection rollouts are **new recordings** made after training. They were recorded with the frozen checkpoints
(`ckpt/learner_<steps>.msgpack`, sha256 in `data/checkpoints.json`) and with the fixed baseline, using the held-out
evaluation protocol of `sch02.train`:

- noise stream `numpy.default_rng([seed, 1])`;
- deterministic masked argmax;
- each checkpoint's own training-time prior strength (1 → 0 between 5k and 50k).

Instrumentation is passive: a frame-capturing env proxy and a policy wrapper that stores the probabilities it already
computes. It consumes no extra RNG and changes no decisions.

Frame t is observation o_t **before** the action of control step t; frame 0 is the reset frame. The decision at tick t
is made on o_t.

Rendering is not bit-deterministic (±1 intensity), so a recording can differ from the episode logged at training time.
Each recording stores both outcomes and is labelled "matches log" or "differs from log".

**Seeds** (`data/seeds.json`, `export/select_seeds.py`): a *curated inspection set, not an unbiased new benchmark*. The
six held-out seeds were chosen from the logged per-episode outcomes. The categories are always success, always fail,
baseline success → final fail, baseline fail → final success, most outcome flips, and one seed drawn at random
(`default_rng(0)`). Within each category the rule takes the lowest seed. Six seeds × (fixed baseline + 14 checkpoints)
gives 90 rollouts.

**Probes.** Each recorded decision state (9,993 states) was evaluated by each of the 14 checkpoints. The evaluation used
the same jitted batch-1 forward pass the policy uses online and stored residual logits, Q1 and Q2. The full inputs are
archived as npz in `viewer/archive/`, which is not served. A first pass with batched evaluation differed from the online
forward pass by up to 0.41 logits (GPU kernel choice), so it was discarded and recomputed one state at a time.

| file (per episode, `data/episodes/<policy>_s<seed>/`) | content |
|---|---|
| `meta.json` | policy/checkpoint provenance, seed, outcome (new and logged), per-tick chunk/p/queue/in-flight/underrun/reward, requests, decisions (t, action, q, p, chunk, reference action, plan age, source N, …), frame offsets |
| `frames.bin` | 320×240 JPEG per control step (t = 0…t_end) |
| `masks.bin` | uint8 [n, 201] feasibility masks |
| `rec.bin` | float32 [n, 2, 201]: recorded probabilities used for the action, residual logits |
| `inputs.bin`, `netimg.bin` | queue / qmask / robot / runtime features; 64×64 img_now (img_src = img_now of the source decision) |
| `probes/<ckpt>.bin` | float32 [n, 3, 201]: residual logits, Q1, Q2 of that checkpoint on these states |

Learning diagnostics (`data/learning.json`, `export/export_logs.py`) come only from `episodes.jsonl`, `updates.jsonl`,
`evals.jsonl`, `reference_eval.jsonl` and `evals_supplemental.jsonl` of this run; no legacy v1 logs are used.

- The discounted return is not logged. It is reconstructed exactly from (t_end, success) because every training episode
  has zero underrun steps: G = −c (1 − γ^T)/(1 − γ) + 1[success] γ^(T−1), with c = 0.1/400 and γ = 0.999. `validate.py`
  checks this formula against the executor's discounted return on all 90 recordings.
- The logged `ret` is undiscounted.
- Missing metrics are listed on the tab.

No model weights, replay buffers or credentials are served.

## Reproduce

```bash
cd experiments/scheduling/fixed_delay10_v2
python viewer/export/select_seeds.py
python viewer/export/export_logs.py
sbatch -J sch2_viewer_export -p gpu-debug --time=00:59:00 viewer/cluster/export_viewer.sbatch
python viewer/export/validate.py
python viewer/export/build_site.py
PLAYWRIGHT_BROWSERS_PATH=/N/scratch/jindjia/ms-playwright /N/scratch/jindjia/envs/viewer_test/bin/python viewer/export/ui_test.py --site /N/scratch/jindjia/arli/runs_scheduling/fixed_delay10_v2/viewer/site
```

- `select_seeds.py` and `export_logs.py` run on CPU and are read-only on the run directory.
- The GPU export job records the rollouts and probes. It is resumable, writes `meta.json` last, and exit 3 means
  "resubmit to continue".
- `validate.py` runs the CPU checks.
- `build_site.py` copies `web/` and writes `index.json`.
- `ui_test.py` is the headless browser test.

A CPU smoke test of the recorder with fake env/sampler: `record_rollouts.py --fake --out <dir>/data --archive <dir>/archive`.
