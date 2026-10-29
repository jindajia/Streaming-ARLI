# When does the handoff delay R change ALOHA Transfer Cube outcomes? Same-state screening pilot

Branch `experiment/r-sensitivity-probe`. This is an exploratory intervention study. Nothing was trained, and no learned
scheduler or critic was used. Three repetitions per cell are a screen: "no detectable difference in this pilot" never
means equivalence, and the best-looking R in a cell is not an established optimum.

## Setup (verified in code)

- **Policy:** frozen π0 ALOHA policy `/N/scratch/jindjia/arli/checkpoints/pi0_aloha_sim`, the same checkpoint and
  preprocessing as `fixed_delay10_v2`. The normalisation-stats sha256 is in `sources_summary.json`.
- **Timing and RTC:**
  - N = 10 full-interval denoising steps; latency d = 10 control steps from the shared delay table; no jitter.
  - H = 50, K = 25.
  - RTC on with the existing port: exponential soft mask, guidance cap 10.
- **Executor convention:** a request at tick t observes o_t and is logically ready at t + 10. R old actions execute,
  and at t + R the slice new[R : R + 25] is installed.
  - The RTC reference is the previous plan shifted by `t − origin`. The committed prefix is R (weight 1 on new[0:R]),
    with a soft overlap until H − shift.
  - Recorded per branch: for example, forced R=10 at p=5 has shift 15, committed 10, overlap end 35, installed
    new[10:35].
  - No learned noise steering is used.
- **Reward and horizon:** reward 1 at success minus 0.00025 per control step (no underruns occurred). Horizon 400 steps
  at 50 Hz; discount 0.999 per step.
- **Full fixed baseline:** KEEP while q > 10, then REPLAN(10, 10) at q = 10 (p = 15), handing off at p = 25.
- **Forced branches:** exactly one forced REPLAN(R, 10) at the snapshot. Afterwards the same baseline rule resumes;
  its next request comes at q = 10 of the newly installed chunk.
- **Control:** the full baseline from the snapshot, which next observes at p = 15.

## Sources, snapshots and coverage (`manifest.json`, frozen before any branch outcome)

The source seeds were fixed from the logged fixed-baseline held-out records:

| role | seed | logged | regenerated now |
|---|---|---|---|
| median-time success | 10000014 | success at 272 | success at 264 |
| latest success | 10000016 | success at 376 | success at 366 |
| lowest-numbered failure | 10000004 | failure | failure |

Phases come from simulator contact events (gym_aloha reward stages 1–4) plus image inspection. The rule anchors each
phase on when the forced handoffs land (see `select_snapshots.py`).

- **10000004 never reaches transfer:** the left finger never touches the cube, and the right gripper loses contact at
  178. Its two transfer snapshots are marked missing.
- **Coverage:** 22 of the 24 target snapshots. Source 10000016's "grasp/lift" chunk is at first contact (92); that
  episode only lifts at 206 after a long struggle.

Images: `snapshot_images/all_snapshots_contact_sheet.jpg`, plus one contact sheet per source.

## Rollouts and compute

| what | rollouts | GPU job | wall time |
|---|---|---|---|
| source rollouts | 3 | 8769035 (gpu-debug, A100-40GB) | 3 min 11 s, together with the restoration controls |
| restoration-control rollouts | 4 | 8769035 | (included above) |
| main matrix: 22 snapshots × 4 choices × 3 continuation seeds | 264 | 8769086 | 15 min 31 s |

Total: about 19 minutes of GPU time over 2 jobs. CPU smoke tests used fake environments and are not counted. There were
no anomalies, no underruns, and no episode terminated before a forced handoff.

## Restoration checks (`restoration_checks.json`)

- On two snapshots, the restored physics state, actuator state, queue, previous plan and counters exactly equal the
  saved snapshot.
- Re-rendering the restored state differs from the saved input image by at most 1 intensity level (the renderer is not
  bit-exact).
- Identical intervention, identical continuation seed, run twice:
  - source 016, chunk 8, p=10, R=12: identical outcomes.
  - source 014, chunk 6, p=5, R=15: success at 271 vs 272, a 1-step difference.
- So repeat variability of about 1 step exists. It is far smaller than the success flips and 10+ step timing gaps
  reported below, but it limits how far small timing differences can be interpreted.
- Continuation-seed pairing: policy noise uses `default_rng([source_seed, t_snap, cont_seed, 303])`, one draw per
  inference request in order, and is shared by all four choices. Different request times mean the observations
  conditioned on differ, so outcomes are paired, not identical in input.

## Results (`results/success_heatmaps.png`, `results/paired_differences_and_times.png`, `results/cells.csv`, `results/paired_comparisons.csv`)

Discordant pairs are pairs where one choice succeeds and the other fails, with the same snapshot and continuation seed.

| source | phase | forced vs forced, p=5 / p=10 (of 9) | forced vs baseline, p=5 / p=10 (of 9) |
|---|---|---|---|
| 014 (success) | all four phases | 0 / 0 | 0 / 0 |
| 016 (late success) | ordinary motion | 2 / 2 | 1 / 2 |
| 016 | approach | 4 / 2 | 3 / 2 |
| 016 | grasp/lift | **6 / 6** | 3 / 5 |
| 016 | transfer | 0 / 0 | 0 / 0 |
| 004 (failure) | ordinary motion | 0 / 2 | 0 / 1 |
| 004 | approach | 0 / 0 | 0 / 0 |
| 004 | grasp/lift | 0 / **6** | 0 / 3 |

### Answers

1. **Where there is evidence that R matters** (observations):
   - **Grasp/lift in the two harder episodes.**
     - Source 016, p=5 (t=80, first contact at 92): R=20 succeeded 3/3, R=10 0/3, R=15 1/3, baseline 1/3. All three
       R=20 vs R=10 pairs are discordant in the same direction.
     - Source 016, p=10 (t=85): R=10 2/3, R=12 0/3, R=15 2/3, baseline 1/3. The pattern is non-monotonic in R.
     - Source 004, p=10 (t=110, just after first contact): R=15 2/3 against 1/3 for R=10, R=12 and the baseline.
       The pairs disagree in both directions: R=15 won 2 of them and lost 1.
   - **A timing effect without any outcome change.** In source 014 at grasp/lift p=5 (t=130, contact at 135), forced
     R=10 completed the task 8–16 steps sooner than R=15, R=20 and the baseline on all 3 paired seeds. The discounted
     return gain is +0.009 to +0.017. R=20 and the baseline, which both hand off at p=25, differ by only 1 step on
     every seed.
2. **Little detectable sensitivity in this pilot:**
   - Transfer, in both sources that reach it: 3/3 for every choice, with completion-time differences of 11 steps or
     less. In source 016 the baseline was slightly faster (36–40 steps vs 39–49).
   - Every phase of the easy source 014, at the outcome level.
   - All p=5 cells of the failing source 004: 0/3 for every choice.
3. **p=5 vs p=10.** The two snapshot positions are different states 5 steps apart with different R sets, so they are
   not directly comparable.
   - At p=5, R=20 hands off at the same time as the baseline but observes 10 steps earlier. In source 014 this made
     about 1 step of difference.
   - Outcome patterns do not carry over between the positions. In 016 grasp/lift, R=10 goes from 0/3 at p=5 to 2/3
     at p=10, and R=15 from 1/3 to 2/3. In 004 grasp/lift, every choice fails at p=5, but some succeed at p=10.
   - So small changes in state and timing around contact can flip outcomes. A single rule of "R effect by phase" is
     not supported.
4. **Improvement over the full baseline?** Nothing consistent.
   - Better than the baseline: 016 grasp/lift p=5 R=20 (3/3 vs 1/3) and 004 grasp/lift p=10 R=15 (2/3 vs 1/3, mixed
     pairs). Source 014 grasp/lift p=5 R=10 was faster on all 3 seeds.
   - Worse than the baseline: 016 approach p=5, where every forced R scored 2/3 vs 3/3, and 016 grasp/lift p=10
     R=12 (0/3 vs 1/3).
5. **Cases that deserve 10–20 paired repetitions and checking on new source episodes:**
   - `s016_grasp_lift_p5`: R=20 vs R=10 vs baseline.
   - `s016_grasp_lift_p10`: the non-monotonic R pattern.
   - `s004_grasp_lift_p10`: R=15.
   - `s014_grasp_lift_p5`: the R=10 timing gain.
   - A background-sensitivity control: `s016_ordinary_motion` (t=30), where interventions also flip outcomes, with
     baseline-only repeats over many continuation seeds. This measures how much outcome variance exists without any R
     change.

### Hypotheses (not established)

- R matters mainly when the episode is marginal and its outcome is decided at the grasp. In easy episodes the policy
  recovers from any of these timings, and at transfer the choices tested here change little.
- In source 016, however, interventions as early as t=30 change success (2 of 9 pairs) and completion time by up to 93
  steps. Continuation seeds alone also change outcomes within a cell (for example the baseline at 016 grasp/lift p=5
  is 1/3).
  - So marginal episodes look chaotic: many small perturbations, not just R at grasp, can change the outcome.
  - The grasp-phase concentration may partly reflect when the marginal episode becomes decidable, rather than a
    special role of R.
- In source 014, a fresher plan installed before contact completes (R=10 at p=5) may speed the grasp. It changed
  timing, not success.

## Files

All of these are in `experiments/scheduling/r_sensitivity_probe/`:

- **Scripts:**
  - `rsens.py` — sources and matrix stages;
  - `select_snapshots.py` — phase rule and manifest;
  - `analyze.py`, `make_videos.py`;
  - `run.sbatch`.
- **Frozen inputs and checks:** `manifest.json`, `snapshot_images/`, `sources_summary.json`, `restoration_checks.json`.
- **Results:** `results/branches.csv` (one row per branch with every requested field), `results/cells.csv`,
  `results/paired_comparisons.{csv,json}`, `results/summary.json`, and the figures.

Bulky data is in `/N/scratch/jindjia/arli/runs_scheduling/r_sensitivity_probe/`:

- `sources/` — snapshot pickles and frames;
- `branches.jsonl` — raw branch records;
- `video_frames/`;
- `videos/s014_grasp_lift_p5_cont0.mp4`, `videos/s014_transfer_p5_cont0.mp4` — 2×2 paired videos, 25 fps (half
  speed), mp4v codec;
- `logs/`.
