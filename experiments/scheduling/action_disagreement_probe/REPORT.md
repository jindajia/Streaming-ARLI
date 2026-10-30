# Action-disagreement probe: queued plan vs fresh plan, per step and future offset

Branch `experiment/action-disagreement-probe` (based on `experiment/r-sensitivity-probe` @ `0c4d2a1`). This is a
diagnostic only: nothing was trained, and no probe output was ever executed. Disagreement is not evidence that
replanning would help.

## What was run (verified in code)

- **Policy and configuration:** the frozen π0 ALOHA checkpoint `/N/scratch/jindjia/arli/checkpoints/pi0_aloha_sim`
  with the same preprocessing as `fixed_delay10_v2`. The instruction is "Transfer cube", the config default prompt.
  - N = 10, d = 10 control steps (no jitter), H = 50, K = 25, RTC on.
  - Fixed baseline: request at p = 15 / q = 10, handoff at p = 25.
  - Existing reward and success criterion, 400-step horizon, 50 Hz.
- **Reference episodes, regenerated.** The saved R-sensitivity sources had no per-step observations or plan state, so
  the three baseline episodes were recorded again with a read-only logging proxy. The renderer is not bit-exact, so
  these are new trajectories:

  | seed | earlier R-sensitivity source | regenerated here |
  |---|---|---|
  | 10000014 | success at 264 | success at 263 |
  | 10000016 | success at 366 | success at 364; lifted at 195 (was 206) |
  | 10000004 | failure; never lifted | failure; lifted briefly at 178 |

  Per step the proxy saved:
  - raw 480×640 image and robot state;
  - t, chunk, p, q, the queue before the pop;
  - the installed full plan (normalized x0, env actions, origin);
  - pending-request metadata;
  - the contact stage.
- **Probes (offline, from the saved records).** At every step t, fresh plans were generated from o_t:
  - RTC off;
  - RTC on with R = d = 10, using the shifted previous full plan as reference, committed prefix 10 and the exponential
    mask until H − shift, exactly as `Executor._launch` does.

  RTC-on probes exist only where a new request is allowed: no inference pending (or the baseline's own launch step)
  and q ≥ d. That is 173 of 263, 238 of 364 and 256 of 400 steps; elsewhere RTC-on is N/A.
- **Probe noise:** two fixed tensors per episode, `default_rng([seed, 4242, k])` for k = 0, 1. They are reused at every
  step, shared between the RTC-off and RTC-on variants, and independent of the reference stream. sha256 hashes are in
  `probes/probe_<seed>.json`.
- **Metrics:**
  - Arm: RMS over the 12 arm joints of (old − fresh) / std_j, where std_j is the policy's own action norm-stat std. The
    aloha↔pi joint flips only change signs.
  - Grippers (dims 6 and 13): absolute difference in env units, normalized gripper position with 0 = closed and
    1 = open.
  - No verified forward kinematics exists for commanded actions, so TCP error was not computed.
  - Discrepancies are computed per sample and then summarized; plans are never averaged.
- **Windows:**
  - committed prefix: 0 ≤ h < min(d, q);
  - replaceable execution window: d ≤ h < q, reported as N/A when q ≤ d;
  - prediction-only tail: q ≤ h < 50 − (t − origin).
- **Validation:**
  - The queue equals the stored plan slice `old[t − origin : t − origin + q]` at all 1,027 steps (maximum error 0).
  - At 9 baseline launch steps, an RTC-on probe using the reference request's own recorded noise reproduces the
    installed plan exactly (maximum difference 0.0), and repeated probes are bit-identical. This checks alignment,
    the RTC reference and mask, and reproducibility together.
  - All probe outputs are finite.
  - The reference was never touched: the proxy only reads, probes run on saved arrays, and the reference noise stream
    was only recorded. The CPU smoke test also checked empty-window and in-flight handling.
- **Compute:** one gpu-debug job, 8772461 (A100-40GB), **8 min 15 s** wall.
  - Inference calls: 43 reference (the episodes' own plans), **3,406** trajectory probe calls (including 18 validation
    calls), and **88** calls on the exact saved R-sensitivity snapshots.

## Figures and data

- **Per-episode figures** `results/action_disagreement_<seed>.png`, from top to bottom:
  - timeline with frames;
  - RTC-off heatmap;
  - RTC-on heatmap;
  - arm curves;
  - gripper curves;
  - fresh–fresh curves.
- **Heatmap encoding:** color = arm discrepancy, the mean of the two samples' discrepancies. The scale is shared across
  all episodes and variants, 0 to 0.52, and the top 1.0% of cells are clipped.
- **Zooms** `results/zoom_<seed>_peak<k>_t<t>.png`: per-offset trajectories of the old plan and the 4 fresh plans for
  the most discrepant joints and the right gripper.
- **Data:**
  - `results/timeseries_<seed>.csv`: per-step metadata and every window scalar;
  - `results/summary.json`: segments, peaks and the color scale;
  - `results/rsens_snapshots_disagreement.csv`: the exact saved snapshots joined to their earlier branch outcomes.
- **Bulky data** in `/N/scratch/jindjia/arli/runs_scheduling/action_disagreement_probe/`:
  - `reference/ref_<seed>.npz`: raw observations, queues and plans;
  - `probes/probe_<seed>.npz`: all raw fresh plans;
  - `discrepancy_<seed>.npz`: D(t, h) for each metric;
  - `snapshot_probes.pkl`;
  - `videos/reference_<seed>.mp4`: a timestamped reference video with t, chunk, p, q, stage, request status and the
    current disagreement.

## Answers

Unless noted, numbers are medians over eligible steps with a non-empty replaceable window, as "old–fresh / fresh–fresh"
arm RMS with RTC off.

**1. Does disagreement concentrate in particular states or phases?** Yes, in contact-related phases.

| episode | phase | old–fresh | fresh–fresh | ratio |
|---|---|---|---|---|
| 10000014 | grasp/lift | 0.101 | 0.033 | 3.1× |
| 10000014 | carry | 0.069 | 0.034 | 2.0× |
| 10000014 | free motion | 0.044 | 0.045 | about 1× |
| 10000014 | approach | 0.036 | 0.030 | about 1× |
| 10000014 | transfer approach | 0.025 | 0.029 | about 1× |
| 10000004 | grasp/lift | 0.087 | 0.041 | 2.1× |
| 10000004 | after losing the cube | 0.117 | 0.041 | 2.9× |
| 10000004 | free motion and approach | about 0.04 | about 0.04 | about 1× |
| 10000016 (marginal) | free motion | 0.064 | 0.034 | 1.9× |
| 10000016 | approach | 0.050 | 0.020 | 2.5× |
| 10000016 | long grasp struggle | 0.059 | 0.021 | 2.8× |
| 10000016 | carry | 0.061 | 0.033 | 1.8× |

The marginal episode is above sampling variation almost everywhere. The largest single peaks fall right after a
handoff (p = 0), when the newly installed plan was computed from an observation 10 steps old:

- 10000004 at t = 175: 0.50 vs fresh–fresh 0.06;
- 10000014 at t = 175 (carry): 0.25 vs 0.03.

There are also peaks at t = 10 in the startup chunk, which was planned from t = 0 without RTC.

**2. Replaceable actions or the unexecuted tail?** Both, with a gradient: disagreement grows with offset h.

| episode | committed prefix | replaceable window | prediction-only tail |
|---|---|---|---|
| 10000014 | 0.036 | 0.048 | 0.077 |
| 10000016 | 0.041 | 0.057 | 0.070 |
| 10000004 | 0.081 | 0.081 | 0.088 |

The prominent events (the 10000004 grasp/release, the 10000014 carry peak) sit inside the replaceable window, not only
in the tail. The tail is shown as context only and is never part of the primary scalars.

**3. How much does RTC suppress it?** Strongly near the old plan, by design.

- Median replaceable-window RMS falls by 29–43%:

  | episode | RTC off | RTC on |
  |---|---|---|
  | 10000014 | 0.048 | 0.034 |
  | 10000016 | 0.057 | 0.036 |
  | 10000004 | 0.081 | 0.046 |

- The committed prefix falls by 44–57%.
- Peaks are suppressed most: 0.50 → 0.09 at 10000004 t = 175, and 0.30 → 0.09 at 10000014 t = 10.
- RTC-on fresh–fresh variation is also smaller (about 0.02–0.03).
- Low RTC-on discrepancy is expected from the guidance and does not show that the fresh observation is uninformative.

**4. Old–fresh vs fresh–fresh.**

- Overall RTC-off medians:

  | episode | old–fresh / fresh–fresh | ratio |
  |---|---|---|
  | 10000014 | 0.048 / 0.035 | 1.4× |
  | 10000016 | 0.057 / 0.024 | 2.4× |
  | 10000004 | 0.081 / 0.041 | 2.0× |

- During free motion, approach and transfer approach of 10000014 and 10000004, old–fresh is about equal to
  fresh–fresh: disagreement there is consistent with sampling variation plus normal progress.
- Contact phases, and most of the marginal episode, are 2–3× above it.
- These are two samples per step, not a confidence interval.

**5. Arm vs gripper.** Arm disagreement is smooth and broadband. Gripper disagreement is rare, large (up to about 0.9
env units) and mode-like, open versus closed:

- **10000004, t = 150–165** (grasp closing): the queued plan closes the right gripper (0.90 → 0.25). Fresh RTC-off
  plans keep it open (about 1.05); RTC-on follows the old plan.
- **10000004, t = 175–185:** the queued plan, computed at t = 165, opens the right gripper (0.29 → 0.73) inside the
  replaceable window. Both RTC-off and RTC-on fresh plans keep it closed (about 0.17–0.27). The cube contact is lost
  around t ≈ 180–185 in the reference (stage drops to 0).
- **10000016, t = 205–215,** just after the first lift: the old plan keeps the right gripper open (about 1.1). Fresh
  RTC-off plans close it (toward 0.2–0.5); RTC-on stays with the old plan.
- **10000014, t = 175–188** (carry, preparing the handover): the old plan opens the left gripper 0.20 → 0.85 within the
  window. Fresh plans open it later and more slowly (for example at t = 185: old 0.53 → 0.85, fresh RTC off
  0.31 → 0.50). RMS reaches about 0.35 with RTC off and less with RTC on. This is a timing difference, not a mode flip,
  and the episode succeeds.
- **Left gripper elsewhere:** disagreement appears only near transfer (10000016 median 0.086 in the last 30 steps).

**6. Timestamps worth a later intervention test.** All of these are states of the regenerated trajectories; their saved
records allow exact restoration.

| episode | t | what happens there |
|---|---|---|
| 10000004 | 175 | chunk 8, p = 0. The queued plan opens the gripper while fresh plans hold it closed, just before contact is lost. Test a replan at t = 175 (R = 10) vs the baseline. |
| 10000004 | 150–158 | Grasp-closing timing; RTC-off and RTC-on disagree with each other. |
| 10000016 | 205–210 | Fresh RTC-off plans want to close the gripper; old and RTC-on keep it open. |
| 10000016 | 110 and 135 | The grasp struggle; arm RMS 0.10 and 0.09, against fresh–fresh 0.015 and 0.018. |
| 10000014 | 175 | Carry, just after a handoff; arm 0.25 vs 0.03. The right gripper agrees; the left-gripper opening is later in fresh plans. A control case where the plans disagree but the episode succeeds anyway. |

**Link to the R-sensitivity results** (the exact saved snapshots, not the regenerated trajectory):

- Disagreement at the 22 snapshots is modest: RTC-off arm RMS 0.02–0.08, with no large gripper events.
- The snapshots whose branch outcomes varied with R are all from 10000016 and 10000004. For example
  `s016_grasp_lift_p5`, where R = 20 gave 3/3 and R = 10 gave 0/3, has 0.046 vs fresh–fresh 0.020. The 10000016 ones mostly
  show old–fresh 2–3× fresh–fresh (the 10000004 ones 1.0–1.8×), while the insensitive 10000014 snapshots sit at
  0.8–1.8×.
- That association is confounded by source episode and has only 22 states, so it is suggestive only.

## Interpretation and caveats (hypotheses)

- Disagreement grows with offset and plan age because the world moves on during the 10-step latency. Much of the
  smooth arm disagreement is normal progress: the fresh plan starts from where the robot actually is.
- Gripper open/closed disagreements may be alternative valid plans, such as different timing of closing or releasing.
  They are not necessarily errors in the old plan.
- With two samples, fresh–fresh only roughly calibrates sampling variation, and multimodal policies can make both
  samples agree while other modes exist.
- This experiment changes the observation and the plan together; it does not isolate the causal effect of the image.
- **The 10000004 t = 175 release event is the most suggestive case, and it is a hypothesis.** Only an intervention test
  can show whether replacing those actions would have kept the cube.
