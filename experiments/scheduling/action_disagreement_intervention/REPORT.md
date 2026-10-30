# Intervention test at the t = 175 disagreement peak (regenerated episode 10000004)

Branch `experiment/action-disagreement-intervention`, based on `experiment/action-disagreement-probe` @ `53218e0`. One
gpu-debug job, 8773720 (A100-40GB): **13 min 43 s** wall. About 8.6 min of that was startup and snapshot
reconstruction; the 33 rollouts took about 4 min. Nothing was trained.

## 1. Did a single early replan help?

**No, not in this test.**

- Branch B (one early replan) succeeded **0/10**, the same as the fixed baseline A at **0/10**.
- Paired B vs A: 0 wins, 0 losses, 10 ties (both fail). The discounted task-return difference is 0.0 on every seed.

## 2. Did repeated replanning help?

**No.**

- Branch C (three early replans) also succeeded **0/10**.
- C vs A and C vs B: 10 ties each, both fail.

## 3. Was t = 175 early enough to prevent the observed failure?

**No.**

- **The failure happens before any possible handoff.** Right-finger contact with the cube is lost at step 178 in all
  30 main rollouts, the 3 controls and the saved reference. A request launched at t = 175 cannot install new actions
  before t = 185.
- **No branch can differ before 185.** Over t = 175–184 every branch executes exactly the same recorded old actions,
  and the cube pose is identical across all rollouts (difference 0.0).
- **The cube was never lifted.** At t = 175 the cube sits at table height: z = 0.0245 m, touching the table, with a
  maximum of 0.0247 m over the whole rollout. The "lifted" stage at step 177 is a one-step contact flicker. This
  corrects the earlier probe report's wording that 10000004 "lifts the cube briefly at t = 178".
- **Any benefit could only have come from recovery after the failure, and none occurred.** No rollout regained
  right-finger contact after t = 185, touched the cube with the left finger, or raised the cube.

## 4. What changed in the actual executed behavior?

- **Only the right-gripper command changed, and only after the cube was gone.**
  - A executes the queued plan's gripper opening over t = 185–199, with the command rising from 0.29 to 0.73. The
    measured gripper opens from 0.31 to 0.51, then closes again after the t = 200 handoff.
  - B and C keep the gripper commanded closed (about 0.19–0.24); the measured gripper settles near 0.23–0.25.
- **The interaction did not change.** The commanded opening the disagreement probe flagged happens after contact was
  already lost, with the cube resting on the table outside the fingers. The cube pose and the contact stage are the
  same in all branches.
- **After the loss, the policy never re-grasps.** From about t = 200 to the horizon, every branch keeps the right
  gripper commanded closed (about 0.15–0.2) and never reopens to try again.

## 5. Most useful next experiment

Move the intervention before the loss. The relevant window is roughly t ≈ 160–168: a replan there hands off by 170–178,
during the closing grasp.

- Use the same reconstructed-prefix method and the same A/B/C design.
- In this episode, check first whether the queued plan and fresh plans disagree about the grasp at those steps. The
  probe showed a closing-timing disagreement at t = 150–165.
- Also worth testing on new source episodes: does losing contact during the closing grasp generally happen before any
  replan could act? That would mean failures of this kind need earlier triggers than a plan-disagreement peak.

An equal-compute timing control is not justified by these results: neither treatment changed anything.

---

## Snapshot reconstruction (`results/snapshot_validation.json`)

The probe's record contains observations, queue, plans and contact stages but **no physics state**. The t = 175 state
was therefore **reconstructed**:

- Reset the env with seed 10000004.
- Replay the recorded executed actions a_t = queue[t, 0] for t = 0..174.
- Compare every step with the record:
  - robot state matches to at most 6e-8 (float32 rounding);
  - images differ by at most 1 intensity level (the renderer is not bit-exact);
  - the contact stage matches at all 175 steps.
- The cube pose was not in the record, so it was validated indirectly, through the matching contact stages and images.

The executor state at t = 175 was built from the record. It matches what the task specified:

- chunk 8, p = 0, q = 25;
- the installed plan has origin 165 and N = 10, and the queue equals plan[10:35];
- no request pending, n_calls = 8, last action a_174;
- current observation = the recorded o_175 pixels.

**Controls** (3 extra rollouts):

- **Branch A driven by the reference's own recorded noise:** commands identical to the saved trajectory through
  t = 224, and the contact-stage sequence identical over all 225 steps. The first command difference (max 0.009)
  appears at t = 225. That is the first handoff whose plan used a re-rendered observation (o_215, ±1 intensity). Same
  outcome as the saved reference: timeout.
- **Branch A seed 0, run twice:** identical through t = 224, first difference at t = 225; both time out.
- **Conclusion:** restoration is exact for physics. Rendering differences enter only through later observations, and
  they did not change any outcome.

## Execution and schedules (verified per rollout in `results/rollouts.csv`)

| branch | requests (observation → handoff) | inference calls after 175 | median calls |
|---|---|---|---|
| A | 190 → 200, then every 25 steps | 9 per rollout | 9 |
| B | 175 → 185, then 200 → 210, then every 25 steps | 10 | 10 (+1, +11% vs A) |
| C | 175 → 185, 185 → 195, 195 → 205, then 220 → 230, every 25 steps | 11 | 11 (+2, +22% vs A) |

- **RTC settings:** early requests have shift 10, committed prefix 10, overlap end 40. Baseline requests have shift
  25, committed 10, overlap end 25.
- **Readiness and installed slice:** ready at observation + 10, and handoff installs new[10:35].
- **One automatic flag, not a schedule error.** B's flag "schedule not as expected" comes from its last request at
  t = 399. With one step left KEEP is infeasible, so the baseline rule issues a recovery REPLAN(R = 11), which the
  horizon then abandons. All specified request times are correct.
- **B and C** issue the identical first request (same observation, plan and noise) and execute identical commands over
  t = 175–194, as expected.
- **A and B** first differ at t = 185 on every seed.

## Outcomes

| branch | success | failures (timeout at 400) | mean discounted return from 175 |
|---|---|---|---|
| A | 0/10 | 10 | −0.0504 |
| B | 0/10 | 10 | −0.0504 |
| C | 0/10 | 10 | −0.0504 |

There are no completion times to report, since all rollouts are failures. The saved reference (kept separate) also
failed.

## Observations vs hypotheses

**Observations:**

- Contact is lost at step 178, before any intervention can act.
- The cube stays on the table.
- The fresh plans' different gripper commands, holding closed instead of opening, act on an empty gripper and change
  nothing.

**Hypotheses:**

- The t = 175 disagreement reflects the policy reacting to an already-failed grasp (the fresh observation shows no cube
  between the fingers) rather than a preventable error.
- The policy's later behavior (gripper held closed, no re-grasp) suggests it does not recover from this kind of slip
  within 200+ steps.
- Ten seeds at one state are exploratory, and with every rollout failing, this test cannot rule out a recovery
  probability of up to about 30% per branch (rule of three for 0/10).

## Files

- **Code:** `adi.py` (reconstruction and branches), `analyze.py`, `run.sbatch`.
- **Results:** `results/summary.json`, `results/rollouts.csv`, `results/rollouts_raw.jsonl`,
  `results/snapshot_validation.json`, `results/noise_manifest.json`, `results/run_8773720.json`.
- **Figures:** `results/outcomes.png`, `results/mechanism_traces.png`.
- **Bulky data** in `/N/scratch/jindjia/arli/runs_scheduling/action_disagreement_intervention/`:
  - `trace_<tag>.npz`: per-step commands, measured gripper positions, cube pose, contact flags, stage, chunk, queue
    length, pending request;
  - `frames/`;
  - `videos/abc_seed0.webm`, `videos/abc_seed1.webm`: A/B/C side by side with t, chunk, queue, request status, gripper
    command and measured position, and stage.

  Every seed has the same outcome, so seeds 0 and 1 are representative.
- **Provenance:** pi0 checkpoint `/N/scratch/jindjia/arli/checkpoints/pi0_aloha_sim`; the norm-stats and
  reference-record sha256 hashes are in `results/run_8773720.json` / `snapshot_validation.json`; code commit
  `d7034be` (job).
- **Rollout counts:** 30 main + 3 controls; no skipped work.
