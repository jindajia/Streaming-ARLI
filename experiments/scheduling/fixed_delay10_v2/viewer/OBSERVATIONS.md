# What the viewer data show (rn_seed0_d10): observations vs hypotheses

Scope and caveats:

- One training seed.
- The 90 inspection rollouts are a curated set of 6 held-out seeds × (fixed baseline + 14 checkpoints). They are not an
  unbiased benchmark, so success counts out of 6 are illustrations, not estimates.
- The held-out success numbers below come from the run's own 50-episode evaluations.
- Numbers come from `data/summary.json` (`export/summarize.py`), `data/validation.json` and the run logs.
- The probe states are the 1,424 decisions of the six fixed-baseline rollouts (KEEP is feasible in all of them).

## Observations

1. **Recordings mostly reproduce the training-time evaluation episodes, but not exactly.**
   - 85/90 new recordings have the same success outcome as the episode logged for that checkpoint and seed during
     training. 61/90 also have the same end step.
   - 5 outcomes flipped: ck150000 s10000007, ck175264 s10000035, ck200252 s10000043, ck250150 s10000043 and
     ck300155 s10000043.
   - Physics, π0 and the actors are deterministic; the renderer is not (±1 intensity). So the per-seed outcomes of
     these policies are sensitive to tiny perturbations.
2. **The learned scheduler moved from "observe 10 steps before the handoff" to "re-plan immediately after every
   handoff".** Over the rollouts:
   - Mean launch phase p falls from 15 (fixed baseline; checkpoints 0 and 5k) to 4.8 (25k), 1.3 (125k) and 0.19
     (300k).
   - The share of requests launched at p = 0 rises from 0 to 0.59 (25k) and 0.95 (300k).
   - Requests per episode rise from 14.8 to 37.3.
   - Discarded old actions per request rise from 0 to about 15.6.
   - Mean R stays near 9–11 and mean N near 6–9, against 10 and 10 for the baseline.
3. **Locked probes show the same trend on identical inputs.** On the frozen fixed-baseline states, the fraction where a
   checkpoint's argmax is KEEP is:

   | checkpoint | argmax is KEEP |
   |---|---|
   | baseline rule | 0.94 |
   | 0 | 0.94 |
   | 5k | 0.94 |
   | 25k | 0.64 |
   | 50k | 0.57 |
   | 100k | 0.39 |
   | 150k | 0.20 |
   | 200k | 0.16 |
   | 250k | 0.07 |
   | 300k | 0.02 |

   The fraction where the argmax equals the fixed-baseline action follows the same curve (1.00 → 0.03).
4. **The prior hid an early network preference for replanning.**
   - At 5,334 steps (334 gradient updates, prior strength 0.993) the policy including the prior picks KEEP in 94% of
     these states. The network-only distribution (prior disabled) picks KEEP in 0%.
   - At 25k (prior strength 0.55) the split is 64% with the prior vs 8% without it.
   - From 50k on the prior strength is 0 and both distributions are identical, as the viewer states.
5. **Before ~240k, the critics' preferences behind departures are smaller than the critics' own disagreement.**
   - These are states where a checkpoint's argmax differs from the baseline action.
   - For checkpoints 25k–225k, the median min-Q margin of the chosen action over the baseline action is 0.002–0.016.
   - The median |Q1 − Q2| is 0.02–0.05 at the chosen action and 0.016–0.05 at the baseline action.
   - Only 4–14% of these departures have a margin larger than the two disagreements combined.
6. **The critics diverged late in training (run logs, independent of the viewer).**
   - Logged mean Q of replay actions (`q_taken`) first exceeds 1 at 242k steps.
   - It reaches 128 at 300k, with critic loss rising from about 2e-3 to 3.8e4, `q_max` reaching 1.7e4, and actor loss
     falling to about −100.
   - The return cannot exceed 1 (success bonus 1, negative step costs).
   - On the recorded rollouts, the mean predicted Q of the selected action is 2.9 / 14.2 / 48.9 at 250k / 275k / 300k,
     against an observed discounted return-to-go of about 0.13–0.25. The correlation is −0.09 to −0.35.
   - For 75k–125k the same comparison is roughly sensible: mean prediction 0.44–0.55 vs observed 0.28–0.48, correlation
     0.67–0.84.
   - The viewer shows a critic-divergence warning for the 250k, 275k and 300k checkpoints.
7. **No success gain.** Held-out success for the learned checkpoints ranges over 31–39/50 and ends at 32/50, against
   36/50 for the fixed baseline (logged). At 300k the extra replanning costs 2.7× the inference calls and 2.3× the modeled inference time per 1,000 steps.
   No underruns occur in any rollout.

## Hypotheses (not established by these data)

- The shift to immediate replanning may come from the actor exploiting critic noise rather than a real return
  difference: margins below the critics' disagreement (observation 5) fit this. Testing it would need same-state
  rollouts with many replicates for KEEP vs immediate REPLAN. The existing diagnostic `diag_v2_coarse` was
  ceiling-limited.
- The late critic divergence may involve rising α together with growing residual logits, and bootstrapping through
  very short decisions (D = 1 step for KEEP) at γ = 0.999. The logs show α rising to about 0.02–0.03 and the residual
  maximum rising together with the divergence. Cause has not been isolated.
- The outcome sensitivity in observation 1 suggests that per-seed differences between checkpoints are partly chaotic
  noise. Evaluating a checkpoint on the same seeds twice would quantify this before attributing per-seed differences
  to the scheduler.
