# SUPERSEDED — two-actor warm-start S1 experiments (stopped 2026-10-28 09:55 EDT)

These runs continued training from the **r2_arli_seed0 600k checkpoint** with a *separate* ARLI initial-noise actor
plus a second steering actor. The user retired this design on 2026-10-28: it does not represent the intended
methods. They are kept for provenance only and must not be mixed into the single-actor comparison
(`results/single_actor_comparison/`).

- v1 `s1_{B,C,D}_seed0`: "chain" credit; critics diverged (Q ≈ +28 with rewards ≤ 0) → invalid (DECISIONS D18).
- v2 `s1_{B,C,D}_seed0_v2`: chunk-level credit; stopped after its first segment.
- `s1_A_seed0`: continued ARLI from the 600k checkpoint (complete, +100k).
- seed-1 S1 (`s1_*_seed1`): never started (jobs cancelled while pending).

Stopped jobs: 8758246 (s1_B/D_seed0_v2 segment, running), 8758247 (s1_C_seed0_v2 segment, running), 8757633 and
8757634 (seed-1 S1 A+C / B+D, pending afterok:8757628). Launcher disabled: `scripts/superseded_launch_s1.sh`
(exits 1). Configs moved to `configs/superseded_warmstart_s1/`. Run dirs (checkpoints, replay, diagnostics) remain at
`/N/scratch/jindjia/arli/runs/s1_*`, each with a `SUPERSEDED.txt` marker.

Eval records (successes out of 50, by additional online env steps from the warm start; NOT from step 0):

| run | last step | evals |
|---|---|---|
| `s1_A_seed0` | 100,122 | 0k:47, 10k:22, 20k:27, 30k:33, 40k:18, 50k:30, 60k:33, 70k:35, 80k:36, 90k:37, 100k:38 |
| `s1_B_seed0` | 40,377 | 0k:15, 10k:0, 20k:1, 30k:0, 40k:1 |
| `s1_B_seed0_v2` | 30,391 | 0k:22, 10k:1, 20k:0, 30k:0 |
| `s1_C_seed0` | 90,020 | 0k:25, 10k:1, 20k:0, 30k:1, 40k:0, 50k:0, 60k:0, 70k:0, 80k:0, 90k:0 |
| `s1_C_seed0_v2` | 40,363 | 0k:19, 10k:0, 20k:0, 30k:0, 40k:0 |
| `s1_D_seed0` | 50,049 | 0k:47, 10k:31, 20k:20, 30k:26, 40k:27, 50k:33 |
| `s1_D_seed0_v2` | 40,255 | 0k:47, 10k:29, 20k:26, 30k:25, 40k:35 |
