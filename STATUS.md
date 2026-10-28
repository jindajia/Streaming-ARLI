# STATUS

_Last updated 2026-10-28 10:40 EDT._ All numbers are measured; nothing is digitised from the paper.

## Stage summary

| Stage | State | Evidence |
|---|---|---|
| R0 source/timing audit | done | DECISIONS.md; no author ARLI code (paper-based reconstruction); `tests/test_schedule.py` |
| R1 frozen baseline + smoke | done | `results/r1/` |
| R2 Figure-6 pilot, seed 0 | **complete**: ARLI 45/50 at 600k; DSRL 0/50 at all 25 evals | `results/r2/` |
| R2 seed-1 replication + prefill-variance control | running (jobs 8757628/30/31/32) | `configs/r2_*_seed1_std030.yaml`, `configs/r2_*_seed0_var030.yaml` |
| **Single-actor comparison** (shared_every_step vs final_step_only vs ARLI, from RL step 0, seeds 0 and 1) | **queued** — 4 chains × 13 one-hour segments on BR200 gpu (seed 1 at nice 200) | `results/single_actor_comparison/STATUS.md` (auto-updated at each 100k milestone), DECISIONS D19–D24 |
| Two-actor warm-start S1 | **superseded 2026-10-28, stopped** | `results/superseded_warmstart_s1/README.md` |

## R1 results (job 8755240/8755266 checks, 8755293 smoke + baselines)

- π0 `pi0_aloha_sim` loads (A100-40GB, 12 s). `infer` 71 ms = VLM prefix 44 ms + 10 × 3.2 ms expert steps. Sim 2.6 ms/step.
- Step-wise sampler vs openpi `while_loop` sampler, same obs + noise: max |Δa| 0.0029 / 0.0053 / 0.0026 rad
  (|a| ≤ 1.17); zero-residual hook path == step-wise path exactly. See DECISIONS D12c.
- **Frozen π0, async d=20 / d_rl=10, no RTC, 50 held-out seeds: 24/50 = 0.48** (95% Wilson 0.35–0.61).
  Paper's dashed line ≈ 0.37 (read off the figure, not used as data). Max-reward histogram [0..4]: 2/13/11/0/24.
- Frozen π0, synchronous zero-delay diagnostic: 33/50 = 0.66 (not the comparison line).
- Smoke: train → eval → checkpoint → resume (counters continue 1200 → 1600) for ARLI and DSRL; S1 branches A/B/C/D
  run end-to-end from a smoke checkpoint, with diagnostics + same-snapshot probes written.

## R2 throughput (measured, A100, first segment)

prefill 73 episodes / 24,740 env steps / 1,007 latent transitions (32 successful episodes); online ≈ 4.7 s rollout
+ 4.2 s updates (10.6 ms per gradient step) per 400 env steps; 50-episode eval ≈ 230 s → ≈ 5.5 GPU-h per run.

## Incident: Quartz node g37 renderer abort (2026-10-28 02:51)

DSRL seed 0 segments 12501064 / 12501065 / 12501875 and debug job 12501915 aborted (SIGABRT, core dump) inside
MuJoCo EGL `physics.render` 1–2 episodes after resuming at 524,000 online steps — all on Quartz g37, which had run
segment 12500083 cleanly. The identical state copied to `/N/scratch/jindjia/arli/debug_runs/` ran cleanly on a BR200
A100 (job 8757635: passed the crash point and the 525k eval). Not the run state, not the RL code; a first guess
(EGL device id) was tested and rejected. The real run state was never modified by the failed segments (state is only
written at evaluations / clean exits). DSRL seed 0 continues on BR200 (job 8757652).

## Jobs (current)

| Job | Cluster / partition | Run(s) | Dependency | State |
|---|---|---|---|---|
| 8757650 | BR200 gpu-debug, 2×A100 | S1 seed 0 branches A + C, segment 1 | — | running |
| 8757651 | BR200 gpu-debug, 2×A100 | S1 seed 0 branches B + D, segment 1 | — | running |
| 8757670 | BR200 gpu-debug, 2×A100 | S1 seed 0 A + C, segment 2 | afterany:8757650 | queued |
| 8757652 | BR200 gpu-debug, A100 | R2 DSRL seed 0, resume from 524k | — | queued (Resources) |
| 8757628 | BR200 gpu, 12 h, nice 500 | r2_arli_seed1_std030 | — | queued (Priority) |
| 8757630 | BR200 gpu, 12 h, nice 500 | r2_dsrl_seed1_std030 | — | queued (Priority) |
| 8757631 | BR200 gpu, 12 h, nice 500 | r2_arli_seed0_var030 | — | queued (Priority) |
| 8757632 | BR200 gpu, 12 h, nice 500 | r2_dsrl_seed0_var030 | — | queued (Priority) |
| 8757633 | BR200 gpu, 2×A100, 6 h, nice 500 | s1_A_seed1 + s1_C_seed1 | afterok:8757628 | queued (Dependency) |
| 8757634 | BR200 gpu, 2×A100, 6 h, nice 500 | s1_B_seed1 + s1_D_seed1 | afterok:8757628 | queued (Dependency) |

Completed R2 seed-0 ARLI segments: 8755407, 8755751, 8756469, 8756470, 8756471, 8756523 (BR200 gpu-debug; each
exits 3 = checkpointed wall budget, shown as FAILED by sacct), 8756898 (found the run complete).
DSRL seed 0 segments: 8755408 (BR200), 12499954/56/57, 12500083 (Quartz H100, clean), 12501064/65, 12501875 (g37 abort).

Hardware note: DSRL seed 0 runs its first segment on A100 and later segments on H100 (each segment's host is in
`launch_*.json`).  Every segment resumes the exact replay/agent/RNG state; numerics differ only at bf16 level.

Resume / extend (any time): BR200 `export CMD="python -m arli.train --config configs/r2_arli_seed0.yaml --max_wall_hours 0.85"; sbatch -J r2_arli_s0_dbgN -p gpu-debug --time=00:59:00 cluster/run.sbatch`;
Quartz: `ssh quartz.uits.iu.edu 'cd /N/slate/jindjia/SD/VLA/streaming_arli && export CMD="python -m arli.train --config configs/r2_dsrl_seed0.yaml --max_wall_hours 0.85" && sbatch -J r2_dsrl_s0_qN --time=00:59:00 cluster/run_quartz.sbatch'`.
