# STATUS

_Last updated 2026-10-27 22:00 EDT._ All numbers below are measured; nothing is digitised from the paper.

## Stage summary

| Stage | State | Evidence |
|---|---|---|
| R0 source/timing audit | done | DECISIONS.md; no author ARLI code (paper-based reconstruction); `tests/test_schedule.py` passes; trace in `results/r1/trace_seed10000000.json` |
| R1 frozen baseline + smoke | done | `results/r1/` (see below) |
| R2 Figure 6 pilot (seed 0) | **running** | job table below; `python scripts/progress.py` |
| S0 steering implementation | implemented, smoke-tested | `src/arli/steering.py`; CPU test `tests/test_rl_cpu.py`; GPU smoke in `results/r1/smoke_summary.json` |
| S1 continuation (A–D) | waiting for the R2 ARLI checkpoint | configs `configs/s1_{A,B,C,D}_seed0.yaml` |

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

## Jobs

| Job | Partition | What | State |
|---|---|---|---|
| 8755240 | gpu-debug | R1 check (attribute bug) | FAILED, fixed |
| 8755266 | gpu-debug | R1 check | COMPLETED |
| 8755293 | gpu-debug | smoke + frozen baselines | COMPLETED |
| 8755398/8755399 | gpu | first R2 submissions | cancelled before start (resubmitted with matching wall budget) |
| 8755407 | gpu-debug | R2 ARLI seed 0, segment 1 (≤0.85 h, then checkpoint) | running |
| 8755408 | gpu-debug | R2 DSRL seed 0, segment 1 | running |
| 8755409 | gpu | R2 ARLI seed 0, resume (12 h) | pending (after 8755407) |
| 8755410 | gpu | R2 DSRL seed 0, resume (12 h) | pending (after 8755408) |

Resume command (any time): `export CMD="python -m arli.train --config configs/r2_arli_seed0.yaml --max_wall_hours 11.5"; sbatch -J r2_arli_s0 --time=12:00:00 cluster/run.sbatch`.
