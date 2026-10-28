# STATUS

_Last updated 2026-10-27 23:30 EDT._ All numbers below are measured; nothing is digitised from the paper.

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

| Job | Cluster / partition | What | State |
|---|---|---|---|
| 8755240 | BR200 gpu-debug | R1 check (attribute bug) | FAILED, fixed |
| 8755266 | BR200 gpu-debug | R1 check | COMPLETED |
| 8755293 | BR200 gpu-debug | smoke + frozen baselines | COMPLETED |
| 8755398/8755399, 8755409/8755410 | BR200 gpu | 12 h R2 jobs | cancelled unstarted (queue est. start 2026-10-29 23:45) |
| 8755407 | BR200 gpu-debug (A100) | R2 ARLI seg 1: 0 → 80k | done (clean wall-budget exit; sacct shows FAILED = exit 3) |
| 8755408 | BR200 gpu-debug (A100) | R2 DSRL seg 1: 0 → 87k | done (same) |
| 8755751 | BR200 gpu-debug (A100) | R2 ARLI seg 2 | running |
| 8756469/70/71 | BR200 gpu-debug (A100) | R2 ARLI seg 3–5 (afterany chain) | pending |
| 12499954 | Quartz h100-debug (H100) | R2 DSRL seg 2 from 87.2k | running |
| 12499956/57 | Quartz h100-debug | R2 DSRL seg 3–4 (chain) | pending |

Hardware note: DSRL seed 0 runs its first segment on A100 and later segments on H100 (each segment's host is in
`launch_*.json`).  Every segment resumes the exact replay/agent/RNG state; numerics differ only at bf16 level.

Resume / extend (any time): BR200 `export CMD="python -m arli.train --config configs/r2_arli_seed0.yaml --max_wall_hours 0.85"; sbatch -J r2_arli_s0_dbgN -p gpu-debug --time=00:59:00 cluster/run.sbatch`;
Quartz: `ssh quartz.uits.iu.edu 'cd /N/slate/jindjia/SD/VLA/streaming_arli && export CMD="python -m arli.train --config configs/r2_dsrl_seed0.yaml --max_wall_hours 0.85" && sbatch -J r2_dsrl_s0_qN --time=00:59:00 cluster/run_quartz.sbatch'`.
