# Single-actor comparison — status

_Auto-generated 2026-10-28 10:14 EDT by `scripts/report_single.py` (smoke test from compute node)._

Methods train from RL step 0 with the frozen π0 backbone; ARLI curves are the R2 runs with the same protocol and prefill std 0.3. Real evaluations only.

| run | latest eval step | successes / 50 | prefill env steps | grad steps | online actor calls | wall h |
|---|---|---|---|---|---|---|
| `sa_shared_every_step_seed0` | — | — | — | — | — | — |
| `sa_final_step_only_seed0` | — | — | — | — | — | — |
| `sa_shared_every_step_seed1` | — | — | — | — | — | — |
| `sa_final_step_only_seed1` | — | — | — | — | — | — |
| `r2_arli_seed0` | 600,124 | 45 | 24,740 | 619,775 | — | — |
| `r2_arli_seed1_std030` | 250,034 | 42 | 24,975 | 255,625 | — | — |

## Slurm jobs (at report time)

| job | name | state | elapsed | dependency |
|---|---|---|---|---|
| 8758475 | sa_smoke | RUNNING | 11:14 | (null) |

Figure: `success_vs_env_steps.png` (vector: `.svg`); data: `evals.csv`, `raw/*.jsonl`.
Design, objective and limitations: `DECISIONS.md` (D19–D24).
