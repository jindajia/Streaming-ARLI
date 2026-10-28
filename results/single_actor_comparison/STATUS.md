# Single-actor comparison — status

_Auto-generated 2026-10-28 11:37 EDT by `scripts/report_single.py` (sa_final_step_only_seed1 milestone 100000)._

Methods train from RL step 0 with the frozen π0 backbone; ARLI curves are the R2 runs with the same protocol and prefill std 0.3. Real evaluations only.

| run | latest eval step | successes / 50 | prefill env steps | grad steps | online actor calls | wall h |
|---|---|---|---|---|---|---|
| `sa_shared_every_step_seed0` | 75,200 | 0 | 24,909 | 75,200 | 31960 | 0.9 |
| `sa_final_step_only_seed0` | 100,241 | 34 | 24,793 | 102,325 | 4254 | 1.0 |
| `sa_shared_every_step_seed1` | 75,041 | 0 | 24,847 | 75,075 | 31896 | 0.9 |
| `sa_final_step_only_seed1` | 100,174 | 31 | 24,854 | 101,675 | 4248 | 1.0 |
| `r2_arli_seed0` | 600,124 | 45 | 24,740 | 619,775 | — | — |
| `r2_arli_seed1_std030` | 425,023 | 48 | 24,975 | 437,200 | — | — |

## Slurm jobs (at report time)

| job | name | state | elapsed | dependency |
|---|---|---|---|---|
| 8758542 | sa_final_step_only_s0_seg13 | PENDING | 0:00 | afterany:8758541(unfulfilled) |
| 8758541 | sa_final_step_only_s0_seg12 | PENDING | 0:00 | afterany:8758540(unfulfilled) |
| 8758540 | sa_final_step_only_s0_seg11 | PENDING | 0:00 | afterany:8758539(unfulfilled) |
| 8758539 | sa_final_step_only_s0_seg10 | PENDING | 0:00 | afterany:8758538(unfulfilled) |
| 8758538 | sa_final_step_only_s0_seg09 | PENDING | 0:00 | afterany:8758537(unfulfilled) |
| 8758537 | sa_final_step_only_s0_seg08 | PENDING | 0:00 | afterany:8758536(unfulfilled) |
| 8758536 | sa_final_step_only_s0_seg07 | PENDING | 0:00 | afterany:8758535(unfulfilled) |
| 8758535 | sa_final_step_only_s0_seg06 | PENDING | 0:00 | afterany:8758534(unfulfilled) |
| 8758534 | sa_final_step_only_s0_seg05 | PENDING | 0:00 | afterany:8758533(unfulfilled) |
| 8758533 | sa_final_step_only_s0_seg04 | PENDING | 0:00 | afterany:8758532(unfulfilled) |
| 8758532 | sa_final_step_only_s0_seg03 | PENDING | 0:00 | afterany:8758531(unfulfilled) |
| 8758529 | sa_shared_every_step_s0_seg13 | PENDING | 0:00 | afterany:8758528(unfulfilled) |
| 8758528 | sa_shared_every_step_s0_seg12 | PENDING | 0:00 | afterany:8758527(unfulfilled) |
| 8758527 | sa_shared_every_step_s0_seg11 | PENDING | 0:00 | afterany:8758526(unfulfilled) |
| 8758526 | sa_shared_every_step_s0_seg10 | PENDING | 0:00 | afterany:8758525(unfulfilled) |
| 8758525 | sa_shared_every_step_s0_seg09 | PENDING | 0:00 | afterany:8758524(unfulfilled) |
| 8758524 | sa_shared_every_step_s0_seg08 | PENDING | 0:00 | afterany:8758523(unfulfilled) |
| 8758523 | sa_shared_every_step_s0_seg07 | PENDING | 0:00 | afterany:8758522(unfulfilled) |
| 8758522 | sa_shared_every_step_s0_seg06 | PENDING | 0:00 | afterany:8758521(unfulfilled) |
| 8758521 | sa_shared_every_step_s0_seg05 | PENDING | 0:00 | afterany:8758520(unfulfilled) |
| 8758520 | sa_shared_every_step_s0_seg04 | PENDING | 0:00 | afterany:8758519(unfulfilled) |
| 8758519 | sa_shared_every_step_s0_seg03 | PENDING | 0:00 | afterany:8758518(unfulfilled) |
| 8758568 | sa_final_step_only_s1_seg13 | PENDING | 0:00 | afterany:8758567(unfulfilled) |
| 8758567 | sa_final_step_only_s1_seg12 | PENDING | 0:00 | afterany:8758566(unfulfilled) |
| 8758566 | sa_final_step_only_s1_seg11 | PENDING | 0:00 | afterany:8758565(unfulfilled) |
| 8758565 | sa_final_step_only_s1_seg10 | PENDING | 0:00 | afterany:8758564(unfulfilled) |
| 8758564 | sa_final_step_only_s1_seg09 | PENDING | 0:00 | afterany:8758563(unfulfilled) |
| 8758563 | sa_final_step_only_s1_seg08 | PENDING | 0:00 | afterany:8758562(unfulfilled) |
| 8758562 | sa_final_step_only_s1_seg07 | PENDING | 0:00 | afterany:8758561(unfulfilled) |
| 8758561 | sa_final_step_only_s1_seg06 | PENDING | 0:00 | afterany:8758560(unfulfilled) |
| 8758560 | sa_final_step_only_s1_seg05 | PENDING | 0:00 | afterany:8758559(unfulfilled) |
| 8758559 | sa_final_step_only_s1_seg04 | PENDING | 0:00 | afterany:8758558(unfulfilled) |
| 8758558 | sa_final_step_only_s1_seg03 | PENDING | 0:00 | afterany:8758557(unfulfilled) |
| 8758555 | sa_shared_every_step_s1_seg13 | PENDING | 0:00 | afterany:8758554(unfulfilled) |
| 8758554 | sa_shared_every_step_s1_seg12 | PENDING | 0:00 | afterany:8758553(unfulfilled) |
| 8758553 | sa_shared_every_step_s1_seg11 | PENDING | 0:00 | afterany:8758552(unfulfilled) |
| 8758552 | sa_shared_every_step_s1_seg10 | PENDING | 0:00 | afterany:8758551(unfulfilled) |
| 8758551 | sa_shared_every_step_s1_seg09 | PENDING | 0:00 | afterany:8758550(unfulfilled) |
| 8758550 | sa_shared_every_step_s1_seg08 | PENDING | 0:00 | afterany:8758549(unfulfilled) |
| 8758549 | sa_shared_every_step_s1_seg07 | PENDING | 0:00 | afterany:8758548(unfulfilled) |
| 8758548 | sa_shared_every_step_s1_seg06 | PENDING | 0:00 | afterany:8758547(unfulfilled) |
| 8758547 | sa_shared_every_step_s1_seg05 | PENDING | 0:00 | afterany:8758546(unfulfilled) |
| 8758546 | sa_shared_every_step_s1_seg04 | PENDING | 0:00 | afterany:8758545(unfulfilled) |
| 8758545 | sa_shared_every_step_s1_seg03 | PENDING | 0:00 | afterany:8758544(unfulfilled) |
| 8758518 | sa_shared_every_step_s0_seg02 | RUNNING | 24:01 | (null) |
| 8758531 | sa_final_step_only_s0_seg02 | RUNNING | 24:01 | (null) |
| 8758557 | sa_final_step_only_s1_seg02 | RUNNING | 19:00 | (null) |
| 8758544 | sa_shared_every_step_s1_seg02 | RUNNING | 24:01 | (null) |

Figure: `success_vs_env_steps.png` (vector: `.svg`); data: `evals.csv`, `raw/*.jsonl`.
Design, objective and limitations: `DECISIONS.md` (D19–D24).
