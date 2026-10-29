# MANIFEST — fixed_delay10_v2

| Item | Location |
|---|---|
| Frozen VLA | `/N/scratch/jindjia/arli/checkpoints/pi0_aloha_sim` (read-only; norm-stats sha256 in every launch json) |
| Online runs | `/N/scratch/jindjia/arli/runs_scheduling/fixed_delay10_v2/online/<run_id>/` (state_latest.pkl, replay/ parts, ckpt/ incl. milestone_0150000, logs) |
| Diagnostic | `/N/scratch/jindjia/arli/runs_scheduling/fixed_delay10_v2/diagnostic/<run_id>/` (snapshots.pkl, branches.jsonl) — never read by training |
| Smoke | `/N/scratch/jindjia/arli/runs_scheduling/fixed_delay10_v2/smoke/` |
| Slurm logs | `/N/scratch/jindjia/arli/runs_scheduling/fixed_delay10_v2/logs/<job name>_<job id>.out` |
| Legacy (unchanged) | see `../LEGACY_BUDGET7.md` |
