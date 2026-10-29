# MANIFEST — SCH-01 external artefacts

| Item | Location |
|---|---|
| Frozen policy | `/N/scratch/jindjia/arli/checkpoints/pi0_aloha_sim` (read-only; provenance in the repo-level MANIFEST.md); norm-stats sha256 logged in every launch json |
| RTC implementation | `src/sch01/rtc_vendored.py` = byte copy (sha256 5ee9014d…) of `streaming_arli_profiling/profiling/src/arli_prof/rtc.py` @ profiling commit 7e58596, verified against Physical-Intelligence/real-time-chunking-kinetix@9296f31 (max |diff| 6.8e-6) |
| Python env | `/N/scratch/jindjia/envs/arli` (shared with Streaming-ARLI, unchanged) |
| Online runs (replay parts, checkpoints, raw logs) | `/N/scratch/jindjia/arli/runs_scheduling/online/<run_id>/` |
| Diagnostic runs | `/N/scratch/jindjia/arli/runs_scheduling/diagnostic/<run_id>/` (kept separate; never loaded by training) |
| Smoke runs | `/N/scratch/jindjia/arli/runs_scheduling/smoke/` |
| Slurm logs | `/N/scratch/jindjia/arli/runs_scheduling/logs/<job name>_<job id>.out` |
