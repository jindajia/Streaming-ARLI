# MANIFEST — sources, checkpoints and external storage

| Item | Location | Provenance |
|---|---|---|
| π0 ALOHA checkpoint | `/N/scratch/jindjia/arli/checkpoints/pi0_aloha_sim` | cited as `s3://openpi-assets/checkpoints/pi0_aloha_sim`; S3 bucket gone (NoSuchBucket, 2026-10-27); fetched from `gs://openpi-assets/checkpoints/pi0_aloha_sim` via public HTTPS, per-object MD5 verified; object list + md5 + GCS generation in `results/manifest/pi0_aloha_sim_objects.json` |
| DSRL code | `third_party/dsrl_pi0` submodule @ `7f48937d4553e95244cd81c79236a3256df80597` | github.com/nakamotoo/dsrl_pi0 |
| openpi (DSRL fork) | `third_party/dsrl_pi0/openpi` @ `a6d2400d2534ce32e7bdf8747709b97aaef8ec04` | github.com/nakamotoo/openpi |
| Python env | `/N/scratch/jindjia/envs/arli` (py3.11.11) | `setup/create_env.sh`, frozen list `setup/pip_freeze_arli.txt` |
| Run directories (replay, checkpoints, videos, raw logs) | `/N/scratch/jindjia/arli/runs/<run name>/` | `launch_*.json` in each holds config, git hash, Slurm job id |
| Slurm stdout | `/N/scratch/jindjia/arli/logs/<job name>_<job id>.out` | |
| R1 checks (video, trace) | `/N/scratch/jindjia/arli/r1_checks/` | `scripts/check_r1.py` |

Scratch is purged after 30 days without access; copy anything worth keeping before then.
