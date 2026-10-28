#!/bin/bash
echo "SUPERSEDED (2026-10-28): the two-actor ARLI-checkpoint warm-start S1 design is retired; see results/superseded_warmstart_s1/README.md" >&2; exit 1
# Launch one 1-h segment of the S1 branches from the common R2 ARLI checkpoint (the 600k evaluation point).
# Usage: bash scripts/launch_s1.sh "A C" [dependency_jobid]   -> one 2-GPU gpu-debug job running both branches
set -e
cd /N/slate/jindjia/SD/VLA/streaming_arli
R=/N/scratch/jindjia/arli/runs/r2_arli_seed0
CK=$(/N/scratch/jindjia/envs/arli/bin/python - <<'PY'
import json
ev = [json.loads(l) for l in open("/N/scratch/jindjia/arli/runs/r2_arli_seed0/evals.jsonl")]
e = [x for x in ev if x["milestone"] == 600000][-1]
print(f"/N/scratch/jindjia/arli/runs/r2_arli_seed0/ckpt/agent_{e['online_env_steps']:07d}.msgpack")
PY
)
test -f "$CK"
CMDS=""
for B in $1; do CMDS+="python -m arli.train_continue --config configs/s1_${B}_seed0${SUFFIX:-}.yaml --set init_agent=$CK --max_wall_hours 0.75;"; done
export CMDS="${CMDS%;}"
N=$(echo $1 | wc -w)
DEP=${2:+--dependency=afterany:$2}
sbatch --parsable -J s1_$(echo $1 | tr -d " ")${SUFFIX:-}_seg -p gpu-debug --gpus-per-node=$N --cpus-per-task=$((12*N)) --mem=$((90*N))g --time=00:59:00 $DEP cluster/multi.sbatch
