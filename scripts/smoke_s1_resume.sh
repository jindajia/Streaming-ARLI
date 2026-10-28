#!/bin/bash
# S1 resume test: branch C from the smoke ARLI checkpoint; first run exits on a tiny wall budget, second resumes.
set -e
SM=/N/scratch/jindjia/arli/smoke; rm -rf $SM/smoke_s1r_C
CK=$(ls $SM/smoke_arli/ckpt/agent_*.msgpack | tail -1)
A="--config configs/s1_C_seed0.yaml --set name=smoke_s1r_C run_root=$SM init_agent=$CK warmup_chunks=4 online_env_steps=2400 eval_interval=1200 eval_episodes=2 diag_episodes=1"
python -m arli.train_continue $A --max_wall_hours 0.04 || echo "first run exit $?"
python -m arli.train_continue $A
python - <<'PY'
import json
L=[json.loads(l) for l in open('/N/scratch/jindjia/arli/smoke/smoke_s1r_C/episodes.jsonl')]
print('episodes', len(L), 'online', [d['online_env_steps'] for d in L])
print('evals', [(json.loads(l)['online_env_steps'], json.loads(l)['n_success']) for l in open('/N/scratch/jindjia/arli/smoke/smoke_s1r_C/evals.jsonl')])
PY
echo S1_RESUME_OK
