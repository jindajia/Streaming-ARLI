#!/bin/bash
# v2 GPU smoke: real-stack checks; short train interrupted mid-evaluation (wall budget) and resumed; episode-stop resume.
set -e
rm -rf /N/scratch/jindjia/arli/runs_scheduling/fixed_delay10_v2/smoke
python scripts/gpu_checks_v2.py
python -m sch02.train --config configs/smoke.yaml --max_wall_hours 0.025 || echo "segment 1 exit $?"
python -m sch02.train --config configs/smoke.yaml --stop_after_episodes 2 || echo "segment 2 exit $?"
python -m sch02.train --config configs/smoke.yaml
python - <<'PY'
import json, os
d='/N/scratch/jindjia/arli/runs_scheduling/fixed_delay10_v2/smoke/smoke'
for f in ('reference_eval','evals','evals_supplemental'):
    for l in open(f'{d}/{f}.jsonl'):
        e=json.loads(l); print(f, e['tag'], e['train_steps'], f"{e['n_success']}/{e['n_episodes']}", 'parity', e['reference_parity'], 'recovery', e['recovery_states'], 'prior', e['prior_strength'])
ep=[json.loads(l) for l in open(f'{d}/episodes.jsonl')]
print('train', [(e['steps'], e['grad_steps'], e['replay']) for e in ep])
print('milestone', os.listdir(f'{d}/ckpt'))
PY
echo SMOKE_V2_DONE
