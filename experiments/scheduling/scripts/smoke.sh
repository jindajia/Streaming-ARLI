#!/bin/bash
# GPU smoke: real-stack checks, then a short online run with a deterministic mid-run stop and resume.
set -e
rm -rf /N/scratch/jindjia/arli/runs_scheduling/smoke
python scripts/gpu_checks.py
python -m sch01.train --config configs/smoke.yaml --stop_after_episodes 3 || echo "first segment exit $?"
python -m sch01.train --config configs/smoke.yaml
python - <<'PY'
import json
d='/N/scratch/jindjia/arli/runs_scheduling/smoke/smoke'
for f in ('reference_eval','evals'):
    for l in open(f'{d}/{f}.jsonl'):
        e=json.loads(l); print(f, e['tag'], e['train_steps'], f"{e['n_success']}/{e['n_episodes']}", 'parity', e['reference_parity'], 'mean_N', e['mean_N'])
ep=[json.loads(l) for l in open(f'{d}/episodes.jsonl')]
print('train episodes', [(e['steps'], e['success'], e['departures'], e['grad_steps'], e['replay']) for e in ep])
up=[json.loads(l) for l in open(f'{d}/updates.jsonl')]
print('updates', [(u['grad_steps'], u['actor_on'], round(u['critic_loss'],4), round(u['entropy'],3), round(u['ref_prob'],3)) for u in up])
PY
echo SMOKE_DONE
