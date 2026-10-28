#!/bin/bash
# GPU smoke for both single-actor methods: prefill -> updates -> evals -> checkpoint -> resume; timings; report test.
set -e
rm -rf /N/scratch/jindjia/arli/smoke_single
for M in shared_every_step final_step_only; do
  A="--config configs/single/smoke.yaml --set name=smoke_$M method=$M"
  python -m arli.train_single $A --max_wall_hours 0.02 || echo "first run exit $?"
  python -m arli.train_single $A
done
python - <<'PY'
import json, glob, os
for d in sorted(glob.glob('/N/scratch/jindjia/arli/smoke_single/*')):
    ev=[json.loads(l) for l in open(f'{d}/evals.jsonl')]; ep=[json.loads(l) for l in open(f'{d}/episodes.jsonl')]
    on=[e for e in ep if e['phase']=='online']
    print(os.path.basename(d), 'evals', [(e['online_env_steps'], e['n_success'], e['grad_steps'], e['actor_calls_online']) for e in ev])
    print('  calls/chunk online', sorted({e['actor_calls']/e['chunks'] for e in on}), 'prefill', json.load(open(f'{d}/prefill_summary.json')))
    print('  segments', ep[-1]['segments'], 'wall', {k: round(ep[-1][k],1) for k in ('wall_prefill','wall_online','wall_update','wall_eval')})
PY
ls -la /N/scratch/jindjia/arli/smoke_single/*/replay | head
python scripts/report_single.py --reason "smoke test from compute node" && echo REPORT_OK
echo SMOKE_SINGLE_DONE
