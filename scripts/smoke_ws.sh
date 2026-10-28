#!/bin/bash
# GPU smoke: real zero-correction equivalence; short training + checkpoint + resume for both branches.
set -e
rm -rf /N/scratch/jindjia/arli/smoke_ws
[ -n "$SKIP_CHECK2" ] || python scripts/check_warmstart_gpu.py
for B in A B; do
  A="--config configs/warmstart/smoke.yaml --set name=smoke_$B branch=$B"
  python -m arli.train_warmstart $A --max_wall_hours 0.03 || echo "first run exit $?"
  python -m arli.train_warmstart $A
done
python - <<'PY'
import json, glob, os, pickle
for d in sorted(glob.glob('/N/scratch/jindjia/arli/smoke_ws/smoke_*')):
    ev=[json.loads(l) for l in open(f'{d}/evals.jsonl')]; ep=[json.loads(l) for l in open(f'{d}/episodes.jsonl')]
    print(os.path.basename(d), 'evals', [(e['additional_env_steps'], e['n_success'], e['grad_steps'], e['arli_actor_hash_ok']) for e in ev])
    print('  phases', [e['phase'][0] for e in ep], 'segments', ep[-1]['segments'], 'chunks', ep[-1]['chunks'], 'grad', ep[-1]['grad_steps'])
    pol=sorted(glob.glob(f'{d}/ckpt/policy_*.pkl'))[-1]; P=pickle.load(open(pol,'rb'))
    print('  policy snapshot keys', sorted(P), 'prov sha', P['provenance']['sha256'][:16])
PY
echo SMOKE_WS_DONE
