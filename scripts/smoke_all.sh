#!/bin/bash
# R1 smoke: short ARLI/DSRL training + resume, S1 branch smoke, frozen baselines.  One gpu-debug job.
set -e
SM=/N/scratch/jindjia/arli/smoke
rm -rf $SM; mkdir -p $SM
echo "== smoke ARLI (prefill -> updates -> evals at 0/500/1000 -> checkpoint)"
python -m arli.train --config configs/smoke_arli.yaml --set run_root=$SM
echo "== resume ARLI to 1500 online steps"
python -m arli.train --config configs/smoke_arli.yaml --set run_root=$SM online_env_steps=1500
echo "== smoke DSRL"
python -m arli.train --config configs/smoke_dsrl.yaml --set run_root=$SM
CK=$(ls $SM/smoke_arli/ckpt/agent_*.msgpack | tail -1)
for B in A B C D; do
  echo "== S1 smoke branch $B from $CK"
  python -m arli.train_continue --config configs/s1_A_seed0.yaml --set name=smoke_s1_$B branch=$B run_root=$SM \
     init_agent=$CK warmup_chunks=4 online_env_steps=600 eval_interval=600 eval_episodes=2 diag_episodes=1
done
echo "== frozen pi0, async d=20, 50 eval seeds"
python scripts/eval_frozen.py --episodes 50 --out results/r1/frozen_async_d20.jsonl
echo "== frozen pi0, synchronous zero-delay diagnostic, 50 eval seeds"
python scripts/eval_frozen.py --episodes 50 --sync --out results/r1/frozen_sync_d0.jsonl
echo SMOKE_ALL_DONE
