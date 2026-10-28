"""Print progress of all runs under /N/scratch/jindjia/arli/runs (evals and latest counters)."""
import glob
import json
import os

for d in sorted(glob.glob("/N/scratch/jindjia/arli/runs/*")):
    name = os.path.basename(d)
    ev = []
    if os.path.exists(f"{d}/evals.jsonl"):
        ev = list({e.get("milestone"): e for e in map(json.loads, open(f"{d}/evals.jsonl"))}.values())
    last = None
    if os.path.exists(f"{d}/episodes.jsonl"):
        with open(f"{d}/episodes.jsonl") as f:
            for line in f:
                last = line
    c = json.loads(last) if last else {}
    pts = " ".join(f"{e['online_env_steps'] // 1000}k:{e['n_success']}/{e['n_episodes']}" for e in ev)
    print(f"{name:24s} online={c.get('online_env_steps', 0):>7} grad={c.get('grad_steps', c.get('grad_iters', 0)):>7} | {pts}")
