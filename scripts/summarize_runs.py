"""Per-run prefill statistics and learning-dynamics windows (R2 runs).

For every run dir with episodes.jsonl: prefill episodes / successes / rate / transitions / env steps, and per 25k
online-env-step window: online (exploration) success rate, mean max reward, and mean actor entropy / temperature /
Q from updates.jsonl.  Used to examine whether DSRL's failure tracks its initial data or its later learning
dynamics (a hypothesis, not an established cause).

    python scripts/summarize_runs.py  -> results/r2/prefill_summary.csv, results/r2/dynamics_<run>.csv
"""
import csv
import glob
import json
import os

import numpy as np

RUNS = os.environ.get("ARLI_RUNS", "/N/scratch/jindjia/arli/runs")
OUT = "results/r2"
W = 25_000
os.makedirs(OUT, exist_ok=True)
pre_rows = []
for d in sorted(glob.glob(f"{RUNS}/r2_*")):
    name = os.path.basename(d)
    p = f"{d}/episodes.jsonl"
    if not os.path.exists(p):
        continue
    eps = [json.loads(l) for l in open(p)]
    cfg = json.load(open(sorted(glob.glob(f"{d}/launch_*.json"))[0]))["config"]
    pre = [e for e in eps if e["phase"] == "prefill"]
    on = {}
    for e in eps:  # dedupe episodes re-logged after a resume: keep the last record per online step count
        if e["phase"] == "online":
            on[e["online_env_steps"]] = e
    on = [on[k] for k in sorted(on)]
    if pre:
        n_s = sum(e["success"] for e in pre)
        pre_rows.append(dict(run=name, method=cfg["method"], seed=cfg["seed"], prefill_std=cfg["prefill_std"],
                             prefill_episodes=len(pre), prefill_successes=n_s, prefill_success_rate=n_s / len(pre),
                             prefill_transitions=pre[-1]["prefill_transitions"], prefill_env_steps=pre[-1]["prefill_env_steps"],
                             online_episodes=len(on), online_env_steps=on[-1]["online_env_steps"] if on else 0))
    upd = {}
    if os.path.exists(f"{d}/updates.jsonl"):
        for u in map(json.loads, open(f"{d}/updates.jsonl")):
            upd[u["grad_steps"]] = u
    upd = [upd[k] for k in sorted(upd)]
    rows = []
    for w0 in range(0, (on[-1]["online_env_steps"] if on else 0) + 1, W):
        ew = [e for e in on if w0 < e["online_env_steps"] <= w0 + W]
        uw = [u for u in upd if w0 < u["online_env_steps"] <= w0 + W]
        if not ew:
            continue
        rows.append(dict(window_start=w0, window_end=w0 + W, online_episodes=len(ew),
                         online_success_rate=float(np.mean([e["success"] for e in ew])),
                         mean_max_reward=float(np.mean([e["max_reward"] for e in ew])),
                         entropy=float(np.mean([u["entropy"] for u in uw])) if uw else None,
                         temperature=float(np.mean([u["temperature"] for u in uw])) if uw else None,
                         q=float(np.mean([u["q"] for u in uw])) if uw else None,
                         critic_loss=float(np.mean([u["critic_loss"] for u in uw])) if uw else None))
    if rows:
        with open(f"{OUT}/dynamics_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
if pre_rows:
    with open(f"{OUT}/prefill_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pre_rows[0]))
        w.writeheader()
        w.writerows(pre_rows)
    for r in pre_rows:
        print(r)
