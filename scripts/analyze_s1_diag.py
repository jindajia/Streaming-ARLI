"""Aggregate S1 diagnostics (bounded subset of eval episodes).

Per branch and eval point: mean |u| and |X~ - X| per expert step k, fraction of steering calls that saw a genuinely
newer frame, observation age at each call, and the same-snapshot probes (max |Δaction| between the all-corrections
chunk and chunks with corrections kept only on subsets of steps) -> do late corrections erase earlier ones, and
does last-only match all-steps?

    python scripts/analyze_s1_diag.py  -> results/s1/diag_summary.json
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

RUNS = os.environ.get("ARLI_RUNS", "/N/scratch/jindjia/arli/runs")
PAT = os.environ.get("ARLI_S1_PATTERN", "s1_{b}_seed0")
out = {}
for b in "ABCD":
    files = sorted(glob.glob(f"{RUNS}/{PAT.format(b=b)}/diag/*.npz"))
    by_tag = defaultdict(list)
    for f in files:
        by_tag[re.search(r"(eval\d+)_ep", f).group(1)].append(f)
    res_b = {}
    for tag, fs in by_tag.items():
        u_k, dx_k, age, fresh, probes, cols = defaultdict(list), defaultdict(list), [], [], [], None
        for f in fs:
            d = np.load(f)
            for key in d.files:
                m = re.match(r"j(\d+)_steer_k_tobs_tnow_fresh", key)
                if not m:
                    continue
                j = m.group(1)
                info = d[key]
                us = d[f"j{j}_u"]
                for row, u in zip(info, us):
                    k, t_obs, t_now, fr = row
                    u_k[int(k)].append(float(np.linalg.norm(u)))
                    age.append(float(t_now - t_obs))
                    fresh.append(bool(fr))
                if f"j{j}_x_in" in d.files:
                    dx = np.linalg.norm((d[f"j{j}_x_tilde"] - d[f"j{j}_x_in"]).reshape(10, -1), axis=1)
                    for k, v in enumerate(dx):
                        dx_k[k].append(float(v))
            if "probe_maxabs_vs_full" in d.files:
                probes.append(d["probe_maxabs_vs_full"][:, 1:])
                cols = [str(c) for c in d["probe_cols"][1:]]
        r = dict(n_episodes=len(fs),
                 u_norm_by_k={k: float(np.mean(v)) for k, v in sorted(u_k.items())},
                 dx_norm_by_k={k: float(np.mean(v)) for k, v in sorted(dx_k.items())},
                 fresh_frame_frac=float(np.mean(fresh)) if fresh else None,
                 obs_age_mean_steps=float(np.mean(age)) if age else None)
        if probes:
            P = np.concatenate(probes)
            r["probe_maxabs_vs_full_mean"] = dict(zip(cols, map(float, P.mean(0))))
            r["probe_n_chunks"] = int(P.shape[0])
        res_b[tag] = r
    if res_b:
        out[b] = res_b
OUT = os.environ.get("ARLI_DIAG_OUT", "results/s1/diag_summary.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1)[:4000])
