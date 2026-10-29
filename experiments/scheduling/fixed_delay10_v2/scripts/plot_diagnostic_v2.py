"""v2 diagnostic heatmaps: per snapshot (rows = chunk/p), success rate per (N, R) and improvement over the SAME
snapshot's fixed-baseline branch.  Infeasible cells are masked (hatched), not failures; each cell is annotated with
successes/replicates.  Improvement panels share one diverging colour scale [-1, 1].

    python scripts/plot_diagnostic_v2.py --run diag_v2_coarse
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = "/N/scratch/jindjia/arli/runs_scheduling/fixed_delay10_v2/diagnostic"
DELAY = {3: 6, 4: 7, 5: 7, 6: 8, 7: 9, 8: 9, 9: 10, 10: 10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="diag_v2_coarse")
    ap.add_argument("--out", default="results/diagnostic")
    args = ap.parse_args()
    rd, out = f"{ROOT}/{args.run}", f"{args.out}/{args.run}"
    os.makedirs(out, exist_ok=True)
    rows = [json.loads(l) for l in open(f"{rd}/branches.jsonl")]
    meta = json.load(open(f"{rd}/snapshots.json"))
    agg = defaultdict(list)
    for r in rows:
        agg[(r["snap_id"], r["branch"])].append(r)
    with open(f"{out}/branch_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snap_id", "seed", "chunk", "p", "q", "t_snap", "branch", "R", "N", "n_reps", "successes",
                    "success_rate", "baseline_success_rate", "improvement", "mean_discounted_return", "obs_t", "ready_t",
                    "handoff_t", "old_executed", "discarded", "obs_age_at_handoff"])
        for (sid, b), rs in sorted(agg.items()):
            base = agg.get((sid, "baseline"), [])
            br = np.mean([x["success"] for x in base]) if base else np.nan
            sr = np.mean([x["success"] for x in rs])
            fr = rs[0]["first_request"] or {}
            w.writerow([sid, rs[0]["seed"], rs[0]["chunk"], rs[0]["p"], rs[0]["q"], rs[0]["t_snap"], b, rs[0]["R"],
                        rs[0]["N"], len(rs), sum(x["success"] for x in rs), sr, br, sr - br,
                        np.mean([x["discounted_return"] for x in rs]), fr.get("obs_t"), fr.get("ready_t"),
                        fr.get("handoff_t"), fr.get("old_executed"), fr.get("discarded"), fr.get("obs_age_at_handoff")])
    with open(f"{out}/branches_raw.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump(meta, open(f"{out}/snapshots.json", "w"), indent=1)

    snaps = sorted({(r["snap_id"], r["seed"], r["chunk"], r["p"], r["t_snap"], r["q"]) for r in rows})
    Ns, Rs = [3, 5, 7, 10], [6, 7, 9, 10, 15, 20, 25]
    n = len(snaps)
    fig, axes = plt.subplots(n, 2, figsize=(10.5, 2.25 * n + 1.2), squeeze=False)
    for i, (sid, seed, chunk, p, t, q) in enumerate(snaps):
        base = agg.get((sid, "baseline"), [])
        bk, bn = sum(x["success"] for x in base), len(base)
        br = bk / bn if bn else np.nan
        for j, kind in enumerate(("success", "improvement")):
            M = np.full((len(Ns), len(Rs)), np.nan)
            ann = {}
            for a, N in enumerate(Ns):
                for b, R in enumerate(Rs):
                    rs = agg.get((sid, f"R{R}_N{N}"))
                    if rs:
                        k = sum(x["success"] for x in rs)
                        sr = k / len(rs)
                        M[a, b] = sr if kind == "success" else sr - br
                        ann[(a, b)] = f"{k}/{len(rs)}"
            ax = axes[i, j]
            if kind == "success":
                im = ax.imshow(M, origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto")
            else:
                im = ax.imshow(M, origin="lower", cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
            for a, N in enumerate(Ns):
                for b, R in enumerate(Rs):
                    if (a, b) in ann:
                        txt = ann[(a, b)] if kind == "success" else f"{M[a, b]:+.1f}"
                        ax.text(b, a, txt, ha="center", va="center", fontsize=6.5,
                                color="white" if (kind == "success" and M[a, b] > 0.6) else "black")
                    else:  # infeasible (R < d(N) or R > q): masked, not failed
                        ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, hatch="///", lw=0,
                                                   ec="#b5b4ae"))
            ax.set_xticks(range(len(Rs)), Rs, fontsize=7)
            ax.set_yticks(range(len(Ns)), [f"{N} (d={DELAY[N]})" for N in Ns], fontsize=7)
            ttl = (f"seed {seed} chunk {chunk} p={p} q={q} (t={t}) · success" if kind == "success" else
                   f"improvement vs fixed baseline ({bk}/{bn} success)")
            ax.set_title(ttl, fontsize=7.5, loc="left")
            if i == n - 1:
                ax.set_xlabel("R = old actions executed before handoff", fontsize=8)
            if j == 0:
                ax.set_ylabel("N (delay)", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    reps = max(r["rep"] for r in rows) + 1
    fig.suptitle(f"SCH-01 fixed_delay10_v2 same-state diagnostic ({args.run}): first decision varied, then the fixed "
                 f"baseline; {reps} paired replicates per cell (successes/replicates). Hatched = infeasible. "
                 f"Baseline = continue the fixed schedule (observe at p=15), NOT an immediate REPLAN(10,10). "
                 f"Analysis only.", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{out}/heatmaps.png", dpi=140)
    fig.savefig(f"{out}/heatmaps.svg")
    print(f"wrote {out}: {len(rows)} rollouts, {n} snapshots")


if __name__ == "__main__":
    main()
