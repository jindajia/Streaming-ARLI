"""SCH-01 diagnostic: per-snapshot R x N heatmaps of branch success (mean over paired replicates) and discounted return.

    python scripts/plot_diagnostic.py --run diag_coarse_v1 [--out results/diagnostic]
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

ROOT = "/N/scratch/jindjia/arli/runs_scheduling/diagnostic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="diag_coarse_v1")
    ap.add_argument("--out", default="results/diagnostic")
    args = ap.parse_args()
    rd, out = f"{ROOT}/{args.run}", f"{args.out}/{args.run}"
    os.makedirs(out, exist_ok=True)
    rows = [json.loads(l) for l in open(f"{rd}/branches.jsonl")]
    snaps = json.load(open(f"{rd}/snapshots.json"))
    agg = defaultdict(list)
    for r in rows:
        agg[(r["snap_id"], r["branch"])].append(r)
    with open(f"{out}/branch_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snap_id", "t_snap", "branch", "R", "N", "n_reps", "successes", "mean_discounted_return",
                    "mean_ticks_to_end", "mean_first_delay"])
        for (sid, b), rs in sorted(agg.items()):
            d = [x["first_request"]["d_actual"] for x in rs if x["first_request"]]
            w.writerow([sid, rs[0]["t_snap"], b, rs[0]["R"], rs[0]["N"], len(rs), sum(x["success"] for x in rs),
                        np.mean([x["discounted_return"] for x in rs]), np.mean([x["ticks"] for x in rs]),
                        np.mean(d) if d else ""])
    with open(f"{out}/branches_raw.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    sids = sorted({r["snap_id"] for r in rows})
    Ns = sorted({r["N"] for r in rows if r["N"]})
    Rs = sorted({r["R"] for r in rows if r["R"]})
    fig, axes = plt.subplots(2, len(sids), figsize=(3.1 * len(sids), 6.4), squeeze=False)
    for col, sid in enumerate(sids):
        t_snap = next(r["t_snap"] for r in rows if r["snap_id"] == sid)
        for row, (key, label, vmin, vmax) in enumerate((("success", "success rate", 0, 1),
                                                        ("discounted_return", "discounted return", None, None))):
            M = np.full((len(Ns), len(Rs)), np.nan)
            for i, n in enumerate(Ns):
                for j, r in enumerate(Rs):
                    rs = agg.get((sid, f"R{r}_N{n}"))
                    if rs:
                        M[i, j] = np.mean([float(x[key]) for x in rs])
            ax = axes[row, col]
            im = ax.imshow(M, origin="lower", cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(Rs)), Rs, fontsize=7)
            ax.set_yticks(range(len(Ns)), Ns, fontsize=7)
            keep = agg.get((sid, "KEEP1"))
            ktxt = f" · KEEP1 {np.mean([float(x[key]) for x in keep]):.2f}" if keep else ""
            ax.set_title(f"snap {sid} (t={t_snap}) {label}{ktxt}", fontsize=7.5)
            for i in range(len(Ns)):
                for j in range(len(Rs)):
                    if not np.isnan(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                                color="white" if (vmax and M[i, j] > 0.6) else "black")
            if col == 0:
                ax.set_ylabel("N (denoising steps)", fontsize=8)
            if row == 1:
                ax.set_xlabel("R (handoff ticks after decision)", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    reps = max(r["rep"] for r in rows) + 1
    fig.suptitle(f"SCH-01 same-state R/N branching diagnostic ({args.run}): {len(rows)} rollouts, {reps} paired "
                 f"replicates per cell, continuation = fixed reference scheduler. Blank = infeasible (R < budget(N)). "
                 f"Diagnostic only, not training data.", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{out}/heatmaps.png", dpi=140)
    with open(f"{out}/snapshots.json", "w") as f:
        json.dump(snaps, f, indent=1)
    print(f"wrote {out}: {len(rows)} rollouts, {len(sids)} snapshots")


if __name__ == "__main__":
    main()
