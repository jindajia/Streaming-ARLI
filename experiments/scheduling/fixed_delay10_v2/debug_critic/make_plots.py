"""Plots + compact tables from a critic_debug.py output directory (CPU).

    python debug_critic/make_plots.py --src <job dir> --dst debug_critic/results
"""
import argparse
import csv
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CK = [100247, 225337, 300155]
COL = {100247: "#2a63c4", 225337: "#1a9a63", 300155: "#d9632b"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    for f in ("status.json", "audit.json", "decomp_fixed_summary.json", "full_replay_summary.json", "pilot.json",
              "ablation.json", "fixed_sets.json"):
        if os.path.exists(f"{a.src}/{f}"):
            shutil.copy(f"{a.src}/{f}", f"{a.dst}/{f}")
    rows = list(csv.DictReader(open(f"{a.src}/decomp_fixed.csv")))
    # compact decomposition table: per ckpt x set, quantiles of each component
    with open(f"{a.dst}/decomp_quantiles.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ckpt", "set", "n", "quantity", "median", "p95", "p99", "max"])
        for c in CK:
            for s in sorted({r["set"] for r in rows}):
                R = [r for r in rows if int(r["ckpt"]) == c and r["set"] == s]
                for k in ("G", "fut", "ent", "y", "Q1", "Q2", "Qdis", "td1_sq", "H_next"):
                    v = np.array([float(r[k]) for r in R])
                    w.writerow([c, s, len(v), k, *[f"{x:.4g}" for x in (np.median(v), np.quantile(v, .95), np.quantile(v, .99), v.max())]])
    # largest-error fixed-set samples at 300k
    R = sorted([r for r in rows if int(r["ckpt"]) == 300155], key=lambda r: -float(r["td1_sq"]))[:25]
    with open(f"{a.dst}/top_td_300155_fixed.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(R[0].keys()))
        w.writeheader()
        w.writerows(R)

    # 1. full-replay Qmin distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for c in CK:
        p = f"{a.src}/full_replay_{c}.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p)
        qmin = np.minimum(z["Q1"], z["Q2"])
        td = (z["Q1"] - z["y"]) ** 2
        x = np.sort(np.abs(qmin))
        axes[0].plot(x, 1 - np.arange(len(x)) / len(x), color=COL[c], label=f"{c:,}")
        s = np.sort(td)[::-1]
        axes[1].plot(np.arange(1, len(s) + 1) / len(s), np.cumsum(s) / s.sum(), color=COL[c], label=f"{c:,}")
        axes[2].scatter(np.arange(len(qmin)), qmin, s=1, color=COL[c], alpha=0.4, label=f"{c:,}")
    axes[0].set(xscale="log", yscale="log", xlabel="|min(Q1,Q2)| at replay action", ylabel="fraction of transitions ≥ x",
                title="Q distribution over ALL replay transitions")
    axes[1].set(xscale="log", xlabel="fraction of transitions (largest TD² first)", ylabel="share of total TD²",
                title="Concentration of squared TD error")
    axes[2].set(yscale="symlog", xlabel="replay decision index (≈ training time)", ylabel="min(Q1,Q2)",
                title="Q by replay position")
    for ax in axes:
        ax.legend(title="checkpoint", fontsize=8)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{a.dst}/q_distribution_full_replay.png", dpi=130)
    plt.close(fig)

    # 2. target decomposition on the fixed ordinary set
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for i, c in enumerate(CK):
        R = [r for r in rows if int(r["ckpt"]) == c and r["set"] == "ordinary_all"]
        data = [np.array([float(r[k]) for r in R]) for k in ("G", "fut", "ent", "y", "Q1")]
        axes[i].boxplot(data, labels=["G", "γ^D·E[minQt]", "γ^D·αH", "y", "Q1"], whis=(1, 99), showfliers=True,
                        flierprops=dict(markersize=2))
        axes[i].set_yscale("symlog", linthresh=1e-3)
        axes[i].set_title(f"checkpoint {c:,}: target components (ordinary set, n={len(R)})", fontsize=9)
        axes[i].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{a.dst}/target_decomposition.png", dpi=130)
    plt.close(fig)

    # 3. pilot
    if os.path.exists(f"{a.src}/pilot.json"):
        P = json.load(open(f"{a.src}/pilot.json"))
        B = [b for b in P["branches"] if not b["duplicate_run"]]
        if B:
            snaps = [s["name"] for s in P["snapshots"]]
            fig, axes = plt.subplots(1, len(snaps), figsize=(6 * len(snaps), 3.6), squeeze=False)
            for j, sn in enumerate(snaps):
                ax = axes[0, j]
                cands = [c["name"] for c in P["snapshots"][j]["candidates"]]
                for k, cn in enumerate(cands):
                    bb = [b for b in B if b["snapshot"] == sn and b["candidate"] == cn]
                    ys = [b["steps_to_termination"] for b in bb]
                    ax.scatter([k] * len(ys), ys, c=["#1a9a63" if b["success"] else "#c43a3a" for b in bb], s=40, zorder=3)
                    for b in bb:
                        ax.annotate(f"{b['inference_calls']}", (k + 0.06, b["steps_to_termination"]), fontsize=7)
                lab = [f"{c['name']}\n{c['decoded']}" for c in P["snapshots"][j]["candidates"]]
                ax.set_xticks(range(len(cands)), lab, fontsize=7)
                ax.set_ylabel("steps from snapshot to termination")
                ax.set_title(f"{sn} (t={P['snapshots'][j]['t']}): green = success, red = timeout; label = inference calls", fontsize=8)
                ax.grid(alpha=.3)
            fig.tight_layout()
            fig.savefig(f"{a.dst}/pilot_branches.png", dpi=130)
            plt.close(fig)

    # 4. ablation
    if os.path.exists(f"{a.src}/ablation.json"):
        A = json.load(open(f"{a.src}/ablation.json"))
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
        for name, br in A["branches"].items():
            tr = br["trace"]
            u = [t["update"] for t in tr]
            for ax, k in zip(axes, ("loss", "q", "y", "gnorm")):
                ax.plot(u, [t[k] for t in tr], label=name, lw=1)
        for ax, k in zip(axes, ("critic loss", "mean Q(s,a) of batch", "mean target y", "gradient global norm")):
            ax.set_title(k, fontsize=9)
            ax.set_xlabel("critic update")
            ax.grid(alpha=.3)
        axes[0].set_yscale("log")
        axes[3].set_yscale("log")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"Critic-only ablation from checkpoint {A['checkpoint']:,} (actor, α, replay rows and batch sequence frozen)", fontsize=9)
        fig.tight_layout()
        fig.savefig(f"{a.dst}/ablation_traces.png", dpi=130)
        plt.close(fig)
    print("wrote", a.dst, sorted(os.listdir(a.dst)))


if __name__ == "__main__":
    main()
