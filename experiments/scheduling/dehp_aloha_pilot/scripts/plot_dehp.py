"""DEHP pilot figures + compact results (copied from the scratch run dirs into results/).

    python scripts/plot_dehp.py
"""
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = "/N/scratch/jindjia/arli/runs_dehp"
OUT = "results"
RUNS = [("run_A_nodelay", "A: no delay, RTC off, h ∈ {1..50}"), ("run_B_delay10", "B: delay 10, RTC on, h ∈ {10..40}")]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, GRAY = "#2a78d6", "#52514e"


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8.5)


def main():
    os.makedirs(OUT, exist_ok=True)
    summ = {}
    for name, label in RUNS:
        d = f"{ROOT}/{name}"
        ev, it = jl(f"{d}/evals.jsonl"), jl(f"{d}/iters.jsonl")
        if not ev:
            continue
        os.makedirs(f"{OUT}/{name}", exist_ok=True)
        with open(f"{OUT}/{name}/evals.jsonl", "w") as f:  # compact: per-episode success/hs kept, no large arrays
            for e in ev:
                f.write(json.dumps(e) + "\n")
        shutil.copy(f"{d}/iters.jsonl", f"{OUT}/{name}/iters.jsonl") if it else None
        for fn in ("launch.json", "SUMMARY.json", "FAILED.json"):
            if os.path.exists(f"{d}/{fn}"):
                shutil.copy(f"{d}/{fn}", f"{OUT}/{name}/{fn}")
        base = [e for e in ev if e["kind"] == "baseline"]
        learned = [e for e in ev if e["policy"] == "learned"]
        summ[name] = dict(baseline=[(e["n_success"], e["n_episodes"]) for e in base],
                          learned=[(e["kind"], e["env_steps"], e["n_success"], e["n_episodes"], round(e["elapsed_min"], 1))
                                   for e in learned])
        fig, axes = plt.subplots(2, 3, figsize=(16, 8.4))
        fig.patch.set_facecolor(SURF)
        n_ep = learned[0]["n_episodes"] if learned else 10
        # (1) learning curve vs env steps
        ax = axes[0, 0]
        style(ax, f"held-out success vs online env steps (n = {n_ep} episodes per point)")
        if base:
            b = base[0]
            ax.axhline(b["n_success"] / b["n_episodes"], color=GRAY, ls=(0, (5, 4)), lw=1.5,
                       label=f"matched fixed h=25 baseline: {b['n_success']}/{b['n_episodes']}")
        ax.plot([e["env_steps"] for e in learned], [e["n_success"] / e["n_episodes"] for e in learned], "-o", color=BLUE,
                lw=2, label="learned horizon actor (argmax h)")
        for e in learned:
            ax.annotate(f"{e['n_success']}/{e['n_episodes']}" + (" final" if e["kind"] == "final" else
                                                                  (" init" if e["kind"] == "initial" else "")),
                        (e["env_steps"], e["n_success"] / e["n_episodes"]), xytext=(4, 6), textcoords="offset points",
                        fontsize=8, color=INK)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("online physical env steps (training)", color=INK2, fontsize=9)
        ax.set_ylabel("success rate", color=INK2, fontsize=9)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        # (2) vs elapsed GPU-job time
        ax = axes[0, 1]
        style(ax, "held-out success vs elapsed job time")
        if base:
            ax.axhline(base[0]["n_success"] / base[0]["n_episodes"], color=GRAY, ls=(0, (5, 4)), lw=1.5, label="baseline h=25")
        ax.plot([e["elapsed_min"] for e in learned], [e["n_success"] / e["n_episodes"] for e in learned], "-o", color=BLUE,
                lw=2, label="learned")
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("minutes since process start (incl. load/compile)", color=INK2, fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        # (3) horizon distribution evolution
        ax = axes[0, 2]
        style(ax, "selected-h distribution per PPO iteration (training samples)")
        if it:
            M = np.asarray([r["h_hist"] for r in it], float)
            M = M / np.maximum(M.sum(1, keepdims=True), 1)
            ax.imshow(M.T, aspect="auto", origin="lower", extent=(0.5, len(it) + 0.5, 0.5, 50.5), cmap="Blues")
            ax.plot(range(1, len(it) + 1), [r["h_mean"] for r in it], color="#eb6834", lw=1.5, label="mean h")
            ax.legend(fontsize=8, frameon=False)
        ax.set_xlabel("PPO iteration", color=INK2, fontsize=9)
        ax.set_ylabel("h", color=INK2, fontsize=9)
        # (4)-(6) diagnostics
        if it:
            x = [r["iter"] for r in it]
            ax = axes[1, 0]
            style(ax, "actor: entropy (nats) and train success per batch")
            ax.plot(x, [r["entropy"] for r in it], color=BLUE, lw=1.5, label="policy entropy")
            ax2 = ax.twinx()
            ax2.plot(x, [r["batch_success"] / max(1, r["batch_episodes_done"]) for r in it], color="#1baf7a", lw=1.2,
                     label="train success (episodes ended in batch)")
            ax2.set_ylim(0, 1)
            ax.legend(fontsize=7.5, frameon=False, loc="upper right")
            ax2.legend(fontsize=7.5, frameon=False, loc="lower right")
            ax = axes[1, 1]
            style(ax, "PPO: approx KL and clip fraction (last epoch)")
            ax.plot(x, [r["approx_kl"] for r in it], color=BLUE, lw=1.5, label="approx KL")
            ax2 = ax.twinx()
            ax2.plot(x, [r["clipfrac"] for r in it], color="#eb6834", lw=1.2, label="clip fraction")
            ax.legend(fontsize=7.5, frameon=False, loc="upper left")
            ax2.legend(fontsize=7.5, frameon=False, loc="upper right")
            ax = axes[1, 2]
            style(ax, "critic: predicted V, target, loss, explained variance")
            ax.plot(x, [r["v_mean"] for r in it], color=BLUE, lw=1.5, label="mean V(s)")
            ax.plot(x, [r["target_mean"] for r in it], color="#1baf7a", lw=1.2, label="mean target")
            ax.plot(x, [r["explained_variance"] for r in it], color="#eda100", lw=1.0, label="explained var.")
            ax.plot(x, [r["critic_loss"] for r in it], color=GRAY, lw=1.0, ls=":", label="CE loss")
            ax.plot(x, [r["out_of_support"] for r in it], color="#e34948", lw=1.0, label="target out-of-support frac")
            ax.legend(fontsize=7, frameon=False)
            ax.set_xlabel("PPO iteration", color=INK2, fontsize=9)
        fig.suptitle(f"DEHP-style horizon PPO pilot on ALOHA Transfer Cube (frozen π0) — {label}", x=0.01, ha="left",
                     fontsize=11, color=INK)
        fig.text(0.01, 0.005, f"Pilot: {n_ep} held-out episodes per evaluation (seeds 10,000,000–10,000,009, fixed π0 noise); "
                 "counts, not significance. One 1-hour gpu-debug job.", fontsize=7.5, color=INK2)
        fig.tight_layout(rect=(0, 0.02, 1, 0.96))
        fig.savefig(f"{OUT}/{name}/dehp_{name}.png", dpi=140, facecolor=SURF)
        fig.savefig(f"{OUT}/{name}/dehp_{name}.svg", facecolor=SURF)
    json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
