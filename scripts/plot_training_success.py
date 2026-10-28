"""Training (online exploration) episode success rate vs online env steps, from the per-episode logs.

Each point = fraction of online training episodes that succeeded among episodes ending in a 25k-online-step window
(stochastic actor, exploration on). Hollow marker at x = 0 = prefill success rate (random initial actions). Not the
evaluation metric (see success_vs_env_steps.png for the 50 held-out eval episodes).

    python scripts/plot_training_success.py --out results/training_success
"""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R2 = "/N/scratch/jindjia/arli/runs"
SA = "/N/scratch/jindjia/arli/runs_single"
INK, INK2, GRID, SURF, AXIS = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#a3a29c"
W = 25_000
SERIES = {  # fixed categorical order: ARLI blue, shared_every_step orange, final_step_only aqua, DSRL yellow
    "seed 0 · prefill std 0.3": [("ARLI", "#2a78d6", "o", f"{R2}/r2_arli_seed0"),
                                 ("shared_every_step", "#eb6834", "s", f"{SA}/sa_shared_every_step_seed0"),
                                 ("final_step_only", "#1baf7a", "^", f"{SA}/sa_final_step_only_seed0"),
                                 ("naive async DSRL", "#eda100", "D", f"{R2}/r2_dsrl_seed0")],
    "seed 1 · prefill std 0.3": [("ARLI", "#2a78d6", "o", f"{R2}/r2_arli_seed1_std030"),
                                 ("shared_every_step", "#eb6834", "s", f"{SA}/sa_shared_every_step_seed1"),
                                 ("final_step_only", "#1baf7a", "^", f"{SA}/sa_final_step_only_seed1"),
                                 ("naive async DSRL", "#eda100", "D", f"{R2}/r2_dsrl_seed1_std030")],
    "seed 0 · prefill variance 0.3 (control)": [("ARLI", "#2a78d6", "o", f"{R2}/r2_arli_seed0_var030"),
                                                ("naive async DSRL", "#eda100", "D", f"{R2}/r2_dsrl_seed0_var030")],
}


def episodes(run):
    p = f"{run}/episodes.jsonl"
    if not os.path.exists(p):
        return [], []
    pre, on = [], {}
    for line in open(p):
        e = json.loads(line)
        if e["phase"] == "prefill":
            pre.append(e)
        else:
            on[e["online_env_steps"]] = e  # dedupe episodes re-logged after a resume (keep last)
    return pre, [on[k] for k in sorted(on)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/training_success")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.9), sharey=True)
    fig.patch.set_facecolor(SURF)
    for ax, (title, series) in zip(axes, SERIES.items()):
        ax.set_facecolor(SURF)
        for label, color, mk, run in series:
            pre, on = episodes(run)
            if not on:
                continue
            xs, ys = [], []
            last = on[-1]["online_env_steps"]
            for w0 in range(0, last, W):
                ew = [e for e in on if w0 < e["online_env_steps"] <= w0 + W]
                if len(ew) < 5:
                    continue
                rate = sum(e["success"] for e in ew) / len(ew)
                xs.append((w0 + W / 2) / 1000)
                ys.append(rate)
                rows.append(dict(panel=title, method=label, run=os.path.basename(run), window_start=w0,
                                 window_end=w0 + W, episodes=len(ew), successes=sum(e["success"] for e in ew),
                                 success_rate=rate))
            ax.plot(xs, ys, color=color, lw=2, marker=mk, ms=3.5,
                    label=f"{label} ({sum(e['success'] for e in on)}/{len(on)} online eps, to {last // 1000}k)")
            if pre:
                pr = sum(e["success"] for e in pre) / len(pre)
                ax.plot([0], [pr], marker=mk, ms=7, mfc="none", mec=color, mew=1.6, ls="none")
                rows.append(dict(panel=title, method=label, run=os.path.basename(run), window_start="prefill",
                                 window_end="prefill", episodes=len(pre), successes=sum(e["success"] for e in pre),
                                 success_rate=pr))
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_xlim(-12, 615)
        ax.set_ylim(-0.02, 1.0)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.set_xlabel("online env steps (k)", color=INK2, fontsize=9)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    axes[0].set_ylabel("training-episode success rate (per 25k window)", color=INK2, fontsize=9)
    fig.suptitle("Training (exploration) success rate · ALOHA Transfer Cube, frozen π0, d=20, d_rl=10, no RTC · "
                 "hollow marker at 0 = prefill success rate", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.text(0.01, 0.005, "Online training episodes use the stochastic actor (exploration on); this is not the "
             "held-out evaluation metric. Single runs; runs still in progress end early.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(f"{args.out}/training_success_vs_env_steps.png", dpi=150, facecolor=SURF)
    fig.savefig(f"{args.out}/training_success_vs_env_steps.svg", facecolor=SURF)
    with open(f"{args.out}/training_success.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
