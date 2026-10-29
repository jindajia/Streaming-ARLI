"""v2 online figures (separate files) + compact CSVs.

1. eval_success.png: held-out success (50 episodes, deterministic) vs training control steps; new fixed baseline as a
   horizontal reference; prior removal at 50k annotated; 25k prior-disabled supplemental evaluation as a separate marker.
2. training_success.png: stochastic training-episode success per window (window size and episode counts annotated).
3. schedule.png: selected N, R, execution length per chunk (old actions executed), launch phase p.
4. timing_compute.png: observation age at handoff, inference calls and modeled compute per 1,000 control steps, underruns.

    python scripts/plot_online_v2.py --run rn_seed0_d10
"""
import argparse
import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = "/N/scratch/jindjia/arli/runs_scheduling/fixed_delay10_v2/online"
INK, INK2, GRID, SURF, AXIS = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#a3a29c"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
W = 25_000


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def style(ax, title, xlabel=True):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.axvline(50, color=INK2, lw=0.8, ls=(0, (2, 3)))
    if xlabel:
        ax.set_xlabel("training environment control steps (k)", color=INK2, fontsize=8.5)


def binned(eps, key, w=W):
    if not eps:
        return [], [], []
    last = eps[-1]["steps"]
    xs, ys, ns = [], [], []
    for w0 in range(0, last, w):
        b = [e for e in eps if w0 < e["steps"] <= w0 + w and e.get(key) is not None]
        if len(b) >= 10:
            xs.append((w0 + w / 2) / 1000)
            ys.append(float(np.mean([e[key] for e in b])))
            ns.append(len(b))
    return xs, ys, ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="rn_seed0_d10")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default="results/online")
    args = ap.parse_args()
    rd, out = f"{args.root}/{args.run}", f"{args.out}/{args.run}"
    os.makedirs(out, exist_ok=True)
    ev = sorted({e["tag"]: e for e in jl(f"{rd}/evals.jsonl")}.values(), key=lambda e: e["train_steps"])
    sup = jl(f"{rd}/evals_supplemental.jsonl")
    ref = jl(f"{rd}/reference_eval.jsonl")
    eps = jl(f"{rd}/episodes.jsonl")
    debug = bool(ev) and ev[0]["n_episodes"] < 50

    keys = ["tag", "policy", "train_steps", "prior_strength", "grad_steps", "n_success", "n_episodes", "success_rate",
            "mean_N", "mean_R", "mean_launch_p", "mean_exec_len", "mean_obs_age_handoff", "calls_per_1k",
            "modeled_ms_per_1k", "departures", "recovery_states", "underrun_ticks", "reference_parity"]
    with open(f"{out}/evals.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys + ["wilson_lo", "wilson_hi"])
        for e in ref + ev + sup:
            w.writerow([e.get(k) for k in keys] + list(wilson(e["n_success"], e["n_episodes"])))
    with open(f"{out}/evals_raw.jsonl", "w") as f:
        for e in ref + ev + sup:
            f.write(json.dumps({k: v for k, v in e.items() if k != "episodes"} |
                               dict(episode_success=[int(x["success"]) for x in e["episodes"]])) + "\n")

    # 1 eval success
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    fig.patch.set_facecolor(SURF)
    style(ax, f"Held-out evaluation success · {args.run} · fixed delay table (N=10 → 10 steps), RTC on"
              + ("  [DEBUG: <50 episodes]" if debug else ""))
    if ref:
        r = ref[-1]
        lo, hi = wilson(r["n_success"], r["n_episodes"])
        ax.axhline(r["success_rate"], color=INK2, ls=(0, (5, 4)), lw=1.4,
                   label=f"fixed baseline (N=10, KEEP to q=10, 25 actions/chunk): {r['n_success']}/{r['n_episodes']}")
        ax.axhspan(lo, hi, color=INK2, alpha=0.06, lw=0)
    if ev:
        x = [e["train_steps"] / 1000 for e in ev]
        ci = [wilson(e["n_success"], e["n_episodes"]) for e in ev]
        ax.fill_between(x, [a for a, _ in ci], [b for _, b in ci], color=BLUE, alpha=0.12, lw=0)
        ax.plot(x, [e["success_rate"] for e in ev], color=BLUE, lw=2, marker="o", ms=4,
                label=f"learned scheduler (deterministic argmax): {ev[-1]['n_success']}/{ev[-1]['n_episodes']} @ "
                      f"{ev[-1]['train_steps'] // 1000}k")
    for s in sup:
        ax.plot([s["train_steps"] / 1000], [s["success_rate"]], marker="D", ms=7, mfc="none", mec=ORANGE, mew=1.6, ls="none",
                label=f"supplemental: same checkpoint with prior disabled ({s['n_success']}/{s['n_episodes']})")
    ax.text(50.8, 0.03, "prior fully removed (50k)", fontsize=7.5, color=INK2)
    ax.set_xlim(-5, 305)
    ax.set_ylim(-0.02, 1.0)
    ax.set_ylabel("success rate (50 held-out episodes)", color=INK2, fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.text(0.01, 0.005, "Deterministic simulation delay table (not a GPU measurement). Shading: 95% Wilson interval "
             "over evaluation episodes of one training seed (not across seeds).", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"{out}/eval_success.png", dpi=150, facecolor=SURF)
    fig.savefig(f"{out}/eval_success.svg", facecolor=SURF)
    plt.close(fig)

    # 2 training success
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    fig.patch.set_facecolor(SURF)
    style(ax, f"Stochastic training-episode success · window {W // 1000}k steps (episode count per window above points)")
    for e in eps:
        e["success_f"] = float(e["success"])
    xs, ys, ns = binned(eps, "success_f")
    ax.plot(xs, ys, color=ORANGE, lw=2, marker="s", ms=4)
    for x_, y_, n_ in zip(xs, ys, ns):
        ax.annotate(str(n_), (x_, y_), xytext=(0, 5), textcoords="offset points", fontsize=7, color=INK2, ha="center")
    ax.set_xlim(-5, 305)
    ax.set_ylim(-0.02, 1.0)
    ax.set_ylabel("training success rate", color=INK2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{out}/training_success.png", dpi=150, facecolor=SURF)
    plt.close(fig)

    # 3 schedule
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7), sharex=True)
    fig.patch.set_facecolor(SURF)
    for ax, key, title in ((axes[0, 0], "mean_N", "selected N (mean per training episode)"),
                           (axes[0, 1], "mean_R", "selected R (mean per training episode)"),
                           (axes[1, 0], "mean_exec_len", "execution length per chunk = old actions executed"),
                           (axes[1, 1], "mean_launch_p", "launch phase p (actions executed when the request starts)")):
        xs, ys, _ = binned(eps, key, 5_000)
        ax.plot(xs, ys, color=BLUE, lw=1.5)
        style(ax, title, xlabel=ax in axes[1])
    axes[0, 0].axhline(10, color=INK2, ls=":", lw=1)
    axes[0, 1].axhline(10, color=INK2, ls=":", lw=1)
    axes[1, 0].axhline(25, color=INK2, ls=":", lw=1)
    axes[1, 1].axhline(15, color=INK2, ls=":", lw=1)
    fig.suptitle("Scheduling choices in training episodes (5k-step windows; dotted = fixed baseline value)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out}/schedule.png", dpi=140, facecolor=SURF)
    plt.close(fig)

    # 4 timing / compute
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7), sharex=True)
    fig.patch.set_facecolor(SURF)
    for ax, key, title, ref_v in ((axes[0, 0], "mean_obs_age_handoff", "observation age at handoff (control steps)", 10),
                                  (axes[0, 1], "calls_per_1k", "inference calls per 1,000 control steps", 40),
                                  (axes[1, 0], "modeled_ms_per_1k", "modeled inference time per 1,000 steps (ms, d(N)·20 ms)",
                                   40 * 200),
                                  (axes[1, 1], "underrun_ticks", "underrun steps per training episode", 0)):
        xs, ys, _ = binned(eps, key, 5_000)
        ax.plot(xs, ys, color=BLUE, lw=1.5)
        ax.axhline(ref_v, color=INK2, ls=":", lw=1)
        style(ax, title, xlabel=ax in axes[1])
    fig.suptitle("Timing and compute in training episodes (5k-step windows; dotted = fixed baseline value)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out}/timing_compute.png", dpi=140, facecolor=SURF)
    plt.close(fig)
    print(f"wrote {out}: {len(ev)} evals, {len(sup)} supplemental, {len(eps)} training episodes")


if __name__ == "__main__":
    main()
