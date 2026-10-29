"""SCH-01 online run figures + compact CSVs.

Figure 1 (primary): held-out eval success (50 episodes, deterministic masked argmax) vs total training control steps
(warm-up included), with the matched RTC-on fixed scheduler as a horizontal reference.
Figure 2: stochastic training-episode success per 25k window (episode counts annotated) and scheduling diagnostics
(mean selected N, mean R at launch, reference departures per episode, entropy).

    python scripts/plot_online.py --run rn_seed0_v1 [--out results/online]
"""
import argparse
import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = "/N/scratch/jindjia/arli/runs_scheduling/online"
INK, INK2, GRID, SURF, AXIS = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#a3a29c"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="rn_seed0_v1")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default="results/online")
    args = ap.parse_args()
    rd = f"{args.root}/{args.run}"
    out = f"{args.out}/{args.run}"
    os.makedirs(out, exist_ok=True)
    ev = sorted({e["tag"]: e for e in jl(f"{rd}/evals.jsonl")}.values(), key=lambda e: e["train_steps"])
    ref = jl(f"{rd}/reference_eval.jsonl")
    eps = jl(f"{rd}/episodes.jsonl")
    upd = jl(f"{rd}/updates.jsonl")
    small = ev and ev[0]["n_episodes"] < 50

    with open(f"{out}/evals.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "train_steps", "prior_strength", "grad_steps", "n_success", "n_episodes", "success_rate",
                    "wilson_lo", "wilson_hi", "mean_N", "mean_R", "departures", "underrun_ticks", "infer_cost_ms",
                    "reference_parity"])
        for e in ref + ev:
            lo, hi = wilson(e["n_success"], e["n_episodes"])
            w.writerow([e["tag"] if e["policy"] == "learned" else "fixed_reference", e["train_steps"], e["prior_strength"],
                        e["grad_steps"], e["n_success"], e["n_episodes"], e["success_rate"], lo, hi, e["mean_N"],
                        e["mean_R"], e["departures"], e["underrun_ticks"], e["infer_cost_ms"], e["reference_parity"]])
    with open(f"{out}/evals_raw.jsonl", "w") as f:
        for e in ref + ev:
            f.write(json.dumps({k: v for k, v in e.items() if k != "episodes"} |
                               dict(episode_success=[int(x["success"]) for x in e["episodes"]])) + "\n")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    fig.patch.set_facecolor(SURF)
    style(ax, f"SCH-01 online joint R/N scheduler · {args.run} · held-out evaluation"
              + (" (DEBUG: <50 episodes)" if small else ""))
    if ev:
        x = [e["train_steps"] / 1000 for e in ev]
        ci = [wilson(e["n_success"], e["n_episodes"]) for e in ev]
        ax.fill_between(x, [a for a, _ in ci], [b for _, b in ci], color=BLUE, alpha=0.12, lw=0)
        ax.plot(x, [e["success_rate"] for e in ev], color=BLUE, lw=2, marker="o", ms=4,
                label=f"learned R/N scheduler (deterministic): {ev[-1]['n_success']}/{ev[-1]['n_episodes']} @ "
                      f"{ev[-1]['train_steps'] // 1000}k")
    if ref:
        r = ref[-1]
        lo, hi = wilson(r["n_success"], r["n_episodes"])
        ax.axhline(r["success_rate"], color=INK2, ls=(0, (5, 4)), lw=1.4,
                   label=f"matched fixed scheduler (N=10, K=25, RTC on): {r['n_success']}/{r['n_episodes']}")
        ax.axhspan(lo, hi, color=INK2, alpha=0.06, lw=0)
    ax.axvline(5, color=GRID, lw=1)
    ax.text(5.5, 0.02, "end of 5k warm-up", fontsize=7.5, color=INK2)
    ax.set_xlim(-5, 305)
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlabel("total training control steps (k), warm-up included", color=INK2, fontsize=9)
    ax.set_ylabel("success rate (held-out episodes)", color=INK2, fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.text(0.01, 0.005, "Proxy-derived logical-time simulation (A100 pi0+RTC profile, synthetic jitter); not measured "
             "native ALOHA latency. Shading: 95% Wilson interval over evaluation episodes (one training seed).",
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"{out}/eval_success_vs_steps.png", dpi=150, facecolor=SURF)
    fig.savefig(f"{out}/eval_success_vs_steps.svg", facecolor=SURF)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    fig.patch.set_facecolor(SURF)
    W = 25_000
    if eps:
        last = eps[-1]["steps"]
        bins = [(w0, [e for e in eps if w0 < e["steps"] <= w0 + W]) for w0 in range(0, last, W)]
        bins = [(w0, b) for w0, b in bins if b]
        xs = [(w0 + W / 2) / 1000 for w0, _ in bins]
        ys = [sum(e["success"] for e in b) / len(b) for _, b in bins]
        axes[0, 0].plot(xs, ys, color=ORANGE, lw=2, marker="s", ms=4)
        for x_, y_, (_, b) in zip(xs, ys, bins):
            axes[0, 0].annotate(str(len(b)), (x_, y_), xytext=(0, 5), textcoords="offset points", fontsize=7,
                                color=INK2, ha="center")
        ex = [e["steps"] / 1000 for e in eps]
        axes[0, 1].plot(ex, [e["mean_N"] for e in eps], color=BLUE, lw=0.8, alpha=0.8, label="mean selected N")
        axes[0, 1].plot(ex, [e["mean_R"] for e in eps], color=ORANGE, lw=0.8, alpha=0.8, label="mean R at launch")
        axes[0, 1].legend(fontsize=8, frameon=False)
        axes[1, 0].plot(ex, [e["departures"] for e in eps], color=BLUE, lw=0.8)
        axes[1, 1].plot(ex, [e["entropy"] for e in eps], color=BLUE, lw=0.8)
    style(axes[0, 0], "stochastic training-episode success per 25k window (episode count above points)")
    style(axes[0, 1], "selected N and R per training episode")
    style(axes[1, 0], "reference-rule departures per training episode")
    style(axes[1, 1], "mean policy entropy per decision (training)")
    axes[0, 0].set_ylim(-0.02, 1.0)
    for a in axes[1]:
        a.set_xlabel("total training control steps (k)", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out}/training_diagnostics.png", dpi=140, facecolor=SURF)
    if upd:
        with open(f"{out}/updates.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(upd[-1]))
            w.writeheader()
            for u in upd:
                w.writerow({k: u.get(k) for k in upd[-1]})
    print(f"wrote {out}: {len(ev)} evals, {len(eps)} training episodes")


if __name__ == "__main__":
    main()
