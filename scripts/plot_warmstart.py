"""Warm-start experiment figure: branch A (continue ARLI) vs branch B (frozen ARLI + final-step residual actor).

Top row: held-out eval success (50 episodes). Bottom row: training-episode success per 25k window. x = ADDITIONAL online
env steps after the common ARLI 600k checkpoint (x = 0). Seeds in separate columns.

    python scripts/plot_warmstart.py --out results/warmstart_final
"""
import argparse
import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WS = "/N/scratch/jindjia/arli/runs_warmstart"
INK, INK2, GRID, SURF, AXIS = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#a3a29c"
BR = [("A", "A: continue ARLI noise actor", "#2a78d6", "o"),
      ("B", "B: frozen ARLI + final-step actor", "#1baf7a", "^")]
W = 25_000


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/warmstart_final")
    args = ap.parse_args()
    os.makedirs(f"{args.out}/raw", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURF)
    rows = []
    for col, seed in enumerate((0, 1)):
        src = None
        for b, label, color, mk in BR:
            run = f"{WS}/ws_{b}_seed{seed}"
            ev = sorted({e["milestone"]: e for e in jl(f"{run}/evals.jsonl")}.values(), key=lambda e: e["milestone"])
            if os.path.exists(f"{run}/provenance.json"):
                src = json.load(open(f"{run}/provenance.json"))
            if ev:
                with open(f"{args.out}/raw/ws_{b}_seed{seed}_evals.jsonl", "w") as f:
                    for e in ev:
                        f.write(json.dumps({k: v for k, v in e.items() if k != "episodes"} |
                                           {"episode_success": [int(x["success"]) for x in e["episodes"]]}) + "\n")
            ax = axes[0, col]
            if ev:
                x = [e["additional_env_steps"] / 1000 for e in ev]
                ci = [wilson(e["n_success"], e["n_episodes"]) for e in ev]
                ax.fill_between(x, [a for a, _ in ci], [c for _, c in ci], color=color, alpha=0.1, lw=0)
                ax.plot(x, [e["success_rate"] for e in ev], color=color, lw=2, marker=mk, ms=4,
                        label=f"{label}: {ev[-1]['n_success']}/50 @ +{ev[-1]['additional_env_steps'] // 1000}k")
                for e, (lo, hi) in zip(ev, ci):
                    rows.append(dict(kind="eval", seed=seed, branch=b, additional_env_steps=e["additional_env_steps"],
                                     total_online_env_steps=e["total_online_env_steps"], grad_steps=e["grad_steps"],
                                     n=e["n_episodes"], successes=e["n_success"], rate=e["success_rate"],
                                     wilson_lo=lo, wilson_hi=hi, wall_total_s=e["wall_total_s"]))
            else:
                ax.plot([], [], color=color, marker=mk, label=f"{label} (no evals yet)")
            on = {}
            for e in jl(f"{run}/episodes.jsonl"):
                on[e["online_env_steps"]] = e
            on = [on[k] for k in sorted(on)]
            xs, ys = [], []
            for w0 in range(0, on[-1]["online_env_steps"] if on else 0, W):
                ew = [e for e in on if w0 < e["online_env_steps"] <= w0 + W]
                if len(ew) >= 5:
                    xs.append((w0 + W / 2) / 1000)
                    ys.append(sum(e["success"] for e in ew) / len(ew))
                    rows.append(dict(kind="train", seed=seed, branch=b, additional_env_steps=w0 + W, n=len(ew),
                                     successes=sum(e["success"] for e in ew), rate=ys[-1]))
            axes[1, col].plot(xs, ys, color=color, lw=2, marker=mk, ms=4, label=label)
        start = (f"common start: {src['source_run']} @ {src['online_env_steps']:,} online steps "
                 f"(sha256 {src['sha256'][:10]}…, eval {src['eval_success']}/50)") if src else "common start: ARLI 600k"
        axes[0, col].set_title(f"seed {seed} · held-out eval (50 episodes)\n{start}", loc="left", fontsize=9, color=INK)
        axes[1, col].set_title(f"seed {seed} · training-episode success per 25k window", loc="left", fontsize=9,
                               color=INK)
    for ax in axes.flat:
        ax.set_facecolor(SURF)
        ax.axvline(0, color=INK2, lw=1, ls=(0, (2, 2)))
        ax.set_xlim(-8, 310)
        ax.set_ylim(-0.02, 1.0)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    for ax in axes[1]:
        ax.set_xlabel("additional online env steps after the 600k ARLI checkpoint (k)", color=INK2, fontsize=9)
    axes[0, 0].set_ylabel("eval success rate", color=INK2, fontsize=9)
    axes[1, 0].set_ylabel("training success rate", color=INK2, fontsize=9)
    fig.suptitle("Warm start from ARLI 600k: continue the noise actor (A) vs freeze it and add a k=9 residual actor (B) · "
                 "ALOHA Transfer Cube, frozen π0, d=20/d_rl=10, no RTC", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.text(0.01, 0.005, "Real evaluations only. Shading: 95% Wilson interval over the 50 eval episodes of one run "
             "(not an across-seed band). Separate from the from-scratch experiments.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(f"{args.out}/warmstart_success.png", dpi=150, facecolor=SURF)
    fig.savefig(f"{args.out}/warmstart_success.svg", facecolor=SURF)
    if rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
        with open(f"{args.out}/warmstart.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
