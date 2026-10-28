"""S1 continuation plot (kept separate from the Figure 6 reproduction plot).

Left: success vs additional online env steps from the common R2 ARLI checkpoint.
Right: same points vs cumulative interaction (R2 prefill + R2 online + additional).
Shaded: 95% Wilson interval over the 50 eval episodes (one training seed; no across-seed band).

    python scripts/plot_s1.py
"""
import argparse
import csv
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_r2 import read_jsonl, wilson  # noqa: E402

RUNS = "/N/scratch/jindjia/arli/runs"
BR = [("A", "A: continued ARLI (initial noise only)", "#6a2bd9"),
      ("B", "B: per-step steering, stale obs", "#e3a47a"),
      ("C", "C: per-step steering, fresh obs", "#2a9d8f"),
      ("D", "D: initial noise + last-step correction", "#d1495b")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/s1")
    ap.add_argument("--r2", default=f"{RUNS}/r2_arli_seed0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    base = 0
    r2 = f"{args.r2}/evals.jsonl"
    if os.path.exists(r2):
        last = read_jsonl(r2)[-1]
        base = last["total_env_steps"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    rows = []
    for b, label, color in BR:
        p = f"{RUNS}/s1_{b}_seed0/evals.jsonl"
        if not os.path.exists(p):
            continue
        ev = list({e["milestone"]: e for e in read_jsonl(p)}.values())
        shutil.copy(p, f"{args.out}/s1_{b}_seed0_evals.jsonl")
        prm = f"{RUNS}/s1_{b}_seed0/params.json"
        if os.path.exists(prm):
            shutil.copy(prm, f"{args.out}/s1_{b}_seed0_params.json")
        xs = [e["online_env_steps"] for e in ev]
        ys = [e["success_rate"] for e in ev]
        ci = [wilson(e["n_success"], e["n_episodes"]) for e in ev]
        for ax, off in zip(axes, (0, base)):
            ax.plot([x + off for x in xs], ys, "-o", ms=3, color=color, label=label)
            ax.fill_between([x + off for x in xs], [c[0] for c in ci], [c[1] for c in ci], color=color, alpha=0.1, lw=0)
        for e, (lo, hi) in zip(ev, ci):
            rows.append(dict(branch=b, additional_env_steps=e["online_env_steps"], cumulative_env_steps=e["online_env_steps"] + base,
                             grad_iters=e["grad_iters"], n_success=e["n_success"], n_episodes=e["n_episodes"],
                             success_rate=e["success_rate"], wilson_lo=lo, wilson_hi=hi, actor_calls_eval=e["actor_calls"],
                             fresh_frame_frac=e["fresh_frame_frac"], actor_s_eval=e["actor_s"], pi0_s_eval=e["pi0_s"]))
    axes[0].set_xlabel("additional online env steps from the common R2 ARLI checkpoint")
    axes[1].set_xlabel(f"cumulative env steps (R2 prefill + online = {base:,} at start)")
    axes[0].set_ylabel("success rate (50 held-out eval episodes)")
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("S1 continuation pilot, ALOHA Transfer Cube, d=20/d_rl=10, no RTC, 1 seed — idealised "
                 "information-timing experiment (actor overhead not charged to the clock)\n"
                 "shaded: 95% Wilson interval over eval episodes", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{args.out}/s1_success.png", dpi=150)
    fig.savefig(f"{args.out}/s1_success.pdf")
    if rows:
        with open(f"{args.out}/s1_evals.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(json.dumps({"rows": len(rows), "base_env_steps": base}))


if __name__ == "__main__":
    main()
