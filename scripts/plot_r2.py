"""Collect R2 evaluations into results/ (CSV + raw JSONL copies) and plot success vs env steps.

Only measured evaluations are plotted.  One training seed -> no across-seed band; the shaded band is a 95%
Wilson interval over the 50 evaluation episodes of each point (labelled as such).

    python scripts/plot_r2.py
"""
import argparse
import csv
import json
import math
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = "/N/scratch/jindjia/arli/runs"
METHODS = [("r2_arli_seed0", "ARLI (reconstruction)", "#6a2bd9"),
           ("r2_dsrl_seed0", "DSRL (naive async)", "#a98bd6"),
           ("r2_arli_actions_seed0", "ARLI (actions only)", "#6fbf8e"),
           ("r2_arli_state_seed0", "ARLI (state only)", "#e3a47a")]


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/r2")
    ap.add_argument("--frozen", default="results/r1/frozen_async_d20.jsonl")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for name, label, color in METHODS:
        p = f"{RUNS}/{name}/evals.jsonl"
        if not os.path.exists(p):
            continue
        ev = list({e["milestone"]: e for e in read_jsonl(p)}.values())  # dedupe re-logged evals after a resume
        shutil.copy(p, f"{args.out}/{name}_evals.jsonl")
        for e in ev:
            lo, hi = wilson(e["n_success"], e["n_episodes"])
            rows.append(dict(run=name, online_env_steps=e["online_env_steps"], prefill_env_steps=e["prefill_env_steps"],
                             total_env_steps=e["total_env_steps"], grad_steps=e["grad_steps"],
                             n_success=e["n_success"], n_episodes=e["n_episodes"], success_rate=e["success_rate"],
                             wilson_lo=lo, wilson_hi=hi))
        for ax, key in zip(axes, ("online_env_steps", "total_env_steps")):
            xs = [e[key] for e in ev]
            ys = [e["success_rate"] for e in ev]
            ci = [wilson(e["n_success"], e["n_episodes"]) for e in ev]
            ax.plot(xs, ys, "-o", ms=3, color=color, lw=2 if name == "r2_arli_seed0" else 1.5,
                    label=f"{label} (n={len(ev)} evals)")
            ax.fill_between(xs, [c[0] for c in ci], [c[1] for c in ci], color=color, alpha=0.12, lw=0)
    if os.path.exists(args.frozen):
        fr = read_jsonl(args.frozen)
        k, n = sum(r["success"] for r in fr), len(fr)
        lo, hi = wilson(k, n)
        for ax in axes:
            ax.axhline(k / n, ls="--", color="#a98bd6", lw=2, label=f"Pretrained π0, async d=20, no RTC ({k}/{n})")
            ax.axhspan(lo, hi, color="#a98bd6", alpha=0.07, lw=0)
        rows.append(dict(run="frozen_pi0_async_d20", online_env_steps=0, n_success=k, n_episodes=n,
                         success_rate=k / n, wilson_lo=lo, wilson_hi=hi))
    axes[0].set_xlabel("online env steps (prefill excluded)")
    axes[1].set_xlabel("total env steps (prefill + online)")
    axes[0].set_ylabel("success rate (50 held-out eval episodes)")
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.set_xlim(left=0)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("ALOHA Transfer Cube, t_delay=20, t_delay^rl=10, no RTC — paper-based ARLI reconstruction, "
                 "1 training seed\nshaded: 95% Wilson interval over eval episodes (not an across-seed band)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{args.out}/r2_success_vs_env_steps.png", dpi=150)
    fig.savefig(f"{args.out}/r2_success_vs_env_steps.pdf")
    if rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        with open(f"{args.out}/r2_evals.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
