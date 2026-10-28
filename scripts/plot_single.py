"""Single-actor comparison: ARLI vs shared_every_step vs final_step_only, all trained from online step 0.

Same protocol and prefill std 0.3 for all three; seeds shown separately (no aggregate until enough seeds exist).
Writes raw JSONL copies, a combined CSV, success_vs_env_steps.png and .svg into --out.

    python scripts/plot_single.py --out results/single_actor_comparison
"""
import argparse
import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R2 = "/N/scratch/jindjia/arli/runs"
SA = "/N/scratch/jindjia/arli/runs_single"
INK, INK2, GRID, SURF, AXIS = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#a3a29c"
# fixed categorical order: ARLI = slot 1 (blue), shared_every_step = slot 2 (orange), final_step_only = slot 3 (aqua)
RUNS = {  # (method label, color, run dir) per seed; ARLI reference = R2 runs with prefill std 0.3
    0: [("ARLI", "#2a78d6", f"{R2}/r2_arli_seed0"),
        ("shared_every_step", "#eb6834", f"{SA}/sa_shared_every_step_seed0"),
        ("final_step_only", "#1baf7a", f"{SA}/sa_final_step_only_seed0")],
    1: [("ARLI", "#2a78d6", f"{R2}/r2_arli_seed1_std030"),
        ("shared_every_step", "#eb6834", f"{SA}/sa_shared_every_step_seed1"),
        ("final_step_only", "#1baf7a", f"{SA}/sa_final_step_only_seed1")],
}
MARK = {"ARLI": "o", "shared_every_step": "s", "final_step_only": "^"}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(run):
    p = f"{run}/evals.jsonl"
    if not os.path.exists(p):
        return []
    return sorted({e["milestone"]: e for e in map(json.loads, open(p))}.values(), key=lambda e: e["milestone"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/single_actor_comparison")
    ap.add_argument("--frozen", default=None, help="frozen pi0 async eval JSONL (dashed reference)")
    args = ap.parse_args()
    os.makedirs(f"{args.out}/raw", exist_ok=True)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    fig.patch.set_facecolor(SURF)
    frozen = None
    if args.frozen and os.path.exists(args.frozen):
        fr = [json.loads(l) for l in open(args.frozen)]
        frozen = (sum(r["success"] for r in fr), len(fr))
    for ax, seed in zip(axes, (0, 1)):
        ax.set_facecolor(SURF)
        for label, color, run in RUNS[seed]:
            ev = load(run)
            name = os.path.basename(run)
            if ev:
                with open(f"{args.out}/raw/{name}_evals.jsonl", "w") as f:
                    for e in ev:
                        e = {k: v for k, v in e.items() if k != "episodes"} | {
                            "episode_success": [int(x["success"]) for x in e["episodes"]]}
                        f.write(json.dumps(e) + "\n")
            for e in ev:
                lo, hi = wilson(e["n_success"], e["n_episodes"])
                rows.append(dict(seed=seed, method=label, run=name, milestone=e["milestone"],
                                 online_env_steps=e["online_env_steps"], prefill_env_steps=e["prefill_env_steps"],
                                 total_env_steps=e["total_env_steps"], grad_steps=e["grad_steps"],
                                 actor_calls_online=e.get("actor_calls_online", e.get("online_transitions")),
                                 wall_total_s=e.get("wall_total_s"), n_success=e["n_success"],
                                 n_episodes=e["n_episodes"], success_rate=e["success_rate"], wilson_lo=lo, wilson_hi=hi))
            if not ev:
                ax.plot([], [], color=color, lw=2, marker=MARK[label], label=f"{label} (no evals yet)")
                continue
            x = [e["online_env_steps"] / 1000 for e in ev]
            y = [e["success_rate"] for e in ev]
            ci = [wilson(e["n_success"], e["n_episodes"]) for e in ev]
            ax.fill_between(x, [a for a, _ in ci], [b for _, b in ci], color=color, alpha=0.10, lw=0)
            ax.plot(x, y, color=color, lw=2, marker=MARK[label], ms=4,
                    label=f"{label}: {ev[-1]['n_success']}/50 @ {ev[-1]['online_env_steps'] // 1000}k")
        if frozen:
            ax.axhline(frozen[0] / frozen[1], color=INK2, ls=(0, (5, 4)), lw=1.3,
                       label=f"frozen π0, same protocol ({frozen[0]}/{frozen[1]})")
        ax.set_title(f"training seed {seed}", loc="left", fontsize=10.5, color=INK)
        ax.set_xlim(-8, 610)
        ax.set_ylim(-0.02, 1.0)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.set_xlabel("online env steps (k), all methods from RL step 0", color=INK2, fontsize=9)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    axes[0].set_ylabel("success rate (50 held-out eval episodes)", color=INK2, fontsize=9)
    fig.suptitle("ALOHA Transfer Cube · frozen π0 · d=20, d_rl=10, 10 denoising iterations, no RTC · prefill std 0.3 · "
                 "ARLI = paper-based reconstruction", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.text(0.01, 0.005, "Real evaluations only; shading = 95% Wilson interval over the 50 eval episodes of one run "
             "(not an across-seed band). Prefill env steps are excluded from the x-axis and listed in the CSV.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(f"{args.out}/success_vs_env_steps.png", dpi=150, facecolor=SURF)
    fig.savefig(f"{args.out}/success_vs_env_steps.svg", facecolor=SURF)
    if rows:
        with open(f"{args.out}/evals.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} eval rows")


if __name__ == "__main__":
    main()
