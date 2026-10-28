"""Overview figure (real evaluations only): R2 seed-0 reproduction | new groups vs seed 0 | S1 continuation.

    python scripts/plot_overview.py  -> results/overview.png
"""
import json
import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = "/N/scratch/jindjia/arli/runs"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY = "#a3a29c"


def evals(name):
    p = f"{RUNS}/{name}/evals.jsonl"
    if not os.path.exists(p):
        return [], []
    ev = sorted({e["milestone"]: e for e in map(json.loads, open(p))}.values(), key=lambda e: e["milestone"])
    return [e["online_env_steps"] / 1000 for e in ev], [e["success_rate"] for e in ev]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRAY)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlabel("online env steps (k)", color=INK2, fontsize=9)


def end_label(ax, x, y, text, dy=0.0):
    if x:
        ax.annotate(text, (x[-1], y[-1]), xytext=(5, dy), textcoords="offset points", va="center",
                    fontsize=8, color=INK)


fr = [json.loads(l) for l in open("results/r1/frozen_async_d20.jsonl")]
k_fr = sum(r["success"] for r in fr)

fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), sharey=True, gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
fig.patch.set_facecolor(SURF)

# --- panel 1: R2 seed 0 --------------------------------------------------------------------
ax = axes[0]
style(ax, "R2 · Figure-6 reproduction, seed 0 (complete)")
for name, label, c in (("r2_arli_seed0", "ARLI", BLUE), ("r2_dsrl_seed0", "naive async DSRL", ORANGE)):
    ev = sorted({e["milestone"]: e for e in map(json.loads, open(f"{RUNS}/{name}/evals.jsonl"))}.values(),
                key=lambda e: e["milestone"])
    x = [e["online_env_steps"] / 1000 for e in ev]
    ci = [wilson(e["n_success"], e["n_episodes"]) for e in ev]
    ax.fill_between(x, [a for a, _ in ci], [b for _, b in ci], color=c, alpha=0.12, lw=0)
    ax.plot(x, [e["success_rate"] for e in ev], color=c, lw=2, marker="o", ms=3.5, label=label)
ax.axhline(k_fr / len(fr), color=INK2, ls=(0, (5, 4)), lw=1.4, label=f"frozen π0, same async protocol ({k_fr}/{len(fr)})")
ax.text(598, k_fr / len(fr) - 0.045, f"frozen π0 {k_fr}/{len(fr)}", fontsize=8, color=INK2, ha="right")
ax.text(560, 0.94, "ARLI 45/50", fontsize=8, color=INK, ha="right")
ax.text(560, 0.035, "DSRL 0/50 at every eval", fontsize=8, color=INK, ha="right")
ax.set_ylabel("success rate (50 held-out eval episodes)", color=INK2, fontsize=9)
ax.legend(fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.1))

# --- panel 2: new groups vs seed 0 -----------------------------------------------------------
ax = axes[1]
style(ax, "New groups (running) vs seed 0, first 150k")
LS = {"seed 0 (original)": "-", "seed 1, std 0.3": (0, (6, 3)), "seed 0, var 0.3 (std .548)": (0, (1.5, 2))}
xmax = 0
for method, c, base in (("ARLI", BLUE, "arli"), ("DSRL", ORANGE, "dsrl")):
    for cond, name in (("seed 0 (original)", f"r2_{base}_seed0"), ("seed 1, std 0.3", f"r2_{base}_seed1_std030"),
                       ("seed 0, var 0.3 (std .548)", f"r2_{base}_seed0_var030")):
        x, y = evals(name)
        if name.endswith("seed0"):
            keep = [i for i, v in enumerate(x) if v <= 150]
            x, y = [x[i] for i in keep], [y[i] for i in keep]
        else:
            xmax = max(xmax, x[-1] if x else 0)
        ax.plot(x, y, color=c, lw=1.8, ls=LS[cond], marker="o", ms=3, label=f"{method} · {cond}")
ax.set_xlim(-4, 155)
ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=1)

# --- panel 3: S1 --------------------------------------------------------------------------
ax = axes[2]
style(ax, "S1 · +100k from seed-0 ARLI @600k")
x, y = evals("s1_A_seed0")
ax.plot(x, y, color=AQUA, lw=2, marker="o", ms=3.5, label="A · continued ARLI (valid)")
end_label(ax, x, y, f"A {round(y[-1] * 50)}/50")
first = True
for b in "BCD":
    x, y = evals(f"s1_{b}_seed0")
    ax.plot(x, y, color=GRAY, lw=1.2, ls=(0, (2, 2)), marker="o", ms=2.5,
            label="v1 B / C / D · INVALID (critics diverged, Q>0)" if first else None)
    end_label(ax, x, y, f"v1 {b}", dy={"B": 6, "C": 8, "D": 0}[b])
    first = False
for b, c in (("B", YELLOW), ("C", MAGENTA), ("D", ORANGE)):
    x, y = evals(f"s1_{b}_seed0_v2")
    if x:
        ax.plot(x, y, color=c, lw=2, marker="o", ms=3.5, label=f"v2 {b}")
ax.text(57, 0.17, "v2 B / C / D (fixed credit)\nlaunched 09:00 — no evals yet", fontsize=8, color=INK2, ha="center")
ax.set_xlim(-4, 118)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")

fig.suptitle("Streaming-ARLI · ALOHA Transfer Cube, t_delay=20, t_delay^rl=10, no RTC · paper-based ARLI reconstruction "
             f"· status {time.strftime('%Y-%m-%d %H:%M')} EDT", x=0.01, ha="left", fontsize=11.5, color=INK, y=0.995)
fig.text(0.01, 0.005, "Real evaluations only. Panel 1 shading: 95% Wilson interval over the 50 eval episodes "
         "(one training seed, not a seed band). Panels 2–3: single runs, in progress.", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
fig.savefig("results/overview.png", dpi=150, facecolor=SURF)
print("wrote results/overview.png")
