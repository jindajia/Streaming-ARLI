"""Analyse policy_diagnostics outputs -> compact tables, statistics and figures in policy_diagnostics/results/.

State dependence (final checkpoint, diagnostic bank of N states, p_s over the run's valid horizons, p_bar = mean p_s):
    mean_s KL(p_s || Uniform) = KL(p_bar || Uniform) + mean_s KL(p_s || p_bar)          (exact identity)
    total deviation            = global (state-independent) part + state-dependent part
Paired success difference: Newcombe (1998) method 10 score interval for paired proportions.

    python policy_diagnostics/analyze.py --src /N/scratch/jindjia/arli/runs_dehp/policy_diagnostics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ARM_COLOR = {"U": "#a3a29c", "S": "#2a78d6", "D": "#6250d6", "F25": "#52514e", "M": "#1baf7a"}
ARM_NAME = {"U": "U: uniform stochastic", "S": "S: trained stochastic", "D": "D: trained argmax",
            "F25": "F25: fixed h=25", "M": "M: global-average stochastic"}
PHASE_COLOR = {"approach": "#cde2fb", "grasp": "#eda100", "lift_transport": "#eb6834", "transfer": "#1baf7a",
               "lost_contact": "#e34948", "success": "#6250d6"}
RUN_NAME = {"A": "Run A (no delay, RTC off, h in 1..50)", "B": "Run B (delay 10, RTC on, h in 10..40)"}


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe_paired(a, b, c, d, z=1.96):
    """a both succeed, b first-only, c second-only, d both fail -> (diff, lo, hi) for p1 - p2."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    if den == 0:
        phi = 0.0
    else:
        num = a * d - b * c
        num = max(num - n / 2, 0) if num > 0 else num
        phi = num / math.sqrt(den)
    dl = math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0))
    du = math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0))
    return p1 - p2, p1 - p2 - dl, p1 - p2 + du


def ent(p):
    q = p[p > 0]
    return float(-(q * np.log(q)).sum())


def kl(p, q):
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


def style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/N/scratch/jindjia/arli/runs_dehp/policy_diagnostics")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    summary = dict(state_dependence={}, evaluation={}, paired={})

    # ------------------------------------------------------------------ state bank distributions
    for run in ("A", "B"):
        bank = jl(f"{args.src}/state_bank_{run}.jsonl")
        traj = jl(f"{args.src}/trajectory_dists_{run}.jsonl")
        if not bank:
            continue
        ck = sorted(k for k in bank[0] if k.startswith("p_"))
        final = "p_learner_final"
        inter = [k for k in ck if k != final][0]
        valid = np.asarray(bank[0]["valid"]) - 1
        hs = valid + 1
        n_valid = len(valid)
        uni = np.zeros(50)
        uni[valid] = 1 / n_valid
        rows, sd = [], {}
        for key in (inter, final):
            P = np.asarray([b[key] for b in bank])
            pbar = P.mean(0)
            Eh = (P * np.arange(1, 51)).sum(1)
            sdh = np.sqrt((P * (np.arange(1, 51) - Eh[:, None]) ** 2).sum(1))
            H = np.asarray([ent(p) for p in P])
            kl_u = np.asarray([kl(p, uni) for p in P])
            kl_bar = np.asarray([kl(p, pbar) for p in P])
            tv_bar = 0.5 * np.abs(P - pbar).sum(1)
            am = P.argmax(1) + 1
            sd[key] = dict(
                n_states=len(P), n_valid=n_valid,
                mean_KL_to_uniform=float(kl_u.mean()), KL_pbar_to_uniform=kl(pbar, uni),
                mean_KL_to_pbar=float(kl_bar.mean()),
                identity_check=float(kl_u.mean() - kl(pbar, uni) - kl_bar.mean()),
                share_state_dependent=float(kl_bar.mean() / max(kl_u.mean(), 1e-15)),
                TV_pbar_uniform=float(0.5 * np.abs(pbar - uni).sum()), mean_TV_state_pbar=float(tv_bar.mean()),
                max_TV_state_pbar=float(tv_bar.max()),
                Eh_mean=float(Eh.mean()), Eh_std_across_states=float(Eh.std()), Eh_min=float(Eh.min()), Eh_max=float(Eh.max()),
                maxprob_mean=float(P.max(1).mean()), maxprob_max=float(P.max(1).max()), uniform_prob=1 / n_valid,
                norm_entropy_mean=float((H / math.log(n_valid)).mean()), norm_entropy_min=float((H / math.log(n_valid)).min()),
                argmax_counts=dict(Counter(int(a) for a in am).most_common()),
                pbar_top5=[(int(h), float(pbar[h - 1])) for h in np.argsort(-pbar)[:5] + 1])
            if key == final:
                for b, p, e, s_, h_, k1, k2 in zip(bank, P, Eh, sdh, H, kl_u, kl_bar):
                    rows.append(dict(run=run, source_seed=b["source_seed"], decision=b["decision"], t=b["t"],
                                     remaining=b["remaining"], phase=b["phase"], plan_obs_age=b["plan_obs_age"],
                                     argmax_h=int(p.argmax() + 1), argmax_p=float(p.max()), expected_h=float(e), sd_h=float(s_),
                                     entropy=float(h_), norm_entropy=float(h_ / math.log(n_valid)), KL_to_uniform=float(k1),
                                     KL_to_pbar=float(k2), p_valid=json.dumps([round(float(p[h - 1]), 6) for h in hs])))
        summary["state_dependence"][run] = sd
        with open(f"{args.out}/state_bank_{run}_final.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        phases_present = Counter(b["phase"] for b in bank)
        summary["state_dependence"][run]["phases_in_bank"] = dict(phases_present)
        summary["state_dependence"][run]["phases_missing"] = [p for p in ("approach", "grasp", "lift_transport", "transfer")
                                                              if p not in phases_present]

        # Figure A/D: heatmaps (intermediate vs final) on the same states
        fig, axes = plt.subplots(2, 1, figsize=(13, 8.2), sharex=True)
        fig.patch.set_facecolor(SURF)
        for ax, key in zip(axes, (inter, final)):
            P = np.asarray([b[key] for b in bank])[:, valid]
            im = ax.imshow(P.T, aspect="auto", origin="lower", cmap="Blues", extent=(-0.5, len(bank) - 0.5, hs[0] - 0.5, hs[-1] + 0.5),
                           vmin=0, vmax=max(2.5 / n_valid, float(P.max())))
            style(ax, f"{key[2:]}: p(h | state) on the shared diagnostic state bank (uniform = {1 / n_valid:.3f})")
            ax.set_ylabel("execution horizon h", color=INK2, fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01).set_label("probability", fontsize=8)
            prev = None
            for i, b in enumerate(bank):
                ax.add_patch(plt.Rectangle((i - 0.5, hs[0] - 0.5 - 2.2), 1, 2.0, color=PHASE_COLOR[b["phase"]], clip_on=False))
                if b["source_seed"] != prev and i:
                    ax.axvline(i - 0.5, color=INK, lw=0.8)
                prev = b["source_seed"]
        axes[1].set_xlabel("decision states (grouped by source episode, time order; color strip = phase)", color=INK2, fontsize=9)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PHASE_COLOR.values()]
        axes[0].legend(handles, list(PHASE_COLOR), fontsize=7.5, ncol=6, frameon=False, loc="upper right",
                       bbox_to_anchor=(1, 1.16))
        fig.suptitle(f"{RUN_NAME[run]} — full horizon distributions (not sampled-h histograms)", x=0.01, ha="left",
                     fontsize=11, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(f"{args.out}/fig_A_heatmap_{run}.png", dpi=140, facecolor=SURF)
        plt.close(fig)

        # Figure B: representative states (first bank state of each phase, in bank order)
        reps, seen = [], set()
        for b in bank:
            if b["phase"] not in seen:
                seen.add(b["phase"])
                reps.append(b)
        fig, ax = plt.subplots(figsize=(9, 4.3))
        fig.patch.set_facecolor(SURF)
        for b in reps[:5]:
            p = np.asarray(b[final])[valid]
            ax.plot(hs, p, lw=1.6, label=f"{b['phase']} (seed {b['source_seed']}, t={b['t']})", color=PHASE_COLOR[b["phase"]])
        ax.axhline(1 / n_valid, color=INK2, ls=(0, (4, 3)), lw=1, label="uniform")
        style(ax, f"{RUN_NAME[run]}: final-checkpoint p(h | state), one state per phase")
        ax.set_xlabel("execution horizon h", color=INK2, fontsize=9)
        ax.set_ylabel("probability", color=INK2, fontsize=9)
        ax.set_ylim(0, max(3 / n_valid, max(float(np.max(np.asarray(b[final]))) for b in reps) * 1.1))
        ax.legend(fontsize=7.5, frameon=False)
        fig.tight_layout()
        fig.savefig(f"{args.out}/fig_B_state_curves_{run}.png", dpi=140, facecolor=SURF)
        plt.close(fig)

        # Figure C: along source trajectories (all decisions of 3 source episodes)
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        fig.patch.set_facecolor(SURF)
        by_ep = defaultdict(list)
        for r in traj:
            by_ep[r["source_seed"]].append(r)
        for seed in list(by_ep)[:3]:
            tr = by_ep[seed]
            P = np.asarray([r[final] for r in tr])
            t = [r["t"] for r in tr]
            H = np.asarray([ent(p) for p in P]) / math.log(n_valid)
            axes[0].plot(t, (P * np.arange(1, 51)).sum(1), "-o", ms=3, label=f"seed {seed}")
            axes[1].plot(t, P.max(1), "-o", ms=3)
            axes[2].plot(t, H, "-o", ms=3)
        for ax, ttl in zip(axes, ("expected h", "maximum probability", "normalised entropy")):
            style(ax, ttl)
            ax.set_xlabel("decision time t (control steps)", color=INK2, fontsize=9)
        axes[1].axhline(1 / n_valid, color=INK2, ls=(0, (4, 3)), lw=1)
        axes[0].legend(fontsize=7.5, frameon=False)
        fig.suptitle(f"{RUN_NAME[run]}: final checkpoint along fixed-h=25 source trajectories", x=0.01, ha="left", fontsize=10.5)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(f"{args.out}/fig_C_trajectories_{run}.png", dpi=140, facecolor=SURF)
        plt.close(fig)

    # ------------------------------------------------------------------ evaluation arms
    eps = jl(f"{args.src}/episodes.jsonl")
    tab = []
    for e in eps:
        hs_sel = [d["h"] for d in e["decisions"]]
        tab.append(dict(run=e["run"], arm=e["arm"], seedset=e["seedset"], seed=e["seed"], success=int(e["success"]),
                        t_end=e["t_end"], discounted_return=round(e["discounted_return"], 4),
                        inference_calls=e["inference_calls"], decisions=len(hs_sel), mean_h=float(np.mean(hs_sel)),
                        selected_h=" ".join(map(str, hs_sel)), times=" ".join(str(d["t"]) for d in e["decisions"]),
                        mean_p_selected=None if e["decisions"][0]["p_selected"] is None else
                        float(np.mean([d["p_selected"] for d in e["decisions"]])),
                        async_violations=e["async_violations"], underruns=e["underruns"]))
    if tab:
        with open(f"{args.out}/episodes.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tab[0]))
            w.writeheader()
            w.writerows(tab)
    res = defaultdict(dict)
    for (run, arm, ss) in sorted({(t["run"], t["arm"], t["seedset"]) for t in tab}):
        T = [t for t in tab if (t["run"], t["arm"], t["seedset"]) == (run, arm, ss)]
        E = [e for e in eps if (e["run"], e["arm"], e["seedset"]) == (run, arm, ss)]
        k, n = sum(t["success"] for t in T), len(T)
        allh = [d["h"] for e in E for d in e["decisions"]]
        res[f"{run}/{ss}"][arm] = dict(successes=k, n=n, wilson95=wilson(k, n), mean_t_end=float(np.mean([t["t_end"] for t in T])),
                                       mean_inference_calls=float(np.mean([t["inference_calls"] for t in T])),
                                       h_counts=dict(sorted(Counter(allh).items())),
                                       distinct_h=len(set(allh)),
                                       violations=int(sum(t["async_violations"] for t in T)),
                                       underruns=int(sum(t["underruns"] for t in T)))
        if arm == "D":
            am = [d["argmax_final"] for e in E for d in e["decisions"]]
            res[f"{run}/{ss}"][arm]["argmax_constant"] = len(set(am)) == 1
            res[f"{run}/{ss}"][arm]["argmax_values"] = dict(Counter(am))
    summary["evaluation"] = res

    def pair(run, ss, a1, a2):
        x = {t["seed"]: t for t in tab if (t["run"], t["arm"], t["seedset"]) == (run, a1, ss)}
        y = {t["seed"]: t for t in tab if (t["run"], t["arm"], t["seedset"]) == (run, a2, ss)}
        common = sorted(set(x) & set(y))
        if not common:
            return None
        a = sum(x[s]["success"] and y[s]["success"] for s in common)
        b = sum(x[s]["success"] and not y[s]["success"] for s in common)
        c = sum(not x[s]["success"] and y[s]["success"] for s in common)
        d = len(common) - a - b - c
        diff, lo, hi = newcombe_paired(a, b, c, d)
        joint = [s for s in common if x[s]["success"] and y[s]["success"]]
        return dict(n_pairs=len(common), both=a, first_only=b, second_only=c, neither=d, diff=diff, ci95=(lo, hi),
                    jointly_successful_mean_t_end=(float(np.mean([x[s]["t_end"] for s in joint])),
                                                   float(np.mean([y[s]["t_end"] for s in joint]))) if joint else None)

    for run in ("A", "B"):
        for ss in ("new20", "orig10"):
            for a1, a2 in (("S", "U"), ("S", "M"), ("S", "F25"), ("D", "F25"), ("M", "U"), ("D", "S")):
                r = pair(run, ss, a1, a2)
                if r:
                    summary["paired"][f"{run}/{ss}/{a1}-vs-{a2}"] = r
    json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1, default=float)

    # Figure E: evaluation summary
    if tab:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.patch.set_facecolor(SURF)
        for col, run in enumerate(("A", "B")):
            ax = axes[0, col]
            arms = [a for a in ("U", "S", "D", "F25", "M") if a in res.get(f"{run}/new20", {})]
            for i, a in enumerate(arms):
                r = res[f"{run}/new20"][a]
                lo, hi = r["wilson95"]
                ax.bar(i, r["successes"] / r["n"], color=ARM_COLOR[a], width=0.6)
                ax.errorbar(i, r["successes"] / r["n"], yerr=[[r["successes"] / r["n"] - lo], [hi - r["successes"] / r["n"]]],
                            color=INK, capsize=3, lw=1)
                ax.text(i, min(hi + 0.03, 1.02), f"{r['successes']}/{r['n']}", ha="center", fontsize=9, color=INK)
            ax.set_xticks(range(len(arms)), [ARM_NAME[a] for a in arms], fontsize=7.5, rotation=12)
            ax.set_ylim(0, 1.1)
            style(ax, f"{RUN_NAME[run]}: success on 20 paired new seeds (95% Wilson)")
            ax = axes[1, col]
            for a in arms:
                hc = res[f"{run}/new20"][a]["h_counts"]
                tot = sum(hc.values())
                ax.plot([int(h) for h in hc], [v / tot for v in hc.values()], "o-" if len(hc) > 1 else "s", ms=3, lw=1.2,
                        color=ARM_COLOR[a], label=ARM_NAME[a])
            style(ax, f"{RUN_NAME[run]}: selected-h distribution during evaluation")
            ax.set_xlabel("execution horizon h", color=INK2, fontsize=9)
            ax.set_ylabel("fraction of decisions", color=INK2, fontsize=9)
            ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(f"{args.out}/fig_E_evaluation.png", dpi=140, facecolor=SURF)
        plt.close(fig)
    print(json.dumps(summary, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
