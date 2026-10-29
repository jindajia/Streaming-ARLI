"""Analysis of the R-sensitivity matrix: tables, paired comparisons, figures (CPU).

    python r_sensitivity_probe/analyze.py [--src <bulky dir>]

Question A: among forced branches observing/launching at the same snapshot, does R change outcomes?
Question B: does a forced intervention differ from the full fixed-baseline control?
Pairing: the same snapshot and continuation seed (shared noise stream indexed by request ordinal).
Completion time is reported only for successful branches; timeouts are listed separately.
"""
import argparse
import csv
import itertools
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PHASES = ["ordinary_motion", "approach", "grasp_lift", "transfer"]
PH_LABEL = {"ordinary_motion": "ordinary motion", "approach": "approach", "grasp_lift": "grasp / lift", "transfer": "transfer"}
SRC_LABEL = {10000014: "src 014 (success)", 10000016: "src 016 (late success)", 10000004: "src 004 (failure)"}
CHOICES = {5: ["R10", "R15", "R20", "baseline"], 10: ["R10", "R12", "R15", "baseline"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/N/scratch/jindjia/arli/runs_scheduling/r_sensitivity_probe")
    a = ap.parse_args()
    out = f"{HERE}/results"
    os.makedirs(out, exist_ok=True)
    man = json.load(open(f"{HERE}/manifest.json"))
    rows = [json.loads(l) for l in open(f"{a.src}/branches.jsonl")]
    seen = {}
    for r in rows:  # keep the first row per key (a resumed job never rewrites finished keys)
        seen.setdefault((r["snapshot_id"], r["choice"], r["cont_seed"]), r)
    rows = list(seen.values())
    B = {(r["snapshot_id"], r["choice"], r["cont_seed"]): r for r in rows}
    snaps = [m for m in man["snapshots"] if not m.get("missing")]
    order = [m for src in (10000014, 10000016, 10000004) for ph in PHASES for m in snaps
             if m["source_seed"] == src and m["phase"] == ph]

    # ---- compact per-branch CSV
    cols = ["snapshot_id", "phase", "source_seed", "cont_seed", "choice", "t", "p", "q", "N", "d", "requested_R",
            "success", "termination", "t_end", "steps_to_success", "steps_to_timeout", "terminated_before_first_handoff",
            "discounted_task_return", "task_return", "inference_calls", "underrun_ticks", "max_stage", "first_stage2",
            "first_stage3", "fr_obs_t", "fr_ready_t", "fr_handoff_t", "fr_shift", "fr_committed", "fr_overlap_end",
            "fr_installed", "anomalies", "wall_s"]
    with open(f"{out}/branches.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in sorted(rows, key=lambda r: (r["snapshot_id"], r["choice"], r["cont_seed"])):
            fr = r["first_request"] or {}
            w.writerow([r.get(c) for c in cols[:24]] + [fr.get("obs_t"), fr.get("ready_t"), fr.get("handoff_t"), fr.get("shift"),
                                                       fr.get("committed"), fr.get("overlap_end"), fr.get("installed_slice"),
                                                       ";".join(r["anomalies"]), round(r["wall_s"], 2)])

    # ---- per cell table
    cell = {}
    with open(f"{out}/cells.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["snapshot_id", "source_seed", "phase", "p", "t", "choice", "n", "successes", "mean_disc_return",
                    "success_steps_from_snapshot", "timeouts", "terminated_before_handoff", "mean_inference_calls"])
        for m in order:
            for ch in CHOICES[m["p"]]:
                rr = [B[k] for k in [(m["snapshot_id"], ch, c) for c in (0, 1, 2)] if k in B]
                if not rr:
                    continue
                cell[(m["snapshot_id"], ch)] = rr
                w.writerow([m["snapshot_id"], m["source_seed"], m["phase"], m["p"], m["t"], ch, len(rr), sum(x["success"] for x in rr),
                            round(float(np.mean([x["discounted_task_return"] for x in rr])), 4),
                            " ".join(str(x["steps_to_success"]) for x in rr if x["success"]),
                            sum(x["termination"] == "timeout" for x in rr), sum(bool(x["terminated_before_first_handoff"]) for x in rr),
                            round(float(np.mean([x["inference_calls"] for x in rr])), 1)])

    # ---- paired comparisons
    def paired(m, ca, cb):
        res = dict(both=0, only_a=0, only_b=0, neither=0, dret=[], dtime=[], n=0)
        for c in (0, 1, 2):
            x, y = B.get((m["snapshot_id"], ca, c)), B.get((m["snapshot_id"], cb, c))
            if not x or not y:
                continue
            res["n"] += 1
            res[{(True, True): "both", (True, False): "only_a", (False, True): "only_b", (False, False): "neither"}[(x["success"], y["success"])]] += 1
            res["dret"].append(x["discounted_task_return"] - y["discounted_task_return"])
            if x["success"] and y["success"]:
                res["dtime"].append(x["steps_to_success"] - y["steps_to_success"])
        return res
    comp = []
    for m in order:
        forced = [c for c in CHOICES[m["p"]] if c != "baseline"]
        for ca, cb in itertools.combinations(forced, 2):
            comp.append(dict(question="A_forced_vs_forced", snapshot_id=m["snapshot_id"], phase=m["phase"], source_seed=m["source_seed"], p=m["p"],
                             a=ca, b=cb, **paired(m, ca, cb)))
        for ca in forced:
            comp.append(dict(question="B_forced_vs_baseline", snapshot_id=m["snapshot_id"], phase=m["phase"], source_seed=m["source_seed"], p=m["p"],
                             a=ca, b="baseline", **paired(m, ca, "baseline")))
    json.dump(comp, open(f"{out}/paired_comparisons.json", "w"), indent=1)
    with open(f"{out}/paired_comparisons.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["question", "snapshot_id", "phase", "source_seed", "p", "a", "b", "n_pairs", "both_succeed", "only_a", "only_b",
                    "both_fail", "disc_return_diff_a_minus_b", "completion_time_diff_a_minus_b_when_both_succeed"])
        for c in comp:
            w.writerow([c["question"], c["snapshot_id"], c["phase"], c["source_seed"], c["p"], c["a"], c["b"], c["n"], c["both"],
                        c["only_a"], c["only_b"], c["neither"], " ".join(f"{d:+.4f}" for d in c["dret"]), " ".join(f"{d:+d}" for d in c["dtime"])])

    # ---- figures: success heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
    for ax, p in zip(axes, (5, 10)):
        ms = [m for m in order if m["p"] == p]
        chs = CHOICES[p]
        M = np.full((len(ms), len(chs)), np.nan)
        for i, m in enumerate(ms):
            for j, ch in enumerate(chs):
                rr = cell.get((m["snapshot_id"], ch))
                if rr:
                    M[i, j] = sum(x["success"] for x in rr) / len(rr)
                    ax.text(j + (0.25 if ch == "baseline" else 0), i, f"{sum(x['success'] for x in rr)}/{len(rr)}", ha="center", va="center",
                            fontsize=9, color="white" if M[i, j] < 0.5 else "black")
        Mx = np.insert(M, len(chs) - 1, np.nan, axis=1)  # visual gap before the baseline column
        ax.imshow(np.where(np.isnan(Mx), np.nan, Mx), cmap="viridis", vmin=0, vmax=1, aspect="auto",
                  extent=(-0.5, len(chs) - 0.5 + 0.5, len(ms) - 0.5, -0.5))
        ax.set_xticks([0, 1, 2, 3.25], [f"forced R={c[1:]}" if c != "baseline" else "full baseline\n(control)" for c in chs], fontsize=8)
        ax.set_yticks(range(len(ms)), [f"{SRC_LABEL[m['source_seed']]} · {PH_LABEL[m['phase']]} (t={m['t']})" for m in ms], fontsize=8)
        ax.set_title(f"p = {p} (q = {25 - p}): success x/3 per cell", fontsize=10)
        for i in range(1, len(ms)):
            if ms[i]["source_seed"] != ms[i - 1]["source_seed"]:
                ax.axhline(i - 0.5, color="white", lw=2)
    fig.suptitle("Same-state forced-R branching (N = 10, d = 10, RTC on); 3 paired continuation seeds per cell. "
                 "Screening pilot: 3 repetitions cannot establish equivalence.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{out}/success_heatmaps.png", dpi=140)
    plt.close(fig)

    # ---- figures: paired return differences and completion times
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw=dict(width_ratios=[1, 1]))
    for col, p in enumerate((5, 10)):
        ms = [m for m in order if m["p"] == p]
        chs = CHOICES[p]
        ax = axes[0, col]
        forced = [c for c in chs if c != "baseline"]
        colors = {forced[0]: "#2a63c4", forced[1]: "#1a9a63", forced[2]: "#d9632b"}
        for i, m in enumerate(ms):
            for k, ch in enumerate(forced):
                c = next((x for x in comp if x["question"] == "B_forced_vs_baseline" and x["snapshot_id"] == m["snapshot_id"] and x["a"] == ch), None)
                if c:
                    ax.scatter(c["dret"], [i + (k - 1) * 0.22] * len(c["dret"]), color=colors[ch], s=18, label=f"forced R={ch[1:]} − baseline" if i == 0 else None)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(ms)), [f"{SRC_LABEL[m['source_seed']]} · {PH_LABEL[m['phase']]}" for m in ms], fontsize=8)
        ax.invert_yaxis()
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlabel("paired difference in discounted task return from the snapshot (forced − full baseline)")
        ax.set_title(f"p = {p}: paired return differences (one point per continuation seed)", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax = axes[1, col]
        for i, m in enumerate(ms):
            for k, ch in enumerate(chs):
                rr = cell.get((m["snapshot_id"], ch), [])
                y = i + (k - 1.5) * 0.18
                succ = [x["steps_to_success"] for x in rr if x["success"]]
                fails = [x for x in rr if not x["success"]]
                ax.scatter(succ, [y] * len(succ), color=(colors.get(ch, "#3d3c38")), s=18, marker="o" if ch != "baseline" else "s",
                           label=(f"forced R={ch[1:]}" if ch != "baseline" else "full baseline") if i == 0 else None)
                if fails:
                    ax.scatter([max(1, max([x["steps_from_snapshot"] for x in fails]))] * len(fails), [y] * len(fails), marker="x",
                               color=colors.get(ch, "#3d3c38"), s=25)
        ax.set_yticks(range(len(ms)), [f"{SRC_LABEL[m['source_seed']]} · {PH_LABEL[m['phase']]}" for m in ms], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("steps from snapshot to SUCCESS (× = failure; plotted at its timeout, not a completion time)")
        ax.set_title(f"p = {p}: individual completion times", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/paired_differences_and_times.png", dpi=130)
    plt.close(fig)

    n_main = len(rows)
    summary = dict(main_matrix_rollouts=n_main, snapshots=len(snaps), missing=[m["snapshot_id"] for m in man["snapshots"] if m.get("missing")],
                   cells=len(cell), anomalies=[(r["snapshot_id"], r["choice"], r["cont_seed"], r["anomalies"]) for r in rows if r["anomalies"]],
                   terminated_before_handoff=[(r["snapshot_id"], r["choice"], r["cont_seed"]) for r in rows if r["terminated_before_first_handoff"]])
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1)[:1500])


if __name__ == "__main__":
    main()
