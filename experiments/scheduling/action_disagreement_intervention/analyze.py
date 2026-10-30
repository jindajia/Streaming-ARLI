"""Analysis of the t=175 intervention test (CPU): outcomes, paired comparisons, temporal checks, mechanism plots, videos.

    python action_disagreement_intervention/analyze.py [--src DIR] [--video_seeds 0 1 2]
"""
import argparse
import csv
import json
import os
import subprocess

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REF = "/N/scratch/jindjia/arli/runs_scheduling/action_disagreement_probe/reference/ref_10000004.npz"
FF = "/N/scratch/jindjia/ms-playwright/ffmpeg-1013/ffmpeg-linux"
T0, NS = 175, 10
COL = {"A": "#5b6573", "B": "#2a63c4", "C": "#d9632b"}
NAME = {"A": "A: fixed baseline", "B": "B: one early replan (175)", "C": "C: three early replans (175, 185, 195)"}


def load(src):
    rows = {}
    for l in open(f"{src}/rollouts.jsonl"):
        r = json.loads(l)
        rows.setdefault(r["tag"], r)
    tr = {t: dict(np.load(f"{src}/trace_{t}.npz")) for t in rows}
    return rows, tr


def paired(rows, a, b):
    o = dict(both_succeed=0, only_a=0, only_b=0, both_fail=0, dret=[], dtime=[])
    for s in range(NS):
        x, y = rows.get(f"{a}_s{s}"), rows.get(f"{b}_s{s}")
        if not x or not y:
            continue
        k = {(True, True): "both_succeed", (True, False): "only_a", (False, True): "only_b", (False, False): "both_fail"}[(x["success"], y["success"])]
        o[k] += 1
        o["dret"].append(round(x["discounted_task_return"] - y["discounted_task_return"], 5))
        if x["success"] and y["success"]:
            o["dtime"].append(x["steps_to_success"] - y["steps_to_success"])
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/N/scratch/jindjia/arli/runs_scheduling/action_disagreement_intervention")
    ap.add_argument("--video_seeds", type=int, nargs="*", default=None)
    a = ap.parse_args()
    out = f"{HERE}/results"
    os.makedirs(out, exist_ok=True)
    rows, tr = load(a.src)
    Z = np.load(REF)
    main_tags = [f"{b}_s{s}" for s in range(NS) for b in "ABC" if f"{b}_s{s}" in rows]
    S = {}
    # --- outcomes
    S["success"] = {b: f"{sum(rows[t]['success'] for t in main_tags if t[0] == b)}/{sum(t[0] == b for t in main_tags)}" for b in "ABC"}
    S["completion_steps_from_175"] = {b: sorted(rows[t]["steps_to_success"] for t in main_tags if t[0] == b and rows[t]["success"]) for b in "ABC"}
    S["failures_timeout_at_400"] = {b: sum(not rows[t]["success"] for t in main_tags if t[0] == b) for b in "ABC"}
    S["mean_discounted_return"] = {b: float(np.mean([rows[t]["discounted_task_return"] for t in main_tags if t[0] == b])) for b in "ABC"}
    S["inference_calls"] = {b: [rows[t]["inference_calls"] for t in main_tags if t[0] == b] for b in "ABC"}
    S["paired"] = {"B_vs_A": paired(rows, "B", "A"), "C_vs_A": paired(rows, "C", "A"), "C_vs_B": paired(rows, "C", "B")}
    S["schedule_as_expected"] = {t: rows[t]["schedule_as_expected"] for t in main_tags if not rows[t]["schedule_as_expected"]} or "all"
    # --- temporal checks
    ref_cmd = Z["queue"][T0, :10]
    same_old = all(np.array_equal(tr[t]["cmd"][:10], ref_cmd) for t in main_tags)
    cube_same = max(float(np.abs(tr[t]["cube"][:10] - tr[main_tags[0]]["cube"][:10]).max()) for t in main_tags)
    bc_same = {s: bool(np.array_equal(tr[f"B_s{s}"]["cmd"][:20], tr[f"C_s{s}"]["cmd"][:20])) for s in range(NS) if f"B_s{s}" in tr and f"C_s{s}" in tr}
    ab_diff_first = {}
    for s in range(NS):
        if f"A_s{s}" in tr and f"B_s{s}" in tr:
            d = np.flatnonzero(np.abs(tr[f"A_s{s}"]["cmd"][:min(len(tr[f"A_s{s}"]["cmd"]), len(tr[f"B_s{s}"]["cmd"]))] - tr[f"B_s{s}"]["cmd"][:min(len(tr[f"A_s{s}"]["cmd"]), len(tr[f"B_s{s}"]["cmd"]))]).max(1) > 0)
            ab_diff_first[s] = int(T0 + d[0]) if len(d) else None
    S["temporal"] = dict(all_branches_execute_recorded_old_actions_175_184=bool(same_old), cube_pose_max_diff_175_184_across_rollouts=cube_same,
                         B_and_C_identical_commands_175_194=bc_same, first_step_where_A_and_B_commands_differ=ab_diff_first,
                         right_contact_lost_step=sorted({rows[t]["first_right_contact_loss"] for t in main_tags}, key=lambda x: (x is None, x)),
                         right_contact_after_step_185={b: sum(bool(rows[t]["right_contact_at_185"]) for t in main_tags if t[0] == b) for b in "ABC"},
                         regained_right_contact_after_185={b: sum(bool(rows[t]["regained_right_contact_after_185"]) for t in main_tags if t[0] == b) for b in "ABC"},
                         reached_left_contact={b: sum(rows[t]["first_left_contact"] is not None for t in main_tags if t[0] == b) for b in "ABC"})
    # --- controls
    c = {}
    if "ctrl_A_refnoise" in tr:
        x = tr["ctrl_A_refnoise"]
        n = min(len(x["cmd"]), len(Z["t"]) - T0)
        dif = np.abs(x["cmd"][:n] - Z["queue"][T0:T0 + n, 0]).max(1)
        c["refnoise_vs_saved_reference"] = dict(steps_compared=int(n), first_command_difference_step=int(T0 + np.flatnonzero(dif > 0)[0]) if (dif > 0).any() else None,
                                                max_command_diff=float(dif.max()), stage_sequence_equal=bool(np.array_equal(x["stage"][:n], Z["stage"][T0:T0 + n])),
                                                outcome=dict(replay_success=rows["ctrl_A_refnoise"]["success"], replay_t_end=rows["ctrl_A_refnoise"]["t_end"],
                                                             saved_t_end=int(len(Z["t"]))))
    if "ctrl_A_s0_rep1" in tr and "ctrl_A_s0_rep2" in tr:
        x, y = tr["ctrl_A_s0_rep1"], tr["ctrl_A_s0_rep2"]
        n = min(len(x["cmd"]), len(y["cmd"]))
        dif = np.abs(x["cmd"][:n] - y["cmd"][:n]).max(1)
        c["A_s0_repeat"] = dict(identical_commands=bool((dif == 0).all()), first_difference_step=int(T0 + np.flatnonzero(dif > 0)[0]) if (dif > 0).any() else None,
                                t_end=[rows["ctrl_A_s0_rep1"]["t_end"], rows["ctrl_A_s0_rep2"]["t_end"]],
                                success=[rows["ctrl_A_s0_rep1"]["success"], rows["ctrl_A_s0_rep2"]["success"]])
        if "A_s0" in tr:
            z = tr["A_s0"]
            m = min(n, len(z["cmd"]))
            c["A_s0_repeat"]["main_A_s0_identical_to_rep1"] = bool(np.array_equal(z["cmd"][:m], x["cmd"][:m]))
    S["controls"] = c
    S["saved_reference_outcome"] = dict(success=False, t_end=int(len(Z["t"])), note="original regenerated reference (separate from the paired continuations)")
    json.dump(S, open(f"{out}/summary.json", "w"), indent=1, default=str)
    with open(f"{out}/rollouts.csv", "w", newline="") as fh:
        keys = ["tag", "branch", "success", "termination", "t_end", "steps_to_success", "discounted_task_return", "inference_calls",
                "schedule_as_expected", "first_right_contact_loss", "right_contact_at_185", "regained_right_contact_after_185",
                "first_left_contact", "max_stage", "max_cube_height", "requests"]
        w = csv.writer(fh)
        w.writerow(keys)
        for t in sorted(rows):
            r = rows[t]
            w.writerow([r.get(k) if k != "requests" else ";".join(f"obs{q['obs_t']}/ready{q['ready_t']}/handoff{q['handoff_t']}/shift{q['shift']}/commit{q['committed']}/ovl{q['overlap_end']}" for q in r["requests"]) for k in keys])
    print(json.dumps(S, indent=1, default=str)[:5000])
    plots(rows, tr, main_tags, out)
    seeds = a.video_seeds if a.video_seeds is not None else pick_video_seeds(rows)
    for s in seeds:
        video(a.src, rows, tr, s)


def pick_video_seeds(rows):
    """Prefer seeds where the branches' outcomes differ; fill with seed 0."""
    diff = [s for s in range(NS) if len({rows.get(f"{b}_s{s}", {}).get("success") for b in "ABC"}) > 1]
    return (diff + [s for s in range(NS) if s not in diff])[:3]


def plots(rows, tr, tags, out):
    # 1 outcome grid
    fig, ax = plt.subplots(figsize=(9, 2.8))
    for i, b in enumerate("ABC"):
        for s in range(NS):
            r = rows.get(f"{b}_s{s}")
            if not r:
                continue
            ax.add_patch(plt.Rectangle((s, 2 - i), 0.92, 0.85, color="#1d7f4e" if r["success"] else "#b23a3a"))
            ax.text(s + 0.46, 2 - i + 0.42, str(r["t_end"]) if r["success"] else "fail", ha="center", va="center", color="white", fontsize=8)
    ax.set_xlim(0, NS)
    ax.set_ylim(0, 3)
    ax.set_yticks([2.42, 1.42, 0.42], [f"{NAME[b]}  ({sum(rows[t]['success'] for t in tags if t[0] == b)}/{sum(t[0] == b for t in tags)})" for b in "ABC"], fontsize=8)
    ax.set_xticks(np.arange(NS) + 0.46, [f"seed {s}" for s in range(NS)], fontsize=8)
    ax.set_title("Outcome per paired continuation seed (cell text = success step, or fail = timeout at 400)", fontsize=9, loc="left")
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{out}/outcomes.png", dpi=140)
    plt.close(fig)
    # 2 mechanism traces
    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex=True)
    for j, b in enumerate("ABC"):
        for s in range(NS):
            x = tr.get(f"{b}_s{s}")
            if x is None:
                continue
            t = x["t"]
            ok = rows[f"{b}_s{s}"]["success"]
            kw = dict(color="#1d7f4e" if ok else "#b23a3a", lw=0.9, alpha=0.75)
            axes[0, j].plot(t, x["cube"][:, 2], **kw)
            axes[1, j].plot(t, x["cmd"][:, 13], **kw)
            axes[2, j].plot(t, x["grip_state_r"], **kw)
            axes[3, j].step(t, x["stage"] + 0.06 * (s - 4.5) / 4.5, where="post", **kw)
        for ax in axes[:, j]:
            ax.axvline(185, color="k", lw=0.8, ls="--")
            ax.axvspan(175, 185, color="#999999", alpha=0.12, lw=0)
            for tt in {"A": [190, 200], "B": [200, 210], "C": [195, 205, 220, 230]}[b]:
                ax.axvline(tt, color=COL[b], lw=0.6, ls=":")
            ax.axvline(178, color="#b23a3a", lw=0.8, ls="-.")
            ax.grid(alpha=0.3)
        axes[0, j].set_title(NAME[b], fontsize=10)
        axes[3, j].set_xlabel("absolute control step t")
        axes[3, j].set_xlim(170, 300)
    axes[0, 0].set_ylabel("cube height z (m)")
    axes[1, 0].set_ylabel("right gripper COMMAND\n(0 closed, 1 open)")
    axes[2, 0].set_ylabel("right gripper ACTUAL\n(normalised position)")
    axes[3, 0].set_ylabel("contact stage\n(1 right touch, 2 lifted,\n3 left touch, 4 success)")
    fig.suptitle("Mechanism around the intervention (green = rollout succeeds, red = fails; 10 seeds per branch).\nGrey band t=175–184: old "
                 "actions, identical in all branches. Dashed = first possible handoff (185); dash-dot = right-contact loss (178); dotted = later "
                 "request / handoff times of that branch.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{out}/mechanism_traces.png", dpi=120)
    plt.close(fig)


def video(src, rows, tr, s):
    tags = [f"{b}_s{s}" for b in "ABC"]
    if not all(t in tr for t in tags):
        return
    panels = []
    for t in tags:
        o = json.load(open(f"{src}/frames/{t}.json"))["offsets"]
        buf = open(f"{src}/frames/{t}.bin", "rb").read()
        panels.append([cv2.imdecode(np.frombuffer(buf[o[k]:o[k + 1]], np.uint8), cv2.IMREAD_COLOR) for k in range(len(o) - 1)])
    L = max(len(p) for p in panels)
    os.makedirs(f"{src}/videos", exist_ok=True)
    mj = f"{src}/videos/tmp_s{s}.mjpeg"
    with open(mj, "wb") as fh:
        for k in range(L):
            tiles = []
            for b, t, fr in zip("ABC", tags, panels):
                x = tr[t]
                i = min(k, len(fr) - 1)
                im = cv2.resize(fr[i], (426, 320))
                cv2.rectangle(im, (0, 0), (426, 92), (0, 0, 0), -1)
                lines = [f"{NAME[b]}", f"t={T0 + i}  chunk {int(x['chunk'][i])}  queue {int(x['q'][i])}  stage {int(x['stage'][i])}",
                         f"R-grip cmd {x['cmd'][i][13]:.2f}  actual {x['grip_state_r'][i]:.2f}"]
                pend = x["pending"][i]
                lines.append(f"request in flight: obs {pend[0]} -> handoff {pend[2]}" if pend[0] >= 0 else "no request in flight")
                for j, ln in enumerate(lines):
                    cv2.putText(im, ln, (6, 18 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255) if j else (0, 220, 255), 1)
                if k >= len(fr) - 1:
                    cv2.putText(im, "SUCCESS" if rows[t]["success"] else "TIMEOUT", (10, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 220, 0) if rows[t]["success"] else (0, 0, 255), 2)
                tiles.append(im)
            fh.write(cv2.imencode(".jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes())
    subprocess.run([FF, "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", "25", "-i", f"file:{mj}",
                    "-c:v", "libvpx-vp9", "-b:v", "900k", "-deadline", "realtime", "-cpu-used", "8", "-pix_fmt", "yuv420p",
                    f"file:{src}/videos/abc_seed{s}.webm"], check=False)
    os.remove(mj)
    print("video", f"{src}/videos/abc_seed{s}.webm")


if __name__ == "__main__":
    main()
