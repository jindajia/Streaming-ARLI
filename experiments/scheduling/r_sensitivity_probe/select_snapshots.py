"""Propose the 4 phase chunks per source episode from simulator contact events, then write manifest.json and images.

Events (gym_aloha TransferCube reward per step, logged in the unmodified source rollout):
  t1 = first step with right-finger/cube contact (stage >= 1), t2 = first step lifted (>= 2),
  t3 = first left-finger/cube contact (>= 3), t4 = success (4).
Under the fixed baseline, chunk c covers control steps [25(c-1), 25c).

Rule (anchored on WHEN the forced handoffs land; forced handoffs fall in [start+15, start+25] for p = 5 and
[start+20, start+25] for p = 10, start = 25(c-1); chunks must be distinct, >= 2, with both snapshots available):
  approach        latest chunk with start <= t1 - 25: every forced handoff installs a new chunk before first contact
  grasp_lift      approach + 1: forced handoffs straddle first right-finger contact (closing / lift)
  transfer        latest chunk with start <= t3 - 30 (all forced handoffs >= 5 steps before first left-finger
                  contact) and after grasp_lift; missing if the left finger never touches the cube
  ordinary_motion chunk 2 (free-space motion from the home pose), if it precedes the approach chunk
Missing phases are recorded as missing; nothing is duplicated.  The phase descriptions in PHASE_NOTES are written
after inspecting the contact sheets (images + event ticks) and BEFORE any branch outcome exists.

    python r_sensitivity_probe/select_snapshots.py --src <bulky dir>
"""
import argparse
import json
import os
import pickle

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE_NOTES = {  # written after inspecting snapshot images + contact events, before any branch outcome
    "10000014:approach": "Right gripper (open) arriving at the cube; no contact yet (first right-finger contact at 135).",
    "10000014:grasp_lift": "Right gripper around the cube, closing; first contact and lift both register at 135 (lift held from 148).",
    "10000014:transfer": "Cube lifted to the centre in the right gripper; open left gripper approaching (left contact = success at 264).",
    "10000014:ordinary_motion": "Right arm in free-space motion from the home pose towards the cube; cube far from both grippers.",
    "10000016:approach": "Right arm moving in towards the cube; gripper not yet at the cube (first contact at 92).",
    "10000016:grasp_lift": "Right gripper reaching the cube; first contact at 92, but the cube is only lifted at 206 after a long struggle.",
    "10000016:transfer": "Cube lifted at the centre in the right gripper; open left gripper approaching (left contact = success at 366).",
    "10000016:ordinary_motion": "Right arm in free-space motion from the home pose; cube far from both grippers.",
    "10000004:approach": "Right gripper arriving at the cube; no contact yet (first contact at 103).",
    "10000004:grasp_lift": "Right gripper at the cube just after first contact (103); in the source the cube is never lifted and contact is lost at 178.",
    "10000004:ordinary_motion": "Right arm in free-space motion from the home pose; cube far from both grippers.",
}
VIDEO = ["s014_grasp_lift_p5", "s014_transfer_p5"]  # fixed before outcomes: paired videos for these snapshots, cont seed 0


def frame(src, seed, t):
    meta = next(s for s in json.load(open(f"{src}/sources/sources.json"))["sources"] if s["seed"] == seed)
    o = meta["frame_offsets"]
    buf = open(f"{src}/sources/frames_{seed}.bin", "rb").read()
    return cv2.imdecode(np.frombuffer(buf[o[t]:o[t + 1]], np.uint8), cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/N/scratch/jindjia/arli/runs_scheduling/r_sensitivity_probe")
    ap.add_argument("--sheets_only", action="store_true")
    a = ap.parse_args()
    S = json.load(open(f"{a.src}/sources/sources.json"))
    os.makedirs(f"{HERE}/snapshot_images", exist_ok=True)
    snaps_out = []
    for s in S["sources"]:
        seed, fs = s["seed"], s["first_stage"]
        t1, t2, t3 = fs.get("1"), fs.get("2"), fs.get("3")
        avail = {(c, p): t for c, p, t in s["snapshot_keys"]}
        ch = lambda t: t // 25 + 1  # noqa: E731
        chosen = {}
        if t1 is not None:
            c = (t1 - 25) // 25 + 1
            if c >= 2:
                chosen["approach"] = c
                chosen["grasp_lift"] = c + 1
        if t3 is not None:
            c = (t3 - 30) // 25 + 1
            if c > chosen.get("grasp_lift", 99):
                chosen["transfer"] = c
        if "approach" in chosen and 2 < chosen["approach"]:
            chosen["ordinary_motion"] = 2
        # contact sheet of the whole source episode: every chunk at p = 5, annotated with the contact stage
        tiles = []
        for t in range(5, s["t_end"], 25):
            im = cv2.resize(frame(a.src, seed, t), (240, 180))
            stg = s["stage_per_step"][t - 1] if t > 0 else 0
            lab = [k for k, v in chosen.items() if v == t // 25 + 1]
            cv2.putText(im, f"chunk {t // 25 + 1} t={t} stage {stg}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            if lab:
                cv2.putText(im, lab[0], (4, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            tiles.append(im)
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
        cv2.imwrite(f"{HERE}/snapshot_images/source_{seed}_contact_sheet.jpg", sheet)
        print(seed, "success", s["success"], "t_end", s["t_end"], "events t1,t2,t3,t4", t1, t2, t3, fs.get("4"), "chosen", chosen)
        for phase in ("approach", "grasp_lift", "transfer", "ordinary_motion"):
            for p in (5, 10):
                sid = f"s{seed % 1000:03d}_{phase}_p{p}"
                if phase not in chosen or (chosen[phase], p) not in avail:
                    snaps_out.append(dict(snapshot_id=sid, source_seed=seed, phase=phase, p=p, missing=True,
                                          reason="phase not reached in the source episode" if phase not in chosen
                                          else "snapshot not available (episode ended)"))
                    continue
                c = chosen[phase]
                t = avail[(c, p)]
                im = frame(a.src, seed, t)
                cv2.imwrite(f"{HERE}/snapshot_images/{sid}.jpg", im)
                snaps_out.append(dict(snapshot_id=sid, source_seed=seed, phase=phase, chunk=c, p=p, q=25 - p, t=t,
                                      remaining_horizon=400 - t, stage_at_snapshot=s["stage_per_step"][t - 1],
                                      events=dict(t1_right_contact=t1, t2_lifted=t2, t3_left_contact=t3, t4_success=fs.get("4")),
                                      source_outcome=dict(success=s["success"], t_end=s["t_end"]),
                                      description=PHASE_NOTES.get(f"{seed}:{phase}", "TO BE FILLED AFTER IMAGE INSPECTION"),
                                      image=f"snapshot_images/{sid}.jpg"))
    if a.sheets_only:
        return
    man = dict(frozen_before_branch_outcomes=True, rule=__doc__.split("Rule")[1].split("Missing")[0].strip(),
               source_seed_rule="fixed-baseline held-out records (reference_eval.jsonl): median-t_end success 10000014 "
                                "(272), latest success 10000016 (376), lowest-numbered failure 10000004",
               R_sets={"p5": [10, 15, 20], "p10": [10, 12, 15]}, continuation_seeds=[0, 1, 2],
               video_snapshots=VIDEO, snapshots=snaps_out)
    json.dump(man, open(f"{HERE}/manifest.json", "w"), indent=1)
    print("valid snapshots", sum(not x.get("missing") for x in snaps_out), "missing", [x["snapshot_id"] for x in snaps_out if x.get("missing")])


if __name__ == "__main__":
    main()
