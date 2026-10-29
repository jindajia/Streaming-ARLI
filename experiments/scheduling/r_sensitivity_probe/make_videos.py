"""Paired 2x2 videos (forced R choices + full baseline, continuation seed 0) for the snapshots fixed in
manifest.video_snapshots.  Frame k of every panel = k control steps after the same snapshot; a panel that has ended
holds its last frame with the outcome.  Written to <bulky dir>/videos/ (not committed).

    python r_sensitivity_probe/make_videos.py [--src <bulky dir>]
"""
import argparse
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/N/scratch/jindjia/arli/runs_scheduling/r_sensitivity_probe")
    ap.add_argument("--fps", type=int, default=25)
    a = ap.parse_args()
    man = json.load(open(f"{HERE}/manifest.json"))
    rows = {(r["snapshot_id"], r["choice"], r["cont_seed"]): r for r in map(json.loads, open(f"{a.src}/branches.jsonl"))}
    os.makedirs(f"{a.src}/videos", exist_ok=True)
    for sid in man["video_snapshots"]:
        m = next(x for x in man["snapshots"] if x["snapshot_id"] == sid)
        chs = (["R10", "R15", "R20"] if m["p"] == 5 else ["R10", "R12", "R15"]) + ["baseline"]
        panels = []
        for ch in chs:
            base = f"{a.src}/video_frames/{sid}_{ch}_c0"
            if not os.path.exists(base + ".bin"):
                panels = None
                break
            o = json.load(open(base + ".json"))["offsets"]
            buf = open(base + ".bin", "rb").read()
            fr = [cv2.imdecode(np.frombuffer(buf[o[k]:o[k + 1]], np.uint8), cv2.IMREAD_COLOR) for k in range(len(o) - 1)]
            r = rows[(sid, ch, 0)]
            panels.append((ch, fr, r))
        if not panels:
            print("missing frames for", sid)
            continue
        L = max(len(f) for _, f, _ in panels)
        path = f"{a.src}/videos/{sid}_cont0.mp4"
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (640, 480))
        for k in range(L):
            tiles = []
            for ch, fr, r in panels:
                im = fr[min(k, len(fr) - 1)].copy()
                lab = ("full baseline" if ch == "baseline" else f"forced R={ch[1:]}")
                cv2.putText(im, f"{lab}  t={m['t'] + min(k, len(fr) - 1)}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                fq = r["first_request"]
                if fq and m["t"] + k >= fq["handoff_t"]:
                    cv2.putText(im, f"after first handoff ({fq['handoff_t']})", (5, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
                if k >= len(fr) - 1:
                    cv2.putText(im, "SUCCESS" if r["success"] else "TIMEOUT", (5, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0) if r["success"] else (0, 0, 255), 2)
                tiles.append(im)
            vw.write(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
        vw.release()
        print("wrote", path, L, "frames")


if __name__ == "__main__":
    main()
