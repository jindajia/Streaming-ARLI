"""Assemble the static site: copy viewer/web/* into SITE_DIR and write data/index.json + data/MANIFEST.json.

The served directory contains only web files, compact JSON and binary inspection arrays (JPEG frames, masks,
probabilities, probe outputs, 64x64 network inputs).  No model weights, replay buffers or credentials.

    python viewer/export/build_site.py [--site DIR]
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SITE_DIR  # noqa: E402

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=SITE_DIR)
    args = ap.parse_args()
    site, data = args.site, f"{args.site}/data"
    for f in os.listdir(WEB):
        shutil.copy(os.path.join(WEB, f), os.path.join(site, f))
    seeds = json.load(open(f"{data}/seeds.json"))
    cks = json.load(open(f"{data}/checkpoints.json"))
    root = f"{data}/episodes"
    episodes, prov = {}, None
    for d in sorted(os.listdir(root)):
        mp = f"{root}/{d}/meta.json"
        if not os.path.exists(mp):
            continue
        M = json.load(open(mp))
        prov = prov or M["provenance"]
        episodes[d] = dict(policy=M["policy"]["id"], seed=M["seed"], success=M["success"], t_end=M["t_end"],
                           ret=M["ret"], n_decisions=M["n_decisions"], n_requests=len(M["requests"]),
                           outcome_matches_log=M["outcome_matches_log"],
                           logged_success=None if M["logged"] is None else M["logged"]["success"],
                           logged_t_end=None if M["logged"] is None else M["logged"]["t_end"],
                           probes=sorted(f[:-4] for f in os.listdir(f"{root}/{d}/probes")) if os.path.isdir(
                               f"{root}/{d}/probes") else [])
    val = json.load(open(f"{data}/validation.json")) if os.path.exists(f"{data}/validation.json") else None
    vlines = []
    if val:
        t = val["totals"]
        vlines.append(f"validate.py: {val['n_failures']} failures over {t['episodes']} recordings, {t['decisions']} decision "
                      f"states, {t['frames']} frames (masks vs delay table + queue, P(KEEP) + ΣP(REPLAN) = 1 (max error "
                      f"{t['max_sum_err']:.1e}), recorded probabilities = viewer formula (max error {t['max_recompute_err']:.1e}), "
                      f"selected action = argmax / baseline rule, planned vs actual handoffs, idle-tick decisions, rewards "
                      f"and discounted returns, served inputs = archived inputs, probe files for every checkpoint)")
    pc = json.load(open(f"{data}/probe_check.json")) if os.path.exists(f"{data}/probe_check.json") else []
    idx = dict(built=time.strftime("%F %T"), checkpoints=cks, seeds=seeds["seeds"], seeds_note=seeds["note"],
               seeds_rule=seeds["rule"], episodes=episodes, probe_check=pc, provenance=prov, validation=vlines)
    json.dump(idx, open(f"{data}/index.json", "w"), separators=(",", ":"))
    sizes = {}
    for dp, _, fs in os.walk(site):
        for f in fs:
            ext = f.split(".")[-1]
            sizes[ext] = sizes.get(ext, 0) + os.path.getsize(os.path.join(dp, f))
    json.dump(dict(built=idx["built"], n_episodes=len(episodes), bytes_by_extension=sizes,
                   total_mb=round(sum(sizes.values()) / 1e6, 1)), open(f"{data}/MANIFEST.json", "w"), indent=1)
    print(f"site {site}: {len(episodes)} episodes, {sum(sizes.values()) / 1e6:.1f} MB", sizes)


if __name__ == "__main__":
    main()
