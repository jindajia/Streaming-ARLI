"""SCH-01 same-state R/N branching diagnostic (understanding only; NEVER used as training data).

For each source episode, run the fixed reference scheduler and capture complete snapshots at selected handoff ticks
(idle worker, q = 25, no request in flight).  For each snapshot and replicate, restore the full simulator/executor
state, apply ONE initial decision (REPLAN(R, N) on the coarse grid, or KEEP for one tick), then continue with the same
frozen reference scheduler (N = 10, K = 25, budget launch rule) to success or the 400-tick limit.  Paired randomness:
replicate r of a snapshot uses the same noise stream (indexed by call number) and jitter stream in every branch.

    python -m sch01.diagnostic --config configs/diagnostic_coarse.yaml
Outputs (separate from online runs): <out_root>/<run_id>/branches.jsonl, snapshots.json, summary.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time

import numpy as np
import yaml


@dataclasses.dataclass
class DCfg:
    run_id: str = "diag_debug"
    out_root: str = "/N/scratch/jindjia/arli/runs_scheduling/diagnostic"
    source_seeds: tuple = (20_000_000, 20_000_001)  # disjoint from the 50 held-out eval seeds
    handoff_ticks: tuple = (25, 100, 175)  # snapshot at these handoff ticks if reached
    N_grid: tuple = (3, 5, 7, 10)
    R_grid: tuple = (4, 5, 7, 10, 15, 25)
    keep_branch: bool = True
    replicates: int = 2
    max_wall_hours: float = 0.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    d = yaml.safe_load(open(args.config))
    for o in args.set:
        k, v = o.split("=", 1)
        d[k] = yaml.safe_load(v)
    d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}
    cfg = DCfg(**d)

    from arli.pi0_policy import configure_caches
    configure_caches()
    from arli import aloha_env
    from arli.pi0_policy import FrozenPi0
    from sch01 import core
    from sch01.executor import Executor, reference_policy, take_snapshot
    from sch01.sampler import RTCSampler

    wall0 = time.time()
    out = f"{cfg.out_root}/{cfg.run_id}"
    os.makedirs(out, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    git = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open(f"{out}/launch_{int(wall0)}.json", "w") as f:
        json.dump(dict(config=dataclasses.asdict(cfg), git=git, argv=sys.argv, host=os.uname().nodename,
                       slurm_job=os.environ.get("SLURM_JOB_ID"), note="diagnostic only; not training data"), f, indent=1)
    env = aloha_env.make_env()
    exe = Executor(env, RTCSampler(FrozenPi0(10)))

    # 1. capture snapshots
    snaps = []
    src_rows = []
    for si, seed in enumerate(cfg.source_seeds):
        def grab(ex, env_, seed=seed, si=si):
            if ex["t"] in cfg.handoff_ticks:
                snaps.append(take_snapshot(ex, env_, extra=dict(env_seed=seed, snap_id=len(snaps), source=si)))
        r = exe.run(seed, reference_policy, np.random.default_rng([seed, 1]), np.random.default_rng([seed, 2]),
                    on_snapshot=grab)
        src_rows.append(dict(seed=seed, success=r["success"], t_end=r["t_end"]))
    with open(f"{out}/snapshots.json", "w") as f:
        json.dump(dict(sources=src_rows, snapshots=[dict(snap_id=s["snap_id"], seed=s["env_seed"], t=s["t"],
                                                         q=len(s["queue"])) for s in snaps]), f, indent=1)
    print("source episodes", src_rows, "snapshots", [(s["snap_id"], s["t"]) for s in snaps], flush=True)

    # 2. branches
    branches = []
    for n in cfg.N_grid:
        for r in cfg.R_grid:
            if core.BUDGET[n] <= r <= core.K_CAP:
                branches.append(("REPLAN", r, n, core.encode(r, n)))
    if cfg.keep_branch:
        branches.append(("KEEP1", None, None, core.KEEP))
    log = open(f"{out}/branches.jsonl", "a", buffering=1)
    done = set()
    if os.path.exists(f"{out}/branches.jsonl"):
        for line in open(f"{out}/branches.jsonl"):
            row = json.loads(line)
            done.add((row["snap_id"], row["branch"], row["rep"]))
    n_roll = n_steps = 0
    for rep in range(cfg.replicates):
        for s in snaps:
            for kind, r, n, a in branches:
                name = "KEEP1" if kind == "KEEP1" else f"R{r}_N{n}"
                if (s["snap_id"], name, rep) in done:
                    continue
                if (time.time() - wall0) / 3600 > cfg.max_wall_hours:
                    print("wall budget reached; partial diagnostic saved", flush=True)
                    return
                nr = np.random.default_rng([s["env_seed"], s["t"], rep, 101])  # paired across branches
                jr = np.random.default_rng([s["env_seed"], s["t"], rep, 102])
                t0 = time.time()
                res = exe.run(s["env_seed"], reference_policy, nr, jr, snapshot=s, forced_first=a,
                              continuation=reference_policy)
                first = res["requests"][0] if res["requests"] else None
                row = dict(snap_id=s["snap_id"], source=s["source"], t_snap=s["t"], branch=name, R=r, N=n, rep=rep,
                           success=res["success"], t_end=res["t_end"], ret=res["ret"],
                           discounted_return=res["discounted_return"], ticks=res["t_end"] - s["t"],
                           first_request=None if first is None else dict(launch_t=first["launch_t"], r=first["r"],
                                                                         n=first["n"], d_actual=first["d_actual"],
                                                                         jitter_ms=first["jitter_ms"]),
                           n_requests=len(res["requests"]), underrun_ticks=res["underrun_ticks"],
                           infer_cost_ms=res["infer_cost_ms"], wall_s=time.time() - t0)
                log.write(json.dumps(row) + "\n")
                n_roll += 1
                n_steps += row["ticks"]
        print(f"replicate {rep} done: {n_roll} rollouts, {n_steps} env steps", flush=True)
    with open(f"{out}/summary.json", "w") as f:
        json.dump(dict(rollouts=n_roll, env_steps=n_steps, wall_s=time.time() - wall0, branches=len(branches),
                       snapshots=len(snaps), replicates=cfg.replicates), f, indent=1)
    print("DIAGNOSTIC DONE", flush=True)


if __name__ == "__main__":
    main()
