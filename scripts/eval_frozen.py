"""Frozen pi0 baseline under the same protocol (async no-RTC by default; --sync is the zero-delay diagnostic).

    python scripts/eval_frozen.py --episodes 50 --out results/frozen_async.jsonl
"""
import argparse
import json
import time

import numpy as np

from arli.pi0_policy import configure_caches

configure_caches()
from arli import aloha_env  # noqa: E402
from arli.controllers import FrozenPolicy  # noqa: E402
from arli.pi0_policy import FrozenPi0  # noqa: E402
from arli.rollout import run_episode  # noqa: E402
from arli.schedule import Schedule  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=50)
ap.add_argument("--seed_base", type=int, default=10_000_000)
ap.add_argument("--noise_seed", type=int, default=0)
ap.add_argument("--sync", action="store_true", help="zero-delay diagnostic: delay=0 (sim pauses for inference)")
ap.add_argument("--out", required=True)
args = ap.parse_args()

sched = Schedule(delay=0, delay_rl=0) if args.sync else Schedule()
env = aloha_env.make_env()
pi0 = FrozenPi0(10)
ctrl = FrozenPolicy(np.random.default_rng([args.noise_seed, 99]))
with open(args.out, "a") as f:
    for e in range(args.episodes):
        t0 = time.time()
        res = run_episode(env, pi0, sched, args.seed_base + e, ctrl)
        row = dict(protocol="sync_delay0" if args.sync else "async_d20", seed=res["seed"], success=res["success"],
                   t_end=res["t_end"], max_reward=res["max_reward"], ret=res["ret"], noise_seed=args.noise_seed,
                   wall_s=time.time() - t0)
        f.write(json.dumps(row) + "\n")
        f.flush()
        print(row, flush=True)
