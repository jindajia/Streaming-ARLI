"""DEHP pilot policy diagnostics + matched evaluation (no gradient updates).

Stage 1 (per run A/B): diagnostic state bank from fixed-h=25 trajectories on dedicated source seeds; full p(h|state)
for an intermediate and the final checkpoint; the frozen average vector p_bar (final checkpoint) for arm M.
Stage 2: matched evaluation arms on a predeclared seed list, in priority order
    U (uniform stochastic), S (trained stochastic) -> D (trained argmax), F25 (fixed 25) -> M (average-vector stochastic),
then the original 10 held-out seeds for continuity.  Every episode is appended to JSONL immediately.

RNG (all explicit, never the checkpoint's Learner.key):
  * environment initial state: env seed (shared by all arms);
  * pi0 initial noise: per-episode stream default_rng([NOISE_SEED, ep_index]); the i-th inference call of every arm
    draws the i-th noise block (matched by call index, NOT by physical time once schedules differ);
  * horizon sampling: per-episode stream default_rng([H_SEED, ep_index]); decision k uses u_k; inverse CDF over valid h
    in ascending order (common random numbers for U, S, M).

    python policy_diagnostics/diagnose.py --out /N/scratch/jindjia/arli/runs_dehp/policy_diagnostics
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

T0 = time.time()
NOISE_SEED, H_SEED = 2026, 4049
BANK_SEEDS = list(range(10_000_200, 10_000_208))  # diagnostic source episodes (disjoint from all evaluation seeds)
BANK_EVERY = 3  # keep decisions 0, 3, 6, ... of each source episode (time-coverage rule, not probability-based)
NEW_SEEDS = list(range(10_000_100, 10_000_120))  # predeclared primary evaluation set (20)
ORIG_SEEDS = list(range(10_000_000, 10_000_010))  # original pilot held-out seeds (reported separately)
CKPTS = {"A": ("run_A_nodelay", ["learner_it0016", "learner_final"]),
         "B": ("run_B_delay10", ["learner_it0013", "learner_final"])}
ROOT = "/N/scratch/jindjia/arli/runs_dehp"


def phase(last_r, max_r):
    if max_r == 0:
        return "approach"
    return {0: "lost_contact", 1: "grasp", 2: "lift_transport", 3: "transfer", 4: "success"}[int(last_r)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline_min", type=float, default=50.0)
    ap.add_argument("--dry_run", action="store_true", help="CPU pipeline check: fake pi0 sampler, tiny seed lists")
    ap.add_argument("--eval_only", action="store_true", help="skip stage 1; load the frozen pbar_*.json from --out")
    args = ap.parse_args()
    global BANK_SEEDS, NEW_SEEDS, ORIG_SEEDS
    if args.dry_run:
        BANK_SEEDS, NEW_SEEDS, ORIG_SEEDS = BANK_SEEDS[:2], NEW_SEEDS[:1], ORIG_SEEDS[:1]
    os.makedirs(args.out, exist_ok=True)

    from arli.pi0_policy import configure_caches
    configure_caches()
    import jax
    import jax.numpy as jnp

    from arli import aloha_env
    from arli.pi0_policy import FrozenPi0
    from dehp.ppo import Learner, actor_input, masked_logits
    from dehp.smdp import STATE_DIM, AsyncHorizonEnv, SyncHorizonEnv
    from sch01.sampler import RTCSampler

    def recording(Cls):
        class Rec(Cls):  # read-only: records the env reward stage; executor semantics unchanged
            def reset(self, seed, noise_rng):
                self.last_r, self.max_r = 0.0, 0.0
                return super().reset(seed, noise_rng)

            def _env_step(self, a):
                obs, r, term = super()._env_step(a)
                self.last_r, self.max_r = float(r), max(self.max_r, float(r))
                return obs, r, term
        return Rec

    t_load = time.time()
    if args.dry_run:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
        from test_dehp_cpu import FakeSampler  # noqa: E402  (runs the module's CPU tests on import; harmless)
        sampler = FakeSampler()
    else:
        sampler = RTCSampler(FrozenPi0(10))
    envs = {"A": recording(SyncHorizonEnv)(aloha_env.make_env(), sampler, 0.999),
            "B": recording(AsyncHorizonEnv)(aloha_env.make_env(), sampler, 0.999)}
    ain = STATE_DIM + 50 * 14 + 50 + 3
    learners = {}
    for run, (rdir, names) in CKPTS.items():
        for nm in names:
            L = Learner(0, ain, ain if run == "B" else STATE_DIM, run == "B")
            L.load_bytes(open(f"{ROOT}/{rdir}/ckpt/{nm}.msgpack", "rb").read())
            learners[(run, nm)] = L
    actor_mod = next(iter(learners.values())).actor
    probs_fn = jax.jit(lambda pa, s: jax.nn.softmax(masked_logits(actor_mod.apply(pa, actor_input(s)), s["hmask"]), -1))
    KEYS = ("phys", "plan", "plan_mask", "meta", "hmask")

    def probs(run, nm, s):
        p = np.asarray(probs_fn(learners[(run, nm)].pa, {k: jnp.asarray(s[k])[None] for k in KEYS}))[0].astype(np.float64)
        p[~s["hmask"]] = 0.0
        return p / p.sum()

    log = open(f"{args.out}/log.jsonl", "a", buffering=1)

    def note(**kw):
        kw["elapsed_min"] = (time.time() - T0) / 60
        log.write(json.dumps(kw) + "\n")
        print(json.dumps(kw)[:300], flush=True)

    note(event="loaded", load_s=time.time() - t_load, jax=jax.__version__)

    # ------------------------------------------------------------------ stage 1: state bank + distributions
    pbar = {}
    if args.eval_only:
        for run in envs:
            pbar[run] = np.asarray(json.load(open(f"{args.out}/pbar_{run}.json"))["p_bar"])
    for run, env in ({} if args.eval_only else envs).items():
        bank_f = open(f"{args.out}/state_bank_{run}.jsonl", "w")
        traj_f = open(f"{args.out}/trajectory_dists_{run}.jsonl", "w")
        P = []
        for i, seed in enumerate(BANK_SEEDS):
            s = env.reset(seed, np.random.default_rng([NOISE_SEED + 1, i]))
            k, info = 0, {}
            while s is not None:
                rec = dict(run=run, source_seed=seed, decision=k, t=int(s["t"]), remaining=400 - int(s["t"]),
                           phase=phase(env.last_r, env.max_r), stage=env.last_r, e0=int(s["e0"]),
                           plan_obs_age=int(s["t"] - s["plan_obs_t"]), valid=[int(h) for h in np.flatnonzero(s["hmask"]) + 1])
                for nm in CKPTS[run][1]:
                    rec[f"p_{nm}"] = probs(run, nm, s).tolist()
                traj_f.write(json.dumps(rec) + "\n")
                if k % BANK_EVERY == 0:
                    bank_f.write(json.dumps(rec) + "\n")
                    P.append(rec["p_learner_final"])
                s, G, D, term, info = env.step(25)
                k += 1
            note(event="bank_episode", run=run, seed=seed, success=bool(info["success"]), t_end=int(info["t_end"]),
                 decisions=k)
        bank_f.close()
        traj_f.close()
        pbar[run] = np.mean(np.asarray(P), 0)
        json.dump(dict(run=run, checkpoint="learner_final", n_states=len(P), p_bar=pbar[run].tolist()),
                  open(f"{args.out}/pbar_{run}.json", "w"))
        note(event="pbar_frozen", run=run, n_states=len(P))

    # ------------------------------------------------------------------ stage 2: matched evaluation
    eval_f = open(f"{args.out}/episodes.jsonl", "a", buffering=1)

    def inv_cdf(p, u):
        c = np.cumsum(p)
        return int(min(np.searchsorted(c, u * c[-1], side="right"), len(p) - 1)) + 1

    def run_episode(run, arm, seed, ep_index, seedset):
        env = envs[run]
        hrng = np.random.default_rng([H_SEED, ep_index])
        st0 = dict(env.stats)
        s = env.reset(seed, np.random.default_rng([NOISE_SEED, ep_index]))
        decs, info = [], {}
        while s is not None:
            valid = s["hmask"]
            u = float(hrng.random())
            p = None
            if arm == "U":
                p = valid / valid.sum()
            elif arm in ("S", "D"):
                p = probs(run, "learner_final", s)
            elif arm == "M":
                p = pbar[run].copy()
                p[~valid] = 0
                p = p / p.sum()
            if arm == "F25":
                h, ph = 25, None
            elif arm == "D":
                h = int(np.argmax(p)) + 1
                ph = float(p[h - 1])
            else:
                h = inv_cdf(p, u)
                ph = float(p[h - 1])
            ent = None if p is None else float(-(p[p > 0] * np.log(p[p > 0])).sum())
            pf = probs(run, "learner_final", s) if arm in ("U", "M", "F25") else p
            decs.append(dict(t=int(s["t"]), h=h, u=u, p_selected=ph, entropy=ent, argmax_final=int(np.argmax(pf)) + 1,
                             phase=phase(env.last_r, env.max_r)))
            s, G, D, term, info = env.step(h)
            decs[-1]["D"] = int(D)
        st1 = env.stats
        ret = (0.999 ** (info["t_end"] - 1)) if info["success"] else 0.0
        row = dict(run=run, arm=arm, seedset=seedset, seed=seed, ep_index=ep_index, success=bool(info["success"]),
                   t_end=int(info["t_end"]), discounted_return=ret,
                   inference_calls=st1["inference_calls"] - st0["inference_calls"],
                   async_violations=st1["async_violations"] - st0["async_violations"],
                   underruns=st1["underruns"] - st0["underruns"], decisions=decs, wall_min=(time.time() - T0) / 60)
        eval_f.write(json.dumps(row) + "\n")
        return row

    def run_arm(run, arm, seeds, seedset, offset):
        n_ok = 0
        for i, seed in enumerate(seeds):
            if (time.time() - T0) / 60 > args.deadline_min:
                note(event="skipped_incomplete", run=run, arm=arm, seedset=seedset, completed=i, planned=len(seeds))
                return False
            r = run_episode(run, arm, seed, offset + i, seedset)
            n_ok += r["success"]
        note(event="arm_done", run=run, arm=arm, seedset=seedset, successes=n_ok, n=len(seeds))
        return True

    plan = [(r, a) for a in ("U", "S") for r in ("A", "B")] + [(r, a) for a in ("D", "F25") for r in ("A", "B")] + \
           [(r, "M") for r in ("A", "B")]
    for run, arm in plan:
        if not run_arm(run, arm, NEW_SEEDS, "new20", 0):
            break
    else:
        for run, arm in plan:
            if not run_arm(run, arm, ORIG_SEEDS, "orig10", 1000):
                break
    note(event="done")


if __name__ == "__main__":
    main()
