"""DEHP-style execution-horizon PPO pilot on ALOHA Transfer Cube (one GPU, one-hour budget, no continuation).

    python -m dehp.train --config configs/run_A_nodelay.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import sys
import time

import numpy as np
import yaml

T_PROC0 = time.time()


@dataclasses.dataclass
class Cfg:
    name: str = "dehp_debug"
    run: str = "A"  # A = no delay, RTC off, h in 1..50 ; B = delay 10, RTC on, h in 10..40
    seed: int = 0
    run_root: str = "/N/scratch/jindjia/arli/runs_dehp"
    gamma: float = 0.999
    lam: float = 0.99
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    clip: float = 0.05
    batch_decisions: int = 256
    epochs: int = 4
    warmup_epochs: int = 20
    eval_seeds: tuple = tuple(range(10_000_000, 10_000_010))  # disjoint from training seeds (< 10_000_000)
    eval_noise_seed: int = 777
    eval_every_min: float = 15.0
    train_until_min: float = 47.0  # minutes after process start: stop collecting, final eval + save
    max_iters: int = 10_000
    baseline_h: int = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    d = yaml.safe_load(open(args.config))
    for o in args.set:
        k, v = o.split("=", 1)
        d[k] = yaml.safe_load(v)
    if "eval_seeds" in d:
        d["eval_seeds"] = tuple(d["eval_seeds"])
    cfg = Cfg(**d)
    run_dir = f"{cfg.run_root}/{cfg.name}"
    os.makedirs(f"{run_dir}/ckpt", exist_ok=True)

    t_load0 = time.time()
    from arli.pi0_policy import configure_caches
    configure_caches()
    import jax

    from arli import aloha_env
    from arli.pi0_policy import FrozenPi0
    from dehp.ppo import Learner, duration_gae
    from dehp.smdp import STATE_DIM, AsyncHorizonEnv, SyncHorizonEnv
    from sch01.sampler import RTCSampler

    env = aloha_env.make_env()
    sampler = RTCSampler(FrozenPi0(10))
    Env = SyncHorizonEnv if cfg.run == "A" else AsyncHorizonEnv
    senv = Env(env, sampler, cfg.gamma)
    eval_env = Env(aloha_env.make_env(), sampler, cfg.gamma)  # separate instance: evals never touch the training episode
    actor_in = STATE_DIM + 50 * 14 + 50 + 3
    critic_with_plan = cfg.run == "B"
    L = Learner(cfg.seed, actor_in, actor_in if critic_with_plan else STATE_DIM, critic_with_plan, cfg.actor_lr,
                cfg.critic_lr, cfg.clip)
    import subprocess
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(__file__)).stdout.strip()
    meta = dict(config=dataclasses.asdict(cfg), git=git, host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"),
                start=time.strftime("%F %T"), checkpoint=sampler.fp.checkpoint_dir.as_posix() if hasattr(
                    sampler.fp.checkpoint_dir, "as_posix") else str(sampler.fp.checkpoint_dir),
                h_range=[senv.h_min, senv.h_max], delay=10 if cfg.run == "B" else 0, rtc=cfg.run == "B",
                actor_in=actor_in, critic_in=actor_in if critic_with_plan else STATE_DIM, jax=jax.__version__)
    json.dump(meta, open(f"{run_dir}/launch.json", "w"), indent=1)
    print(json.dumps(meta, indent=1), flush=True)
    W = dict(load_compile_s=0.0, eval_s=0.0, collect_s=0.0, update_s=0.0)

    log_it = open(f"{run_dir}/iters.jsonl", "a", buffering=1)
    log_ev = open(f"{run_dir}/evals.jsonl", "a", buffering=1)
    log_ep = open(f"{run_dir}/train_episodes.jsonl", "a", buffering=1)
    C = dict(env_steps=0, decisions=0, episodes=0, successes=0, iters=0)

    def elapsed_min():
        return (time.time() - T_PROC0) / 60

    def save(tag):
        with open(f"{run_dir}/ckpt/learner_{tag}.msgpack", "wb") as f:
            f.write(L.state_bytes())

    def evaluate(kind, policy):
        """policy: 'baseline' (h = 25) or 'learned' (argmax).  Same seeds and the same pi0 noise stream per episode."""
        t0 = time.time()
        calls0 = eval_env.stats["inference_calls"]
        eps = []
        for i, seed in enumerate(cfg.eval_seeds):
            rng = np.random.default_rng([cfg.eval_noise_seed, i])
            s = eval_env.reset(seed, rng)
            hs, durs = [], []
            info = {}
            while s is not None:
                h = cfg.baseline_h if policy == "baseline" else L.act(s, deterministic=True)[0]
                hs.append(h)
                s, G, D, term, info = eval_env.step(h)
                durs.append(D)
            eps.append(dict(seed=seed, success=bool(info["success"]), t_end=int(info["t_end"]), hs=hs, durations=durs))
        W["eval_s"] += time.time() - t0
        n = sum(e["success"] for e in eps)
        row = dict(kind=kind, policy=policy, env_steps=C["env_steps"], decisions=C["decisions"], iters=C["iters"],
                   n_success=n, n_episodes=len(eps), elapsed_min=elapsed_min(), eval_wall_s=time.time() - t0,
                   inference_calls=eval_env.stats["inference_calls"] - calls0, episodes=eps)
        log_ev.write(json.dumps(row) + "\n")
        print(f"[eval {kind} {policy}] {n}/{len(eps)} at env_steps={C['env_steps']} ({elapsed_min():.1f} min)", flush=True)
        return n

    def stack(states):
        return {k: np.stack([s[k] for s in states]) for k in ("phys", "plan", "plan_mask", "meta", "hmask")}

    def fail(msg):
        save("diagnostic")
        json.dump(dict(error=msg, counters=C, elapsed_min=elapsed_min()), open(f"{run_dir}/FAILED.json", "w"))
        print("STOP:", msg, flush=True)
        sys.exit(2)

    W["load_compile_s"] = time.time() - t_load0
    evaluate("baseline", "baseline")
    evaluate("initial", "learned")
    last_eval = time.time()
    train_rng = np.random.default_rng([cfg.seed, 11])
    noise_rng = np.random.default_rng([cfg.seed, 12])
    cur, ep_succ = None, False
    warmed = False
    while C["iters"] < cfg.max_iters and elapsed_min() < cfg.train_until_min:
        # ---------------- collect ~batch_decisions decisions with the current policy -----------------------------
        t0 = time.time()
        B = dict(s=[], a=[], logp=[], G=[], D=[], h=[], term=[], s_next=[])
        n_ep_done, n_succ_done = 0, 0
        while len(B["a"]) < cfg.batch_decisions:
            if cur is None:
                cur = senv.reset(int(train_rng.integers(10_000_000)), noise_rng)
            h, logp, _ = L.act(cur)
            nxt, G, D, term, info = senv.step(h)
            if D != h and not term:
                fail(f"duration {D} != selected h {h} on a non-terminal decision")
            B["s"].append(cur); B["a"].append(h - 1); B["logp"].append(logp); B["G"].append(G); B["D"].append(D)
            B["h"].append(h); B["term"].append(term); B["s_next"].append(nxt if nxt is not None else cur)
            C["env_steps"] += D
            C["decisions"] += 1
            if term:
                C["episodes"] += 1
                n_ep_done += 1
                C["successes"] += int(info["success"])
                n_succ_done += int(info["success"])
                log_ep.write(json.dumps(dict(env_steps=C["env_steps"], success=info["success"], t_end=info["t_end"],
                                             deadline=info["deadline"])) + "\n")
                cur = None
            else:
                cur = nxt
        W["collect_s"] += time.time() - t0
        # ---------------- targets ---------------------------------------------------------------------------------
        t0 = time.time()
        S, Sn = stack(B["s"]), stack(B["s_next"])
        v, v_next = L.values(S), L.values(Sn)
        term = np.asarray(B["term"], bool)
        adv = duration_gae(np.asarray(B["G"]), np.asarray(B["D"]), term, v, v_next, cfg.gamma, cfg.lam)
        targets_raw = v + adv
        out_of_support = float(np.mean((targets_raw < 0) | (targets_raw > 1)))
        targets = np.clip(targets_raw, 0, 1).astype(np.float32)
        a_idx = np.asarray(B["a"], np.int32)
        old_logp = np.asarray(B["logp"], np.float32)
        adv_n = ((adv - adv.mean()) / (adv.std() + 1e-8)).astype(np.float32)
        if not (np.all(np.isfinite(adv)) and np.all(np.isfinite(v))):
            fail("non-finite advantages/values")
        warm_log = None
        if not warmed:  # DEHP: warm up the critic before the first actor update
            warm_log = L.critic_warmup(S, targets, cfg.warmup_epochs)
            warmed = True
            v = L.values(S)
            v_next = L.values(Sn)
            adv = duration_gae(np.asarray(B["G"]), np.asarray(B["D"]), term, v, v_next, cfg.gamma, cfg.lam)
            targets_raw = v + adv
            targets = np.clip(targets_raw, 0, 1).astype(np.float32)
            adv_n = ((adv - adv.mean()) / (adv.std() + 1e-8)).astype(np.float32)
        logs = L.update(S, a_idx, old_logp, adv_n, targets, cfg.epochs)
        W["update_s"] += time.time() - t0
        if not L.finite() or not all(np.isfinite(list(x.values())).all() for x in logs):
            fail("non-finite parameters or losses")
        C["iters"] += 1
        hs = np.asarray(B["h"])
        ev_ = 1 - np.var(targets_raw - v) / (np.var(targets_raw) + 1e-12)
        row = dict(iter=C["iters"], **C, elapsed_min=elapsed_min(), batch_episodes_done=n_ep_done,
                   batch_success=n_succ_done, h_mean=float(hs.mean()), h_hist=np.bincount(hs, minlength=51)[1:].tolist(),
                   D_mismatch_nonterminal=int(((np.asarray(B["D"]) != hs) & ~term).sum()),
                   D_short_terminal=int(((np.asarray(B["D"]) < hs) & term).sum()),
                   v_mean=float(v.mean()), target_mean=float(targets_raw.mean()), adv_std=float(adv.std()),
                   out_of_support=out_of_support, explained_variance=float(ev_), **logs[-1],
                   kl_first=logs[0]["approx_kl"], warmup=None if warm_log is None else warm_log[-1],
                   stats=dict(senv.stats), wall=dict(W))
        log_it.write(json.dumps(row) + "\n")
        print(f"[it {C['iters']}] steps={C['env_steps']} dec={C['decisions']} succ={n_succ_done}/{n_ep_done} "
              f"h_mean={hs.mean():.1f} ent={logs[-1]['entropy']:.3f} kl={logs[-1]['approx_kl']:.2e} "
              f"clip={logs[-1]['clipfrac']:.2f} closs={logs[-1]['critic_loss']:.3f} ev={ev_:.2f} "
              f"oos={out_of_support:.3f} t={elapsed_min():.1f}m", flush=True)
        if (time.time() - last_eval) / 60 >= cfg.eval_every_min and elapsed_min() < cfg.train_until_min - 3:
            evaluate("periodic", "learned")
            save(f"it{C['iters']:04d}")
            last_eval = time.time()
    evaluate("final", "learned")
    save("final")
    json.dump(dict(counters=C, wall=W, stats=senv.stats, elapsed_min=elapsed_min()), open(f"{run_dir}/SUMMARY.json", "w"),
              indent=1)
    print("DONE", C, W, flush=True)


if __name__ == "__main__":
    main()
