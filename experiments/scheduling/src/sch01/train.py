"""SCH-01 online joint R/N scheduler training (discrete SAC, reference-prior initialisation, RTC ON, frozen pi0).

    python -m sch01.train --config configs/online_rn_seed0.yaml [--max_wall_hours 0.75]

Fresh actor/critics/optimizers and an EMPTY replay; all training data come from this run's own interactions.
Total budget counts every training control tick, including the warm-up.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import yaml


@dataclasses.dataclass
class Cfg:
    run_id: str = "debug"
    seed: int = 0
    run_root: str = "/N/scratch/jindjia/arli/runs_scheduling/online"
    total_steps: int = 300_000
    warmup_steps: int = 5_000
    critic_burnin_updates: int = 256
    updates_per_step: float = 1.0
    batch_size: int = 256
    gamma: float = 0.999
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    alpha_init: float = 0.01
    target_entropy_ratio: float = 0.1
    p0: float = 0.999
    prior_hold_until: int = 5_000
    prior_zero_at: int = 50_000
    eval_steps: tuple = (0, 5_000)  # plus every eval_interval
    eval_interval: int = 25_000
    eval_episodes: int = 50
    eval_seed_base: int = 10_000_000
    eval_reference: bool = True  # evaluate the matched fixed scheduler once (horizontal reference)
    cap_frames: int = 340_000
    cap_states: int = 340_000
    cap_decisions: int = 320_000
    log_update_every: int = 500


def _hash_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--max_wall_hours", type=float, default=1e9)
    ap.add_argument("--stop_after_episodes", type=int, default=0)
    args = ap.parse_args()
    d = yaml.safe_load(open(args.config))
    for o in args.set:
        k, v = o.split("=", 1)
        d[k] = yaml.safe_load(v)
    if "eval_steps" in d:
        d["eval_steps"] = tuple(d["eval_steps"])
    cfg = Cfg(**d)
    run_dir = f"{cfg.run_root}/{cfg.run_id}"
    if os.path.exists(f"{run_dir}/DONE"):
        print("already complete:", open(f"{run_dir}/DONE").read().strip(), flush=True)
        return

    from arli.pi0_policy import configure_caches
    configure_caches()
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
    import jax

    from arli import aloha_env, rl
    from arli.pi0_policy import CHECKPOINT, FrozenPi0
    from arli.train import JsonlLog, claim_run_dir
    from sch01 import core
    from sch01.executor import Executor, reference_policy
    from sch01.learner import Learner
    from sch01.replay import Replay
    from sch01.sampler import RTCSampler

    wall0 = time.time()
    os.makedirs(f"{run_dir}/ckpt", exist_ok=True)
    claim_run_dir(run_dir)
    here = os.path.dirname(os.path.abspath(__file__))
    git = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = dict(config=dataclasses.asdict(cfg), git=git, argv=sys.argv, host=os.uname().nodename,
                slurm_job=os.environ.get("SLURM_JOB_ID"), start=time.strftime("%F %T"),
                pi0_checkpoint=CHECKPOINT, norm_stats_sha256=_hash_file(
                    f"{CHECKPOINT}/assets/lerobot/aloha_sim_transfer_cube_human/norm_stats.json"),
                timing_model=dict(lat_ms="42.5+8.17N", jitter="clip(N(0,5),-10,10)",
                                  budget={n: core.BUDGET[n] for n in core.N_VALUES}, label="proxy-derived simulation"))
    with open(f"{run_dir}/launch_{int(wall0)}.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1), flush=True)

    env = aloha_env.make_env()
    sampler = RTCSampler(FrozenPi0(10))
    exe = Executor(env, sampler, gamma=cfg.gamma)
    L = Learner(cfg.seed, dict(actor_lr=cfg.actor_lr, critic_lr=cfg.critic_lr, alpha_lr=cfg.alpha_lr,
                               alpha_init=cfg.alpha_init, gamma=cfg.gamma, tau=cfg.tau, p0=cfg.p0,
                               target_entropy_ratio=cfg.target_entropy_ratio))
    R = Replay(cfg.cap_frames, cfg.cap_states, cfg.cap_decisions, cfg.seed)
    with open(f"{run_dir}/params.json", "w") as f:
        json.dump(L.n_params, f)

    C = dict(steps=0, episodes=0, successes=0, decisions=0, replans=0, departures=0, grad_steps=0, actor_updates=0,
             evals_done=[], wall_train=0.0, wall_update=0.0, wall_eval=0.0, wall_total=0.0, segments=0,
             infer_cost_ms=0.0, eval_episodes=0, eval_steps_env=0)
    rng_train = np.random.default_rng([cfg.seed, 31])
    state_path = f"{run_dir}/state_latest.pkl"
    if os.path.exists(state_path):
        S = rl.load_pickle(state_path)
        C.update(S["counters"])
        rng_train.bit_generator.state = S["rng_train"]
        L.load_bytes(S["learner"])
        R.load(f"{run_dir}/replay", S["replay"])
        print(f"RESUMED {C}", flush=True)
    else:
        assert len(R) == 0, "replay must start empty"
    C["segments"] += 1
    seg0, total0 = time.time(), C["wall_total"]
    log_ep, log_eval, log_upd = (JsonlLog(f"{run_dir}/{n}.jsonl") for n in ("episodes", "evals", "updates"))

    def strength(step):
        return core.prior_strength(step, cfg.prior_hold_until, cfg.prior_zero_at)

    def make_policy(mode, rng=None, step_fn=None, parity=None):
        def pol(st, mask, t):
            ref = core.reference_action(st["q"], mask)
            s = strength(step_fn(t))
            p, _ = L.policy_probs(st, mask, ref, s)
            a = int(np.argmax(p)) if mode == "eval" else int(rng.choice(core.NUM_ACTIONS, p=p))
            if parity is not None:
                parity.append(a == ref)
            ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
            return a, dict(p=float(p[a]), ref=ref, departed=a != ref, entropy=ent, ref_prob=float(p[ref]))
        return pol

    def summarize(res):
        reps = res["requests"]
        dec = res["decisions"]
        return dict(success=res["success"], t_end=res["t_end"], ret=res["ret"], decisions=len(dec), replans=len(reps),
                    mean_N=float(np.mean([q["n"] for q in reps])) if reps else None,
                    N_hist={int(n): int(sum(q["n"] == n for q in reps)) for n in core.N_VALUES},
                    mean_R=float(np.mean([q["r"] for q in reps])) if reps else None,
                    launch_q=[q["q_at_launch"] for q in reps][:40],
                    playback=[int(b["handoff_t"] - a["handoff_t"]) for a, b in zip(reps, reps[1:])][:40],
                    departures=int(sum(d["info"].get("departed", False) for d in dec)),
                    entropy=float(np.mean([d["info"].get("entropy", 0.0) for d in dec])),
                    underrun_ticks=res["underrun_ticks"], infer_cost_ms=res["infer_cost_ms"],
                    mean_delay=float(np.mean([q["d_actual"] for q in reps])) if reps else None)

    def evaluate(tag, use_reference=False):
        t0 = time.time()
        eps, parity = [], []
        for e in range(cfg.eval_episodes):
            seed = cfg.eval_seed_base + e
            nr, jr = np.random.default_rng([seed, 1]), np.random.default_rng([seed, 2])
            pol = reference_policy if use_reference else make_policy("eval", step_fn=lambda t: C["steps"], parity=parity)
            res = exe.run(seed, pol, nr, jr)
            eps.append(dict(seed=seed, **summarize(res)))
            C["eval_steps_env"] += res["t_end"]
        C["eval_episodes"] += len(eps)
        C["wall_eval"] += time.time() - t0
        succ = [x["success"] for x in eps]
        row = dict(tag=tag, policy="fixed_reference" if use_reference else "learned", train_steps=C["steps"],
                   prior_strength=strength(C["steps"]), grad_steps=C["grad_steps"], n_success=int(sum(succ)),
                   n_episodes=len(succ), success_rate=float(np.mean(succ)),
                   mean_N=float(np.nanmean([x["mean_N"] for x in eps if x["mean_N"] is not None])),
                   mean_R=float(np.nanmean([x["mean_R"] for x in eps if x["mean_R"] is not None])),
                   departures=int(sum(x["departures"] for x in eps)), underrun_ticks=int(sum(x["underrun_ticks"] for x in eps)),
                   infer_cost_ms=float(sum(x["infer_cost_ms"] for x in eps)),
                   reference_parity=(float(np.mean(parity)) if parity else None), episodes=eps,
                   wall_total_s=total0 + time.time() - seg0, eval_wall_s=time.time() - t0)
        (JsonlLog(f"{run_dir}/reference_eval.jsonl") if use_reference else log_eval)(**row)
        print(f"[eval {tag} {'REF' if use_reference else 'learned'}] step {C['steps']}  {sum(succ)}/{len(succ)}  "
              f"parity {row['reference_parity']}", flush=True)

    def save_state(tag=None):
        C["wall_total"] = total0 + time.time() - seg0
        rm = R.save_new(f"{run_dir}/replay")
        S = dict(counters=dict(C), rng_train=rng_train.bit_generator.state, learner=L.state_bytes(), replay=rm,
                 config=dataclasses.asdict(cfg))
        rl.save_pickle(state_path, S)
        if tag:
            with open(f"{run_dir}/ckpt/learner_{tag}.msgpack", "wb") as f:
                f.write(S["learner"])

    def due_evals():
        sched_pts = sorted(set(cfg.eval_steps) | set(range(cfg.eval_interval, cfg.total_steps + 1, cfg.eval_interval)))
        return [p for p in sched_pts if p <= C["steps"] and p not in C["evals_done"]]

    if cfg.eval_reference and not os.path.exists(f"{run_dir}/reference_eval.jsonl"):
        evaluate("reference", use_reference=True)
    upd, seg_eps = {}, 0
    while True:
        for p in due_evals():
            evaluate(f"{p}")
            C["evals_done"] = C["evals_done"] + [p]
            save_state(tag=f"{C['steps']:07d}")
        if C["steps"] >= cfg.total_steps:
            break
        if (time.time() - wall0) / 3600 > args.max_wall_hours or (args.stop_after_episodes and seg_eps >= args.stop_after_episodes):
            save_state()
            print("wall-clock budget reached; checkpointed; exiting for resume", flush=True)
            sys.exit(3)
        t0 = time.time()
        step0 = C["steps"]
        seed = int(rng_train.integers(cfg.eval_seed_base))
        nr = np.random.default_rng(rng_train.integers(2**63))
        jr = np.random.default_rng(rng_train.integers(2**63))
        prng = np.random.default_rng(rng_train.integers(2**63))
        res = exe.run(seed, make_policy("train", rng=prng, step_fn=lambda t: step0 + t), nr, jr)
        R.add_episode(res["decisions"])  # training interactions only
        sm = summarize(res)
        C["steps"] += res["t_end"]
        C["episodes"] += 1
        seg_eps += 1
        C["successes"] += int(res["success"])
        C["decisions"] += sm["decisions"]
        C["replans"] += sm["replans"]
        C["departures"] += sm["departures"]
        C["infer_cost_ms"] += sm["infer_cost_ms"]
        C["wall_train"] += time.time() - t0
        t0 = time.time()
        new_post_warmup = max(0, C["steps"] - max(step0, cfg.warmup_steps))
        n_upd = int(new_post_warmup * cfg.updates_per_step) if len(R) >= cfg.batch_size else 0
        for _ in range(n_upd):
            actor_on = C["grad_steps"] >= cfg.critic_burnin_updates
            info = L.update(R.sample(cfg.batch_size), strength(C["steps"]), actor_on)
            C["grad_steps"] += 1
            C["actor_updates"] += int(actor_on)
            if C["grad_steps"] % 25 == 0:
                for k, v in info.items():
                    upd.setdefault(k, []).append(float(v))
            if C["grad_steps"] % cfg.log_update_every == 0:
                log_upd(grad_steps=C["grad_steps"], steps=C["steps"], prior_strength=strength(C["steps"]),
                        actor_on=actor_on, **{k: float(np.mean(v)) for k, v in upd.items()})
                upd = {}
        C["wall_update"] += time.time() - t0
        log_ep(steps=C["steps"], prior_strength_start=strength(step0), prior_strength_end=strength(C["steps"]),
               grad_steps=C["grad_steps"], replay=len(R), **{k: v for k, v in sm.items()})
    with open(f"{run_dir}/DONE", "w") as f:
        f.write(f"steps={C['steps']} job={os.environ.get('SLURM_JOB_ID')} {time.strftime('%F %T')}\n")
    print(f"DONE {C}", flush=True)


if __name__ == "__main__":
    main()
