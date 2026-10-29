"""SCH-01 v2 online joint R/N scheduler training (fixed delay table, RTC ON, frozen pi0, discrete SAC).

    python -m sch02.train --config configs/online_rn_seed0.yaml --max_wall_hours 0.78

Fresh actor/critics/optimizers and an EMPTY replay (asserted); no legacy weights or replay; diagnostic data are never
read.  Evaluations at 0, 5k and every 25k (50 held-out episodes, deterministic masked argmax, frozen, no replay
writes) are logged per episode and resume mid-evaluation if a segment ends.  A supplemental prior-disabled evaluation
is run at 25k (reported separately).  A complete milestone copy is saved at 150k; the run then continues unchanged.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml


@dataclasses.dataclass
class Cfg:
    run_id: str = "debug"
    seed: int = 0
    run_root: str = "/N/scratch/jindjia/arli/runs_scheduling/fixed_delay10_v2/online"
    total_steps: int = 300_000
    milestone_steps: int = 150_000
    warmup_steps: int = 5_000
    critic_burnin_updates: int = 256
    updates_per_step: float = 1.0
    batch_size: int = 256
    gamma: float = 0.999
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    alpha_init: float = 0.001
    target_entropy_ratio: float = 0.1
    p0: float = 0.999
    prior_hold_until: int = 5_000
    prior_zero_at: int = 50_000
    eval_steps: tuple = (0, 5_000)
    eval_interval: int = 25_000
    supplemental_prior_off_at: tuple = (25_000,)
    eval_episodes: int = 50
    eval_seed_base: int = 10_000_000
    cap_frames: int = 340_000
    cap_states: int = 340_000
    cap_decisions: int = 320_000
    log_update_every: int = 500


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
    d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}
    cfg = Cfg(**d)
    run_dir = f"{cfg.run_root}/{cfg.run_id}"
    if os.path.exists(f"{run_dir}/DONE"):
        print("already complete:", open(f"{run_dir}/DONE").read().strip(), flush=True)
        return

    from arli.pi0_policy import configure_caches
    configure_caches()
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
    from arli import aloha_env, rl
    from arli.pi0_policy import CHECKPOINT, FrozenPi0
    from arli.train import JsonlLog, claim_run_dir
    from sch01.learner import Learner
    from sch01.sampler import RTCSampler
    from sch02 import core
    from sch02.executor import Executor, reference_policy
    from sch02.replay import Replay

    wall0 = time.time()
    os.makedirs(f"{run_dir}/ckpt", exist_ok=True)
    claim_run_dir(run_dir)
    here = os.path.dirname(os.path.abspath(__file__))
    git = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    ns = f"{CHECKPOINT}/assets/lerobot/aloha_sim_transfer_cube_human/norm_stats.json"
    meta = dict(config=dataclasses.asdict(cfg), git=git, argv=sys.argv, host=os.uname().nodename,
                slurm_job=os.environ.get("SLURM_JOB_ID"), partition=os.environ.get("SLURM_JOB_PARTITION"),
                start=time.strftime("%F %T"), pi0_checkpoint=CHECKPOINT,
                norm_stats_sha256=hashlib.sha256(open(ns, "rb").read()).hexdigest(),
                delay_table={str(k): v for k, v in core.DELAY.items()}, jitter=None, margin=0,
                timing_label="deterministic simulation table, N=10 -> 10 control steps (not a GPU measurement)")
    with open(f"{run_dir}/launch_{int(wall0)}.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1), flush=True)

    env = aloha_env.make_env()
    exe = Executor(env, RTCSampler(FrozenPi0(10)), gamma=cfg.gamma)
    L = Learner(cfg.seed, dict(actor_lr=cfg.actor_lr, critic_lr=cfg.critic_lr, alpha_lr=cfg.alpha_lr,
                               alpha_init=cfg.alpha_init, gamma=cfg.gamma, tau=cfg.tau, p0=cfg.p0,
                               target_entropy_ratio=cfg.target_entropy_ratio))
    R = Replay(cfg.cap_frames, cfg.cap_states, cfg.cap_decisions, cfg.seed)
    with open(f"{run_dir}/params.json", "w") as f:
        json.dump(L.n_params, f)

    C = dict(steps=0, episodes=0, successes=0, decisions=0, replans=0, departures=0, recovery_states=0,
             grad_steps=0, actor_updates=0, evals_done=[], supp_done=[], wall_train=0.0, wall_update=0.0,
             wall_eval=0.0, wall_total=0.0, segments=0, modeled_ms=0.0, wall_infer_s=0.0, eval_episodes=0,
             eval_env_steps=0, milestone_saved=False)
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
        assert len(R) == 0 and R.nf == 0, "replay must start empty (no legacy / diagnostic data)"
    C["segments"] += 1
    seg0, total0 = time.time(), C["wall_total"]
    log_ep, log_eval, log_upd = (JsonlLog(f"{run_dir}/{n}.jsonl") for n in ("episodes", "evals", "updates"))

    def out_of_time():
        return (time.time() - wall0) / 3600 > args.max_wall_hours

    def strength(step):
        return core.prior_strength(step, cfg.prior_hold_until, cfg.prior_zero_at)

    def make_policy(mode, rng=None, step_fn=None, parity=None, force_strength=None):
        def pol(st, mask, t):
            ref, ref_kind = core.reference_action(st["q"], mask)
            s = strength(step_fn(t)) if force_strength is None else force_strength
            p, _ = L.policy_probs(st, mask, ref, s)
            a = int(np.argmax(p)) if mode == "eval" else int(rng.choice(core.NUM_ACTIONS, p=p))
            if parity is not None:
                parity.append(a == ref)
            ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
            return a, dict(kind="learned", p=float(p[a]), ref=ref, ref_kind=ref_kind, departed=a != ref,
                           entropy=ent, ref_prob=float(p[ref]))
        return pol

    def summarize(res):
        reps, dec = res["requests"], res["decisions"]

        def mean(key):
            v = [q[key] for q in reps]
            return float(np.mean(v)) if v else None
        return dict(success=res["success"], t_end=res["t_end"], ret=res["ret"], decisions=len(dec), replans=len(reps),
                    N_hist={int(n): int(sum(q["n"] == n for q in reps)) for n in core.N_VALUES},
                    mean_N=mean("n"), mean_R=mean("r"), mean_launch_p=mean("p_at_launch"),
                    mean_exec_len=mean("old_executed"), mean_discarded=mean("discarded"),
                    mean_obs_age_handoff=mean("obs_age_at_handoff"),
                    departures=int(sum(d["info"].get("departed", False) for d in dec)),
                    recovery_states=int(sum(d["info"].get("ref_kind") == "recovery" or d["info"].get("kind") == "recovery"
                                            for d in dec)),
                    entropy=float(np.mean([d["info"].get("entropy", 0.0) for d in dec])),
                    underrun_ticks=res["underrun_ticks"], modeled_ms=res["modeled_ms"], wall_infer_s=res["wall_infer_s"],
                    calls_per_1k=1000.0 * len(reps) / max(1, res["t_end"]),
                    modeled_ms_per_1k=1000.0 * res["modeled_ms"] / max(1, res["t_end"]))

    def evaluate(tag, mode="learned"):
        """Returns True when complete, False if the segment ran out of time (partial episodes kept on disk)."""
        part = f"{run_dir}/eval_partial_{tag}.jsonl"
        done = [json.loads(l) for l in open(part)] if os.path.exists(part) else []
        parity_all = [x.get("parity") for x in done if x.get("parity") is not None]
        t0 = time.time()
        with open(part, "a", buffering=1) as fp:
            for e in range(len(done), cfg.eval_episodes):
                if out_of_time():
                    return False
                seed = cfg.eval_seed_base + e
                nr = np.random.default_rng([seed, 1])
                parity = []
                if mode == "reference":
                    pol = reference_policy
                else:
                    pol = make_policy("eval", step_fn=lambda t: C["steps"], parity=parity,
                                      force_strength=0.0 if mode == "prior_off" else None)
                res = exe.run(seed, pol, nr)
                row = dict(seed=seed, parity=float(np.mean(parity)) if parity else None, **summarize(res))
                fp.write(json.dumps(row) + "\n")
                done.append(row)
                C["eval_env_steps"] += res["t_end"]
                C["eval_episodes"] += 1
        C["wall_eval"] += time.time() - t0
        succ = [x["success"] for x in done]
        par = [x["parity"] for x in done if x["parity"] is not None]

        def agg(key):
            v = [x[key] for x in done if x[key] is not None]
            return float(np.mean(v)) if v else None
        row = dict(tag=tag, policy=mode, train_steps=C["steps"],
                   prior_strength=(0.0 if mode == "prior_off" else strength(C["steps"])), grad_steps=C["grad_steps"],
                   n_success=int(sum(succ)), n_episodes=len(succ), success_rate=float(np.mean(succ)),
                   mean_N=agg("mean_N"), mean_R=agg("mean_R"), mean_launch_p=agg("mean_launch_p"),
                   mean_exec_len=agg("mean_exec_len"), mean_obs_age_handoff=agg("mean_obs_age_handoff"),
                   calls_per_1k=agg("calls_per_1k"), modeled_ms_per_1k=agg("modeled_ms_per_1k"),
                   departures=int(sum(x["departures"] for x in done)),
                   recovery_states=int(sum(x["recovery_states"] for x in done)),
                   underrun_ticks=int(sum(x["underrun_ticks"] for x in done)),
                   reference_parity=float(np.mean(par)) if par else None, episodes=done,
                   wall_total_s=total0 + time.time() - seg0)
        target = {"reference": "reference_eval", "prior_off": "evals_supplemental", "learned": "evals"}[mode]
        JsonlLog(f"{run_dir}/{target}.jsonl")(**row)
        os.remove(part)
        print(f"[eval {tag} {mode}] step {C['steps']}  {sum(succ)}/{len(succ)}  parity {row['reference_parity']}  "
              f"recovery {row['recovery_states']}", flush=True)
        return True

    def save_state(tag=None):
        C["wall_total"] = total0 + time.time() - seg0
        rm = R.save_new(f"{run_dir}/replay")
        S = dict(counters=dict(C), rng_train=rng_train.bit_generator.state, learner=L.state_bytes(), replay=rm,
                 config=dataclasses.asdict(cfg))
        rl.save_pickle(state_path, S)
        if tag:
            with open(f"{run_dir}/ckpt/learner_{tag}.msgpack", "wb") as f:
                f.write(S["learner"])
        return S

    def exit_for_resume():
        save_state()
        print("wall-clock budget reached; checkpointed; exiting for resume", flush=True)
        sys.exit(3)

    def due(points, done_list):
        return [p for p in sorted(points) if p <= C["steps"] and p not in done_list]

    if not os.path.exists(f"{run_dir}/reference_eval.jsonl"):
        if not evaluate("fixed_baseline", mode="reference"):
            exit_for_resume()
    eval_pts = set(cfg.eval_steps) | set(range(cfg.eval_interval, cfg.total_steps + 1, cfg.eval_interval))
    upd, seg_eps = {}, 0
    while True:
        for p in due(eval_pts, C["evals_done"]):
            if not evaluate(f"{p}"):
                exit_for_resume()
            C["evals_done"] = C["evals_done"] + [p]
            save_state(tag=f"{C['steps']:07d}")
        for p in due(cfg.supplemental_prior_off_at, C["supp_done"]):
            if not evaluate(f"{p}_prior_off", mode="prior_off"):
                exit_for_resume()
            C["supp_done"] = C["supp_done"] + [p]
            save_state()
        if not C["milestone_saved"] and C["steps"] >= cfg.milestone_steps and cfg.milestone_steps in C["evals_done"]:
            S = save_state()
            md = f"{run_dir}/ckpt/milestone_{cfg.milestone_steps:07d}"
            os.makedirs(md, exist_ok=True)
            shutil.copy(state_path, f"{md}/state.pkl")  # replay rows up to S['replay']['nd'] live in replay/ parts
            C["milestone_saved"] = True
            save_state()
            print(f"MILESTONE {cfg.milestone_steps} saved at {C['steps']} steps (replay rows {S['replay']['nd']})", flush=True)
        if C["steps"] >= cfg.total_steps:
            break
        if out_of_time() or (args.stop_after_episodes and seg_eps >= args.stop_after_episodes):
            exit_for_resume()
        t0 = time.time()
        step0 = C["steps"]
        seed = int(rng_train.integers(cfg.eval_seed_base))
        nr = np.random.default_rng(rng_train.integers(2**63))
        prng = np.random.default_rng(rng_train.integers(2**63))
        res = exe.run(seed, make_policy("train", rng=prng, step_fn=lambda t: step0 + t), nr)
        R.add_episode(res["decisions"])  # training interactions only
        sm = summarize(res)
        C["steps"] += res["t_end"]
        C["episodes"] += 1
        seg_eps += 1
        C["successes"] += int(res["success"])
        for k in ("decisions", "replans", "departures", "recovery_states"):
            C[k] += sm[k]
        C["modeled_ms"] += sm["modeled_ms"]
        C["wall_infer_s"] += sm["wall_infer_s"]
        C["wall_train"] += time.time() - t0
        t0 = time.time()
        new_post = max(0, C["steps"] - max(step0, cfg.warmup_steps))
        n_upd = int(new_post * cfg.updates_per_step) if len(R) >= cfg.batch_size else 0
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
               grad_steps=C["grad_steps"], replay=len(R), **sm)
    with open(f"{run_dir}/DONE", "w") as f:
        f.write(f"steps={C['steps']} job={os.environ.get('SLURM_JOB_ID')} {time.strftime('%F %T')}\n")
    print(f"DONE {C}", flush=True)


if __name__ == "__main__":
    main()
