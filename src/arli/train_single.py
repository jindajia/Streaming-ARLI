"""Train shared_every_step / final_step_only from RL step zero (frozen pi0, ALOHA Transfer Cube, no RTC).

Protocol = R2: H 50, execute 25, t_delay 20, t_delay^rl 10, 10 denoising iterations, 600k online env steps, eval at 0
and every 25k on the same 50 held-out seeds, prefill until > 1000 executed chunks with prefill std 0.3, 25 gradient
updates per executed online chunk (= ARLI's update budget; NOT multiplied by the number of actor calls).

    python -m arli.train_single --config configs/single/shared_every_step_seed0.yaml --max_wall_hours 0.75
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
class SConfig:
    name: str = "single_debug"
    method: str = "shared_every_step"
    seed: int = 0
    run_root: str = "/N/scratch/jindjia/arli/runs_single"
    gamma: float = 0.999
    batch_size: int = 256
    grad_steps_per_chunk: int = 25
    prefill_chunks: int = 1000  # executed chunks (chunk-level transitions); collect while count <= this
    prefill_std: float = 0.3
    online_env_steps: int = 600_000
    eval_interval: int = 25_000
    eval_episodes: int = 50
    eval_seed_base: int = 10_000_000
    report_every: int = 100_000  # milestone reporting to GitHub (also at completion)
    report: bool = True
    capacity: int = 300_000
    log_update_every: int = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--max_wall_hours", type=float, default=1e9)
    args = ap.parse_args()
    d = yaml.safe_load(open(args.config))
    for o in args.set:
        k, v = o.split("=", 1)
        d[k] = yaml.safe_load(v)
    cfg = SConfig(**d)

    done_flag = os.path.join(cfg.run_root, cfg.name, "DONE")
    if os.path.exists(done_flag):  # surplus chained segment: exit before loading pi0
        print(f"{cfg.name} already complete ({open(done_flag).read().strip()}); nothing to do", flush=True)
        return

    from arli.pi0_policy import configure_caches
    configure_caches()
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
    import jax

    from arli import aloha_env, rl
    from arli.pi0_policy import FrozenPi0
    from arli.rollout import run_episode
    from arli.schedule import Schedule
    from arli.single_actor import Controller, DecisionReplay, SingleActorLearner, transitions
    from arli.train import JsonlLog, claim_run_dir, git_rev

    wall0 = time.time()
    run_dir = os.path.join(cfg.run_root, cfg.name)
    os.makedirs(f"{run_dir}/ckpt", exist_ok=True)
    claim_run_dir(run_dir)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    meta = dict(config=dataclasses.asdict(cfg), git=git_rev(repo), repo=repo, argv=sys.argv, host=os.uname().nodename,
                slurm_job=os.environ.get("SLURM_JOB_ID"), start=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(f"{run_dir}/launch_{int(wall0)}.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1), flush=True)

    sched = Schedule()
    env = aloha_env.make_env()
    pi0 = FrozenPi0(sched.num_denoise)
    L = SingleActorLearner(cfg.seed)
    R = DecisionReplay(cfg.capacity, cfg.seed)
    with open(f"{run_dir}/params.json", "w") as f:
        json.dump(L.n_params, f)
    print("params", L.n_params, flush=True)

    C = dict(prefill_env_steps=0, online_env_steps=0, prefill_chunks=0, online_chunks=0, prefill_episodes=0,
             prefill_successes=0, online_episodes=0, online_successes=0, actor_calls_prefill=0, actor_calls_online=0,
             actor_calls_eval=0, decisions=0, grad_steps=0, next_eval=0, n_evals=0, wall_prefill=0.0,
             wall_online=0.0, wall_update=0.0, wall_eval=0.0, wall_total=0.0, segments=0)
    rng_env = np.random.default_rng([cfg.seed, 1])
    rng_prefill = np.random.default_rng([cfg.seed, 2])
    rng_base = np.random.default_rng([cfg.seed, 3])  # final_step_only's fixed N(0, I) initial noise (training)
    train_key = jax.random.PRNGKey(cfg.seed + 2024)
    eval_key0 = jax.random.PRNGKey(cfg.seed + 456)

    state_path = f"{run_dir}/state_latest.pkl"
    if os.path.exists(state_path):
        S = rl.load_pickle(state_path)
        C.update(S["counters"])
        for g, s in ((rng_env, "rng_env"), (rng_prefill, "rng_prefill"), (rng_base, "rng_base")):
            g.bit_generator.state = S[s]
        train_key = jax.numpy.asarray(S["train_key"])
        L.load_bytes(S["learner"])
        R.load(f"{run_dir}/replay", S["replay"])
        print(f"RESUMED {C}", flush=True)
    C["segments"] += 1
    seg_wall0, total0 = time.time(), C["wall_total"]

    log_ep = JsonlLog(f"{run_dir}/episodes.jsonl")
    log_eval = JsonlLog(f"{run_dir}/evals.jsonl")
    log_upd = JsonlLog(f"{run_dir}/updates.jsonl")

    def save_state(tag=None):
        C["wall_total"] = total0 + time.time() - seg_wall0
        rmeta = R.save_new(f"{run_dir}/replay")
        S = dict(counters=dict(C), rng_env=rng_env.bit_generator.state, rng_prefill=rng_prefill.bit_generator.state,
                 rng_base=rng_base.bit_generator.state, train_key=np.asarray(train_key), learner=L.state_bytes(),
                 replay=rmeta, config=dataclasses.asdict(cfg))
        rl.save_pickle(state_path, S)
        if tag is not None:
            with open(f"{run_dir}/ckpt/learner_{tag}.msgpack", "wb") as f:
                f.write(S["learner"])

    def run(ctrl, seed, **kw):
        return run_episode(env, pi0, sched, seed, ctrl, **kw)

    def evaluate():
        t0 = time.time()
        ctrl = Controller(cfg.method, L, sched, eval_key0, np.random.default_rng([cfg.seed, 456]), mode="eval")
        eps = []
        for e in range(cfg.eval_episodes):
            n0 = len(ctrl.calls)
            res = run(ctrl, cfg.eval_seed_base + e)
            eps.append(dict(seed=res["seed"], success=res["success"], t_end=res["t_end"], max_reward=res["max_reward"],
                            chunks=len(res["executed_decisions"]), actor_calls=len(ctrl.calls) - n0,
                            timing=res["timing"]))
        C["actor_calls_eval"] += len(ctrl.calls)
        succ = [e["success"] for e in eps]
        C["wall_eval"] += time.time() - t0
        log_eval(method=cfg.method, seed=cfg.seed, milestone=C["next_eval"], online_env_steps=C["online_env_steps"],
                 prefill_env_steps=C["prefill_env_steps"], total_env_steps=C["online_env_steps"] + C["prefill_env_steps"],
                 grad_steps=C["grad_steps"], online_chunks=C["online_chunks"], decisions=C["decisions"],
                 actor_calls_online=C["actor_calls_online"], actor_calls_prefill=C["actor_calls_prefill"],
                 n_success=int(sum(succ)), n_episodes=len(succ), success_rate=float(np.mean(succ)),
                 wall_total_s=total0 + time.time() - seg_wall0, wall_update_s=C["wall_update"],
                 wall_online_s=C["wall_online"], wall_eval_s=C["wall_eval"], episodes=eps, eval_wall_s=time.time() - t0)
        print(f"[eval {cfg.method} s{cfg.seed}] {C['online_env_steps']}  {sum(succ)}/{len(succ)}", flush=True)

    def report(reason):
        if not cfg.report:
            return
        try:
            out = subprocess.run([sys.executable, f"{repo}/scripts/report_single.py", "--reason", reason],
                                 capture_output=True, text=True, timeout=900)
            print("[report]", reason, out.returncode, out.stdout[-800:], out.stderr[-800:], flush=True)
        except Exception as ex:  # noqa: BLE001  (reporting must never kill training)
            print("[report] failed:", ex, flush=True)

    def add(res, phase):
        tr = transitions(res, sched, cfg.gamma)
        R.add_episode(tr)
        n_chunks = len(res["executed_decisions"])
        C["decisions"] += len(tr)
        return n_chunks

    # ---------------- prefill ---------------------------------------------------------------------
    pre = Controller(cfg.method, L, sched, None, rng_prefill, mode="prefill", prefill_std=cfg.prefill_std)
    if cfg.method == "final_step_only":
        pre.noise_rng = rng_prefill  # base N(0,I) noise and prefill actions share the prefill stream
    while C["prefill_chunks"] <= cfg.prefill_chunks and C["online_env_steps"] == 0:
        t0 = time.time()
        n0 = len(pre.calls)
        res = run(pre, int(rng_env.integers(cfg.eval_seed_base)))
        n = add(res, "prefill")
        C["prefill_env_steps"] += res["t_end"]
        C["prefill_chunks"] += n
        C["prefill_episodes"] += 1
        C["prefill_successes"] += int(res["success"])
        C["actor_calls_prefill"] += len(pre.calls) - n0
        C["wall_prefill"] += time.time() - t0
        log_ep(phase="prefill", seed=res["seed"], success=res["success"], t_end=res["t_end"], chunks=n,
               actor_calls=len(pre.calls) - n0, timing=res["timing"], **C)
        if C["prefill_chunks"] > cfg.prefill_chunks:
            with open(f"{run_dir}/prefill_summary.json", "w") as f:
                json.dump(dict(prefill_std=cfg.prefill_std, episodes=C["prefill_episodes"],
                               successes=C["prefill_successes"],
                               success_rate=C["prefill_successes"] / C["prefill_episodes"],
                               chunk_transitions=C["prefill_chunks"], decision_transitions=C["decisions"],
                               env_steps=C["prefill_env_steps"], actor_calls=C["actor_calls_prefill"],
                               initial_noise=("N(0,0.3^2) via actor-output prefill" if cfg.method == "shared_every_step"
                                              else "fixed pi0 N(0,I); prefill randomises only the k=9 correction")),
                          f, indent=1)
            save_state()

    # ---------------- online ----------------------------------------------------------------------
    upd = {}
    while True:
        if C["online_env_steps"] >= C["next_eval"]:
            evaluate()
            C["n_evals"] += 1
            ms = C["next_eval"]
            C["next_eval"] += cfg.eval_interval
            save_state(tag=f"{C['online_env_steps']:07d}")
            if ms > 0 and (ms % cfg.report_every == 0 or C["online_env_steps"] >= cfg.online_env_steps):
                report(f"{cfg.name} milestone {ms}")
        if C["online_env_steps"] >= cfg.online_env_steps:
            break
        if (time.time() - wall0) / 3600 > args.max_wall_hours:
            save_state()
            print("wall-clock budget reached; checkpointed; exiting for resume", flush=True)
            sys.exit(3)
        t0 = time.time()
        train_key, k = jax.random.split(train_key)
        ctrl = Controller(cfg.method, L, sched, k, rng_base, mode="train")
        res = run(ctrl, int(rng_env.integers(cfg.eval_seed_base)))
        n = add(res, "online")
        C["online_env_steps"] += res["t_end"]
        C["online_chunks"] += n
        C["online_episodes"] += 1
        C["online_successes"] += int(res["success"])
        C["actor_calls_online"] += len(ctrl.calls)
        C["wall_online"] += time.time() - t0
        t0 = time.time()
        for _ in range(n * cfg.grad_steps_per_chunk):
            info = L.update(R.sample(cfg.batch_size))
            C["grad_steps"] += 1
            if C["grad_steps"] % 50 == 0:
                for kk, v in info.items():
                    upd.setdefault(kk, []).append(float(v))
            if C["grad_steps"] % cfg.log_update_every == 0:
                log_upd(grad_steps=C["grad_steps"], online_env_steps=C["online_env_steps"],
                        **{kk: float(np.mean(v)) for kk, v in upd.items()},
                        q_max_max=float(np.max(upd.get("q_max", [np.nan]))))
                upd = {}
        C["wall_update"] += time.time() - t0
        log_ep(phase="online", seed=res["seed"], success=res["success"], t_end=res["t_end"], chunks=n,
               actor_calls=len(ctrl.calls), timing=res["timing"], **C)
    with open(done_flag, "w") as f:
        f.write(f"online_env_steps={C['online_env_steps']} job={os.environ.get('SLURM_JOB_ID')} {time.strftime('%F %T')}\n")
    print(f"DONE {C}", flush=True)


if __name__ == "__main__":
    main()
