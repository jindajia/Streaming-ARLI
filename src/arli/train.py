"""Paper-based ARLI reconstruction / naive async DSRL training on ALOHA Transfer Cube (frozen pi0, no RTC).

Loop structure follows dsrl_pi0 ``trajwise_alternating_training_loop``: collect one episode, then take
``grad_steps_per_transition`` (25) gradient steps per latent transition added.  Differences, all recorded in
DECISIONS.md: evaluation/budget are indexed by *online environment steps* (not gradient steps), the initial
collection uses N(0, prefill_std^2) noise (ARLI Table 6), evaluation uses a fixed held-out seed list.

    python -m arli.train --config configs/r2_arli_seed0.yaml [--resume]
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
class Config:
    name: str = "debug"
    method: str = "arli"  # dsrl | arli | arli_actions | arli_state
    seed: int = 0
    run_root: str = "/N/scratch/jindjia/arli/runs"
    # schedule (ARLI §5.1 / App. A)
    horizon: int = 50
    n_exec: int = 25
    delay: int = 20
    delay_rl: int = 10
    num_denoise: int = 10
    max_steps: int = 400
    # RL (ARLI Table 3 / Table 6)
    batch_size: int = 256
    gamma: float = 0.999
    grad_steps_per_transition: int = 25
    prefill_transitions: int = 1000  # latent-MDP transitions; updates start once len(buffer) > this
    prefill_std: float = 0.3  # N(0, 0.3) as printed; interpreted as std (see DECISIONS.md D7)
    online_env_steps: int = 600_000
    replay_capacity: int = 60_000
    # evaluation
    eval_interval: int = 25_000
    eval_episodes: int = 50
    eval_seed_base: int = 10_000_000  # train seeds are drawn from [0, eval_seed_base)
    video_episodes: int = 3  # first k eval episodes saved as mp4 at selected evals
    video_every: int = 100_000
    log_update_every: int = 1000


def load_config(path, overrides=()):
    with open(path) as f:
        d = yaml.safe_load(f)
    for o in overrides:
        k, v = o.split("=", 1)
        d[k] = yaml.safe_load(v)
    return Config(**d)


def git_rev(repo):
    try:
        rev = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", repo, "diff", "--quiet", "HEAD", "--", "src", "configs"]) != 0
        return rev + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


class JsonlLog:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def __call__(self, **kw):
        self.f.write(json.dumps(kw, default=float) + "\n")


def claim_run_dir(run_dir):
    """Refuse to run if another live Slurm job owns this run directory (resume-chains must not overlap)."""
    owner = f"{run_dir}/OWNER"
    me = os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")
    if os.path.exists(owner):
        other = open(owner).read().strip()
        if other != me and other.isdigit():
            alive = subprocess.run(["squeue", "-h", "-j", other], capture_output=True, text=True).stdout.strip()
            if alive:
                sys.exit(f"run dir {run_dir} is owned by running job {other}; refusing to start")
    with open(owner, "w") as f:
        f.write(me)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--max_wall_hours", type=float, default=1e9, help="checkpoint and exit before this")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    from arli.pi0_policy import configure_caches
    configure_caches()
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"

    import jax

    from arli import aloha_env, rl
    from arli.controllers import GaussianNoise, NoiseActor
    from arli.pi0_policy import FrozenPi0
    from arli.rollout import noise_transitions, run_episode
    from arli.schedule import Schedule

    wall0 = time.time()
    run_dir = os.path.join(cfg.run_root, cfg.name)
    os.makedirs(f"{run_dir}/ckpt", exist_ok=True)
    os.makedirs(f"{run_dir}/videos", exist_ok=True)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    claim_run_dir(run_dir)
    meta = dict(config=dataclasses.asdict(cfg), git=git_rev(repo), argv=sys.argv, host=os.uname().nodename,
                slurm_job=os.environ.get("SLURM_JOB_ID"), start=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(f"{run_dir}/launch_{int(wall0)}.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1), flush=True)

    sched = Schedule(cfg.horizon, cfg.n_exec, cfg.delay, cfg.delay_rl, cfg.num_denoise, cfg.max_steps)
    env = aloha_env.make_env()
    pi0 = FrozenPi0(cfg.num_denoise)
    agent = rl.make_agent(cfg.method, cfg.seed)
    replay = rl.Replay(rl.observation_space(cfg.method), (1, rl.NOISE_DIM), cfg.replay_capacity, cfg.seed)

    C = dict(prefill_env_steps=0, online_env_steps=0, prefill_transitions=0, online_transitions=0,
             prefill_episodes=0, online_episodes=0, grad_steps=0, next_eval=0, n_evals=0, next_eval_video=0,
             wall_prefill=0.0, wall_online=0.0, wall_update=0.0, wall_eval=0.0)
    rng_env = np.random.default_rng([cfg.seed, 1])
    rng_prefill = np.random.default_rng([cfg.seed, 2])
    eval_key0 = jax.random.PRNGKey(cfg.seed + 456)  # same key at every evaluation (as dsrl_pi0)

    state_path = f"{run_dir}/state_latest.pkl"
    if os.path.exists(state_path):
        S = rl.load_pickle(state_path)
        C.update(S["counters"])
        rng_env.bit_generator.state = S["rng_env"]
        rng_prefill.bit_generator.state = S["rng_prefill"]
        replay.load_state_dict(S["replay"])
        rl.load_agent_state(agent, S["agent"])
        print(f"RESUMED from {state_path}: {C}", flush=True)

    log_ep = JsonlLog(f"{run_dir}/episodes.jsonl")
    log_eval = JsonlLog(f"{run_dir}/evals.jsonl")
    log_upd = JsonlLog(f"{run_dir}/updates.jsonl")

    def save_state(tag=None):
        S = dict(counters=dict(C), rng_env=rng_env.bit_generator.state, rng_prefill=rng_prefill.bit_generator.state,
                 replay=replay.state_dict(), agent=rl.agent_state_bytes(agent), config=dataclasses.asdict(cfg))
        rl.save_pickle(state_path, S)
        if tag is not None:  # small agent-only checkpoints for every evaluation point
            with open(f"{run_dir}/ckpt/agent_{tag}.msgpack", "wb") as f:
                f.write(S["agent"])

    def add(res):
        trans = noise_transitions(res, cfg.gamma, cfg.n_exec)
        for tr in trans:
            replay.insert(tr["obs"], tr["action"], tr["reward"], tr["mask"], tr["discount"], tr["next_obs"])
        return len(trans)

    def evaluate():
        t0 = time.time()
        ctrl = NoiseActor(cfg.method, agent, eval_key=eval_key0)
        eps = []
        save_video = C["online_env_steps"] >= C["next_eval_video"]
        for e in range(cfg.eval_episodes):
            res = run_episode(env, pi0, sched, cfg.eval_seed_base + e, ctrl, video=save_video and e < cfg.video_episodes)
            eps.append(dict(seed=res["seed"], success=res["success"], t_end=res["t_end"], max_reward=res["max_reward"],
                            ret=res["ret"]))
            if res["frames"] is not None:
                import imageio
                imageio.mimsave(f"{run_dir}/videos/eval_{C['online_env_steps']:07d}_ep{e}.mp4", res["frames"], fps=50)
        if save_video:
            C["next_eval_video"] = C["online_env_steps"] + cfg.video_every
        succ = [e["success"] for e in eps]
        C["wall_eval"] += time.time() - t0
        row = dict(milestone=C["next_eval"], online_env_steps=C["online_env_steps"],
                   prefill_env_steps=C["prefill_env_steps"],
                   total_env_steps=C["online_env_steps"] + C["prefill_env_steps"],
                   grad_steps=C["grad_steps"], online_transitions=C["online_transitions"],
                   prefill_transitions=C["prefill_transitions"], online_episodes=C["online_episodes"],
                   n_success=int(sum(succ)), n_episodes=len(succ), success_rate=float(np.mean(succ)),
                   episodes=eps, wall_s=time.time() - wall0, eval_wall_s=time.time() - t0)
        log_eval(**row)
        print(f"[eval] step {C['online_env_steps']}  success {sum(succ)}/{len(succ)}  ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- initial collection (prefill) --------------------------------------------------
    gauss = GaussianNoise(cfg.method, rng_prefill, cfg.prefill_std)
    while len(replay) <= cfg.prefill_transitions:
        t0 = time.time()
        res = run_episode(env, pi0, sched, int(rng_env.integers(cfg.eval_seed_base)), gauss)
        n = add(res)
        C["prefill_env_steps"] += res["t_end"]
        C["prefill_transitions"] += n
        C["prefill_episodes"] += 1
        C["wall_prefill"] += time.time() - t0
        log_ep(phase="prefill", seed=res["seed"], success=res["success"], t_end=res["t_end"], ret=res["ret"],
               max_reward=res["max_reward"], n_trans=n, buffer=len(replay), timing=res["timing"], **C)
    if C["online_env_steps"] == 0 and C["prefill_episodes"] and not os.path.exists(f"{run_dir}/ckpt/prefill_done"):
        save_state()
        open(f"{run_dir}/ckpt/prefill_done", "w").close()
        print(f"[prefill] done: {C['prefill_episodes']} episodes, {C['prefill_env_steps']} env steps, "
              f"{C['prefill_transitions']} transitions", flush=True)

    # ---------------- online -------------------------------------------------------------------
    actor = NoiseActor(cfg.method, agent)
    upd_acc: dict[str, list] = {}
    while True:
        if C["online_env_steps"] >= C["next_eval"]:
            evaluate()
            C["n_evals"] += 1
            C["next_eval"] += cfg.eval_interval
            save_state(tag=f"{C['online_env_steps']:07d}")
        if C["online_env_steps"] >= cfg.online_env_steps:
            break
        if (time.time() - wall0) / 3600 > args.max_wall_hours:
            save_state()
            print("wall-clock budget reached; checkpointed; exiting for resume", flush=True)
            sys.exit(3)
        t0 = time.time()
        res = run_episode(env, pi0, sched, int(rng_env.integers(cfg.eval_seed_base)), actor)
        n = add(res)
        C["online_env_steps"] += res["t_end"]
        C["online_transitions"] += n
        C["online_episodes"] += 1
        C["wall_online"] += time.time() - t0
        t0 = time.time()
        for _ in range(n * cfg.grad_steps_per_transition):
            info = agent.update(replay.sample(cfg.batch_size))
            C["grad_steps"] += 1
            if C["grad_steps"] % 50 == 0:
                for k, v in info.items():
                    if np.ndim(v) == 0:
                        upd_acc.setdefault(k, []).append(float(v))
            if C["grad_steps"] % cfg.log_update_every == 0:
                log_upd(grad_steps=C["grad_steps"], online_env_steps=C["online_env_steps"],
                        **{k: float(np.mean(v)) for k, v in upd_acc.items()})
                upd_acc = {}
        C["wall_update"] += time.time() - t0
        log_ep(phase="online", seed=res["seed"], success=res["success"], t_end=res["t_end"], ret=res["ret"],
               max_reward=res["max_reward"], n_trans=n, buffer=len(replay), timing=res["timing"], **C)
    print(f"DONE {C}", flush=True)


if __name__ == "__main__":
    main()
