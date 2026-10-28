"""S1: matched continuation from a common R2 ARLI checkpoint.

Branches (one seed, same additional online env-step budget, same eval protocol):
  A  continued ARLI (initial-noise steering only)
  B  shared steering actor at expert steps 1..9, observation held at the initial-noise read (o_{m_j})
  C  same architecture, latest causal observation o_t at each step
  D  initial noise + one correction before the final expert call (k = 9), latest observation

Common to all branches (DECISIONS.md D14-D17): the ARLI actor/critic/temperature *and their optimizer states*
are restored from the same checkpoint; the replay buffer is reset for every branch (B-D tuples are semantically
incompatible with R2's chunk-level tuples; A is reset too so the four branches are treated identically); updates
start after ``warmup_chunks`` executed chunks (counted inside the budget); 25 gradient iterations per executed
chunk (= 1 per online env step).  In B-D one iteration updates (Q_w, pi_w) and (Q_u, pi_u) on separate
256-transition batches; A updates (Q_w, pi_w) only.  Disclosed in STATUS/results.

    python -m arli.train_continue --config configs/s1_C_seed0.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import yaml


@dataclasses.dataclass
class CConfig:
    name: str = "s1_debug"
    branch: str = "A"
    seed: int = 0
    init_agent: str = ""  # path to an R2 agent_*.msgpack
    run_root: str = "/N/scratch/jindjia/arli/runs"
    gamma: float = 0.999
    batch_size: int = 256
    grad_steps_per_chunk: int = 25
    warmup_chunks: int = 100
    online_env_steps: int = 100_000
    eval_interval: int = 10_000
    eval_episodes: int = 50
    eval_seed_base: int = 10_000_000
    diag_episodes: int = 5  # full X/V/u tensors saved for the first k eval episodes at first and last eval
    capacity_w: int = 8_000
    capacity_u: int = 72_000
    steer_bound: float = 1.0
    log_update_every: int = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    with open(args.config) as f:
        d = yaml.safe_load(f)
    for o in args.set:
        k, v = o.split("=", 1)
        d[k] = yaml.safe_load(v)
    cfg = CConfig(**d)

    from arli.pi0_policy import configure_caches
    configure_caches()
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
    import flax.serialization
    import jax

    from arli import aloha_env, rl
    from arli.controllers import NoiseActor, stepwise
    from arli.pi0_policy import FrozenPi0
    from arli.rollout import noise_transitions, run_episode
    from arli.schedule import Schedule
    from arli.steering import SteeringController, SteeringLearner, SteerSpec, steering_transitions, z_space
    from arli.train import JsonlLog, claim_run_dir, git_rev

    wall0 = time.time()
    run_dir = os.path.join(cfg.run_root, cfg.name)
    os.makedirs(f"{run_dir}/ckpt", exist_ok=True)
    os.makedirs(f"{run_dir}/diag", exist_ok=True)
    claim_run_dir(run_dir)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    meta = dict(config=dataclasses.asdict(cfg), git=git_rev(repo), argv=sys.argv, host=os.uname().nodename,
                slurm_job=os.environ.get("SLURM_JOB_ID"), start=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(f"{run_dir}/launch_{int(wall0)}.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1), flush=True)

    sched = Schedule()
    env = aloha_env.make_env()
    pi0 = FrozenPi0(sched.num_denoise)
    agent = rl.make_agent("arli", cfg.seed)
    with open(cfg.init_agent, "rb") as f:
        rl.load_agent_state(agent, f.read())
    steer = cfg.branch in ("B", "C", "D")
    if steer:
        L = SteeringLearner(agent, SteerSpec(cfg.branch, bound=cfg.steer_bound), cfg.seed)
        n_params = L.n_params
        RW = rl.Replay(rl.observation_space("arli"), (1, rl.NOISE_DIM), cfg.capacity_w, cfg.seed, next_space=z_space())
        RU = rl.Replay(z_space(), (1, rl.NOISE_DIM), cfg.capacity_u, cfg.seed + 1, extra=("next_is_w",))
    else:
        from arli.steering import count_params
        n_params = dict(actor_w=count_params(agent._actor.params), critic_w=count_params(agent._critic.params))
        RW = rl.Replay(rl.observation_space("arli"), (1, rl.NOISE_DIM), cfg.capacity_w, cfg.seed)
    with open(f"{run_dir}/params.json", "w") as f:
        json.dump(n_params, f, indent=1)
    print("params", n_params, flush=True)

    C = dict(online_env_steps=0, chunks=0, w_transitions=0, u_transitions=0, episodes=0, grad_iters=0,
             next_eval=0, actor_calls=0, wall_online=0.0, wall_update=0.0, wall_eval=0.0, n_evals=0)
    rng_env = np.random.default_rng([cfg.seed, 11])
    train_key = jax.random.PRNGKey(cfg.seed + 2024)
    eval_key0 = jax.random.PRNGKey(cfg.seed + 456)

    state_path = f"{run_dir}/state_latest.pkl"

    def learner_blob():
        return flax.serialization.to_bytes(L.state()) if steer else rl.agent_state_bytes(agent)

    def save_state(tag=None):
        S = dict(counters=dict(C), rng_env=rng_env.bit_generator.state, train_key=np.asarray(train_key),
                 RW=RW.state_dict(), RU=RU.state_dict() if steer else None, learner=learner_blob())
        rl.save_pickle(state_path, S)
        if tag is not None:
            with open(f"{run_dir}/ckpt/learner_{tag}.msgpack", "wb") as f:
                f.write(S["learner"])

    if os.path.exists(state_path):
        S = rl.load_pickle(state_path)
        C.update(S["counters"])
        rng_env.bit_generator.state = S["rng_env"]
        train_key = jax.numpy.asarray(S["train_key"])
        RW.load_state_dict(S["RW"])
        if steer:
            RU.load_state_dict(S["RU"])
            L.load(flax.serialization.from_bytes(L.state(), S["learner"]))
        else:
            rl.load_agent_state(agent, S["learner"])
        print(f"RESUMED {C}", flush=True)

    log_ep = JsonlLog(f"{run_dir}/episodes.jsonl")
    log_eval = JsonlLog(f"{run_dir}/evals.jsonl")
    log_upd = JsonlLog(f"{run_dir}/updates.jsonl")

    def controller(key):
        if steer:
            return SteeringController(L, sched, key)
        return stepwise(NoiseActor("arli", agent, eval_key=key))

    def save_diag(res, tag, e):
        """Bounded diagnostic subset: per-step X, V, corrections, observation times, queue alignment."""
        out = {"seed": res["seed"], "success": res["success"], "t_end": res["t_end"],
               "executed_actions": res["executed_actions"], "trace": np.asarray(res["trace"])}
        for j, dj in res["decisions"].items():
            out[f"j{j}_w"] = dj["rec"]["action"]
            out[f"j{j}_committed"] = dj["ctx"]["committed"]
            out[f"j{j}_t_old_mid"] = np.asarray([dj["ctx"]["t_old"], dj["ctx"]["t_mid"]])
            if "diag" in dj:
                out[f"j{j}_x_in"] = np.stack([s["x_in"][0] for s in dj["diag"]])
                out[f"j{j}_x_tilde"] = np.stack([s["x_tilde"][0] for s in dj["diag"]])
                out[f"j{j}_v"] = np.stack([s["v"][0] for s in dj["diag"]])
                out[f"j{j}_tau"] = np.asarray([s["tau"] for s in dj["diag"]])
                out[f"j{j}_t_step"] = np.asarray([s["t"] for s in dj["diag"]])
            if dj["steer"]:
                out[f"j{j}_u"] = np.stack([s["action"][0] for s in dj["steer"]])
                out[f"j{j}_steer_k_tobs_tnow_fresh"] = np.asarray(
                    [[s["k"], s["t_obs"], s["t_now"], s["fresh_frame"]] for s in dj["steer"]])
        # same-snapshot offline probe: re-run chunk j's sampler from the same w with the recorded corrections kept
        # only on subsets of steps; distances of final actions to the all-corrections result
        if steer:
            from arli.pi0_policy import obs_to_pi0_input
            probes = []
            for j, dj in list(res["decisions"].items())[:6]:
                if not dj["steer"] or dj["ctx"]["raw_old"] is None:
                    continue
                inp = obs_to_pi0_input(dj["ctx"]["raw_old"])
                us = {s["k"]: s["action"] for s in dj["steer"]}
                ks = sorted(us)
                noise = np.repeat(np.reshape(dj["rec"]["action"], (1, 1, 32)), 50, axis=1)

                def run(keep):
                    return pi0.sample_stepwise(inp, noise, hook=lambda k, x: (
                        x + float(pi0.taus[k]) * np.repeat(us[k][None], 50, axis=1) if k in keep else x))
                full = run(set(ks))
                subsets = {"none": set(), "first_half": set(ks[:len(ks) // 2]), "second_half": set(ks[len(ks) // 2:]),
                           "last_only": {ks[-1]}, "all_but_last": set(ks[:-1])}
                row = [j] + [float(np.abs(run(v) - full).max()) for v in subsets.values()]
                probes.append(row)
            if probes:
                out["probe_cols"] = np.asarray(["j"] + list(subsets))
                out["probe_maxabs_vs_full"] = np.asarray(probes)
        np.savez_compressed(f"{run_dir}/diag/{tag}_ep{e}.npz", **out)

    def evaluate(final=False):
        t0 = time.time()
        ctrl = controller(eval_key0)
        eps, fresh, calls = [], [], 0
        diag = C["n_evals"] == 0 or final
        for e in range(cfg.eval_episodes):
            rec = diag and e < cfg.diag_episodes
            res = run_episode(env, pi0, sched, cfg.eval_seed_base + e, ctrl, record=rec)
            for dj in res["decisions"].values():
                calls += 1 + len(dj["steer"])
                fresh += [s["fresh_frame"] for s in dj["steer"]]
            eps.append(dict(seed=res["seed"], success=res["success"], t_end=res["t_end"], max_reward=res["max_reward"],
                            ret=res["ret"], timing=res["timing"]))
            if rec:
                save_diag(res, f"eval{C['online_env_steps']:07d}", e)
        succ = [e["success"] for e in eps]
        C["wall_eval"] += time.time() - t0
        log_eval(branch=cfg.branch, online_env_steps=C["online_env_steps"], milestone=C["next_eval"],
                 grad_iters=C["grad_iters"], chunks=C["chunks"], n_success=int(sum(succ)), n_episodes=len(succ),
                 success_rate=float(np.mean(succ)), actor_calls=calls,
                 fresh_frame_frac=float(np.mean(fresh)) if fresh else None,
                 actor_s=float(sum(e["timing"]["actor_s"] for e in eps)),
                 pi0_s=float(sum(e["timing"]["pi0_s"] for e in eps)), episodes=eps, eval_wall_s=time.time() - t0)
        print(f"[eval {cfg.branch}] +{C['online_env_steps']}  success {sum(succ)}/{len(succ)}", flush=True)

    upd_acc: dict[str, list] = {}
    while True:
        if C["online_env_steps"] >= C["next_eval"]:
            final = C["online_env_steps"] >= cfg.online_env_steps
            evaluate(final=final)
            C["n_evals"] += 1
            C["next_eval"] += cfg.eval_interval
            save_state(tag=f"{C['online_env_steps']:07d}")
        if C["online_env_steps"] >= cfg.online_env_steps:
            break
        t0 = time.time()
        train_key, k = jax.random.split(train_key)
        res = run_episode(env, pi0, sched, int(rng_env.integers(cfg.eval_seed_base)), controller(k))
        n_chunks = len(res["executed_decisions"])
        if steer:
            W, U = steering_transitions(res, sched, cfg.gamma)
            for tr in W:
                RW.insert(tr["obs"], tr["action"], tr["reward"], tr["mask"], tr["discount"], tr["next_obs"])
            for tr in U:
                RU.insert(tr["obs"], tr["action"], tr["reward"], tr["mask"], tr["discount"], tr["next_obs"],
                          next_is_w=tr["next_is_w"])
            C["u_transitions"] += len(U)
        else:
            W = noise_transitions(res, cfg.gamma, sched.n_exec)
            for tr in W:
                RW.insert(tr["obs"], tr["action"], tr["reward"], tr["mask"], tr["discount"], tr["next_obs"])
        C["w_transitions"] += len(W)
        C["actor_calls"] += sum(1 + len(dj["steer"]) for dj in res["decisions"].values())
        C["online_env_steps"] += res["t_end"]
        C["chunks"] += n_chunks
        C["episodes"] += 1
        C["wall_online"] += time.time() - t0
        t0 = time.time()
        if C["chunks"] >= cfg.warmup_chunks:
            for _ in range(n_chunks * cfg.grad_steps_per_chunk):
                if steer:
                    info = L.update(RW.sample(cfg.batch_size), RU.sample(cfg.batch_size))
                else:
                    info = agent.update(RW.sample(cfg.batch_size))
                C["grad_iters"] += 1
                if C["grad_iters"] % 50 == 0:
                    for kk, v in info.items():
                        if np.ndim(v) == 0:
                            upd_acc.setdefault(kk, []).append(float(v))
                if C["grad_iters"] % cfg.log_update_every == 0:
                    log_upd(grad_iters=C["grad_iters"], online_env_steps=C["online_env_steps"],
                            **{kk: float(np.mean(v)) for kk, v in upd_acc.items()})
                    upd_acc = {}
        C["wall_update"] += time.time() - t0
        log_ep(branch=cfg.branch, seed=res["seed"], success=res["success"], t_end=res["t_end"], ret=res["ret"],
               n_chunks=n_chunks, timing=res["timing"], **C)
    print(f"DONE {C}", flush=True)


if __name__ == "__main__":
    main()
