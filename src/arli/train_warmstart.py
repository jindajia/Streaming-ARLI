"""Warm-start continuation from the ARLI 600k checkpoint (branches A / B, see warmstart_final.py).

300k ADDITIONAL online env steps per run; eval before training and every 25k additional steps on the same 50 held-out
seeds; fresh replay; the first 256 executed chunks are collected without updates (inside the 300k budget); then 25
gradient updates per executed chunk with batch 256 -- one learner per branch (A: ARLI actor+critic; B: residual
actor+critic only), so B does not get a double update budget.

    python -m arli.train_warmstart --config configs/warmstart/ws_B_seed0.yaml --max_wall_hours 0.75
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
class WConfig:
    name: str = "ws_debug"
    branch: str = "A"  # A = continue ARLI actor; B = frozen ARLI + final-step residual actor
    seed: int = 0
    source_run: str = "r2_arli_seed0"
    source_milestone: int = 600_000
    source_root: str = "/N/scratch/jindjia/arli/runs"
    run_root: str = "/N/scratch/jindjia/arli/runs_warmstart"
    gamma: float = 0.999
    batch_size: int = 256
    grad_steps_per_chunk: int = 25
    warmup_chunks: int = 256
    additional_env_steps: int = 300_000
    eval_interval: int = 25_000
    eval_episodes: int = 50
    eval_seed_base: int = 10_000_000
    capacity_a: int = 14_000  # chunk transitions (300k / 25 = 12k executed chunks max)
    capacity_b: int = 30_000  # rows: decision + stored successor obs per chunk (~2 x 12k)
    log_update_every: int = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--max_wall_hours", type=float, default=1e9)
    ap.add_argument("--stop_after_episodes", type=int, default=0, help="testing: checkpoint+exit(3) after N episodes")
    args = ap.parse_args()
    d = yaml.safe_load(open(args.config))
    for o in args.set:
        k, v = o.split("=", 1)
        d[k] = yaml.safe_load(v)
    cfg = WConfig(**d)
    run_dir = os.path.join(cfg.run_root, cfg.name)
    if os.path.exists(f"{run_dir}/DONE"):
        print(f"{cfg.name} already complete: {open(f'{run_dir}/DONE').read().strip()}", flush=True)
        return

    from arli.pi0_policy import configure_caches
    configure_caches()
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
    import jax

    from arli import aloha_env, rl
    from arli.pi0_policy import FrozenPi0
    from arli.rollout import noise_transitions, run_episode
    from arli.schedule import Schedule
    from arli.single_actor import DecisionReplay
    from arli.train import JsonlLog, claim_run_dir, git_rev
    from arli.warmstart_final import Controller, ResidualLearner, resolve_checkpoint, transitions_B, tree_hash

    wall0 = time.time()
    os.makedirs(f"{run_dir}/ckpt", exist_ok=True)
    claim_run_dir(run_dir)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    prov, blob = resolve_checkpoint(cfg.source_root, cfg.source_run, cfg.source_milestone)
    if os.path.exists(f"{run_dir}/provenance.json"):
        old = json.load(open(f"{run_dir}/provenance.json"))
        assert old["sha256"] == prov["sha256"], "source checkpoint changed between segments"
    else:
        with open(f"{run_dir}/provenance.json", "w") as f:
            json.dump(prov, f, indent=1)
    meta = dict(config=dataclasses.asdict(cfg), provenance=prov, git=git_rev(repo), repo=repo, argv=sys.argv,
                host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"), start=time.strftime("%F %T"))
    with open(f"{run_dir}/launch_{int(wall0)}.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1), flush=True)

    sched = Schedule()
    env = aloha_env.make_env()
    pi0 = FrozenPi0(sched.num_denoise)
    agent = rl.make_agent("arli", cfg.seed)
    rl.load_agent_state(agent, blob)
    arli_actor_hash = tree_hash(agent._actor.params)
    if cfg.branch == "A":
        Rp = rl.Replay(rl.observation_space("arli"), (1, rl.NOISE_DIM), cfg.capacity_a, cfg.seed)
        res_learner = None
        params = dict(trainable=["arli_actor", "arli_critic", "arli_temperature"])
    else:
        Rp = DecisionReplay(cfg.capacity_b, cfg.seed)
        res_learner = ResidualLearner(cfg.seed)
        params = dict(trainable=["residual_actor", "residual_critic", "residual_temperature"],
                      frozen=["arli_actor (initial noise)"], residual_actor_params=res_learner.n_params["actor"],
                      residual_critic_params=res_learner.n_params["critic_ensemble"])
    params["arli_actor_params_hash_at_load"] = arli_actor_hash
    with open(f"{run_dir}/params.json", "w") as f:
        json.dump(params, f, indent=1)

    def state_hashes():
        h = dict(arli_actor=tree_hash(agent._actor.params))
        if cfg.branch == "A":
            h.update(arli_critic=tree_hash(agent._critic.params), arli_opt=tree_hash(agent._actor.opt_state),
                     replay_size=len(Rp))
        else:
            h.update(res_actor=tree_hash(res_learner.A.params), res_critic=tree_hash(res_learner.C.params),
                     res_opt=tree_hash(res_learner.A.opt_state), res_temp=tree_hash(res_learner.T.params),
                     replay_size=Rp.size)
        return h

    C = dict(online_env_steps=0, chunks=0, episodes=0, successes=0, transitions=0, grad_steps=0, next_eval=0,
             n_evals=0, wall_online=0.0, wall_update=0.0, wall_eval=0.0, wall_total=0.0, segments=0,
             residual_calls=0)
    rng_env = np.random.default_rng([cfg.seed, 61])
    train_key = jax.random.PRNGKey(cfg.seed + 6100)
    eval_key0 = jax.random.PRNGKey(cfg.seed + 456)

    state_path = f"{run_dir}/state_latest.pkl"
    if os.path.exists(state_path):
        S = rl.load_pickle(state_path)
        C.update(S["counters"])
        rng_env.bit_generator.state = S["rng_env"]
        train_key = jax.numpy.asarray(S["train_key"])
        if cfg.branch == "A":
            rl.load_agent_state(agent, S["arli_agent"])
            Rp.load_state_dict(S["replay"])
        else:
            res_learner.load_bytes(S["residual"])
            Rp.load(f"{run_dir}/replay", S["replay"])
            assert tree_hash(agent._actor.params) == S["arli_actor_hash"], "frozen ARLI actor differs on resume"
        print(f"RESUMED {C}", flush=True)
        print("RESUMED_HASHES", state_hashes(), flush=True)
    C["segments"] += 1
    seg0, total0 = time.time(), C["wall_total"]
    seg_episodes = 0

    log_ep = JsonlLog(f"{run_dir}/episodes.jsonl")
    log_eval = JsonlLog(f"{run_dir}/evals.jsonl")
    log_upd = JsonlLog(f"{run_dir}/updates.jsonl")

    def save_state(tag=None):
        C["wall_total"] = total0 + time.time() - seg0
        S = dict(counters=dict(C), rng_env=rng_env.bit_generator.state, train_key=np.asarray(train_key),
                 config=dataclasses.asdict(cfg), provenance=prov)
        if cfg.branch == "A":
            S["arli_agent"] = rl.agent_state_bytes(agent)
            S["replay"] = Rp.state_dict()
        else:
            assert tree_hash(agent._actor.params) == arli_actor_hash, "frozen ARLI actor changed!"
            S["residual"] = res_learner.state_bytes()
            S["arli_actor_hash"] = arli_actor_hash
            S["replay"] = Rp.save_new(f"{run_dir}/replay")
        rl.save_pickle(state_path, S)
        if tag is not None:  # deployable policy snapshot: B stores BOTH actors
            pol = dict(branch=cfg.branch, arli_agent=rl.agent_state_bytes(agent), provenance=prov,
                       online_env_steps_additional=C["online_env_steps"])
            if cfg.branch == "B":
                pol["residual"] = res_learner.state_bytes()
            rl.save_pickle(f"{run_dir}/ckpt/policy_{tag}.pkl", pol)

    def controller(key, zero=False):
        if cfg.branch == "A":
            return Controller("A", agent, sched, key)
        return Controller("B", agent, sched, key, residual=res_learner, res_key=jax.random.fold_in(key, 77),
                          zero_correction=zero)

    def evaluate():
        t0 = time.time()
        ctrl = controller(eval_key0)
        eps = []
        for e in range(cfg.eval_episodes):
            res = run_episode(env, pi0, sched, cfg.eval_seed_base + e, ctrl)
            eps.append(dict(seed=res["seed"], success=res["success"], t_end=res["t_end"], max_reward=res["max_reward"],
                            chunks=len(res["executed_decisions"])))
        succ = [x["success"] for x in eps]
        C["wall_eval"] += time.time() - t0
        log_eval(branch=cfg.branch, seed=cfg.seed, milestone=C["next_eval"], additional_env_steps=C["online_env_steps"],
                 source_online_env_steps=prov["online_env_steps"],
                 total_online_env_steps=prov["online_env_steps"] + C["online_env_steps"], grad_steps=C["grad_steps"],
                 chunks=C["chunks"], n_success=int(sum(succ)), n_episodes=len(succ), success_rate=float(np.mean(succ)),
                 train_episodes=C["episodes"], train_successes=C["successes"],
                 wall_total_s=total0 + time.time() - seg0, wall_update_s=C["wall_update"],
                 wall_online_s=C["wall_online"], wall_eval_s=C["wall_eval"], episodes=eps,
                 arli_actor_hash_ok=tree_hash(agent._actor.params) == arli_actor_hash if cfg.branch == "B" else None)
        print(f"[eval {cfg.name}] +{C['online_env_steps']}  {sum(succ)}/{len(succ)}", flush=True)

    upd = {}
    while True:
        if C["online_env_steps"] >= C["next_eval"]:
            evaluate()
            C["n_evals"] += 1
            C["next_eval"] += cfg.eval_interval
            save_state(tag=f"{C['online_env_steps']:07d}")
        if C["online_env_steps"] >= cfg.additional_env_steps:
            break
        if (time.time() - wall0) / 3600 > args.max_wall_hours or (
                args.stop_after_episodes and seg_episodes >= args.stop_after_episodes):
            save_state()
            print("SAVED_HASHES", state_hashes(), flush=True)
            print("wall-clock budget reached; checkpointed; exiting for resume", flush=True)
            sys.exit(3)
        t0 = time.time()
        train_key, k = jax.random.split(train_key)
        ctrl = controller(k)
        res = run_episode(env, pi0, sched, int(rng_env.integers(cfg.eval_seed_base)), ctrl)
        n = len(res["executed_decisions"])
        if cfg.branch == "A":
            tr = noise_transitions(res, cfg.gamma, sched.n_exec)
            for t in tr:
                Rp.insert(t["obs"], t["action"], t["reward"], t["mask"], t["discount"], t["next_obs"])
        else:
            tr = transitions_B(res, sched, cfg.gamma)
            Rp.add_episode(tr)
            C["residual_calls"] += sum(1 for c in ctrl.calls if c[0] == "residual")
        updating = C["chunks"] >= cfg.warmup_chunks  # updates start once 256 executed chunks were collected
        C["transitions"] += len(tr)
        C["online_env_steps"] += res["t_end"]
        C["chunks"] += n
        C["episodes"] += 1
        seg_episodes += 1
        C["successes"] += int(res["success"])
        C["wall_online"] += time.time() - t0
        t0 = time.time()
        if updating or C["chunks"] >= cfg.warmup_chunks:
            n_upd = n if updating else C["chunks"] - cfg.warmup_chunks  # only chunks collected after warmup count
            for _ in range(n_upd * cfg.grad_steps_per_chunk):
                if cfg.branch == "A":
                    info = agent.update(Rp.sample(cfg.batch_size))
                else:
                    info = res_learner.update(Rp.sample(cfg.batch_size))
                C["grad_steps"] += 1
                if C["grad_steps"] % 50 == 0:
                    for kk, v in info.items():
                        if np.ndim(v) == 0:
                            upd.setdefault(kk, []).append(float(v))
                if C["grad_steps"] % cfg.log_update_every == 0:
                    log_upd(grad_steps=C["grad_steps"], additional_env_steps=C["online_env_steps"],
                            **{kk: float(np.mean(v)) for kk, v in upd.items()})
                    upd = {}
        C["wall_update"] += time.time() - t0
        log_ep(phase="warmup" if not updating else "online", seed=res["seed"], success=res["success"],
               t_end=res["t_end"], episode_chunks=n, max_reward=res["max_reward"], **C)
    with open(f"{run_dir}/DONE", "w") as f:
        f.write(f"additional_env_steps={C['online_env_steps']} job={os.environ.get('SLURM_JOB_ID')} {time.strftime('%F %T')}\n")
    print(f"DONE {C}", flush=True)


if __name__ == "__main__":
    main()
