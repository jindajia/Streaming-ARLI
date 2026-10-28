"""Asynchronous (no-RTC) ALOHA episode runner following ``schedule.Schedule``.

A *controller* chooses the initial noise for each inference (and, for the steering extension, may modify X
between frozen expert steps).  The runner owns the physical clock: the simulator advances one step per action
and the robot keeps executing the previous chunk's committed actions while inference j is in flight.

Controller interface
--------------------
``initial_noise(ctx) -> (noise[1,50,32], action_record)``  called at m_j with
    ctx = dict(j, t, img_old, q_old, committed, img_mid, q_mid, pi0_old)
``steering`` (optional attribute, default None) -> object with
    ``steps``: set of expert steps k (>=1) that receive a correction
    ``correct(ctx, k, x, tau, latest) -> (x_tilde, record)`` called at the physical time of step k.
If ``steering`` is None the chunk is produced in one ``pi0.infer`` call at m_j (identical maths, faster).
"""

from __future__ import annotations

import time

import numpy as np

from arli import aloha_env
from arli.pi0_policy import obs_to_pi0_input
from arli.schedule import Schedule


class EpisodeResult(dict):
    pass


def run_episode(env, pi0, sched: Schedule, seed: int, controller, *, record: bool = False,
                video: bool = False) -> EpisodeResult:
    obs, _ = env.reset(seed=seed)
    n, d, d_rl = sched.n_exec, sched.delay, sched.delay_rl
    steering = getattr(controller, "steering", None)

    def snap(o, t):
        return dict(t=t, img=aloha_env.rl_image(o), q=aloha_env.qpos(o), raw=o)

    chunks: dict[int, np.ndarray] = {}
    starts: dict[int, dict] = {}  # j -> snapshot at s_j
    decisions: dict[int, dict] = {}
    inflight: dict | None = None
    rewards, executed, frames, trace = [], [], [], []
    timing = dict(pi0_s=0.0, actor_s=0.0, sim_s=0.0, n_pi0=0)
    success, t = False, 0
    cur = snap(obs, 0)

    def committed_for(j):
        if j == 0:  # startup placeholder: robot at rest; "hold current pose" for d steps
            return np.repeat(cur["q"][None], d, axis=0)
        a, b = sched.committed_window(j)
        prev = chunks[j - 1]
        return prev[a - sched.s(j - 1): b - sched.s(j - 1)].astype(np.float32)

    def decide(j, latest):
        nonlocal inflight
        old = starts[j]
        ctx = dict(j=j, t=latest["t"], img_old=old["img"], q_old=old["q"], committed=committed_for(j),
                   img_mid=latest["img"], q_mid=latest["q"], t_old=old["t"], t_mid=latest["t"])
        t0 = time.perf_counter()
        noise, rec = controller.initial_noise(ctx)
        timing["actor_s"] += time.perf_counter() - t0
        decisions[j] = dict(ctx=ctx, rec=rec, steer=[])
        if steering is None:
            t0 = time.perf_counter()
            chunks[j] = pi0.infer(obs_to_pi0_input(old["raw"]), noise)
            timing["pi0_s"] += time.perf_counter() - t0
            timing["n_pi0"] += 1
        else:
            t0 = time.perf_counter()
            st = pi0.begin(obs_to_pi0_input(old["raw"]))
            timing["pi0_s"] += time.perf_counter() - t0
            timing["n_pi0"] += 1
            inflight = dict(j=j, st=st, x=np.asarray(noise, np.float32), k=0, ctx=ctx)

    def expert_step(latest):
        """Run expert step k of the in-flight inference at the current physical time."""
        nonlocal inflight
        f = inflight
        k = f["k"]
        tau = float(pi0.taus[k])
        x = f["x"]
        if k in steering.steps:
            t0 = time.perf_counter()
            x, rec = steering.correct(f["ctx"], k, x, tau, latest)
            timing["actor_s"] += time.perf_counter() - t0
            decisions[f["j"]]["steer"].append(dict(k=k, tau=tau, t=latest["t"], **rec))
        t0 = time.perf_counter()
        x_next, v = pi0.step(f["st"], x, k)
        timing["pi0_s"] += time.perf_counter() - t0
        if record:
            decisions[f["j"]].setdefault("diag", []).append(
                dict(k=k, tau=tau, t=latest["t"], x_in=np.asarray(f["x"]), x_tilde=np.asarray(x), v=np.asarray(v)))
        f["x"], f["k"] = np.asarray(x_next), k + 1
        if f["k"] == sched.num_denoise:
            t0 = time.perf_counter()
            chunks[f["j"]] = pi0.finish(f["st"], x_next)
            timing["pi0_s"] += time.perf_counter() - t0
            inflight = None

    while t < sched.max_steps:
        # ---- events at time t (before env.step number t) ------------------------------------------
        if t == 0:  # synchronous startup, j = 0
            starts[0] = cur
            decide(0, cur)
            while inflight is not None:  # clock frozen: all steps see o_0
                expert_step(cur)
        else:
            if (t + d) % n == 0:
                starts[(t + d) // n] = cur
            if (t + d_rl) % n == 0:
                decide((t + d_rl) // n, cur)
            if inflight is not None and inflight["j"] > 0:
                expert_step(cur)
        # ---- execute -------------------------------------------------------------------------------
        j = sched.executing_chunk(t)
        assert j in chunks, f"chunk {j} not ready at t={t} (deadline violated)"
        idx = sched.chunk_index(j, t)
        a = chunks[j][idx]
        if record:
            trace.append((t, j, idx))
        if video:
            frames.append(obs["pixels"]["top"])
        t0 = time.perf_counter()
        obs, r, terminated, truncated, _ = env.step(a)
        timing["sim_s"] += time.perf_counter() - t0
        rewards.append(float(r))
        executed.append(a)
        t += 1
        cur = snap(obs, t)
        if terminated:
            success = r == aloha_env.MAX_REWARD
            break

    if video:
        frames.append(obs["pixels"]["top"])
    t_end = t
    exec_js = sorted(j for j in decisions if sched.T(j) < t_end)
    return EpisodeResult(
        seed=seed, success=bool(success), t_end=t_end, max_reward=float(max(rewards)), ret=float(sum(rewards)),
        decisions=decisions, executed_decisions=exec_js, timing=timing,
        trace=trace if record else None, frames=frames if video else None,
        executed_actions=np.asarray(executed) if record else None,
    )


def noise_transitions(res: EpisodeResult, gamma: float, n_exec: int):
    """Chunk-level (latent-MDP) transitions, DSRL reward convention.

    reward -1 per executed chunk; the chunk executing when the task succeeds gets 0 and mask 0;
    discount gamma ** n_exec per chunk (DSRL: discount ** query_freq).  A decision whose chunk never started
    executing (episode ended while it was in flight) is dropped: its noise never affected the environment.
    """
    js = res["executed_decisions"]
    dec = res["decisions"]
    out = []
    for i, j in enumerate(js):
        last = i == len(js) - 1
        if last and res["success"]:
            r, mask = 0.0, 0.0
        else:
            r, mask = -1.0, 1.0
        nxt = dec[j + 1]["rec"]["obs"] if (j + 1) in dec else dec[j]["rec"]["obs"]
        out.append(dict(obs=dec[j]["rec"]["obs"], action=dec[j]["rec"]["action"], reward=r, mask=mask,
                        discount=gamma ** n_exec, next_obs=nxt))
    return out
