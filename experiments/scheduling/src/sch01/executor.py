"""SCH-01 executor: one robot, one inference worker, at most one request in flight, RTC ON.

Clock: ``t`` counts env.step calls (50 Hz, 20 ms).  At tick t (before env.step number t) the newest frame is o_t.

Startup (fixed, outside the scheduler): the robot is at rest after reset; the first chunk is computed synchronously
with the clock frozen (N = 10, no RTC: nothing to align to) and new[0:25] is installed.

Scheduler decisions happen only when no request is in flight ("unlocked" ticks):
* KEEP: execute the next queued action; decide again next tick.  Duration D = 1.
* REPLAN(R, N): launch a request at tick t on o_t (observation origin t).  The old queue keeps executing for R ticks.
  The result is computed immediately (implementation convenience) but stays inaccessible until its handoff: it is
  logically ready at t + d_actual (d_actual = ceil((L(N) + jitter)/20) <= budget(N) <= R) and is installed atomically
  at the handoff tick t + R as new[R : R + 25].  Decisions are locked until the handoff.  D = R (or shorter if the
  episode ends first).
* RTC alignment for a request launched at t: previous full normalised prediction (origin t_prev) is shifted by
  shift = t - t_prev: y[i] = prev[i + shift].  Committed (full-weight) prefix = R (the old actions that will execute
  before the handoff, also when the result is ready earlier and held), soft overlap until H - shift, zero after.
* Underrun (queue empty at an unlocked tick): unreachable under the feasibility masks; if it happened the last
  command would be held, u_k = 1 logged, and the emergency REPLAN(budget(3), 3) launched.

Reward per tick k (DECISIONS S5):  rho_k = 1[first success at k] - (0.1/400) - (0.5/400) * underrun_k  (GPU cost 0).
Each decision j accumulates G_j = sum_{k<D_j} gamma^k rho_{t_j+k} over its actual duration; timeout and success are
terminal (finite-horizon episode; remaining time is in the state).
"""

from __future__ import annotations

import time

import numpy as np

from arli import aloha_env
from arli.pi0_policy import obs_to_pi0_input
from sch01 import core
from sch01.sampler import rtc_weights, shift_reference

MAX_TICKS = 400
SUCCESS_BONUS, TIME_PEN, UNDERRUN_PEN = 1.0, 0.1 / MAX_TICKS, 0.5 / MAX_TICKS


def runtime_features(t, q, plan_age, plan_n):
    return np.asarray([q / core.K_CAP, plan_age / core.H, plan_n / core.N_MAX, (MAX_TICKS - t) / MAX_TICKS,
                       1.0], np.float32)


class Executor:
    """Runs one episode.  ``policy(state, mask, t) -> (action, info)``; ``record`` keeps per-decision traces."""

    def __init__(self, env, sampler, gamma=0.999):
        self.env, self.sampler, self.gamma = env, sampler, gamma

    # -- state for the scheduler (no privileged simulator state) ------------------------------------------------
    def _state(self, ex):
        q = len(ex["queue"])
        queue = np.zeros((core.K_CAP, 14), np.float32)
        qmask = np.zeros(core.K_CAP, np.float32)
        if q:
            queue[:q] = np.asarray(ex["queue"])
            qmask[:q] = 1
        prev = ex["prev"]
        return dict(img_now=ex["img"], img_src=prev["img"], queue=queue, qmask=qmask,
                    robot=np.concatenate([ex["qpos"], prev["qpos"]]).astype(np.float32),
                    runtime=runtime_features(ex["t"], q, ex["t"] - prev["origin"], prev["n"]),
                    q=q, t=ex["t"])

    def _launch(self, ex, r, n, noise, jitter_ms, rtc_on=True):
        prev = ex["prev"]
        shift = ex["t"] - prev["origin"]
        y = shift_reference(prev["x0"], shift)
        w = rtc_weights(r, core.H - shift)
        t0 = time.perf_counter()
        x0, actions, taus = self.sampler.sample(obs_to_pi0_input(ex["raw"]), n, noise, y if rtc_on else None,
                                                w if rtc_on else None)
        wall = time.perf_counter() - t0
        d = core.actual_delay_steps(n, jitter_ms)
        assert d <= core.BUDGET[n] <= r, (d, n, r)
        return dict(launch_t=ex["t"], r=r, n=n, handoff_t=ex["t"] + r, ready_t=ex["t"] + d, d_actual=d,
                    jitter_ms=jitter_ms, x0=x0, actions=actions, origin=ex["t"], img=ex["img"], qpos=ex["qpos"],
                    shift=shift, committed=r, overlap_end=core.H - shift, taus=taus, wall_s=wall,
                    est_ms=core.estimated_latency_ms(n) + jitter_ms)

    def run(self, seed, policy, noise_rng, jitter_rng, record=False, snapshot=None, forced_first=None,
            continuation=None, max_ticks=MAX_TICKS, on_snapshot=None):
        """Returns dict(success, t_end, ret, decisions, requests, ...).

        snapshot: dict from ``take_snapshot`` -> restore complete simulator + executor state and start there.
        forced_first: first action at the (restored) state, then ``continuation`` policy for the rest (diagnostics).
        on_snapshot(ex, env) is called at every handoff tick (idle, q = 25) for diagnostic snapshot capture.
        """
        env = self.env
        if snapshot is None:
            obs, _ = env.reset(seed=seed)
            ex = dict(t=0, raw=obs, img=aloha_env.rl_image(obs), qpos=aloha_env.qpos(obs), queue=[], last_a=None,
                      inflight=None, success=False, n_calls=0)
            noise = noise_rng.standard_normal((core.H, 32)).astype(np.float32)
            ex["n_calls"] += 1
            x0, actions, _ = self.sampler.sample(obs_to_pi0_input(obs), core.REF_N, noise)  # synchronous startup
            ex["prev"] = dict(x0=x0, actions=actions, origin=0, n=core.REF_N, img=ex["img"], qpos=ex["qpos"])
            ex["queue"] = [a for a in actions[0:core.K_CAP]]
            startup = dict(n=core.REF_N, rtc=False, est_ms=core.estimated_latency_ms(core.REF_N))
        else:
            ex = restore_snapshot(env, snapshot)  # caller supplies the branch's (paired) noise / jitter streams
            startup = None
        decisions, requests, rewards, ticks = [], [], [], []
        open_dec = None
        underrun_ticks = 0
        n_decisions = 0
        while ex["t"] < max_ticks:
            t = ex["t"]
            inf = ex["inflight"]
            if inf is not None and t == inf["handoff_t"]:
                assert t >= inf["ready_t"], "result used before it is ready"
                ex["queue"] = [a for a in inf["actions"][inf["r"]:inf["r"] + core.K_CAP]]  # expired new[0:R] dropped
                ex["prev"] = dict(x0=inf["x0"], actions=inf["actions"], origin=inf["origin"], n=inf["n"],
                                  img=inf["img"], qpos=inf["qpos"])
                inf["installed"] = (inf["r"], inf["r"] + core.K_CAP)
                ex["inflight"] = inf = None
                if on_snapshot is not None:
                    on_snapshot(ex, env)
            if inf is None:  # unlocked decision tick
                st = self._state(ex)
                mask = core.feasible_mask(st["q"], max_ticks - t)
                if open_dec is not None:
                    open_dec.update(next_state=st, next_mask=mask, terminal=False)
                    decisions.append(open_dec)
                if not mask.any():  # unreachable under the masks; emergency path
                    a, info = core.encode(core.BUDGET[core.N_MIN], core.N_MIN), dict(fallback=True)
                elif forced_first is not None and n_decisions == 0:
                    a, info = forced_first, dict(forced=True)
                    assert mask[a], "forced first action infeasible"
                else:
                    pol = continuation if (forced_first is not None and continuation is not None) else policy
                    a, info = pol(st, mask, t)
                    assert mask[a], "policy chose an infeasible action"
                n_decisions += 1
                kind, r, n = core.decode(a)
                open_dec = dict(state=st, mask=mask, action=a, t=t, G=0.0, D=0, info=info)
                if kind == "REPLAN":
                    noise = noise_rng.standard_normal((core.H, 32)).astype(np.float32)
                    ex["n_calls"] += 1
                    jit = core.sample_jitter_ms(jitter_rng)
                    ex["inflight"] = self._launch(ex, r, n, noise, jit)
                    requests.append({k: v for k, v in ex["inflight"].items() if k not in ("x0", "actions", "img", "qpos")}
                                    | dict(q_at_launch=st["q"]))
            # ---- execute one control tick ----
            if ex["queue"]:
                a_cmd = ex["queue"].pop(0)
                u = 0
            else:
                a_cmd, u = ex["last_a"], 1
                underrun_ticks += 1
            ex["last_a"] = a_cmd
            obs, r_env, terminated, _, _ = env.step(a_cmd)
            succ = bool(terminated and r_env == aloha_env.MAX_REWARD)
            rho = SUCCESS_BONUS * succ - TIME_PEN - UNDERRUN_PEN * u
            rewards.append(rho)
            if record:
                ticks.append((t, len(ex["queue"]), int(ex["inflight"] is not None), u))
            open_dec["G"] += (self.gamma ** open_dec["D"]) * rho
            open_dec["D"] += 1
            ex.update(t=t + 1, raw=obs, img=aloha_env.rl_image(obs), qpos=aloha_env.qpos(obs))
            if succ:
                ex["success"] = True
                break
        st = self._state(ex)
        open_dec.update(next_state=st, next_mask=core.feasible_mask(st["q"], max(1, max_ticks - ex["t"])), terminal=True)
        decisions.append(open_dec)
        if ex["inflight"] is not None:  # abandoned by termination: its compute still counts
            requests[-1]["abandoned"] = True
        return dict(seed=seed, success=ex["success"], t_end=ex["t"], ret=float(sum(rewards)),
                    discounted_return=float(sum(self.gamma ** k * r for k, r in enumerate(rewards))),
                    decisions=decisions, requests=requests, underrun_ticks=underrun_ticks, startup=startup,
                    n_calls=ex["n_calls"], ticks=ticks if record else None,
                    infer_cost_ms=float(sum(q["est_ms"] for q in requests)))


# ---------------------------------------------------------------------------------------------------------------
def take_snapshot(ex, env, extra=None):
    """Complete simulator + executor state at an idle tick (no request in flight)."""
    assert ex["inflight"] is None
    physics = env._env.physics
    snap = dict(physics_state=physics.get_state().copy(), ctrl=physics.data.ctrl.copy(),
                t=ex["t"], raw=_copy_obs(ex["raw"]), img=ex["img"].copy(), qpos=ex["qpos"].copy(),
                queue=[a.copy() for a in ex["queue"]], last_a=None if ex["last_a"] is None else ex["last_a"].copy(),
                prev={k: (v.copy() if hasattr(v, "copy") else v) for k, v in ex["prev"].items()},
                n_calls=ex["n_calls"])
    if extra:
        snap.update(extra)
    return snap


def restore_snapshot(env, snap):
    from gym_aloha.tasks import sim as aloha_sim
    env.reset(seed=snap["env_seed"])  # sets BOX_POSE and task RNG, then overwrite the full physics state
    physics = env._env.physics
    with physics.reset_context():
        physics.set_state(snap["physics_state"])
    physics.data.ctrl[:] = snap["ctrl"]
    physics.forward()
    del aloha_sim
    return dict(t=snap["t"], raw=_copy_obs(snap["raw"]), img=snap["img"].copy(), qpos=snap["qpos"].copy(),
                queue=[a.copy() for a in snap["queue"]], last_a=None if snap["last_a"] is None else snap["last_a"].copy(),
                inflight=None, success=False, n_calls=snap["n_calls"],
                prev={k: (v.copy() if hasattr(v, "copy") else v) for k, v in snap["prev"].items()})


def _copy_obs(o):
    return {"pixels": {"top": o["pixels"]["top"].copy()}, "agent_pos": np.asarray(o["agent_pos"]).copy()}


def reference_policy(st, mask, t):
    return core.reference_action(st["q"], mask), dict(reference=True)
