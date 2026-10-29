"""SCH-01 v2 executor (fixed_delay10_v2): one robot, one inference worker, at most one request in flight, RTC ON.

Clock: ``t`` counts env.step calls (50 Hz).  At tick t (before env.step t) the newest frame is o_t.
Chunk bookkeeping: chunk 1 = synchronous startup chunk (clock frozen, N = 10, no RTC -- no previous chunk to align
to); each handoff installs the next chunk.  p = actions already executed from the current installed chunk,
q = selected old actions remaining (q = 25 - p in normal operation).

REPLAN(R, N) at tick t: observation o_t; N full-interval denoising steps with RTC; logically ready at t + d(N) (shared
table ``core.DELAY``, no jitter); R old actions execute; at t + R the aligned suffix new[R : R + 25] is installed.
Feasibility d(N) <= R <= q and R + 25 <= H.  A ready result stays hidden until the handoff.  RTC reference = previous
full normalised prediction shifted by (t - its observation tick); committed full-weight prefix = R; soft overlap until
H - shift.  No decisions while a request is in flight.  KEEP executes one action.

Fixed baseline (core.reference_action): KEEP while q > 10, REPLAN(10, 10) at q = 10 (p = 15), handoff 10 ticks later:
25 actions per chunk, observation taken 10 ticks before each handoff.

Reward per tick: 1[first success] - 0.1/400 - (0.5/400) * underrun.  Decisions accumulate G over their actual
duration D (gamma per tick); success and timeout are terminal.
"""

from __future__ import annotations

import time

import numpy as np

from arli import aloha_env
from arli.pi0_policy import obs_to_pi0_input
from sch01.sampler import rtc_weights, shift_reference
from sch02 import core

MAX_TICKS = 400
SUCCESS_BONUS, TIME_PEN, UNDERRUN_PEN = 1.0, 0.1 / MAX_TICKS, 0.5 / MAX_TICKS


def runtime_features(t, q, plan_age, plan_n):
    return np.asarray([q / core.K_CAP, plan_age / core.H, plan_n / core.N_MAX, (MAX_TICKS - t) / MAX_TICKS, 1.0],
                      np.float32)


class Executor:
    def __init__(self, env, sampler, gamma=0.999):
        self.env, self.sampler, self.gamma = env, sampler, gamma

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
                    q=q, t=ex["t"], p=ex["p"], chunk=ex["chunk"])

    def _launch(self, ex, r, n, noise, rtc_on=True):
        prev = ex["prev"]
        shift = ex["t"] - prev["origin"]
        y = shift_reference(prev["x0"], shift)
        w = rtc_weights(r, core.H - shift)
        t0 = time.perf_counter()
        x0, actions, taus = self.sampler.sample(obs_to_pi0_input(ex["raw"]), n, noise, y if rtc_on else None,
                                                w if rtc_on else None)
        wall = time.perf_counter() - t0
        d = core.delay_steps(n)
        assert d <= r <= len(ex["queue"]) and r + core.K_CAP <= core.H, (d, r, len(ex["queue"]))
        return dict(launch_t=ex["t"], obs_t=ex["t"], r=r, n=n, ready_t=ex["t"] + d, handoff_t=ex["t"] + r, d=d,
                    x0=x0, actions=actions, origin=ex["t"], img=ex["img"], qpos=ex["qpos"], shift=shift, committed=r,
                    overlap_end=core.H - shift, n_taus=len(taus), wall_s=wall, modeled_ms=core.modeled_inference_ms(n),
                    p_at_launch=ex["p"], q_at_launch=len(ex["queue"]), chunk_at_launch=ex["chunk"],
                    old_executed=ex["p"] + r, discarded=len(ex["queue"]) - r, obs_age_at_handoff=r)

    def run(self, seed, policy, noise_rng, record=False, snapshot=None, forced_first=None, continuation=None,
            max_ticks=MAX_TICKS, on_tick=None):
        env = self.env
        if snapshot is None:
            obs, _ = env.reset(seed=seed)
            ex = dict(t=0, raw=obs, img=aloha_env.rl_image(obs), qpos=aloha_env.qpos(obs), queue=[], last_a=None,
                      inflight=None, success=False, n_calls=1, p=0, chunk=1)
            noise = noise_rng.standard_normal((core.H, 32)).astype(np.float32)
            t0 = time.perf_counter()
            x0, actions, _ = self.sampler.sample(obs_to_pi0_input(obs), core.REF_N, noise)  # startup: no RTC
            ex["prev"] = dict(x0=x0, actions=actions, origin=0, n=core.REF_N, img=ex["img"], qpos=ex["qpos"])
            ex["queue"] = [a for a in actions[0:core.K_CAP]]
            startup = dict(n=core.REF_N, rtc=False, wall_s=time.perf_counter() - t0)
        else:
            ex = restore_snapshot(env, snapshot)  # caller supplies the paired noise stream
            startup = None
        decisions, requests, rewards, ticks = [], [], [], []
        open_dec, underrun_ticks, n_decisions = None, 0, 0
        while ex["t"] < max_ticks:
            t = ex["t"]
            inf = ex["inflight"]
            if inf is not None and t == inf["handoff_t"]:
                assert t >= inf["ready_t"], "result used before it is ready"
                ex["queue"] = [a for a in inf["actions"][inf["r"]:inf["r"] + core.K_CAP]]
                ex["prev"] = dict(x0=inf["x0"], actions=inf["actions"], origin=inf["origin"], n=inf["n"],
                                  img=inf["img"], qpos=inf["qpos"])
                ex.update(inflight=None, p=0, chunk=ex["chunk"] + 1)
                inf = None
            if inf is None:
                if on_tick is not None:
                    on_tick(ex, env)
                st = self._state(ex)
                mask = core.feasible_mask(st["q"], max_ticks - t)
                if open_dec is not None:
                    open_dec.update(next_state=st, next_mask=mask, terminal=False)
                    decisions.append(open_dec)
                if not mask.any():  # unreachable under the masks
                    a, info = core.encode(core.MIN_DELAY, core.N_MIN), dict(kind="fallback")
                elif forced_first is not None and n_decisions == 0:
                    a, info = forced_first, dict(kind="forced")
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
                    ex["inflight"] = self._launch(ex, r, n, noise)
                    requests.append({k: v for k, v in ex["inflight"].items() if k not in ("x0", "actions", "img", "qpos")}
                                    | dict(decision_kind=info.get("kind")))
            if ex["queue"]:
                a_cmd, u = ex["queue"].pop(0), 0
            else:
                a_cmd, u = ex["last_a"], 1
                underrun_ticks += 1
            ex["last_a"] = a_cmd
            obs, r_env, terminated, _, _ = env.step(a_cmd)
            succ = bool(terminated and r_env == aloha_env.MAX_REWARD)
            rho = SUCCESS_BONUS * succ - TIME_PEN - UNDERRUN_PEN * u
            rewards.append(rho)
            if record:
                ticks.append((t, ex["chunk"], ex["p"], len(ex["queue"]), int(ex["inflight"] is not None), u))
            open_dec["G"] += (self.gamma ** open_dec["D"]) * rho
            open_dec["D"] += 1
            ex.update(t=t + 1, raw=obs, img=aloha_env.rl_image(obs), qpos=aloha_env.qpos(obs), p=ex["p"] + 1 - u)
            if succ:
                ex["success"] = True
                break
        st = self._state(ex)
        open_dec.update(next_state=st, next_mask=core.feasible_mask(st["q"], max(1, max_ticks - ex["t"])), terminal=True)
        decisions.append(open_dec)
        if ex["inflight"] is not None:
            requests[-1]["abandoned"] = True
        return dict(seed=seed, success=ex["success"], t_end=ex["t"], ret=float(sum(rewards)),
                    discounted_return=float(sum(self.gamma ** k * r for k, r in enumerate(rewards))),
                    decisions=decisions, requests=requests, underrun_ticks=underrun_ticks, startup=startup,
                    n_calls=ex["n_calls"], ticks=ticks if record else None,
                    modeled_ms=float(sum(q["modeled_ms"] for q in requests)),
                    wall_infer_s=float(sum(q["wall_s"] for q in requests)))


def take_snapshot(ex, env, extra=None):
    """Complete simulator + executor state at an idle tick (any p), including the old full prediction, queue,
    observation origin and execution progress."""
    assert ex["inflight"] is None
    physics = env._env.physics
    snap = dict(physics_state=physics.get_state().copy(), ctrl=physics.data.ctrl.copy(), t=ex["t"],
                raw=_copy_obs(ex["raw"]), img=ex["img"].copy(), qpos=ex["qpos"].copy(),
                queue=[a.copy() for a in ex["queue"]], last_a=None if ex["last_a"] is None else ex["last_a"].copy(),
                prev={k: (v.copy() if hasattr(v, "copy") else v) for k, v in ex["prev"].items()},
                n_calls=ex["n_calls"], p=ex["p"], chunk=ex["chunk"])
    if extra:
        snap.update(extra)
    return snap


def restore_snapshot(env, snap):
    env.reset(seed=snap["env_seed"])
    physics = env._env.physics
    with physics.reset_context():
        physics.set_state(snap["physics_state"])
    physics.data.ctrl[:] = snap["ctrl"]
    physics.forward()
    return dict(t=snap["t"], raw=_copy_obs(snap["raw"]), img=snap["img"].copy(), qpos=snap["qpos"].copy(),
                queue=[a.copy() for a in snap["queue"]], last_a=None if snap["last_a"] is None else snap["last_a"].copy(),
                inflight=None, success=False, n_calls=snap["n_calls"], p=snap["p"], chunk=snap["chunk"],
                prev={k: (v.copy() if hasattr(v, "copy") else v) for k, v in snap["prev"].items()})


def _copy_obs(o):
    return {"pixels": {"top": o["pixels"]["top"].copy()}, "agent_pos": np.asarray(o["agent_pos"]).copy()}


def reference_policy(st, mask, t):
    a, kind = core.reference_action(st["q"], mask)
    return a, dict(kind=kind, reference=True)
