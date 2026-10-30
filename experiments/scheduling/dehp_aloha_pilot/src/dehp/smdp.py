"""DEHP-style execution-horizon decisions on ALOHA Transfer Cube as a semi-MDP (frozen pi0_aloha_sim).

Common: H = 50, 10 Euler denoising steps (sch01.sampler.RTCSampler, verified RTC port), 14-d absolute joint actions,
gym_aloha 0.1.1 transfer_cube, success = reward 4 (terminates), deadline 400 control steps (terminal).
Reward: +1 on the successful step, 0 otherwise.  A decision executes h actions; its record holds the actual duration D
(D < h only if the episode terminated inside the interval), the discounted in-interval reward G = sum_j gamma^j r_{T+j},
and terminal status.

Run A (``SyncHorizonEnv``, no simulated delay, RTC off): original DEHP order.  At decision time T the clock is frozen:
observe o_T, generate the full chunk new[0:50] from o_T, the actor picks h in {1..50}, new[0:h] executes, the rest is
discarded.  Physics never advances during inference.

Run B (``AsyncHorizonEnv``, simulated delay d = 10 control steps, RTC on): at handoff time T the next chunk is installed;
its executable positions are new[e0:50], e0 = T - origin (10 in steady state).  The actor reads o_T and the time-aligned
executable plan and picks h in {10..40}.  Then automatically: next inference launch L = T + h - 10 on o_L (actions keep
executing), result ready at L + 10 = T + h = next handoff, installed suffix new[10:50].  RTC reference = current plan's
normalised prediction shifted by (L - origin); committed full-weight prefix = 10 (the actions executing during the
request); soft exp mask until H - shift.  Exactly one request in flight; no queue underrun by construction (asserted).
Startup (both B policies): t = 0 synchronous chunk on o_0 without RTC (no previous plan), origin 0, so e0 = 0; the same
action space h in {10..40} is used and the first launch is at h - 10.
"""

from __future__ import annotations

import time

import numpy as np

from arli import aloha_env
from arli.pi0_policy import obs_to_pi0_input

H = 50
MAX_T = 400
DELAY = 10
STATE_DIM = 42


def physical_state(env, obs, t):
    """42-d current physical state (no future information, no phase labels):
    agent_pos 14 (6 joints + normalised gripper per arm) | robot joint velocities 14 (gym_aloha get_qvel, /5) |
    cube position 3 | cube quaternion 4 | cube linear+angular velocity 6 (/5) | remaining time (400 - t)/400 1."""
    phys = env._env.physics
    qvel_robot = env._env.task.get_qvel(phys)
    cube_q = np.asarray(phys.data.qpos[16:23], np.float32)
    cube_v = np.asarray(phys.data.qvel[16:22], np.float32)
    return np.concatenate([np.asarray(obs["agent_pos"], np.float32), np.asarray(qvel_robot, np.float32) / 5.0,
                           cube_q, cube_v / 5.0, [np.float32((MAX_T - t) / MAX_T)]]).astype(np.float32)


class _Base:
    h_min, h_max = 1, 50
    delayed = False

    def __init__(self, env, sampler, gamma, n_denoise=10):
        self.env, self.sampler, self.gamma, self.n = env, sampler, gamma, n_denoise
        self.stats = dict(inference_calls=0, infer_wall_s=0.0, sim_wall_s=0.0, async_violations=0, underruns=0)

    def valid_mask(self):
        m = np.zeros(H, bool)
        m[self.h_min - 1:self.h_max] = True  # index i <-> h = i + 1
        return m

    def _infer(self, obs, noise, ref=None, w=None):
        t0 = time.perf_counter()
        x0, actions, _ = self.sampler.sample(obs_to_pi0_input(obs), self.n, noise, ref, w)
        self.stats["infer_wall_s"] += time.perf_counter() - t0
        self.stats["inference_calls"] += 1
        return np.asarray(x0, np.float32), np.asarray(actions, np.float32)

    def _env_step(self, a):
        t0 = time.perf_counter()
        obs, r, term, _, _ = self.env.step(a)
        self.stats["sim_wall_s"] += time.perf_counter() - t0
        return obs, r, term

    def _decision_state(self):
        e0 = self.T - self.plan["origin"]
        exe = self.plan["actions"][e0:]
        plan = np.zeros((H, 14), np.float32)
        pmask = np.zeros(H, np.float32)
        plan[:len(exe)] = exe
        pmask[:len(exe)] = 1.0
        meta = np.asarray([e0 / H, float(self.plan["origin"] == 0 and self.T == 0),
                           (self.T - self.plan["obs_t"]) / H], np.float32)
        return dict(phys=physical_state(self.env, self.obs, self.T), plan=plan, plan_mask=pmask, meta=meta,
                    hmask=self.valid_mask(), t=self.T, e0=e0, plan_obs_t=self.plan["obs_t"])


class SyncHorizonEnv(_Base):
    """Run A."""

    h_min, h_max = 1, 50

    def reset(self, seed, noise_rng):
        self.noise_rng = noise_rng
        self.obs, _ = self.env.reset(seed=seed)
        self.T, self.done = 0, False
        self._replan()
        return self._decision_state()

    def _replan(self):
        x0, act = self._infer(self.obs, self.noise_rng.standard_normal((H, 32)).astype(np.float32))
        self.plan = dict(x0=x0, actions=act, origin=self.T, obs_t=self.T)

    def step(self, h):
        assert self.h_min <= h <= self.h_max, h
        assert not self.done
        G, D, success = 0.0, 0, False
        for j in range(h):
            t = self.T + j
            a = self.plan["actions"][t - self.plan["origin"]]
            self.obs, r, term = self._env_step(a)
            success = bool(term and r == aloha_env.MAX_REWARD)
            G += (self.gamma ** j) * float(success)
            D += 1
            if success or t + 1 >= MAX_T:
                break
        self.T += D
        terminal = success or self.T >= MAX_T
        info = dict(h=h, D=D, success=success, deadline=(not success and self.T >= MAX_T), t_start=self.T - D,
                    t_end=self.T)
        if terminal:
            self.done = True
            return None, G, D, True, info
        self._replan()  # clock frozen during inference
        return self._decision_state(), G, D, False, info


class AsyncHorizonEnv(_Base):
    """Run B (delay 10, RTC on)."""

    h_min, h_max = 10, 40
    delayed = True

    def __init__(self, env, sampler, gamma, n_denoise=10, rtc=True):
        super().__init__(env, sampler, gamma, n_denoise)
        self.rtc = rtc
        from sch01.sampler import rtc_weights, shift_reference
        self._w, self._shift = rtc_weights, shift_reference
        self.requests = []

    def reset(self, seed, noise_rng):
        self.noise_rng = noise_rng
        self.obs, _ = self.env.reset(seed=seed)
        self.T, self.done = 0, False
        self.inflight = None
        self.requests = []
        x0, act = self._infer(self.obs, noise_rng.standard_normal((H, 32)).astype(np.float32))  # startup, no RTC
        self.plan = dict(x0=x0, actions=act, origin=0, obs_t=0)
        return self._decision_state()

    def _launch(self, t):
        if self.inflight is not None:
            self.stats["async_violations"] += 1
            raise AssertionError("overlapping inference requests")
        shift = t - self.plan["origin"]
        assert 0 <= shift <= H - DELAY, shift
        ref = self._shift(self.plan["x0"], shift) if self.rtc else None
        w = self._w(DELAY, H - shift) if self.rtc else None
        x0, act = self._infer(self.obs, self.noise_rng.standard_normal((H, 32)).astype(np.float32), ref, w)
        self.inflight = dict(x0=x0, actions=act, origin=t, obs_t=t, ready_t=t + DELAY)
        self.requests.append(dict(launch_t=t, obs_t=t, ready_t=t + DELAY, shift=shift, committed=DELAY,
                                  overlap_end=H - shift, plan_origin=self.plan["origin"]))

    def step(self, h):
        assert self.h_min <= h <= self.h_max, h
        assert not self.done
        e0 = self.T - self.plan["origin"]
        assert e0 + h <= H, (e0, h)  # every executed action exists in the installed plan
        L, handoff = self.T + h - DELAY, self.T + h
        G, D, success = 0.0, 0, False
        for j in range(h):
            t = self.T + j
            if t == L:
                self._launch(t)
            idx = t - self.plan["origin"]
            if not (0 <= idx < H):
                self.stats["underruns"] += 1
                raise AssertionError(f"queue underrun at t={t} (plan origin {self.plan['origin']})")
            self.obs, r, term = self._env_step(self.plan["actions"][idx])
            success = bool(term and r == aloha_env.MAX_REWARD)
            G += (self.gamma ** j) * float(success)
            D += 1
            if success or t + 1 >= MAX_T:
                break
        self.T += D
        terminal = success or self.T >= MAX_T
        info = dict(h=h, D=D, success=success, deadline=(not success and self.T >= MAX_T), t_start=self.T - D,
                    t_end=self.T, launch_t=L, handoff_t=handoff)
        if terminal:
            self.done = True
            self.inflight = None  # an abandoned in-flight request at termination is never installed
            return None, G, D, True, info
        assert self.T == handoff and self.inflight is not None and self.inflight["ready_t"] == handoff, \
            "handoff without a ready result"
        self.plan, self.inflight = self.inflight, None  # install: executable suffix new[10:50]
        assert self.T - self.plan["origin"] == DELAY
        return self._decision_state(), G, D, False, info
