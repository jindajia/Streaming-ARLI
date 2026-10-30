"""CPU tests: executor timing (A, B), RTC reference/mask derivation, alignment, terminal handling, duration-aware
GAE (vs brute-force Eq. 6), two-hot projection, masks / categorical distribution, PPO update sanity.

    JAX_PLATFORMS=cpu PYTHONPATH=<wt>/src:<wt>/experiments/scheduling/src:src python tests/test_dehp_cpu.py
"""
import numpy as np

from dehp import smdp
from dehp.ppo import ATOMS, Learner, duration_gae, two_hot, value_from_logits

G_ = 0.999


class FakeTask:
    @staticmethod
    def get_qvel(physics):
        return np.zeros(14)


class FakePhysics:
    class data:
        qpos = np.zeros(23)
        qvel = np.zeros(22)


class FakeInner:
    physics = FakePhysics()
    task = FakeTask()


class FakeEnv:
    """agent_pos[0] = t; executed actions are recorded with their tick."""

    def __init__(self, success_at=None):
        self.success_at, self._env, self.executed = success_at, FakeInner(), []

    def _obs(self):
        return {"pixels": {"top": np.full((480, 640, 3), self.t % 256, np.uint8)}, "agent_pos": np.full(14, float(self.t))}

    def reset(self, seed=None):
        self.t, self.executed = 0, []
        return self._obs(), {}

    def step(self, a):
        self.executed.append((self.t, np.asarray(a).copy()))
        self.t += 1
        s = self.success_at is not None and self.t >= self.success_at
        return self._obs(), 4.0 if s else 0.0, s, False, {}


class FakeSampler:
    """actions[i] = [call_id, i, obs_t, ...]; x0[i] = [call_id, i, ...]; records RTC reference / weights."""

    def __init__(self):
        self.calls = []

    def sample(self, obs_pi0, n, noise, ref=None, w=None):
        cid = len(self.calls)
        obs_t = float(np.asarray(obs_pi0["state"])[0])
        self.calls.append(dict(obs_t=obs_t, ref=None if ref is None else np.asarray(ref), w=None if w is None else np.asarray(w)))
        a = np.zeros((50, 14), np.float32)
        a[:, 0], a[:, 1], a[:, 2] = cid, np.arange(50), obs_t
        x0 = np.zeros((50, 32), np.float32)
        x0[:, 0], x0[:, 1] = cid, np.arange(50)
        return x0, a, list(range(n))


def run(EnvCls, hs, success_at=None):
    env, sp = FakeEnv(success_at), FakeSampler()
    e = EnvCls(env, sp, G_)
    s = e.reset(0, np.random.default_rng(0))
    recs, states = [], [s]
    for h in hs:
        s, G, D, term, info = e.step(h)
        recs.append((G, D, term, info))
        states.append(s)
        if term:
            break
    return e, env, sp, recs, states


# ---- 1. Run A: frozen clock, new[0:h], replan at T+h -------------------------------------------------------------
e, env, sp, recs, _ = run(smdp.SyncHorizonEnv, [1, 50, 7, 25])
assert [c["obs_t"] for c in sp.calls] == [0, 1, 51, 58, 83] and all(c["ref"] is None for c in sp.calls)
ex = env.executed
assert len(ex) == 83 and [int(a[1]) for _, a in ex[:1]] == [0]
assert [int(a[1]) for _, a in ex[1:51]] == list(range(50)) and all(int(a[0]) == 1 for _, a in ex[1:51])
assert [int(a[1]) for _, a in ex[51:58]] == list(range(7))
assert all(int(a[2]) == t - int(a[1]) for t, a in ex), "action index i <-> time obs_t + i"
assert e.valid_mask().sum() == 50 and e.valid_mask()[0] and e.valid_mask()[49]
print("[1] run A: synchronous replans at T+h on o_T, executes new[0:h], no RTC, h in 1..50")

# ---- 2. Run B baseline h = 25: launches/handoffs/RTC ----------------------------------------------------------------
e, env, sp, recs, states = run(smdp.AsyncHorizonEnv, [25] * 5)
assert [c["obs_t"] for c in sp.calls] == [0, 15, 40, 65, 90, 115]
assert [r["launch_t"] for r in e.requests] == [15, 40, 65, 90, 115]
assert [rec[3]["handoff_t"] for rec in recs] == [25, 50, 75, 100, 125]
assert [r["shift"] for r in e.requests] == [15, 25, 25, 25, 25] and all(r["committed"] == 10 for r in e.requests)
assert [r["overlap_end"] for r in e.requests] == [35, 25, 25, 25, 25]
assert all(r["ready_t"] == r["launch_t"] + 10 for r in e.requests)
ex = env.executed
assert [int(a[1]) for _, a in ex[:25]] == list(range(25))  # startup chunk: positions 0..24
assert [int(a[1]) for _, a in ex[25:50]] == list(range(10, 35)) and all(int(a[0]) == 1 for _, a in ex[25:50])
assert all(int(a[2]) == t - int(a[1]) for t, a in ex), "index alignment: action i of a chunk observed at s is for s+i"
c1 = sp.calls[1]
assert np.array_equal(c1["ref"][:35, 1], np.arange(15, 50)) and np.all(c1["ref"][:35, 0] == 0)
assert np.allclose(c1["w"][:10], 1) and np.all(c1["w"][35:] == 0)
c2 = sp.calls[2]
assert np.array_equal(c2["ref"][:25, 1], np.arange(25, 50)) and np.all(c2["ref"][:25, 0] == 1) and np.all(c2["w"][25:] == 0)
assert all(s_["e0"] == 10 and np.isclose(s_["meta"][2], 10 / 50) for s_ in states[1:] if s_ is not None) and states[0]["e0"] == 0
assert e.stats["underruns"] == 0 and e.stats["async_violations"] == 0
print("[2] run B baseline: launch 15/40/65, handoff 25/50/75, obs at launch, ready == handoff, new[10:35] executed, "
      "RTC shift 15 then 25, committed 10, overlap_end 35 then 25")

# ---- 3. Run B variable h: alignment, shifts derived from the actual plan origin, no underrun -------------------------
hs = [10, 40, 17, 33, 10, 40, 25]
e, env, sp, recs, states = run(smdp.AsyncHorizonEnv, hs)
T, origin = 0, 0
for h, rec, req in zip(hs, recs, e.requests):
    assert req["launch_t"] == T + h - 10 and req["shift"] == req["launch_t"] - origin and req["overlap_end"] == 50 - req["shift"]
    assert rec[3]["handoff_t"] == T + h and rec[1] == h
    origin, T = req["launch_t"], T + h
assert all(int(a[2]) == t - int(a[1]) for t, a in env.executed) and len(env.executed) == sum(hs)
assert e.stats["underruns"] == 0
assert e.valid_mask().sum() == 31 and not e.valid_mask()[8] and e.valid_mask()[9] and e.valid_mask()[39] and not e.valid_mask()[40]
for bad in (9, 41):
    try:
        run(smdp.AsyncHorizonEnv, [bad])
        raise SystemExit("infeasible h accepted")
    except AssertionError:
        pass
print("[3] run B variable h in 10..40: shift = launch - plan origin, overlap_end = 50 - shift, no underrun; "
      "h outside 10..40 rejected")

# ---- 4. termination inside an interval: actual duration, G, terminal, deadline ------------------------------------
for Cls in (smdp.SyncHorizonEnv, smdp.AsyncHorizonEnv):
    e, env, sp, recs, _ = run(Cls, [25, 25, 25], success_at=37)
    G, D, term, info = recs[-1]
    assert term and info["success"] and D == 37 - 25 and np.isclose(G, G_ ** (D - 1)) and info["t_end"] == 37
    assert recs[0][0] == 0.0 and not recs[0][2]
    e, env, sp, recs, _ = run(Cls, [25] * 20)
    G, D, term, info = recs[-1]
    assert term and info["deadline"] and not info["success"] and info["t_end"] == 400 and G == 0.0
    assert sum(r[1] for r in recs) == 400
print("[4] success mid-interval -> D = actual steps, G = gamma^(D-1), terminal; deadline at 400 terminal, G = 0")

# ---- 5. duration-aware GAE vs brute-force Eq. 6 ----------------------------------------------------------------------
rng = np.random.default_rng(0)
n = 9
G = rng.random(n) * (rng.random(n) < 0.3)
D = rng.integers(10, 41, n)
term = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0], bool)  # two episodes end, the last decision is a non-terminal cut
v, v_next = rng.random(n), rng.random(n)
lam = 0.99
adv = duration_gae(G, D, term, v, v_next, G_, lam)
delta = G + np.where(term, 0, G_ ** D * v_next) - v
for k in range(n):
    ref, el = 0.0, 0
    for j in range(k, n):
        ref += (G_ * lam) ** el * delta[j]
        if term[j]:
            break
        el += D[j]
    assert np.isclose(adv[k], ref), (k, adv[k], ref)
assert np.isclose(adv[-1], delta[-1])  # cut: bootstrap V(s') inside delta, nothing beyond
print("[5] duration-aware GAE == brute-force sum_l (gamma*lambda)^(t_{k+l}-t_k) delta_{k+l}; zero bootstrap at "
      "terminal, V(s') at the buffer cut")

# ---- 6. two-hot / categorical / masks / PPO update --------------------------------------------------------------------
import jax.numpy as jnp  # noqa: E402

for x in (0.0, 0.3337, 0.5, 0.999, 1.0):
    p = np.asarray(two_hot(jnp.asarray([x])))[0]
    assert np.isclose(p.sum(), 1) and np.isclose((p * np.asarray(ATOMS)).sum(), x, atol=1e-6) and (p > 0).sum() <= 2
assert np.isclose(float(value_from_logits(jnp.zeros((1, 201)))[0]), 0.5)
for Cls, with_plan in ((smdp.SyncHorizonEnv, False), (smdp.AsyncHorizonEnv, True)):
    e, env, sp, recs, states = run(Cls, [25] * 3)
    ain = smdp.STATE_DIM + 700 + 50 + 3
    L = Learner(0, ain, ain if with_plan else smdp.STATE_DIM, with_plan)
    s0 = states[0]
    _, _, probs = L.act(s0)
    valid = s0["hmask"]
    assert np.allclose(probs[valid], 1 / valid.sum(), atol=1e-6) and np.all(probs[~valid] < 1e-12)
    for _ in range(300):
        h, _, _ = L.act(s0)
        assert valid[h - 1]
    S = {k: np.stack([s[k] for s in states[:3]]) for k in ("phys", "plan", "plan_mask", "meta", "hmask")}
    logs = L.update(S, np.array([24, 24, 9]) if not with_plan else np.array([24, 24, 9]), np.log(np.full(3, 1 / valid.sum(), np.float32)),
                    np.array([1.0, -1.0, 0.5], np.float32), np.array([0.2, 0.7, 1.0], np.float32), 4)
    assert L.finite() and all(np.isfinite(list(r.values())).all() for r in logs)
    assert logs[0]["approx_kl"] == 0.0 and logs[0]["clipfrac"] == 0.0
print("[6] two-hot mean == target; initial actor exactly uniform over valid h (0 elsewhere); samples always valid; "
      "PPO update finite, first-epoch KL 0")
print("DEHP CPU TESTS OK")
