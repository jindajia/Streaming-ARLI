"""SCH-01 CPU checks (fake simulator + fake sampler; real masks, executor, learner, replay).

    JAX_PLATFORMS=cpu PYTHONPATH=<repo>/src:src python tests/test_core_cpu.py
"""
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from sch01 import core, rtc_vendored
from sch01.executor import Executor, reference_policy
from sch01.learner import Learner
from sch01.replay import Replay
from sch01.sampler import rtc_weights, shift_reference

# ---- 1. timing model ----------------------------------------------------------------------------------
assert [core.BUDGET[n] for n in core.N_VALUES] == [4, 5, 5, 6, 6, 6, 7, 7]
assert core.estimated_latency_ms(10) == 124.2 and core.NUM_ACTIONS == 201
rng = np.random.default_rng(0)
for _ in range(2000):
    n = int(rng.integers(3, 11))
    j = core.sample_jitter_ms(rng)
    assert -10 <= j <= 10 and core.actual_delay_steps(n, j) <= core.BUDGET[n]
print("[1] timing: budgets", [core.BUDGET[n] for n in core.N_VALUES], "; delay <= budget for 2000 jitter draws")

# ---- 2. masks and reference rule --------------------------------------------------------------------------
m = core.feasible_mask(25, 300)
assert m[core.KEEP] and m[core.encode(25, 10)] and not m[core.encode(6, 10)] and m[core.encode(4, 3)]
m4 = core.feasible_mask(4, 300)
assert not m4[core.KEEP] and m4.sum() == 1 and m4[core.encode(4, 3)]
assert core.reference_action(25, m) == core.KEEP and core.reference_action(8, core.feasible_mask(8, 300)) == core.KEEP
assert core.reference_action(7, core.feasible_mask(7, 300)) == core.encode(7, 10)
assert core.reference_action(6, core.feasible_mask(6, 300)) == core.encode(6, 8)
assert core.reference_action(5, core.feasible_mask(5, 300)) == core.encode(5, 5)
for q in range(4, 26):
    mq = core.feasible_mask(q, 300)
    for a in np.flatnonzero(mq):
        kind, r, n = core.decode(int(a))
        if kind == "REPLAN":
            assert core.BUDGET[n] <= r <= q and r + core.K_CAP <= core.H
print("[2] masks: KEEP masked when q-1 < 4; REPLAN needs budget(N) <= R <= q; reference rule q>7 KEEP, q=7 REPLAN(7,10)")

# ---- 3. executor event timeline with a fake simulator / sampler -------------------------------------------
class FakeEnv:
    def reset(self, seed=None):
        self.t, self.cmds = 0, []
        return self._o(), {}

    def _o(self):
        return {"pixels": {"top": np.full((480, 640, 3), self.t % 255, np.uint8)}, "agent_pos": np.full(14, float(self.t))}

    def step(self, a):
        self.cmds.append(np.asarray(a).copy())
        self.t += 1
        return self._o(), 0.0, False, False, {}


class FakeSampler:
    """action[i] = [call_id, i, 0...]; x0 normalised = action-coded; records every call's RTC inputs."""

    def __init__(self):
        self.calls = []

    def sample(self, obs_pi0, n, noise, y=None, w=None):
        cid = len(self.calls)
        self.calls.append(dict(n=n, y=None if y is None else y.copy(), w=None if w is None else w.copy(),
                               obs_t=float(obs_pi0["state"][0])))
        act = np.zeros((core.H, 14), np.float32)
        act[:, 0], act[:, 1] = cid, np.arange(core.H)
        x0 = np.zeros((core.H, 32), np.float32)
        x0[:, 0], x0[:, 1] = cid, np.arange(core.H)
        return x0, act, [1 - k / n for k in range(n)]


env, fs = FakeEnv(), FakeSampler()
exe = Executor(env, fs)
res = exe.run(0, reference_policy, np.random.default_rng(1), np.random.default_rng(2), record=True)
reqs = res["requests"]
assert [q["launch_t"] for q in reqs[:4]] == [18, 43, 68, 93], [q["launch_t"] for q in reqs[:4]]
assert all(q["r"] == 7 and q["n"] == 10 for q in reqs[:-1])
assert [q["shift"] for q in reqs[:3]] == [18, 25, 25] and all(q["committed"] == 7 for q in reqs)
assert [q["overlap_end"] for q in reqs[:2]] == [32, 25]
assert all(q["ready_t"] <= q["handoff_t"] for q in reqs)
cmds = np.asarray(env.cmds)
assert cmds.shape[0] == 400
assert np.array_equal(cmds[:25, 1], np.arange(25)) and np.all(cmds[:25, 0] == 0)  # startup chunk new[0:25]
assert np.array_equal(cmds[25:50, 1], np.arange(7, 32)) and np.all(cmds[25:50, 0] == 1)  # first replan new[7:32]
assert np.all(cmds[50:75, 0] == 2) and np.array_equal(cmds[50:75, 1], np.arange(7, 32))
for c in fs.calls[1:4]:  # RTC reference = previous prediction shifted by its observation-origin difference
    assert c["y"] is not None and np.allclose(c["w"][:7], 1.0, atol=1e-6)
assert fs.calls[1]["obs_t"] == 18 and fs.calls[2]["obs_t"] == 43
assert np.array_equal(fs.calls[1]["y"][:32, 1], np.arange(18, 50)) and np.all(fs.calls[1]["y"][:32, 0] == 0)
assert np.array_equal(fs.calls[2]["y"][:25, 1], np.arange(25, 50)) and np.all(fs.calls[2]["y"][:25, 0] == 1)
assert res["underrun_ticks"] == 0
print("[3] executor: launches 18,43,68,93; handoffs every 25; installs new[7:32]; RTC shift 18/25, committed 7, "
      "overlap end 32/25; no use before ready; 400 commands, no underrun")

# ---- 4. duration-aware returns ------------------------------------------------------------------------------
g = 0.999
dec = res["decisions"]
assert sum(d["D"] for d in dec) == 400
for d in dec:
    k, r, n = core.decode(d["action"])
    if k == "REPLAN" and not d["terminal"]:
        assert d["D"] == r
    ref = sum(g ** i * (-0.1 / 400) for i in range(d["D"]))
    assert abs(d["G"] - ref) < 1e-9
assert dec[-1]["terminal"] and not any(d["terminal"] for d in dec[:-1])
print(f"[4] returns: {len(dec)} decisions; D = R for REPLAN, 1 for KEEP; G matches the tick-level sum; timeout terminal")

# ---- 5. RTC weights / reference shift / zero-weight guidance ----------------------------------------------------
w = rtc_weights(7, 25)
assert np.allclose(w[:7], 1, atol=1e-6) and np.all(np.diff(w[7:25]) < 0) and np.all(w[25:] == 0)
prev = np.arange(50 * 32, dtype=np.float32).reshape(50, 32)
y = shift_reference(prev, 18)
assert np.array_equal(y[:32], prev[18:]) and not y[32:].any()
f = lambda x: jnp.tanh(x) * 0.7 + 0.1  # noqa: E731
x = jnp.asarray(np.random.default_rng(3).normal(size=(1, 50, 32)), jnp.float32)
v0 = f(x)
vg = rtc_vendored.guided_velocity(f, x, jnp.float32(0.6), jnp.zeros((1, 50, 32)), jnp.zeros(50), 10.0)
assert np.allclose(np.asarray(v0), np.asarray(vg))
print("[5] RTC: W = 1 on committed prefix, decreasing soft overlap, 0 after; y = prev shifted; zero weights = plain")

# ---- 6. prior / deterministic parity ----------------------------------------------------------------------------
cfg = dict(actor_lr=1e-4, critic_lr=3e-4, alpha_lr=3e-4, alpha_init=0.01, gamma=g, tau=0.005, p0=0.999,
           target_entropy_ratio=0.1)
L = Learner(0, cfg)
for d in dec[:60]:
    st, mk = d["state"], d["mask"]
    ref = core.reference_action(st["q"], mk)
    p, resid = L.policy_probs(st, mk, ref, 1.0)
    assert np.all(resid == 0) and int(np.argmax(p)) == ref and abs(p[ref] - 0.999) < 1e-9 or mk.sum() == 1
    p0, _ = L.policy_probs(st, mk, ref, 0.0)
    assert np.allclose(p0[mk], 1.0 / mk.sum())  # prior removed + zero head -> uniform over feasible actions
assert core.prior_strength(0) == 1 and core.prior_strength(5000) == 1 and core.prior_strength(27500) == 0.5
assert core.prior_strength(50000) == 0
print("[6] initial policy: zero residual head; argmax = reference; P(reference) = 0.999 at strength 1; uniform at 0")

# ---- 7. learner updates (critic first), replay + resume -----------------------------------------------------------
Rp = Replay(5000, 5000, 5000, 0)
assert len(Rp) == 0
Rp.add_episode(dec)
for s_ in range(3):
    env2 = FakeEnv()
    r2 = Executor(env2, FakeSampler()).run(s_, reference_policy, np.random.default_rng(s_), np.random.default_rng(9))
    Rp.add_episode(r2["decisions"])
h = lambda t: float(sum(np.abs(np.asarray(l)).sum() for l in jax.tree_util.tree_leaves(t)))  # noqa: E731
a0, c0 = h(L.A.params), h(L.C.params)
info = L.update(Rp.sample(64), 1.0, actor_on=False)
assert h(L.A.params) == a0 and h(L.C.params) != c0
info = L.update(Rp.sample(64), 1.0, actor_on=True)
assert h(L.A.params) != a0
assert np.isfinite([float(v) for v in info.values()]).all()
with tempfile.TemporaryDirectory() as tmp:
    meta = Rp.save_new(tmp)
    blob = L.state_bytes()
    R2 = Replay(5000, 5000, 5000, 7)
    R2.load(tmp, meta)
    L2 = Learner(5, cfg)
    L2.load_bytes(blob)
    b1, b2 = Rp.sample(32), R2.sample(32)
    for k in b1:
        assert np.array_equal(np.asarray(b1[k]), np.asarray(b2[k])), k
    i1, i2 = L.update(b1, 0.7, True), L2.update(b2, 0.7, True)
    assert float(i1["critic_loss"]) == float(i2["critic_loss"]) and h(L.A.params) == h(L2.A.params)
print("[7] learner: critic burn-in leaves actor unchanged; actor updates after; replay+learner resume exact")
print("SCH-01 CPU CHECKS OK")
