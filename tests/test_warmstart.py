"""CPU checks 1-6 for the warm-start final-step experiment (fake simulator + fake frozen sampler; real networks).

    JAX_PLATFORMS=cpu PYTHONPATH=src python tests/test_warmstart.py
"""
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from arli import rl
from arli.rollout import noise_transitions, run_episode
from arli.schedule import Schedule
from arli.single_actor import C_SCALE, DecisionReplay
from arli.warmstart_final import FINAL_K, RES_INIT_STD, Controller, ResidualLearner, transitions_B, tree_hash

S, G = Schedule(), 0.999


class FakeEnv:  # agent_pos[0] = t, so recorded states reveal their capture time
    def __init__(self, success_at=None):
        self.success_at = success_at

    def _obs(self):
        return {"pixels": {"top": np.full((480, 640, 3), self.t % 256, np.uint8)}, "agent_pos": np.full(14, float(self.t))}

    def reset(self, seed=None):
        self.t = 0
        return self._obs(), {}

    def step(self, a):
        self.t += 1
        s = self.success_at is not None and self.t >= self.success_at
        return self._obs(), 4.0 if s else 0.0, s, False, {}


class FakePi0:  # deterministic, non-trivial dynamics; records the X entering every iteration
    def __init__(self):
        self.taus = [jnp.float32(1.0 - 0.1 * k) for k in range(10)]
        self.seen = []

    def begin(self, inp):
        return None

    def step(self, st, x, k):
        self.seen.append((k, np.asarray(x).copy()))
        x = jnp.asarray(x)
        return x - 0.1 * (0.5 * x + 0.01 * k), x

    def finish(self, st, x):
        return np.asarray(x)[0, :, :14]

    def infer(self, inp, noise):
        raise AssertionError("both branches must use the step-wise sampler")


agent = rl.make_agent("arli", 0)
R = ResidualLearner(0)


def run(branch, key, zero=False, success_at=None, record=True):
    pi0 = FakePi0()
    ctrl = Controller(branch, agent, S, key, residual=R if branch == "B" else None,
                      res_key=jax.random.fold_in(key, 77) if branch == "B" else None, zero_correction=zero)
    res = run_episode(FakeEnv(success_at), pi0, S, 5, ctrl, record=record)
    return res, ctrl, pi0


# 1. zero-mean, small-noise initialisation (zero mean != zero sample)
rng = np.random.default_rng(0)
zs = []
res, ctrl, _ = run("B", jax.random.PRNGKey(1))
for j in res["executed_decisions"]:
    (s,) = res["decisions"][j]["steer"]
    zs.append(s["obs"])
dist = R.A.apply_fn({"params": R.A.params}, {k: np.stack([z[k] for z in zs]) for k in zs[0]})
mean = np.asarray(dist.distribution._loc)
std = np.asarray(dist.distribution._scale_diag)
acts = np.concatenate([res["decisions"][j]["steer"][0]["action"] for j in res["executed_decisions"]])
assert np.all(mean == 0.0) and np.allclose(std, RES_INIT_STD)
assert np.all(acts != 0.0) and np.abs(acts).std() < 2 * RES_INIT_STD
print(f"[1] residual init: mean exactly 0, std {RES_INIT_STD}; sampled |a| mean {np.abs(acts).mean():.3f} (non-zero); "
      f"correction scale {C_SCALE} -> |dX| ~ {C_SCALE * np.abs(acts).mean():.3f}/dim")

# 2. zero-correction B == A with matched initial-noise randomness and env seed
for success_at in (None, 150):
    key = jax.random.PRNGKey(11)
    rA, cA, pA = run("A", key, success_at=success_at)
    rB, cB, pB = run("B", key, zero=True, success_at=success_at)
    assert np.array_equal(rA["executed_actions"], rB["executed_actions"])
    wA = [rA["decisions"][j]["rec"]["action"] for j in sorted(rA["decisions"])]
    wB = [rB["decisions"][j]["rec"]["action"] for j in sorted(rB["decisions"])]
    assert all(np.array_equal(a, b) for a, b in zip(wA, wB)) and len(wA) == len(wB)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(pA.seen, pB.seen))
print("[2] zero-correction branch B reproduces branch A exactly (noise, every X_k, executed actions)")

# 3. residual updates change only the residual actor; ARLI actor bit-identical
h_arli = tree_hash(agent._actor.params)
h_res = tree_hash(R.A.params)
RB = DecisionReplay(5000, 0)
for s in range(3):
    r, _, _ = run("B", jax.random.PRNGKey(20 + s))
    RB.add_episode(transitions_B(r, S, G))
for _ in range(5):
    info = R.update(RB.sample(64))
assert tree_hash(R.A.params) != h_res and float(info["actor_param_step"]) > 0
assert tree_hash(agent._actor.params) == h_arli
print("[3] residual actor params changed; frozen ARLI actor hash unchanged")

# 4. k=9 call sees the actual k=9 frame and X_9 (no stale substitution, no future frame)
res, ctrl, pi0 = run("B", jax.random.PRNGKey(3))
x_entering = {}
it = iter(pi0.seen)
for j in sorted(res["decisions"]):
    for k in range(10):
        x_entering[(j, k)] = next(it)[1]
for j in res["executed_decisions"]:
    (s,) = res["decisions"][j]["steer"]
    t_expect = 0 if j == 0 else S.m(j) + FINAL_K
    assert s["k"] == FINAL_K and s["t_now"] == t_expect
    assert float(s["obs"]["state"][14, 0]) == t_expect  # newest frame captured exactly at the call
    assert float(s["obs"]["state"][0, 0]) == S.s(j)  # old frame = inference-start observation
    assert t_expect <= max(0, S.T(j) - 1)  # before the chunk executes: no future information
    assert np.allclose(s["obs"]["x_noisy"], s["x_in"]) and np.allclose(s["x_in"], C_SCALE * 0 + s["x_in"])
    # X_9 handed to the actor == the trajectory produced by 9 frozen iterations (before the correction)
    xk9_corrected = x_entering[(j, 9)][0]
    assert np.allclose(xk9_corrected - s["x_in"], C_SCALE * np.repeat(s["action"], 50, axis=0), atol=1e-6)
print("[4] k=9 actor input = frame o_{m_j+9} and actual X_9; correction applied before the 10th iteration only")

# 5. rewards, masks, bootstrap states, discount interval
for success_at in (None, 150):
    res, ctrl, _ = run("B", jax.random.PRNGKey(4), success_at=success_at)
    tr = transitions_B(res, S, G)
    js = res["executed_decisions"]
    assert [t["j"] for t in tr] == js
    for t in tr:
        last = t["j"] == js[-1]
        assert t["reward"] == (0.0 if last and res["success"] else -1.0)
        assert t["mask"] == (0.0 if last and res["success"] else 1.0)
        assert np.isclose(t["discount"], G ** 25)
        if t["mask"] > 0:
            t_now = float(t["obs"]["state"][14, 0])
            t_next = float(t["next_obs"]["state"][14, 0])
            assert t_next == S.m(t["j"] + 1) + FINAL_K  # successor = next chunk's k=9 call
            if t["j"] > 0:
                assert t_next - t_now == 25  # physical interval between consecutive decisions
    trA = noise_transitions(run("A", jax.random.PRNGKey(4), success_at=success_at)[0], G, 25)
    assert all(np.isclose(t["discount"], G ** 25) for t in trA)
    print(f"[5] {'success' if success_at else 'truncated'}: {len(tr)} k=9 decisions; rewards/masks/successors OK; "
          f"discount gamma^25 = interval between consecutive k=9 calls")

# 6. checkpoint/resume: residual learner + replay + RNG restore exactly; next update identical
with tempfile.TemporaryDirectory() as tmp:
    blob = R.state_bytes()
    meta = RB.save_new(f"{tmp}/replay")
    R2 = ResidualLearner(9)
    R2.load_bytes(blob)
    RB2 = DecisionReplay(5000, 123)
    RB2.load(f"{tmp}/replay", meta)
    assert tree_hash(R2.A.params) == tree_hash(R.A.params) and tree_hash(R2.C.opt_state) == tree_hash(R.C.opt_state)
    b1, b2 = RB.sample(32), RB2.sample(32)
    assert np.array_equal(b1["observations"]["pixels"], b2["observations"]["pixels"])
    i1, i2 = R.update(b1), R2.update(b2)
    assert float(i1["critic_loss"]) == float(i2["critic_loss"])
    assert tree_hash(R.A.params) == tree_hash(R2.A.params)
    a_blob = rl.agent_state_bytes(agent)
    ag2 = rl.make_agent("arli", 5)
    rl.load_agent_state(ag2, a_blob)
    assert tree_hash(ag2._actor.params) == h_arli and tree_hash(ag2._critic.opt_state) == tree_hash(agent._critic.opt_state)
print("[6] resume: residual actor/critic/optimizers/rng + replay rows/sampler rng + ARLI agent (incl. optimizer) exact; "
      "post-resume update identical")
print("WARMSTART CHECKS OK")
