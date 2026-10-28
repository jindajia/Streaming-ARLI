"""Focused CPU checks for the single-actor methods (fake simulator + fake frozen sampler, real actor/critic/replay).

    JAX_PLATFORMS=cpu PYTHONPATH=src python tests/test_single_actor.py
"""
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from arli import rl
from arli.rollout import run_episode
from arli.schedule import Schedule
from arli.single_actor import (C_SCALE, W0_SCALE, Controller, DecisionReplay, SingleActorLearner, call_steps,
                               transitions)

S, G = Schedule(), 0.999


class FakeEnv:
    """Robot state encodes time: agent_pos[0] = t, so recorded observations reveal their capture time."""

    def __init__(self, success_at=None):
        self.success_at = success_at

    def _obs(self):
        img = np.full((480, 640, 3), self.t % 256, np.uint8)
        return {"pixels": {"top": img}, "agent_pos": np.full(14, float(self.t))}

    def reset(self, seed=None):
        self.t = 0
        return self._obs(), {}

    def step(self, a):
        self.t += 1
        succ = self.success_at is not None and self.t >= self.success_at
        return self._obs(), 4.0 if succ else 0.0, succ, False, {}


class FakePi0:
    def __init__(self):
        self.taus = [jnp.float32(1.0 - 0.1 * k) for k in range(10)]
        self.n_begin = self.n_step = 0

    def begin(self, inp):
        self.n_begin += 1
        return None

    def step(self, st, x, k):
        self.n_step += 1
        v = 0.5 * jnp.asarray(x)
        return jnp.asarray(x) - 0.1 * v, v

    def finish(self, st, x):
        return np.asarray(x)[0, :, :14]

    def infer(self, inp, noise):
        raise AssertionError("single-actor methods must use the step-wise sampler")


def run(method, L, mode="train", success_at=None, seed=0):
    ctrl = Controller(method, L, S, jax.random.PRNGKey(seed), np.random.default_rng(seed), mode=mode)
    pi0 = FakePi0()
    res = run_episode(FakeEnv(success_at), pi0, S, 123, ctrl, record=True)
    return res, ctrl, pi0


L = SingleActorLearner(0)
print("actor/critic params:", L.n_params)

# 1. call counts: exactly 10 vs 1 per chunk, at the right iterations, and pi0 stays frozen/stepwise
for method, want in (("shared_every_step", list(range(10))), ("final_step_only", [9])):
    res, ctrl, pi0 = run(method, L)
    by_chunk = {}
    for j, k, t_obs, t_now in ctrl.calls:
        by_chunk.setdefault(j, []).append(k)
    for j in res["executed_decisions"]:
        assert by_chunk[j] == want, (method, j, by_chunk[j])
    assert call_steps(method) == tuple(want)
    assert pi0.n_step == 10 * pi0.n_begin  # 10 frozen expert iterations per inference
    print(f"[1] {method}: {len(want)} call(s)/chunk at k={want}; {pi0.n_begin} inferences x 10 iterations")

    # 4. causal timestamps: call k of chunk j>=1 at t = m_j + k sees o_t (fake state encodes t); never later
    for j, k, t_obs, t_now in ctrl.calls:
        want_t = 0 if j == 0 else S.m(j) + k
        assert t_now == want_t and t_obs == t_now, (j, k, t_obs, t_now)
    for j in res["executed_decisions"]:
        for (k, z, a, t_now) in __import__("arli.single_actor", fromlist=["x"]).decision_list(res["decisions"][j]):
            q_lat = float(z["state"][14, 0])  # first coordinate of the latest-frame state = capture time
            q_old = float(z["state"][0, 0])
            assert q_lat == t_now and q_old == S.s(j) and q_lat <= max(0, S.T(j) - 1), (j, k, q_old, q_lat)
            assert z["td"][2] == (1.0 if k == 0 else 0.0)
            if k == 0:
                assert not np.any(z["x_noisy"]), "k=0 placeholder must be zeros"
    print(f"[4] {method}: observation timestamps causal (old = s_j, latest = m_j + k <= T_j - 1)")

# 2. parameter sharing: every call uses the same params object; one output head
res, ctrl, _ = run("shared_every_step", L)
leaves = jax.tree_util.tree_leaves(L.A.params)
assert len([k for k in L.A.params["network"] if k.startswith("Dense")]) == 2  # mean + log-std heads only
print("[2] one actor TrainState; heads:", sorted(L.A.params["network"]))

# 3. fresh initialisation, independent of any ARLI checkpoint: two seeds differ; init does not read files
L1 = SingleActorLearner(1)
d = sum(float(jnp.abs(a - b).sum()) for a, b in zip(jax.tree_util.tree_leaves(L.A.params),
                                                    jax.tree_util.tree_leaves(L1.A.params)))
assert d > 0
assert float(L.T.apply_fn({"params": L.T.params})) == 1.0
print("[3] fresh init (temperature 1.0, seeds differ; no checkpoint path in the constructor)")

# 5. transitions: rewards once per chunk, discounts, terminal handling, successor = first call of chunk j+1
for method in ("shared_every_step", "final_step_only"):
    for success_at in (None, 140):
        res, ctrl, _ = run(method, L, success_at=success_at)
        tr = transitions(res, S, G)
        js = res["executed_decisions"]
        per = len(call_steps(method))
        assert len(tr) == per * len(js)
        for t in tr:
            last = t["j"] == js[-1]
            assert t["reward"] == (0.0 if (last and res["success"]) else -1.0)
            assert t["mask"] == (0.0 if (last and res["success"]) else 1.0)
            el = (S.m(t["j"]) + t["k"] - S.m(t["j"])) if t["j"] > 0 else 0
            assert np.isclose(t["discount"], G ** (25 - el))
            if t["mask"] > 0:
                nxt_k = 0 if method == "shared_every_step" else 9
                assert t["next_obs"]["td"][2] == (1.0 if nxt_k == 0 else 0.0)
                tn = float(t["next_obs"]["state"][14, 0])
                assert tn == (S.m(t["j"] + 1) + nxt_k), (t["j"], tn)
        # reward counted once per chunk: sum over chunk-level rewards equals -(#non-terminal executed chunks)
        assert sum(t["reward"] for t in tr if t["k"] == call_steps(method)[0]) == -(len(js) - int(res["success"]))
        print(f"[5] {method} {'success' if success_at else 'truncated'}: {len(js)} chunks, {len(tr)} decisions OK")

# 6. updates reach the shared actor; critic target sane on this data
R = DecisionReplay(20000, 0)
for s in range(3):
    res, _, _ = run("shared_every_step", L, mode="prefill", seed=s)
    R.add_episode(transitions(res, S, G))
p0 = jax.tree_util.tree_map(lambda x: x, L.A.params)
for _ in range(3):
    info = L.update(R.sample(64))
moved = sum(float(jnp.abs(a - b).sum()) for a, b in zip(jax.tree_util.tree_leaves(p0), jax.tree_util.tree_leaves(L.A.params)))
assert moved > 0 and float(info["actor_param_step"]) > 0
print("[6] actor params moved by updates; info:", {k: round(float(v), 4) for k, v in info.items()})

# 7. checkpoint/resume preserves learner + replay + RNG exactly
with tempfile.TemporaryDirectory() as tmp:
    blob = L.state_bytes()
    meta = R.save_new(f"{tmp}/replay")
    R2 = DecisionReplay(20000, 99)
    R2.load(f"{tmp}/replay", meta)
    Lr = SingleActorLearner(5)
    Lr.load_bytes(blob)
    for a, b in zip(jax.tree_util.tree_leaves(L.A.params), jax.tree_util.tree_leaves(Lr.A.params)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    for a, b in zip(jax.tree_util.tree_leaves(L.C.opt_state), jax.tree_util.tree_leaves(Lr.C.opt_state)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert np.array_equal(np.asarray(L.rng), np.asarray(Lr.rng))
    b1, b2 = R.sample(32), R2.sample(32)
    for k in ("actions", "rewards", "discount"):
        assert np.array_equal(b1[k], b2[k])
    assert np.array_equal(b1["observations"]["pixels"], b2["observations"]["pixels"])
    i1, i2 = L.update(b1), Lr.update(b1)
    assert np.isclose(float(i1["critic_loss"]), float(i2["critic_loss"]))
    # orphan part written after the checkpoint is dropped on resume
    res, _, _ = run("shared_every_step", L, mode="prefill", seed=7)
    R.add_episode(transitions(res, S, G))
    R.save_new(f"{tmp}/replay")
    R3 = DecisionReplay(20000, 0)
    R3.load(f"{tmp}/replay", meta)
    assert R3.size == meta["size"] and len(os.listdir(f"{tmp}/replay")) == 1
print("[7] checkpoint/resume: learner (params, opt states, rng) + replay (rows, sampler rng) identical; orphan part dropped")
print("SINGLE-ACTOR CHECKS OK", "W0_SCALE", W0_SCALE, "C_SCALE", C_SCALE)
