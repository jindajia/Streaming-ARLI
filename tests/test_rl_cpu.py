"""CPU smoke test of RL pieces (no simulator / pi0).  JAX_PLATFORMS=cpu python tests/test_rl_cpu.py"""
import numpy as np
import jax
from flax.core import frozen_dict

from arli import rl
from arli.steering import SteeringLearner, SteerSpec, z_space, count_params


def rand_obs(space, rng):
    return {k: (rng.integers(0, 255, s.shape).astype(np.uint8) if s.dtype == np.uint8 else
                rng.standard_normal(s.shape).astype(np.float32)) for k, s in space.spaces.items()}


rng = np.random.default_rng(0)
for method in rl.METHODS:
    ag = rl.make_agent(method, 0)
    R = rl.Replay(rl.observation_space(method), (1, 32), 300, 0)
    for _ in range(300):
        o, n = rand_obs(R.obs_space, rng), rand_obs(R.obs_space, rng)
        R.insert(o, rng.standard_normal(32), -1.0, 1.0, 0.999 ** 25, n)
    info = ag.update(R.sample(256))
    w = rl.sample_noise_action(ag, rand_obs(R.obs_space, rng), jax.random.PRNGKey(1))
    print(method, "actor params", count_params(ag._actor.params), "critic", count_params(ag._critic.params),
          "q", float(info["q"]), "w", w.shape, float(np.abs(w).max()))
    blob = rl.agent_state_bytes(ag); rl.load_agent_state(ag, blob)

arli = rl.make_agent("arli", 0)
for br in "BCD":
    L = SteeringLearner(arli, SteerSpec(br), 0)
    print(br, L.n_params)
    RW = rl.Replay(rl.observation_space("arli"), (1, 32), 300, 0, next_space=z_space())
    RU = rl.Replay(z_space(), (1, 32), 300, 1, extra=("next_is_w",))
    for _ in range(300):
        RW.insert(rand_obs(RW.obs_space, rng), rng.standard_normal(32), 0., 1., .99, rand_obs(z_space(), rng))
        RU.insert(rand_obs(z_space(), rng), rng.standard_normal(32), -1., 1., .99, rand_obs(z_space(), rng),
                  next_is_w=float(rng.random() < .2))
    # warm start check: steering actor/critic, zero new rows -> critic Q_u(z, u) independent of u at init
    z = {k: v[None] for k, v in rand_obs(z_space(), rng).items()}
    q1 = L.Cu.apply_fn({"params": L.Cu.params}, z, np.zeros((1, 32), np.float32))
    q2 = L.Cu.apply_fn({"params": L.Cu.params}, z, np.ones((1, 32), np.float32))
    zw = {k: z[k] for k in ("pixels", "prev_action", "state")}
    qw = L.Cw.apply_fn({"params": L.Cw.params}, zw, np.zeros((1, 1, 32), np.float32))
    print("  Q_u(u=0) == Q_u(u=1):", bool(np.allclose(q1, q2)), " Q_u(z,0) == Q_w(s,0):", bool(np.allclose(q1, qw, atol=1e-5)))
    info = L.update(RW.sample(256), RU.sample(256))
    print("  update ok", {k: round(float(v), 3) for k, v in info.items()})
    blob = __import__("flax").serialization.to_bytes(L.state())
    L.load(__import__("flax").serialization.from_bytes(L.state(), blob))
print("RL CPU TEST OK")
