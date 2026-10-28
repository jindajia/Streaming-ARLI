"""Where do two runs with identical keys/seeds first diverge on the real stack?  A vs A, A vs zero-B."""
import numpy as np

from arli.pi0_policy import configure_caches, obs_to_pi0_input

configure_caches()
import jax  # noqa: E402

from arli import aloha_env, rl  # noqa: E402
from arli.pi0_policy import FrozenPi0  # noqa: E402
from arli.rollout import run_episode  # noqa: E402
from arli.schedule import Schedule  # noqa: E402
from arli.warmstart_final import Controller, ResidualLearner, resolve_checkpoint  # noqa: E402

prov, blob = resolve_checkpoint("/N/scratch/jindjia/arli/runs", "r2_arli_seed0")
agent = rl.make_agent("arli", 0)
rl.load_agent_state(agent, blob)
S, pi0, R = Schedule(), FrozenPi0(10), ResidualLearner(0)

# (a) simulator/renderer determinism: same seed, same fixed action sequence, two env instances and one instance twice
acts = np.random.default_rng(0).normal(0, 0.05, (60, 14)) + aloha_env.qpos(aloha_env.make_env().reset(seed=3)[0])
def roll(env):
    o, _ = env.reset(seed=3); ims, qs = [o["pixels"]["top"]], [o["agent_pos"]]
    for a in acts:
        o = env.step(a)[0]; ims.append(o["pixels"]["top"]); qs.append(o["agent_pos"])
    return np.stack(ims), np.stack(qs)
e1, e2 = aloha_env.make_env(), aloha_env.make_env()
i1, q1 = roll(e1); i1b, q1b = roll(e1); i2, q2 = roll(e2)
print("sim same-instance: qpos equal", np.array_equal(q1, q1b), "images equal", np.array_equal(i1, i1b))
print("sim two instances: qpos equal", np.array_equal(q1, q2), "images equal", np.array_equal(i1, i2),
      "max |dimg|", int(np.abs(i1.astype(int) - i2.astype(int)).max()))

# (b) pi0 determinism: same input + noise twice
o, _ = e1.reset(seed=3); inp = obs_to_pi0_input(o); nz = np.random.default_rng(1).standard_normal((1, 50, 32)).astype(np.float32)
a1 = pi0.sample_stepwise(inp, nz); a2 = pi0.sample_stepwise(inp, nz)
print("pi0 stepwise repeat equal:", np.array_equal(a1, a2), float(np.abs(a1 - a2).max()))
w1 = rl.sample_noise_action(agent, rl.make_rl_obs("arli", aloha_env.rl_image(o), aloha_env.qpos(o), np.zeros((20, 14)), aloha_env.rl_image(o), aloha_env.qpos(o)), jax.random.PRNGKey(0))
w2 = rl.sample_noise_action(agent, rl.make_rl_obs("arli", aloha_env.rl_image(o), aloha_env.qpos(o), np.zeros((20, 14)), aloha_env.rl_image(o), aloha_env.qpos(o)), jax.random.PRNGKey(0))
print("ARLI actor repeat equal:", np.array_equal(w1, w2))

# (c) closed loop: A vs A, A vs zero-B; first divergent step
def ep(kind, env):
    key = jax.random.PRNGKey(456)
    c = Controller("A", agent, S, key) if kind == "A" else Controller("B", agent, S, key, residual=R, res_key=jax.random.fold_in(key, 77), zero_correction=True)
    r = run_episode(env, pi0, S, 10_000_000, c, record=True)
    return r
def first_div(r1, r2):
    a, b = r1["executed_actions"], r2["executed_actions"]; n = min(len(a), len(b))
    d = [t for t in range(n) if not np.array_equal(a[t], b[t])]
    return (d[0] if d else None), len(a), len(b)
env = aloha_env.make_env()
rA1, rA2 = ep("A", env), ep("A", env)
rB1 = ep("B", env)
envB = aloha_env.make_env()
rA3 = ep("A", envB)
print("A vs A (same env instance): first divergent t, lens", first_div(rA1, rA2))
print("A vs zero-B (same instance):  first divergent t, lens", first_div(rA1, rB1))
print("A vs A (other env instance):  first divergent t, lens", first_div(rA1, rA3))
for name, r in (("A2", rA2), ("B", rB1)):
    t, _, _ = first_div(rA1, r)
    if t is not None:
        j = S.executing_chunk(t)
        wA = rA1["decisions"][j]["rec"]["action"]; wX = r["decisions"][j]["rec"]["action"]
        print(f"  {name}: divergence at t={t} (chunk {j}); that chunk's initial noise equal: {np.array_equal(wA, wX)}; "
              f"max |d action| at t: {float(np.abs(rA1['executed_actions'][t] - r['executed_actions'][t]).max()):.2e}")
