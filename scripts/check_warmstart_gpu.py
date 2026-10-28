"""GPU check 2 on the real stack: zero-correction branch B == branch A (real sim, real pi0, seed-0 600k checkpoint)."""
import numpy as np

from arli.pi0_policy import configure_caches

configure_caches()
import jax  # noqa: E402

from arli import aloha_env, rl  # noqa: E402
from arli.pi0_policy import FrozenPi0  # noqa: E402
from arli.rollout import run_episode  # noqa: E402
from arli.schedule import Schedule  # noqa: E402
from arli.warmstart_final import Controller, ResidualLearner, resolve_checkpoint, tree_hash  # noqa: E402

prov, blob = resolve_checkpoint("/N/scratch/jindjia/arli/runs", "r2_arli_seed0")
print("provenance", {k: v for k, v in prov.items()})
agent = rl.make_agent("arli", 0)
rl.load_agent_state(agent, blob)
h0 = tree_hash(agent._actor.params)
S, env, pi0 = Schedule(), aloha_env.make_env(), FrozenPi0(10)
R = ResidualLearner(0)
for ep in range(3):
    key = jax.random.PRNGKey(456 + ep)
    rA = run_episode(env, pi0, S, 10_000_000 + ep, Controller("A", agent, S, key), record=True)
    cB = Controller("B", agent, S, key, residual=R, res_key=jax.random.fold_in(key, 77), zero_correction=True)
    rB = run_episode(env, pi0, S, 10_000_000 + ep, cB, record=True)
    same = rA["executed_actions"].shape == rB["executed_actions"].shape and np.array_equal(rA["executed_actions"], rB["executed_actions"])
    print(f"episode {ep}: A success={rA['success']} t_end={rA['t_end']} | zero-B success={rB['success']} t_end={rB['t_end']} "
          f"| executed actions bit-identical: {same} | residual calls {sum(c[0] == 'residual' for c in cB.calls)}")
    assert same
assert tree_hash(agent._actor.params) == h0
print("GPU ZERO-CORRECTION CHECK OK")
