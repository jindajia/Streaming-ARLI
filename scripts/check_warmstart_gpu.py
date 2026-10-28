"""GPU check 2 on the real stack: zero-correction branch B == branch A (real sim, real pi0, seed-0 600k checkpoint),\nwith observations held fixed: A's rendered frames are replayed to B while the physical state matches exactly."""
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


class Recorder:
    """Wraps the env and records every rendered frame + qpos."""
    def __init__(self, env):
        self.env, self.frames, self.qpos = env, [], []
    def reset(self, seed=None):
        o, i = self.env.reset(seed=seed); self.frames, self.qpos = [o["pixels"]["top"].copy()], [o["agent_pos"].copy()]; return o, i
    def step(self, a):
        o, r, te, tr, i = self.env.step(a); self.frames.append(o["pixels"]["top"].copy()); self.qpos.append(o["agent_pos"].copy()); return o, r, te, tr, i


class Replayer:
    """Replays recorded frames whenever the physical state matches the recording exactly (the renderer is not
    bit-deterministic: identical states render with +-1 intensity differences, see scripts/diag_determinism.py)."""
    def __init__(self, env, frames, qpos):
        self.env, self.frames, self.qpos, self.t, self.mismatch = env, frames, qpos, 0, []
    def _sub(self, o):
        if self.t < len(self.qpos) and np.array_equal(o["agent_pos"], self.qpos[self.t]):
            o["pixels"]["top"] = self.frames[self.t].copy()
        else:
            self.mismatch.append(self.t)
        return o
    def reset(self, seed=None):
        o, i = self.env.reset(seed=seed); self.t = 0; return self._sub(o), i
    def step(self, a):
        o, r, te, tr, i = self.env.step(a); self.t += 1; return self._sub(o), r, te, tr, i


for ep in range(3):
    key = jax.random.PRNGKey(456 + ep)
    rec = Recorder(env)
    rA = run_episode(rec, pi0, S, 10_000_000 + ep, Controller("A", agent, S, key), record=True)
    rep = Replayer(env, rec.frames, rec.qpos)
    cB = Controller("B", agent, S, key, residual=R, res_key=jax.random.fold_in(key, 77), zero_correction=True)
    rB = run_episode(rep, pi0, S, 10_000_000 + ep, cB, record=True)
    same = rA["executed_actions"].shape == rB["executed_actions"].shape and np.array_equal(rA["executed_actions"], rB["executed_actions"])
    print(f"episode {ep}: A success={rA['success']} t_end={rA['t_end']} | zero-B (A's frames replayed) success={rB['success']} "
          f"t_end={rB['t_end']} | executed actions bit-identical: {same} | state mismatches: {len(rep.mismatch)} | "
          f"residual calls {sum(c[0] == 'residual' for c in cB.calls)}")
    assert same and not rep.mismatch
assert tree_hash(agent._actor.params) == h0
print("GPU ZERO-CORRECTION CHECK OK")
