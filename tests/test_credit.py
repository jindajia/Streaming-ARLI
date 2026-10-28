"""CPU test of S1 'chunk' credit assignment (DECISIONS D18).  JAX_PLATFORMS=cpu python tests/test_credit.py"""
import numpy as np

from arli import rl
from arli.schedule import Schedule
from arli.steering import steering_transitions

S, g = Schedule(), 0.999
rng = np.random.default_rng(0)


def obs(tag):
    o = {k: np.full(s.shape, tag, s.dtype) for k, s in rl.observation_space("arli").spaces.items()}
    return o


def z(tag):
    return dict(obs(tag), x_noisy=np.full((50, 32), tag, np.float32), td=np.zeros(2, np.float32))


def fake(success, n_exec_chunks, steps):
    dec = {}
    for j in range(n_exec_chunks + 1):  # one extra decision made but never executed
        st = [dict(k=k, t_now=(S.m(j) + k if j else 0), obs=z(100 * j + k), action=np.ones((1, 32)))
              for k in sorted(steps)]
        dec[j] = dict(rec=dict(obs=obs(100 * j), action=np.zeros((1, 32))), steer=st)
    return dict(executed_decisions=list(range(n_exec_chunks)), decisions=dec, success=success)


for br, steps in (("C", range(1, 10)), ("D", [9])):
    for success in (True, False):
        W, U = steering_transitions(fake(success, 4, steps), S, g)
        assert len(W) == 4 and len(U) == 4 * len(steps)
        # one physical reward per chunk; U shares the chunk's reward/terminal, never adds extra reward
        assert [t["reward"] for t in W] == ([-1, -1, -1, 0] if success else [-1] * 4)
        assert W[-1]["mask"] == (0.0 if success else 1.0)
        assert all(np.isclose(t["discount"], g ** 25) for t in W)
        for t in U:
            j = int(t["obs"]["state"][0, 0]) // 100
            k = int(t["obs"]["state"][0, 0]) % 100
            assert t["reward"] == W[j]["reward"] and t["mask"] == W[j]["mask"]
            assert np.isclose(t["discount"], g ** (25 - (k if j else 0)))  # time left to the next noise decision
            assert t["next_is_w"] == 1.0
            assert t["next_obs"]["state"][0, 0] == (100 * (j + 1)), "bootstraps from s_{j+1}"
            assert not np.any(t["next_obs"]["x_noisy"])
        print(br, "success" if success else "fail", "ok:", len(W), "W,", len(U), "U")

# learner update with chunk credit runs and never bootstraps Q_u from Q_u
import jax  # noqa: E402
from arli.steering import SteeringLearner, SteerSpec, z_space  # noqa: E402
ag = rl.make_agent("arli", 0)
L = SteeringLearner(ag, SteerSpec("C"), 0, credit="chunk")
RW = rl.Replay(rl.observation_space("arli"), (1, 32), 300, 0, next_space=z_space())
RU = rl.Replay(z_space(), (1, 32), 400, 1, extra=("next_is_w",))
W, U = steering_transitions(fake(False, 4, range(1, 10)), S, g)
for _ in range(10):
    for t in W:
        RW.insert(t["obs"], t["action"], t["reward"], t["mask"], t["discount"], t["next_obs"])
    for t in U:
        RU.insert(t["obs"], t["action"], t["reward"], t["mask"], t["discount"], t["next_obs"], next_is_w=t["next_is_w"])
info = L.update(RW.sample(64), RU.sample(64))
print("update ok", {k: round(float(v), 3) for k, v in info.items() if k.startswith(("q_", "target"))})
print("CREDIT TEST OK")
