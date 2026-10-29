"""SCH-01 v2 CPU checks: delay table, masks, new fixed-baseline timeline, RTC alignment, result gating, mid-chunk
snapshot branch example (p = 5, REPLAN(7, 5)), step-0 prior parity, replay reference labels.

    JAX_PLATFORMS=cpu PYTHONPATH=<repo>/src:<repo>/experiments/scheduling/src:src python tests/test_v2_cpu.py
"""
import numpy as np

from sch01.learner import Learner
from sch02 import core
from sch02.executor import Executor, reference_policy, restore_snapshot, take_snapshot
from sch02.replay import Replay

# 1. shared table, masks, baseline rule
assert core.DELAY == {3: 6, 4: 7, 5: 7, 6: 8, 7: 9, 8: 9, 9: 10, 10: 10} and core.MIN_DELAY == 6
m = core.feasible_mask(20, 300)
assert m[core.KEEP] and m[core.encode(7, 5)] and not m[core.encode(6, 5)] and not m[core.encode(9, 10)]
assert not core.feasible_mask(6, 300)[core.KEEP] and core.feasible_mask(7, 300)[core.KEEP]
assert core.feasible_mask(6, 300).sum() == 1 and core.feasible_mask(6, 300)[core.encode(6, 3)]
for q in range(6, 26):
    for a in np.flatnonzero(core.feasible_mask(q, 300)):
        k, r, n = core.decode(int(a))
        if k == "REPLAN":
            assert core.DELAY[n] <= r <= q and r + 25 <= 50
assert core.reference_action(11, core.feasible_mask(11, 300)) == (core.KEEP, "keep")
assert core.reference_action(10, core.feasible_mask(10, 300)) == (core.encode(10, 10), "baseline_replan")
assert core.reference_action(9, core.feasible_mask(9, 300)) == (core.encode(9, 8), "recovery")
assert core.reference_action(7, core.feasible_mask(7, 300)) == (core.encode(7, 5), "recovery")
print("[1] delay table 6,7,7,8,9,9,10,10; KEEP iff q-1 >= 6; baseline q>10 KEEP, q=10 REPLAN(10,10); recovery q<10")


class Physics:
    """Minimal stand-in for dm_control physics: the whole 'state' is the tick counter."""

    def __init__(self, env):
        self.env = env

        class D:
            ctrl = np.zeros(1)
        self.data = D()

    def get_state(self):
        return np.asarray([self.env.t], float)

    def set_state(self, s):
        self.env.t = int(s[0])

    def reset_context(self):
        import contextlib
        return contextlib.nullcontext()

    def forward(self):
        pass


class FakeEnv:
    def __init__(self):
        self.t, self.cmds = 0, []

        class E:
            pass
        self._env = E()
        self._env.physics = Physics(self)

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
    def __init__(self):
        self.calls = []

    def sample(self, obs_pi0, n, noise, y=None, w=None):
        cid = len(self.calls)
        self.calls.append(dict(n=n, y=None if y is None else y.copy(), w=None if w is None else w.copy(),
                               obs_t=float(obs_pi0["state"][0])))
        act = np.zeros((50, 14), np.float32)
        act[:, 0], act[:, 1] = cid, np.arange(50)
        x0 = np.zeros((50, 32), np.float32)
        x0[:, 0], x0[:, 1] = cid, np.arange(50)
        return x0, act, [1 - k / n for k in range(n)]


# 2. fixed-baseline timeline, RTC alignment, gating
env, fs = FakeEnv(), FakeSampler()
exe = Executor(env, fs)
res = exe.run(0, reference_policy, np.random.default_rng(1), record=True)
reqs = res["requests"]
assert [q["obs_t"] for q in reqs[:4]] == [15, 40, 65, 90] and all(q["r"] == 10 and q["n"] == 10 for q in reqs[:-1])
assert [q["handoff_t"] for q in reqs[:3]] == [25, 50, 75] and all(q["ready_t"] == q["obs_t"] + 10 for q in reqs)
assert [q["shift"] for q in reqs[:3]] == [15, 25, 25] and all(q["committed"] == 10 for q in reqs)
assert [q["overlap_end"] for q in reqs[:2]] == [35, 25]
assert all(q["p_at_launch"] == 15 and q["old_executed"] == 25 and q["discarded"] == 0 for q in reqs[:-1])
assert all(q["decision_kind"] == "baseline_replan" for q in reqs) and res["underrun_ticks"] == 0
cmds = np.asarray(env.cmds)
assert np.array_equal(cmds[:25, 1], np.arange(25)) and np.all(cmds[25:50, 0] == 1)
assert np.array_equal(cmds[25:50, 1], np.arange(10, 35)) and np.array_equal(cmds[50:75, 1], np.arange(10, 35))
assert np.array_equal(fs.calls[1]["y"][:35, 1], np.arange(15, 50)) and np.allclose(fs.calls[1]["w"][:10], 1, atol=1e-6)
assert np.array_equal(fs.calls[2]["y"][:25, 1], np.arange(25, 50)) and np.all(fs.calls[2]["y"][:25, 0] == 1)
print("[2] baseline: observe at p=15 (t=15,40,65,90), ready +10, handoff at p=25, install new[10:35]; RTC shift 15/25, "
      "committed 10, overlap end 35/25; 25 old actions per chunk; no recovery, no underrun")

# 3. mid-chunk snapshot (chunk 2, p = 5) and the REPLAN(7, 5) example
snaps = []
env2 = FakeEnv()
exe2 = Executor(env2, FakeSampler())


def grab(ex, e):
    if ex["chunk"] == 2 and ex["p"] == 5 and not snaps:
        snaps.append(take_snapshot(ex, e, extra=dict(env_seed=0)))


exe2.run(0, reference_policy, np.random.default_rng(1), on_tick=grab)
s = snaps[0]
assert s["t"] == 30 and len(s["queue"]) == 20 and s["prev"]["origin"] == 15
fs3 = FakeSampler()
fs3.calls = [None] * s["n_calls"]  # continue call numbering after the snapshot
exe3 = Executor(env2, fs3)
rb = exe3.run(0, reference_policy, np.random.default_rng(5), snapshot=s)
ra = exe3.run(0, reference_policy, np.random.default_rng(5), snapshot=s, forced_first=core.encode(7, 5),
              continuation=reference_policy)
b0, a0 = rb["requests"][0], ra["requests"][0]
assert (b0["obs_t"], b0["handoff_t"], b0["p_at_launch"], b0["old_executed"]) == (40, 50, 15, 25)
assert (a0["obs_t"], a0["ready_t"], a0["handoff_t"], a0["p_at_launch"], a0["old_executed"], a0["discarded"]) == (
    30, 37, 37, 5, 12, 13)
assert a0["shift"] == 15 and a0["committed"] == 7 and fs3.calls[-len(ra["requests"])]["obs_t"] == 30
a1 = ra["requests"][1]  # after the early replacement the baseline resumes: next request 15 actions into the new chunk
assert a1["obs_t"] == 37 + 15 and a1["shift"] == 52 - 30 and a1["r"] == 10
restore_snapshot(env2, s)
assert env2.t == 30
print("[3] snapshot at chunk 2, p=5 (t=30, q=20): baseline observes t=40 (p=15), hands off t=50; REPLAN(7,5) observes "
      "t=30, ready t=37, hands off t=37: 12 old actions executed, 13 discarded; next RTC shift 22 after the early swap")

# 4. step-0 parity with the new rule; replay stores the new reference
L = Learner(0, dict(actor_lr=1e-4, critic_lr=3e-4, alpha_lr=3e-4, alpha_init=0.001, gamma=0.999, tau=0.005, p0=0.999,
                    target_entropy_ratio=0.1))
for d_ in res["decisions"][:80]:
    st, mk = d_["state"], d_["mask"]
    ref = core.reference_index(st["q"], mk)
    p, resid = L.policy_probs(st, mk, ref, 1.0)
    assert int(np.argmax(p)) == ref
Rp = Replay(4000, 4000, 4000, 0)
assert len(Rp) == 0
Rp.add_episode(res["decisions"])
for i in range(Rp.ns):
    q = int(round(Rp.st["qmask"][i].sum()))
    if Rp.st["mask"][i].any():  # terminal states have no feasible action (stored ref 0)
        assert Rp.st["ref"][i] == core.reference_index(q, Rp.st["mask"][i])
print("[4] step-0 deterministic policy = new fixed baseline; replay starts empty and stores the new reference labels")
print("SCH-01 v2 CPU CHECKS OK")
