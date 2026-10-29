"""CPU stand-ins for the ALOHA env and the pi0 RTC sampler (smoke tests of the viewer exporter only; the same
fakes as tests/test_v2_cpu.py, with a frame that changes over time so the video path is exercised)."""
import contextlib

import numpy as np


class _Physics:
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
        return contextlib.nullcontext()

    def forward(self):
        pass


class FakeEnv:
    def __init__(self):
        self.t = 0

        class E:
            pass
        self._env = E()
        self._env.physics = _Physics(self)

    def reset(self, seed=None):
        self.t = 0
        return self._o(), {}

    def _o(self):
        img = np.zeros((480, 640, 3), np.uint8)
        img[:, : (self.t * 640) // 400] = (40, 120, 200)
        return {"pixels": {"top": img}, "agent_pos": np.full(14, float(self.t) / 400)}

    def step(self, a):
        self.t += 1
        return self._o(), 0.0, False, False, {}


class FakeSampler:
    def __init__(self):
        self.n = 0

    def sample(self, obs_pi0, n, noise, y=None, w=None):
        self.n += 1
        act = np.zeros((50, 14), np.float32)
        act[:, 0], act[:, 1] = self.n, np.arange(50)
        x0 = np.zeros((50, 32), np.float32)
        x0[:, 0], x0[:, 1] = self.n, np.arange(50)
        return x0, act, [1 - k / n for k in range(n)]
