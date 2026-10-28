"""Initial-noise controllers for ``rollout.run_episode``."""

from __future__ import annotations

import jax
import numpy as np

from arli import rl


def broadcast_noise(w: np.ndarray, horizon: int = 50) -> np.ndarray:
    """DSRL: one 32-dim noise vector repeated over the action horizon -> (1, 50, 32)."""
    w = np.reshape(np.asarray(w, np.float32), (1, rl.NOISE_DIM))
    return np.repeat(w[None], horizon, axis=1)


def rl_obs_from_ctx(method: str, ctx: dict) -> dict:
    return rl.make_rl_obs(method, ctx["img_old"], ctx["q_old"], ctx["committed"], ctx["img_mid"], ctx["q_mid"])


class FrozenPolicy:
    """Pretrained pi0 as deployed: full (50 x 32) i.i.d. N(0, I) noise, no RL."""

    steering = None

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def initial_noise(self, ctx):
        return self.rng.standard_normal((1, 50, rl.NOISE_DIM)).astype(np.float32), {}


class GaussianNoise:
    """Initial data collection (ARLI Table 6): w ~ N(0, std^2) broadcast like the actor's action, unclipped."""

    steering = None

    def __init__(self, method: str, rng: np.random.Generator, std: float):
        self.method, self.rng, self.std = method, rng, std

    def initial_noise(self, ctx):
        w = (self.std * self.rng.standard_normal((1, rl.NOISE_DIM))).astype(np.float32)
        return broadcast_noise(w), dict(obs=rl_obs_from_ctx(self.method, ctx), action=w)


class NoiseActor:
    """DSRL / ARLI actor.  train: agent.sample_actions (agent RNG, as dsrl_pi0); eval: explicit key stream."""

    steering = None

    def __init__(self, method: str, agent, eval_key=None):
        self.method, self.agent = method, agent
        self.eval_key = eval_key

    def initial_noise(self, ctx):
        obs = rl_obs_from_ctx(self.method, ctx)
        if self.eval_key is None:
            w = np.asarray(self.agent.sample_actions(rl.batchify(obs)))[0]
        else:
            self.eval_key, k = jax.random.split(self.eval_key)
            w = rl.sample_noise_action(self.agent, obs, k)
        w = np.reshape(w, (1, rl.NOISE_DIM)).astype(np.float32)
        return broadcast_noise(w), dict(obs=obs, action=w)
