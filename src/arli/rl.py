"""RL state construction, replay buffer and the DSRL/ARLI noise agent (jaxrl2 PixelSACLearner, unchanged).

RL observation per method (ARLI App. A.1, "Intermediate Information Processing"):

=================  ========================================  ==================  =========================
method             pixels (64x64, channel-concatenated)      state               prev_action (committed)
=================  ========================================  ==================  =========================
dsrl (naive async) img(o_s)                       3 ch       q(o_s)      14      --
arli               img(o_s) ++ img(o_m)           6 ch       q(o_s)++q(o_m) 28   a_[s, T)  20 x 14
arli_actions       img(o_s)                       3 ch       q(o_s)      14      a_[s, T)  20 x 14
arli_state         img(o_s) ++ img(o_m)           6 ch       q(o_s)++q(o_m) 28   --
=================  ========================================  ==================  =========================

jaxrl2's MLP flattens the dict in sorted key order (pixels-latent, prev_action, state), i.e. committed
actions are flattened and concatenated to the state vector before the MLP, as the paper describes.
"""

from __future__ import annotations

import functools
import pickle

import flax.serialization
import gym.spaces
import jax
import numpy as np
from flax.core import frozen_dict

NOISE_DIM = 32  # pi0 padded action dim; DSRL noise action = one 32-vector repeated over the 50-step horizon
METHODS = ("dsrl", "arli", "arli_actions", "arli_state")


def uses_mid_obs(method: str) -> bool:
    return method in ("arli", "arli_state")


def uses_actions(method: str) -> bool:
    return method in ("arli", "arli_actions")


def num_images(method: str) -> int:
    return 2 if uses_mid_obs(method) else 1


def make_rl_obs(method: str, img_old, q_old, committed=None, img_mid=None, q_mid=None) -> dict:
    """Single (unbatched) RL observation."""
    if uses_mid_obs(method):
        pixels = np.concatenate([img_old, img_mid], axis=-1)
        state = np.concatenate([q_old, q_mid])
    else:
        pixels, state = img_old, q_old
    obs = {"pixels": pixels[..., None].astype(np.uint8), "state": state[:, None].astype(np.float32)}
    if uses_actions(method):
        obs["prev_action"] = np.asarray(committed, np.float32)
    return obs


def observation_space(method: str, image_size: int = 64, n_committed: int = 20, act_dim: int = 14):
    sp = {
        "pixels": gym.spaces.Box(0, 255, (image_size, image_size, 3 * num_images(method), 1), np.uint8),
        "state": gym.spaces.Box(-np.inf, np.inf, (act_dim * num_images(method), 1), np.float32),
    }
    if uses_actions(method):
        sp["prev_action"] = gym.spaces.Box(-np.inf, np.inf, (n_committed, act_dim), np.float32)
    return gym.spaces.Dict(sp)


def batchify(obs: dict) -> dict:
    return {k: v[None] for k, v in obs.items()}


# ------------------------------------------------------------------------------------------------
class Replay:
    """Flat numpy transition store with per-transition discount (needed later for multi-rate steering)."""

    def __init__(self, obs_space: gym.spaces.Dict, act_shape, capacity: int, seed: int,
                 next_space: gym.spaces.Dict | None = None, extra: tuple = ()):
        self.capacity = capacity
        self.obs_space = obs_space
        next_space = next_space or obs_space
        self.act_shape = tuple(act_shape)
        self.extra = tuple(extra)
        self.data = {
            "observations": {k: np.empty((capacity, *s.shape), s.dtype) for k, s in obs_space.spaces.items()},
            "next_observations": {k: np.empty((capacity, *s.shape), s.dtype) for k, s in next_space.spaces.items()},
            "actions": np.empty((capacity, *self.act_shape), np.float32),
            "rewards": np.empty((capacity,), np.float32),
            "masks": np.empty((capacity,), np.float32),
            "discount": np.empty((capacity,), np.float32),
            **{k: np.empty((capacity,), np.float32) for k in extra},
        }
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def insert(self, obs, action, reward, mask, discount, next_obs, **extra):
        if self.size == self.capacity:
            raise RuntimeError("replay buffer full; increase capacity")
        i = self.size
        for k in self.data["observations"]:
            self.data["observations"][k][i] = obs[k]
        for k in self.data["next_observations"]:
            self.data["next_observations"][k][i] = next_obs[k]
        for k in self.extra:
            self.data[k][i] = extra[k]
        self.data["actions"][i] = np.reshape(action, self.act_shape)
        self.data["rewards"][i] = reward
        self.data["masks"][i] = mask
        self.data["discount"][i] = discount
        self.size += 1

    def sample(self, batch_size: int):
        idx = self.rng.integers(self.size, size=batch_size)
        b = {k: ({kk: vv[idx] for kk, vv in v.items()} if isinstance(v, dict) else v[idx])
             for k, v in self.data.items()}
        return frozen_dict.freeze(b)

    def state_dict(self):
        n = self.size
        return {
            "size": n, "rng": self.rng.bit_generator.state,
            "data": {k: ({kk: vv[:n] for kk, vv in v.items()} if isinstance(v, dict) else v[:n])
                     for k, v in self.data.items()},
        }

    def load_state_dict(self, sd):
        n = sd["size"]
        for k, v in sd["data"].items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    self.data[k][kk][:n] = vv
            else:
                self.data[k][:n] = v
        self.size = n
        self.rng.bit_generator.state = sd["rng"]


# ------------------------------------------------------------------------------------------------
# ARLI Table 1 / Table 3 (AlohaTransferCube) mapped onto dsrl_pi0's PixelSACLearner arguments.
AGENT_KWARGS = dict(
    actor_lr=1e-4, critic_lr=3e-4, temp_lr=3e-4,
    hidden_dims=(128, 128, 128), cnn_features=(32, 32, 32, 32), cnn_strides=(2, 1, 1, 1), cnn_padding="VALID",
    latent_dim=50, discount=0.999, tau=0.005, critic_reduction="mean",  # mean over 10 Qs = no clipped Q
    dropout_rate=0.0, aug_next=1, use_bottleneck=True, encoder_type="small", encoder_norm="group",
    use_spatial_softmax=True, softmax_temperature=-1, target_entropy="auto",  # auto = -d/2 = -16
    num_qs=10, action_magnitude=2.0,
)


def make_agent(method: str, seed: int, **overrides):
    from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner

    kw = dict(AGENT_KWARGS, num_cameras=num_images(method))
    kw.update(overrides)
    space = observation_space(method)
    sample_obs = {k: s.sample()[None] for k, s in space.spaces.items()}
    sample_action = np.zeros((1, 1, NOISE_DIM), np.float32)
    return PixelSACLearner(seed, sample_obs, sample_action, **kw)


@functools.partial(jax.jit, static_argnames="apply_fn")
def _sample(key, apply_fn, params, obs):
    return apply_fn({"params": params}, obs).sample(seed=key)


def sample_noise_action(agent, obs: dict, key) -> np.ndarray:
    """Stochastic actor sample with an explicit key (eval does not consume the training RNG)."""
    return np.asarray(_sample(key, agent._actor.apply_fn, agent._actor.params, batchify(obs)))[0]


def agent_state_bytes(agent) -> bytes:
    sd = {"actor": agent._actor, "critic": agent._critic, "target_critic_params": agent._target_critic_params,
          "temp": agent._temp, "rng": agent._rng}
    return flax.serialization.to_bytes(sd)


def load_agent_state(agent, blob: bytes):
    target = {"actor": agent._actor, "critic": agent._critic, "target_critic_params": agent._target_critic_params,
              "temp": agent._temp, "rng": agent._rng}
    sd = flax.serialization.from_bytes(target, blob)
    agent._actor, agent._critic = sd["actor"], sd["critic"]
    agent._target_critic_params, agent._temp, agent._rng = sd["target_critic_params"], sd["temp"], sd["rng"]


def save_pickle(path, obj):
    import os
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
