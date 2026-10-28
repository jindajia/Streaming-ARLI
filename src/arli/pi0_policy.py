"""Frozen pi0 ALOHA policy (openpi @ a6d2400, the DSRL fork) with a step-wise denoising sampler.

* ``infer(obs, noise)`` is exactly the DSRL path (``Policy.infer`` -> ``Pi0.sample_actions`` while_loop).
* ``begin(obs)`` + ``step(state, x, k)`` + ``finish(x)`` split the same computation so that the steering
  extension can read/modify the noisy action X between frozen action-expert calls.  With no
  modification the step-wise path must reproduce ``infer`` (checked in ``tests``/``scripts/check_sampler.py``).

No parameter of pi0 is ever passed to an optimizer: the model state is captured once by ``jax.jit`` closures.
"""

from __future__ import annotations

import dataclasses
import os
import time

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as openpi_config
from openpi_client import image_tools

# Paper: s3://openpi-assets/checkpoints/pi0_aloha_sim.  That bucket is gone; the same object path is served from
# gs://openpi-assets and fetched with MD5 checks by scripts/download_checkpoint.py (see MANIFEST.md).
CHECKPOINT_CITED = "s3://openpi-assets/checkpoints/pi0_aloha_sim"
CHECKPOINT = os.environ.get("ARLI_PI0_CKPT", "/N/scratch/jindjia/arli/checkpoints/pi0_aloha_sim")
CONFIG_NAME = "pi0_aloha_sim"


def obs_to_pi0_input(obs: dict) -> dict:
    """Identical to dsrl_pi0 examples/train_utils_sim.obs_to_pi_zero_input for aloha_cube."""
    img = np.ascontiguousarray(obs["pixels"]["top"])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
    return {"state": obs["agent_pos"], "images": {"cam_high": np.transpose(img, (2, 0, 1))}}


@dataclasses.dataclass
class InferenceState:
    observation: object  # preprocessed _model.Observation (batch 1)
    kv_cache: object
    prefix_mask: object
    raw_state: np.ndarray  # pi0 "state" input, needed by output transforms


class FrozenPi0:
    def __init__(self, num_steps: int = 10):
        self.num_steps = num_steps
        self.config = openpi_config.get_config(CONFIG_NAME)
        self.checkpoint_dir = download.maybe_download(CHECKPOINT)
        self.policy = policy_config.create_trained_policy(self.config, self.checkpoint_dir)
        self.model = self.policy._model
        self.action_horizon = self.model.action_horizon
        self.action_dim = self.model.action_dim
        assert (self.action_horizon, self.action_dim) == (50, 32)
        self.dt = -1.0 / num_steps
        # flow time used at expert step k, reproduced exactly as the float32 while_loop accumulates it
        t = jnp.float32(1.0)
        taus = []
        for _ in range(num_steps):
            taus.append(t)
            t = t + jnp.float32(self.dt)
        self.taus = [jnp.asarray(x, jnp.float32) for x in taus]
        self._build_jits()

    # -------------------------------------------------------------------------------------------
    def _build_jits(self):
        graphdef, state = nnx.split(self.model)
        self._state = state

        def prefix(state, observation):
            model = nnx.merge(graphdef, state)
            observation = _model.preprocess_observation(None, observation, train=False)
            prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
            return observation, kv_cache, prefix_mask

        def velocity(state, observation, kv_cache, prefix_mask, x_t, time):
            # body of Pi0.sample_actions.step, returning v_t
            model = nnx.merge(graphdef, state)
            batch_size = x_t.shape[0]
            suffix_tokens, suffix_mask, suffix_ar_mask = model.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size))
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = model.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache)
            return model.action_out_proj(suffix_out[:, -model.action_horizon:])

        def euler(state, observation, kv_cache, prefix_mask, x_t, time, dt):
            v = velocity(state, observation, kv_cache, prefix_mask, x_t, time)
            return x_t + dt * v, v

        self._prefix = jax.jit(prefix)
        self._euler = jax.jit(euler)

    def _transform_inputs(self, obs_pi0: dict):
        inputs = jax.tree.map(lambda x: x, obs_pi0)
        inputs = self.policy._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        return inputs

    # --- reference (DSRL) path --------------------------------------------------------------------
    def infer(self, obs_pi0: dict, noise) -> np.ndarray:
        """pi0(o, noise) -> (50, 14) ALOHA joint-position actions.  noise: (1, 50, 32)."""
        return np.asarray(self.policy.infer(obs_pi0, noise=jnp.asarray(noise, jnp.float32))["actions"])

    # --- step-wise path ----------------------------------------------------------------------------
    def begin(self, obs_pi0: dict) -> InferenceState:
        """VLM prefix pass on the (old) observation; returns the cache the expert steps reuse."""
        inputs = self._transform_inputs(obs_pi0)
        observation, kv_cache, prefix_mask = self._prefix(self._state, _model.Observation.from_dict(inputs))
        return InferenceState(observation, kv_cache, prefix_mask, np.asarray(inputs["state"]))

    def step(self, st: InferenceState, x, k: int):
        """One frozen action-expert Euler step at flow time tau_k.  Returns (x_next, v)."""
        return self._euler(self._state, st.observation, st.kv_cache, st.prefix_mask,
                           jnp.asarray(x, jnp.float32), self.taus[k], jnp.float32(self.dt))

    def finish(self, st: InferenceState, x0) -> np.ndarray:
        outputs = {"state": st.raw_state, "actions": x0}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        return np.asarray(self.policy._output_transform(outputs)["actions"])

    def sample_stepwise(self, obs_pi0: dict, noise, hook=None):
        """Full step-wise sample; ``hook(k, x_k) -> x_tilde_k`` may modify X before expert call k."""
        st = self.begin(obs_pi0)
        x = jnp.asarray(noise, jnp.float32)
        for k in range(self.num_steps):
            if hook is not None:
                x = hook(k, x)
            x, _ = self.step(st, x, k)
        return self.finish(st, x)


def configure_caches():
    """Point openpi / XLA caches at scratch (home is full)."""
    root = os.environ.setdefault("ARLI_SCRATCH", "/N/scratch/jindjia/arli")
    os.environ.setdefault("OPENPI_DATA_HOME", f"{root}/openpi_cache")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")
    jax.config.update("jax_compilation_cache_dir", f"{root}/jax_cache")


def time_call(fn, *args, n=5):
    out = fn(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(*args)
        jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n
