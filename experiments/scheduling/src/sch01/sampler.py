"""Frozen pi0 (openpi fork a6d2400, as used by Streaming-ARLI) with N-step full-interval Euler sampling and RTC.

* N in 3..10: tau_k = 1 - k/N, dt = -1/N, k = 0..N-1, so every N integrates the complete flow interval tau 1 -> 0.
* VLM prefix computed once per request on the request's observation.
* RTC (inference-time guidance, VJP through the frozen action expert): ``rtc_vendored.guided_velocity`` -- verbatim
  copy of the profiling port (Streaming-ARLI profiling commit 7e58596), verified against
  Physical-Intelligence/real-time-chunking-kinetix@9296f31 (max |diff| 6.8e-6).  Soft mask = official ``exp`` schedule,
  max guidance weight 10.
* RTC operates in the model's NORMALISED padded action space (50 x 32).  The ALOHA config uses absolute joint actions
  and state-independent output transforms, so the previous normalised prediction is directly comparable.
"""

from __future__ import annotations

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask

from sch01 import rtc_vendored as rtc
from sch01.core import H

MAX_GUIDANCE = 10.0


def rtc_weights(committed: int, overlap_end: int) -> np.ndarray:
    """Official exp soft mask: W[0:committed] = 1, decreasing on [committed, overlap_end), 0 from overlap_end on."""
    return np.asarray(rtc.get_prefix_weights(committed, overlap_end, H, "exp"), np.float32)


def shift_reference(prev_norm: np.ndarray, shift: int) -> np.ndarray:
    """y[i] = prev[i + shift] where it exists (zeros elsewhere; their weights are zero)."""
    y = np.zeros_like(prev_norm)
    if shift < H:
        y[: H - shift] = prev_norm[shift:]
    return y


class RTCSampler:
    def __init__(self, frozen_pi0):
        self.fp = frozen_pi0
        model = frozen_pi0.model
        graphdef, state = nnx.split(model)
        self._state = state

        def velocity(state, observation, kv_cache, prefix_mask, x_t, time):
            m = nnx.merge(graphdef, state)
            b = x_t.shape[0]
            suffix_tokens, suffix_mask, suffix_ar_mask = m.embed_suffix(observation, x_t, jnp.broadcast_to(time, b))
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, out), _ = m.PaliGemma.llm([None, suffix_tokens], mask=full, positions=positions, kv_cache=kv_cache)
            return m.action_out_proj(out[:, -m.action_horizon:])

        def plain_step(state, observation, kv, pm, x, tau, dt):
            v = velocity(state, observation, kv, pm, x, tau)
            return x + dt * v

        def rtc_step(state, observation, kv, pm, x, tau, dt, y, w):
            f = lambda xx: velocity(state, observation, kv, pm, xx, tau)  # noqa: E731
            v = rtc.guided_velocity(f, x, tau, y, w, MAX_GUIDANCE)
            return x + dt * v

        self._plain = jax.jit(plain_step)
        self._rtc = jax.jit(rtc_step)

    def sample(self, obs_pi0: dict, n: int, noise: np.ndarray, rtc_ref: np.ndarray | None = None,
               rtc_w: np.ndarray | None = None, return_trace: bool = False):
        """-> (normalised x0 [50, 32], actions [50, 14], taus).  noise: [50, 32]."""
        st = self.fp.begin(obs_pi0)  # VLM prefix on the request observation
        x = jnp.asarray(noise, jnp.float32)[None]
        dt = jnp.float32(-1.0 / n)
        taus = []
        for k in range(n):
            tau = jnp.float32(1.0 - k / n)
            taus.append(float(tau))
            if rtc_ref is None:
                x = self._plain(self._state, st.observation, st.kv_cache, st.prefix_mask, x, tau, dt)
            else:
                x = self._rtc(self._state, st.observation, st.kv_cache, st.prefix_mask, x, tau, dt,
                              jnp.asarray(rtc_ref, jnp.float32)[None], jnp.asarray(rtc_w, jnp.float32))
        x0 = np.asarray(x[0], np.float32)
        actions = self.fp.finish(st, x)
        return x0, np.asarray(actions, np.float32), taus
