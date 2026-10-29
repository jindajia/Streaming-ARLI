"""Real-Time Chunking (RTC) guided denoising, ported to openpi's flow-time convention.

Source (verified, see profiling/tests/test_rtc_port.py):
    Physical-Intelligence/real-time-chunking-kinetix @ 9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b, src/model.py
    ``get_prefix_weights`` and ``FlowPolicy.realtime_action.step.pinv_corrected_velocity``.

Conventions
-----------
kinetix: flow time t_k runs 0 (noise) -> 1 (data), velocity v_k = dx/dt_k, data estimate x_hat = x + (1 - t_k) v_k.
openpi : flow time t_o runs 1 (noise) -> 0 (data), velocity v_o = dx/dt_o = -v_k, t_o = 1 - t_k,
         so the same data estimate is x_hat = x - t_o v_o.
The official guided velocity v_k + w J^T e therefore becomes v_o - w J^T e here, with
    e = (y - x_hat) * W[prefix],    J = d x_hat / d x  (VJP through the full action-expert forward pass),
    w = min(c * inv_r2, w_max),  c = nan_to_num((1 - t_k) / t_k, posinf=w_max),
    inv_r2 = (t_k^2 + (1 - t_k)^2) / (1 - t_k)^2.
``y`` is the previous chunk shifted so that y[i] is the action previously planned for the i-th step of the new chunk
(official eval_flow.py: ``prefix_attention_horizon = H - execute_horizon``), in the model's normalised action space.

Note: LeRobot 0.5.1's ``RTCProcessor.denoise_step`` computes v_t *before* enabling grad on x_t, so its "correction" is the
weighted error itself (no Jacobian / VJP).  That is a cheaper approximation, not the official algorithm, and is not used.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

RTC_SOURCE = "https://github.com/Physical-Intelligence/real-time-chunking-kinetix"
RTC_COMMIT = "9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b"


@dataclasses.dataclass(frozen=True)
class RTCSpec:
    inference_delay: int  # d: actions of the new chunk already committed from the previous chunk
    execute_horizon: int  # s: actions executed per chunk; prefix_attention_horizon = H - s
    max_guidance_weight: float = 10.0
    schedule: str = "exp"  # "exp" = soft masking (paper Table 4 "soft masking: on")

    def prefix_horizon(self, H: int) -> int:
        return H - self.execute_horizon


def get_prefix_weights(start: int, end: int, total: int, schedule: str) -> jax.Array:
    """Verbatim port of the official function (see module docstring)."""
    start = jnp.minimum(start, end)
    if schedule == "ones":
        w = jnp.ones(total)
    elif schedule == "zeros":
        w = (jnp.arange(total) < start).astype(jnp.float32)
    elif schedule == "linear" or schedule == "exp":
        w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * jnp.expm1(w) / (jnp.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return jnp.where(jnp.arange(total) >= end, 0, w)


def guidance_weight(t_o, max_guidance_weight: float):
    t = 1.0 - t_o  # kinetix time
    inv_r2 = (t**2 + (1 - t) ** 2) / ((1 - t) ** 2)
    c = jnp.nan_to_num((1 - t) / t, posinf=max_guidance_weight)
    return jnp.minimum(c * inv_r2, max_guidance_weight)


def guided_velocity(velocity_fn, x_t, t_o, y, weights, max_guidance_weight: float):
    """openpi-convention RTC velocity.  velocity_fn(x) -> v_o (full action-expert forward, prefix KV cache reused).

    Named scopes split the work for the device trace: ``rtc_fwd`` (expert forward, linearised for the VJP),
    ``rtc_vjp`` (the backward pass through the expert = the extra RTC cost) and ``rtc_guidance`` (elementwise).
    """

    def denoiser(x):
        v = velocity_fn(x)
        return x - t_o * v, v

    with jax.named_scope("rtc_fwd"):
        x_hat, vjp_fun, v = jax.vjp(denoiser, x_t, has_aux=True)
    with jax.named_scope("rtc_guidance"):
        err = (y - x_hat) * weights[None, :, None].astype(x_hat.dtype)
    with jax.named_scope("rtc_vjp"):
        correction = vjp_fun(err)[0]
    with jax.named_scope("rtc_guidance"):
        w = guidance_weight(t_o, max_guidance_weight)
        return v - w * correction


def shift_prev_chunk(prev_chunk, execute_horizon: int):
    """y[i] = prev[i + s] for i < H - s, zero afterwards (weights are zero there anyway)."""
    H = prev_chunk.shape[-2]
    tail = prev_chunk[..., execute_horizon:, :]
    pad = jnp.zeros(prev_chunk.shape[:-2] + (execute_horizon, prev_chunk.shape[-1]), prev_chunk.dtype)
    out = jnp.concatenate([tail, pad], axis=-2)
    assert out.shape[-2] == H
    return out
