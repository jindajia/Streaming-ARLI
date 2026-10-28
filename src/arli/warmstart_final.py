"""Warm-start experiment: from the ARLI 600k checkpoint, spend 300k more online env steps either

* Branch A (``continue_arli``): keep training the ARLI initial-noise actor + critic (restored model, optimizer, temperature
  and target states; fresh replay), or
* Branch B (``frozen_arli_plus_final``): freeze the trained ARLI actor (all parameters, incl. encoder) -- it keeps
  generating the initial noise exactly as in ARLI -- and train a separate residual actor called once, immediately before
  the final denoising iteration k = 9, with its own critic / temperature / optimizers.

Both branches run pi0 through the same step-wise sampler (VLM prefix once per inference on o_{s_j}, 10 frozen action-expert
Euler iterations), so a zero correction in B is numerically identical to A (DECISIONS D12c, D25).

Residual actor = the from-scratch ``final_step_only`` architecture (``single_actor.make_defs``: CNN on old ++ newest frame,
q_old ++ q_newest, committed actions, X encoder, (tau, delta, f) features, 3x128 MLP, tanh-Gaussian head 32) and the same
residual parameterisation: X~_9 = X_9 + C_SCALE * broadcast_50(a), a in [-1, 1]^32, C_SCALE = 0.2.
Initialisation: mean head kernel and bias exactly 0 -> mean correction exactly 0; log-std head kernel 0 and bias
log(RES_INIT_STD) -> sampled a = tanh(N(0, RES_INIT_STD^2)), i.e. a small but non-zero sampled correction
(|C_SCALE * a| ~ 0.02 per dim at init).  ``zero_correction=True`` forces a = 0 exactly (validation only).

Branch B transition (one decision per executed chunk): z_{j,9} --a--> z_{j+1,9}, reward r_j (-1, or 0 with mask 0 for the
chunk executing at success), discount gamma^25: consecutive k = 9 calls are one chunk = 25 env steps apart in steady state
(t = m_j + 9 -> m_{j+1} + 9).  The startup chunk is computed with the clock frozen (its k = 9 call is at t = 0, the next at
t = 24); ARLI's per-chunk convention gamma^25 is kept for it.
"""

from __future__ import annotations

import hashlib
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import frozen_dict

from arli import rl
from arli.controllers import broadcast_noise, rl_obs_from_ctx
from arli.single_actor import C_SCALE, SingleActorLearner, make_z

RES_INIT_STD = 0.1
RES_INIT_TEMPERATURE = 0.01  # SAC temperature of the NEW residual actor at start (D32); auto-tuned afterwards
FINAL_K = 9


def resolve_checkpoint(run_root, run, milestone=600_000):
    """The agent checkpoint saved at the given evaluation milestone (filename = actual online step)."""
    ev = [json.loads(l) for l in open(f"{run_root}/{run}/evals.jsonl")]
    hit = [e for e in ev if e["milestone"] == milestone]
    if len(hit) != 1:
        raise RuntimeError(f"{run}: expected exactly one evaluation at milestone {milestone}, found {len(hit)}")
    e = hit[0]
    path = f"{run_root}/{run}/ckpt/agent_{e['online_env_steps']:07d}.msgpack"
    blob = open(path, "rb").read()
    return dict(source_run=run, milestone=milestone, path=path, online_env_steps=e["online_env_steps"],
                prefill_env_steps=e["prefill_env_steps"], grad_steps=e["grad_steps"], eval_success=e["n_success"],
                sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob)), blob


def tree_hash(tree):
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        h.update(np.asarray(leaf).tobytes())
    return h.hexdigest()


class ResidualLearner(SingleActorLearner):
    """Fresh residual actor + 10-critic ensemble + temperature; zero-mean, small-noise actor initialisation."""

    def __init__(self, seed: int, init_std: float = RES_INIT_STD, init_temperature: float = RES_INIT_TEMPERATURE):
        super().__init__(seed + 1_000_003)  # independent of the ARLI agent's init stream
        from jaxrl2.agents.pixel_sac.temperature import Temperature
        from jaxrl2.agents.pixel_sac.pixel_sac_learner import TrainState
        tdef = Temperature(init_temperature)
        self.T = TrainState.create(apply_fn=tdef.apply, params=tdef.init(jax.random.PRNGKey(0))["params"],
                                   tx=self.T.tx, batch_stats=None)
        p = frozen_dict.unfreeze(self.A.params)
        net = p["network"]
        net["Dense_0"]["kernel"] = jnp.zeros_like(net["Dense_0"]["kernel"])  # mean head
        net["Dense_0"]["bias"] = jnp.zeros_like(net["Dense_0"]["bias"])
        net["Dense_1"]["kernel"] = jnp.zeros_like(net["Dense_1"]["kernel"])  # log-std head
        net["Dense_1"]["bias"] = jnp.full_like(net["Dense_1"]["bias"], np.log(init_std))
        self.A = self.A.replace(params=frozen_dict.freeze(p), opt_state=self.A.tx.init(frozen_dict.freeze(p)))


class Controller:
    """Rollout controller for both branches.

    noise_key: key stream for the ARLI initial-noise actor -- split exactly like ``controllers.NoiseActor(eval_key=...)``
    so that branch A and a zero-correction branch B draw identical initial noise.  res_key: separate stream for the
    residual actor (never consumes noise_key).  mode 'warmup' still samples both actors (fresh replay is collected with the
    branch's own policy).
    """

    def __init__(self, branch, agent, sched, noise_key, residual=None, res_key=None, zero_correction=False):
        assert branch in ("A", "B")
        self.branch, self.agent, self.sched, self.noise_key = branch, agent, sched, noise_key
        self.R, self.res_key, self.zero_correction = residual, res_key, zero_correction
        self.steering = self  # always the step-wise sampler (A: no corrections)
        self.steps = frozenset([FINAL_K]) if branch == "B" else frozenset()
        self.calls = []

    def initial_noise(self, ctx):
        obs = rl_obs_from_ctx("arli", ctx)
        self.noise_key, k = jax.random.split(self.noise_key)
        w = np.reshape(rl.sample_noise_action(self.agent, obs, k), (1, rl.NOISE_DIM)).astype(np.float32)
        self.calls.append(("noise", ctx["j"], 0, ctx["t_mid"]))
        return broadcast_noise(w), dict(obs=obs, action=w)

    def correct(self, ctx, k, x, tau, latest):
        assert self.branch == "B" and k == FINAL_K
        j = ctx["j"]
        delta = (self.sched.T(j) - latest["t"]) / self.sched.delay if j > 0 else 0.0
        z = make_z(ctx, latest["img"], latest["q"], np.asarray(x)[0], tau, delta, first=False)
        if self.zero_correction:
            a = np.zeros((1, rl.NOISE_DIM), np.float32)
        else:
            self.res_key, kk = jax.random.split(self.res_key)
            a = np.reshape(self.R.sample(z, kk), (1, rl.NOISE_DIM)).astype(np.float32)
        self.calls.append(("residual", j, k, latest["t"]))
        x_tilde = np.asarray(x, np.float32) + C_SCALE * np.repeat(a[None], 50, axis=1)
        return x_tilde, dict(obs=z, action=a, t_obs=latest["t"], t_now=latest["t"],
                             x_in=np.asarray(x, np.float32)[0], dx_norm=float(np.linalg.norm(x_tilde - x)))


def transitions_B(res, sched, gamma):
    """One k = 9 decision per executed chunk; successor = chunk j+1's k = 9 call; discount gamma^25 (module docstring)."""
    js = res["executed_decisions"]
    dec = res["decisions"]
    out = []
    for i, j in enumerate(js):
        last = i == len(js) - 1
        r, mask = (0.0, 0.0) if (last and res["success"]) else (-1.0, 1.0)
        (s,) = [s for s in dec[j]["steer"] if s["k"] == FINAL_K]
        nxt = [s2 for s2 in dec.get(j + 1, {"steer": []})["steer"] if s2["k"] == FINAL_K]
        if mask > 0 and not nxt:
            raise AssertionError(f"non-terminal chunk {j} lacks its successor k=9 decision")
        if nxt and j > 0:
            assert nxt[0]["t_now"] - s["t_now"] == sched.n_exec, "consecutive k=9 calls must be 25 env steps apart"
        out.append(dict(j=j, k=FINAL_K, obs=s["obs"], action=s["action"], reward=r, mask=mask,
                        discount=gamma ** sched.n_exec, next_obs=nxt[0]["obs"] if nxt else None))
    return out
