"""Discrete SAC scheduler: reference-prior actor with zero-initialised residual head, two critics, duration-aware targets.

policy(u | s) = softmax over feasible u of  residual(s)[u] + prior(s, step)[u]
prior: reference action logit = strength(step) * log(p0 (M-1)/(1-p0)), other feasible logits 0 (core.prior_logits).
Critic target: y = G + gamma^D (1 - terminal) * sum_u' pi(u'|s') [min_i Qbar_i(s', u') - alpha log pi(u'|s')].
Actor loss:    sum_u pi(u|s) [alpha log pi(u|s) - min_i Q_i(s, u)]  over feasible u.
Temperature:   log-alpha descends on (H(pi(.|s)) - target(s)), target(s) = min(H(prior-only policy at the current
               strength), target_ratio * log M(s)), M = number of feasible actions (DECISIONS S12).
All expectations use the same feasibility masks as sampling.
"""

from __future__ import annotations

import functools
import math

import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from sch01 import core

NEG = -1e9


class Encoder(nn.Module):
    @nn.compact
    def __call__(self, img):
        x = img.astype(jnp.float32) / 255.0
        for f, s in ((32, 2), (32, 2), (32, 1), (32, 1)):
            x = nn.relu(nn.Conv(f, (3, 3), (s, s), padding="VALID")(x))
        x = x.reshape(x.shape[0], -1)
        return nn.tanh(nn.LayerNorm()(nn.Dense(128)(x)))


class Net(nn.Module):
    """Shared-CNN visual features (now, source, difference) + queue MLP + runtime MLP -> 256-256 -> out."""
    out: int
    zero_head: bool = False

    @nn.compact
    def __call__(self, s):
        enc = Encoder()
        f_now, f_src = enc(s["img_now"]), enc(s["img_src"])
        f_vis = jnp.concatenate([f_now, f_src, f_now - f_src], -1)
        qf = jnp.concatenate([s["queue"].reshape(s["queue"].shape[0], -1), s["qmask"]], -1)
        f_q = nn.relu(nn.Dense(128)(qf))
        f_r = nn.relu(nn.Dense(64)(jnp.concatenate([s["robot"], s["runtime"]], -1)))
        h = jnp.concatenate([f_vis, f_q, f_r], -1)
        h = nn.relu(nn.Dense(256)(h))
        h = nn.relu(nn.Dense(256)(h))
        init = nn.initializers.zeros if self.zero_head else nn.initializers.lecun_normal()
        return nn.Dense(self.out, kernel_init=init, bias_init=nn.initializers.zeros, name="head")(h)


class Critics(nn.Module):
    @nn.compact
    def __call__(self, s):
        # zero-initialised heads: every action starts with the same Q (no invented preference for untried R/N;
        # random heads gave O(0.1-1) arbitrary gaps that the actor exploited -- DECISIONS S14)
        return jnp.stack([Net(core.NUM_ACTIONS, zero_head=True, name=f"q{i}")(s) for i in range(2)])  # [2, B, A]


STATE_KEYS = ("img_now", "img_src", "queue", "qmask", "robot", "runtime")


def dummy_state(b=1):
    return dict(img_now=np.zeros((b, 64, 64, 3), np.uint8), img_src=np.zeros((b, 64, 64, 3), np.uint8),
                queue=np.zeros((b, core.K_CAP, 14), np.float32), qmask=np.zeros((b, core.K_CAP), np.float32),
                robot=np.zeros((b, 28), np.float32), runtime=np.zeros((b, 5), np.float32))


def prior_logits_batch(ref, nfeas, mask, strength, p0):
    """Vectorised core.prior_logits.  ref: [B] int, nfeas: [B], mask: [B, A] bool."""
    big = strength * jnp.log(p0 * jnp.maximum(nfeas - 1, 1) / (1 - p0))
    big = jnp.where(nfeas > 1, big, 0.0)
    onehot = jax.nn.one_hot(ref, core.NUM_ACTIONS)
    return jnp.where(mask, onehot * big[:, None], NEG)


def prior_entropy(nfeas, strength, p0):
    """Entropy of the prior-only policy (zero residual) with M = nfeas feasible actions at the given strength.
    Used as the per-state entropy target while the prior is annealed (DECISIONS S12)."""
    m = jnp.maximum(nfeas, 1.0)
    b = strength * jnp.log(p0 * jnp.maximum(m - 1, 1) / (1 - p0))
    p_ref = jnp.exp(b) / (jnp.exp(b) + m - 1)
    p_oth = (1 - p_ref) / jnp.maximum(m - 1, 1)
    h = -p_ref * jnp.log(p_ref) - (m - 1) * p_oth * jnp.log(jnp.maximum(p_oth, 1e-30))
    return jnp.where(m > 1, h, 0.0)


def masked_policy(residual, ref, nfeas, mask, strength, p0):
    logits = residual + prior_logits_batch(ref, nfeas, mask, strength, p0)
    logits = jnp.where(mask, logits, NEG)
    logp = jax.nn.log_softmax(logits, -1)
    p = jnp.where(mask, jnp.exp(logp), 0.0)
    return p, jnp.where(mask, logp, 0.0)


def _crop(key, img, pad=4):
    """Paired random shift: the same offset for img_now and img_src of a sample."""
    b = img.shape[0]
    off = jax.random.randint(key, (b, 2), 0, 2 * pad + 1)
    padded = jnp.pad(img, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")
    return jax.vmap(lambda im, o: jax.lax.dynamic_slice(im, (o[0], o[1], 0), (64, 64, 3)))(padded, off)


def augment(key, s):
    """Paired random shift: one key -> identical per-sample offsets for img_now and img_src."""
    return dict(s, img_now=_crop(key, s["img_now"]), img_src=_crop(key, s["img_src"]))


class Learner:
    def __init__(self, seed, cfg):
        self.cfg = cfg
        k = jax.random.PRNGKey(seed)
        k, ka, kc = jax.random.split(k, 3)
        self.actor_def, self.critic_def = Net(core.NUM_ACTIONS, zero_head=True), Critics()
        s = dummy_state()
        ap = self.actor_def.init(ka, s)["params"]
        cp = self.critic_def.init(kc, s)["params"]
        self.A = train_state.TrainState.create(apply_fn=self.actor_def.apply, params=ap, tx=optax.adam(cfg["actor_lr"]))
        self.C = train_state.TrainState.create(apply_fn=self.critic_def.apply, params=cp, tx=optax.adam(cfg["critic_lr"]))
        self.Ct = cp
        la = jnp.asarray(math.log(cfg["alpha_init"]), jnp.float32)
        self.T = train_state.TrainState.create(apply_fn=None, params=dict(log_alpha=la), tx=optax.adam(cfg["alpha_lr"]))
        self.rng = k
        self.n_params = dict(actor=int(sum(x.size for x in jax.tree_util.tree_leaves(ap))),
                             critics=int(sum(x.size for x in jax.tree_util.tree_leaves(cp))))
        self._act = jax.jit(self._residual)

    def _residual(self, params, s):
        return self.actor_def.apply({"params": params}, s)

    # ---- acting ------------------------------------------------------------------------------------
    def policy_probs(self, st, mask, ref, strength):
        s = {k: np.asarray(st[k])[None] for k in STATE_KEYS}
        res = np.asarray(self._act(self.A.params, s))[0].astype(np.float64)
        logits = res + core.prior_logits(ref, mask, strength, self.cfg["p0"]).astype(np.float64)
        logits[~mask] = -np.inf
        z = logits - logits.max()
        p = np.exp(z)
        p /= p.sum()
        return p, res

    # ---- learning -----------------------------------------------------------------------------------
    def update(self, batch, strength, actor_on: bool):
        (self.rng, self.A, self.C, self.Ct, self.T, info) = _update(
            self.rng, self.A, self.C, self.Ct, self.T, batch, jnp.float32(strength), actor_on,
            self.cfg["gamma"], self.cfg["tau"], self.cfg["p0"], self.cfg["target_entropy_ratio"])
        return info

    def state_bytes(self):
        return flax.serialization.to_bytes(dict(A=self.A, C=self.C, Ct=self.Ct, T=self.T, rng=self.rng))

    def load_bytes(self, blob):
        sd = flax.serialization.from_bytes(dict(A=self.A, C=self.C, Ct=self.Ct, T=self.T, rng=self.rng), blob)
        self.A, self.C, self.Ct, self.T, self.rng = sd["A"], sd["C"], sd["Ct"], sd["T"], sd["rng"]


@functools.partial(jax.jit, static_argnames=("actor_on", "gamma", "tau", "p0", "ratio"))
def _update(rng, A, C, Ct, T, b, strength, actor_on, gamma, tau, p0, ratio):
    rng, k1, k2 = jax.random.split(rng, 3)
    s = augment(k1, {k: b["s_" + k] for k in STATE_KEYS})
    s2 = augment(k2, {k: b["n_" + k] for k in STATE_KEYS})
    alpha = jnp.exp(T.params["log_alpha"])
    # ---- target ----
    res2 = A.apply_fn({"params": A.params}, s2)
    p2, logp2 = masked_policy(res2, b["n_ref"], b["n_nfeas"], b["n_mask"], strength, p0)
    q2 = C.apply_fn({"params": Ct}, s2).min(0)
    v2 = jnp.sum(p2 * jnp.where(b["n_mask"], q2 - alpha * logp2, 0.0), -1)
    y = b["G"] + (gamma ** b["D"]) * (1.0 - b["terminal"]) * v2
    y = jax.lax.stop_gradient(y)
    a = b["action"]

    def critic_loss(cp):
        qs = C.apply_fn({"params": cp}, s)  # [2, B, A]
        qa = jnp.take_along_axis(qs, a[None, :, None], -1)[..., 0]
        return ((qa - y[None]) ** 2).mean(), (qa, qs)

    (lc, (qa, qs)), g = jax.value_and_grad(critic_loss, has_aux=True)(C.params)
    C = C.apply_gradients(grads=g)
    Ct = optax.incremental_update(C.params, Ct, tau)
    info = dict(critic_loss=lc, q_taken=qa.mean(), q_disagreement=jnp.abs(qa[0] - qa[1]).mean(), target=y.mean(),
                q_max=qa.max())
    # ---- actor + temperature ----
    qmin = jax.lax.stop_gradient(C.apply_fn({"params": C.params}, s).min(0))

    def actor_loss(ap):
        res = A.apply_fn({"params": ap}, s)
        p, logp = masked_policy(res, b["s_ref"], b["s_nfeas"], b["s_mask"], strength, p0)
        loss = jnp.sum(p * jnp.where(b["s_mask"], alpha * logp - qmin, 0.0), -1).mean()
        ent = -jnp.sum(p * logp, -1)
        pref = jnp.take_along_axis(p, b["s_ref"][:, None], -1)[:, 0]
        rmag = jnp.abs(jnp.where(b["s_mask"], res, 0.0)).max(-1)
        return loss, (ent, pref, rmag)

    (la, (ent, pref, rmag)), ga = jax.value_and_grad(actor_loss, has_aux=True)(A.params)
    if actor_on:
        A = A.apply_gradients(grads=ga)
        target_ent = jnp.minimum(prior_entropy(b["s_nfeas"], strength, p0),
                                 ratio * jnp.log(jnp.maximum(b["s_nfeas"], 1.0)))
        gt = jax.grad(lambda tp: (tp["log_alpha"] * jax.lax.stop_gradient(ent - target_ent)).mean())(T.params)
        T = T.apply_gradients(grads=gt)
    qgap = (jnp.where(b["s_mask"], qmin, -jnp.inf).max(-1) - jnp.where(b["s_mask"], qmin, jnp.inf).min(-1)).mean()
    info.update(actor_loss=la, entropy=ent.mean(), ref_prob=pref.mean(), alpha=alpha, residual_max=rmag.mean(),
                q_gap_feasible=qgap,
                target_entropy=jnp.minimum(prior_entropy(b["s_nfeas"], strength, p0),
                                           ratio * jnp.log(jnp.maximum(b["s_nfeas"], 1.0))).mean())
    return rng, A, C, Ct, T, info
