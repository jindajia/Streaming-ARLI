"""DEHP horizon learner: categorical actor over h, distributional (201-atom, [0,1]) state-value critic, duration-aware
GAE and clipped PPO.

Paper-specified (DEHP, arXiv 2606.11408v2, Sec. 3-4, Table 3): categorical pi_len over h, uniform initialisation,
actor input = state ++ flattened chunk, critic V(s) state-only, 201 atoms on [0,1] with two-hot projection +
cross-entropy, delta_k = R_k + gamma^{h_k} V(s_{k+1}) - V(s_k), A_k = sum_l (gamma*lambda)^{t_{k+l}-t_k} delta_{k+l}
(i.e. A_k = delta_k + (gamma*lambda)^{D_k} A_{k+1}), value target V_old(s_k) + A_k, PPO clip 0.05, DEHP LR 1e-4, critic
warm-up before actor updates.
Pilot choices (not paper-specified): separate 2x256 ReLU MLPs; critic LR 3e-4 (paper 1e-3); 4 epochs; one full-batch
minibatch; per-batch advantage normalisation; no entropy bonus; global-norm gradient clip 1.0; zero bootstrap at
success and at the 400-step deadline; bootstrap with V(s') at a non-terminal rollout-buffer cut.
"""

from __future__ import annotations

import functools

import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

N_ATOMS = 201
ATOMS = jnp.linspace(0.0, 1.0, N_ATOMS)
H = 50


class MLP(nn.Module):
    out: int
    width: int = 256
    zero_last: bool = True

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.width)(x))
        x = nn.relu(nn.Dense(self.width)(x))
        init = nn.initializers.zeros if self.zero_last else nn.initializers.lecun_normal()
        return nn.Dense(self.out, kernel_init=init, bias_init=nn.initializers.zeros)(x)


def actor_input(s):
    return jnp.concatenate([s["phys"], s["plan"].reshape(*s["plan"].shape[:-2], -1), s["plan_mask"], s["meta"]], -1)


def critic_input(s, with_plan):
    if with_plan:  # run B: async adaptation -- installed plan + timing metadata are part of the scheduler state
        return actor_input(s)
    return s["phys"]  # run A: paper's state-only critic


def masked_logits(logits, hmask):
    return jnp.where(hmask, logits, -1e9)


def two_hot(target):
    """Project scalar targets in [0,1] onto ATOMS (linear interpolation between the two straddling atoms)."""
    target = jnp.clip(target, 0.0, 1.0)
    pos = target * (N_ATOMS - 1)
    lo = jnp.floor(pos).astype(jnp.int32)
    hi = jnp.minimum(lo + 1, N_ATOMS - 1)
    w_hi = pos - lo
    p = jax.nn.one_hot(lo, N_ATOMS) * (1 - w_hi)[..., None] + jax.nn.one_hot(hi, N_ATOMS) * w_hi[..., None]
    return p


def value_from_logits(logits):
    return (jax.nn.softmax(logits, -1) * ATOMS).sum(-1)


def duration_gae(G, D, terminal, v, v_next, gamma, lam):
    """Duration-aware GAE (DEHP Eqs. 5-6) for one contiguous segment of decisions.
    G: discounted in-interval rewards; D: actual durations; terminal: 1 at success / deadline; v: V(s_k);
    v_next: V(s_{k+1}) (ignored where terminal; at a non-terminal buffer cut it is the bootstrap value).
    Decisions must be in time order within each episode; an episode boundary is always terminal."""
    n = len(G)
    adv = np.zeros(n, np.float64)
    nxt = 0.0
    for k in reversed(range(n)):
        cont = 0.0 if terminal[k] else 1.0
        delta = G[k] + cont * (gamma ** D[k]) * v_next[k] - v[k]
        # advance the recursion only within an episode: a non-terminal successor is decision k+1 unless k is the cut
        nxt = delta + cont * ((gamma * lam) ** D[k]) * (nxt if k + 1 < n and not terminal[k] else 0.0)
        adv[k] = nxt
    return adv


class Learner:
    def __init__(self, seed, actor_in, critic_in, critic_with_plan, actor_lr=1e-4, critic_lr=3e-4, clip=0.05,
                 grad_clip=1.0):
        self.actor, self.critic = MLP(H), MLP(N_ATOMS)
        k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
        self.pa = self.actor.init(k1, jnp.zeros((1, actor_in)))
        self.pc = self.critic.init(k2, jnp.zeros((1, critic_in)))
        self.oa = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(actor_lr))
        self.oc = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(critic_lr))
        self.sa, self.sc = self.oa.init(self.pa), self.oc.init(self.pc)
        self.clip, self.with_plan = clip, critic_with_plan
        self.key = jax.random.PRNGKey(seed + 1)
        self._build()

    def _build(self):
        actor, critic, wp, clip = self.actor, self.critic, self.with_plan, self.clip

        @jax.jit
        def policy(pa, s, key):
            lg = masked_logits(actor.apply(pa, actor_input(s)), s["hmask"])
            a = jax.random.categorical(key, lg)
            logp = jax.nn.log_softmax(lg)
            return a, jnp.take_along_axis(logp, a[..., None], -1)[..., 0], jnp.argmax(lg, -1), logp

        @jax.jit
        def value(pc, s):
            return value_from_logits(critic.apply(pc, critic_input(s, wp)))

        def critic_loss(pc, s, target):
            lg = critic.apply(pc, critic_input(s, wp))
            ce = -(two_hot(target) * jax.nn.log_softmax(lg)).sum(-1)
            return ce.mean(), value_from_logits(lg)

        def actor_loss(pa, s, a, old_logp, adv):
            lg = masked_logits(actor.apply(pa, actor_input(s)), s["hmask"])
            logp_all = jax.nn.log_softmax(lg)
            logp = jnp.take_along_axis(logp_all, a[:, None], -1)[:, 0]
            ratio = jnp.exp(logp - old_logp)
            l = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip, 1 + clip) * adv).mean()
            p = jnp.exp(logp_all)
            ent = -(jnp.where(s["hmask"], p * logp_all, 0.0)).sum(-1).mean()
            approx_kl = ((ratio - 1) - (logp - old_logp)).mean()
            clipfrac = (jnp.abs(ratio - 1) > clip).mean()
            return l, dict(entropy=ent, approx_kl=approx_kl, clipfrac=clipfrac)

        @jax.jit
        def c_update(pc, sc, s, target):
            (l, v), g = jax.value_and_grad(critic_loss, has_aux=True)(pc, s, target)
            u, sc = self.oc.update(g, sc, pc)
            return optax.apply_updates(pc, u), sc, l, optax.global_norm(g), v

        @jax.jit
        def a_update(pa, sa, s, a, old_logp, adv):
            (l, info), g = jax.value_and_grad(actor_loss, has_aux=True)(pa, s, a, old_logp, adv)
            u, sa = self.oa.update(g, sa, pa)
            return optax.apply_updates(pa, u), sa, l, optax.global_norm(g), info

        self._policy, self._value, self._c_update, self._a_update = policy, value, c_update, a_update

    # --- acting ---------------------------------------------------------------------------------
    def act(self, s, deterministic=False):
        sb = {k: v[None] for k, v in s.items() if k in ("phys", "plan", "plan_mask", "meta", "hmask")}
        self.key, k = jax.random.split(self.key)
        a, logp, am, logp_all = self._policy(self.pa, sb, k)
        idx = int(am[0]) if deterministic else int(a[0])
        return idx + 1, float(logp_all[0, idx]), np.asarray(jnp.exp(logp_all[0]))

    def values(self, batch_states):
        return np.asarray(self._value(self.pc, batch_states))

    # --- updates --------------------------------------------------------------------------------
    def critic_warmup(self, S, targets, epochs):
        out = []
        for _ in range(epochs):
            self.pc, self.sc, l, gn, _ = self._c_update(self.pc, self.sc, S, targets)
            out.append((float(l), float(gn)))
        return out

    def update(self, S, a_idx, old_logp, adv, targets, epochs):
        logs = []
        for _ in range(epochs):
            self.pc, self.sc, lc, gc, v = self._c_update(self.pc, self.sc, S, targets)
            self.pa, self.sa, la, ga, info = self._a_update(self.pa, self.sa, S, a_idx, old_logp, adv)
            logs.append(dict(critic_loss=float(lc), critic_grad_norm=float(gc), actor_loss=float(la),
                             actor_grad_norm=float(ga), **{k: float(v_) for k, v_ in info.items()}))
        return logs

    def state_bytes(self):
        return flax.serialization.to_bytes(dict(pa=self.pa, pc=self.pc, sa=self.sa, sc=self.sc, key=self.key))

    def load_bytes(self, b):
        d = flax.serialization.from_bytes(dict(pa=self.pa, pc=self.pc, sa=self.sa, sc=self.sc, key=self.key), b)
        self.pa, self.pc, self.sa, self.sc, self.key = d["pa"], d["pc"], d["sa"], d["sc"], d["key"]

    def finite(self):
        return all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves((self.pa, self.pc)))
