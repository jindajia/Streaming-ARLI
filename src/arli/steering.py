"""S0: one shared lightweight steering actor called between frozen pi0 action-expert steps.

Per chunk j the decision sequence is (augmented-state MDP):

    s_j --w_j (ARLI initial noise, pi_w)--> z_{j,k1} --u_{k1}--> ... z_{j,kL} --u_{kL}--> s_{j+1}

``z_{j,k} = (o_old, o_lat(k), committed, X_k, tau_k, delta_k)``;  ``u_k in [-b, b]^32`` (tanh-Gaussian);
``X~_k = X_k + alpha(tau_k) * broadcast_H(u_k)`` with ``alpha(tau) = tau`` (a correction expressed in noise
units: X_tau = tau * eps + (1 - tau) * a), then ``X_{k+1} = X~_k + dt * V(X~_k, tau_k | c_old)``, dt = -0.1.
The frozen VLM context c_old is never recomputed; fresh information enters only through the actor.

Transitions (DECISIONS.md D13): physical reward -1 per executed chunk (0 + mask 0 for the chunk executing at
success) is attached to the last steering transition of the chunk (u_{kL} -> s_{j+1}); every internal transition
has reward 0 and mask 1; discounts are gamma ** (elapsed env steps) inside the chunk and the remainder on the last
transition, so each chunk still discounts by gamma ** 25 in total, as in the baseline.

Critics/actors: (Q_w, pi_w) = ARLI networks (warm start); (Q_u, pi_u) = steering networks, shared over k.
Q_w targets bootstrap through (pi_u, Q_u) at z_{j,k1}; Q_u targets bootstrap through (pi_u, Q_u) at the next
steering state or through (pi_w, Q_w) at s_{j+1}.  No entropy bonus in the backup (jaxrl2 default).
"""

from __future__ import annotations

import copy
import dataclasses
import functools
from typing import Any

import flax.linen as nn
import gym.spaces
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import frozen_dict
from jaxrl2.agents.pixel_sac.actor_updater import update_actor
from jaxrl2.agents.pixel_sac.pixel_sac_learner import TrainState
from jaxrl2.agents.pixel_sac.temperature import Temperature
from jaxrl2.agents.pixel_sac.temperature_updater import update_temperature
from jaxrl2.data.augmentations import batched_random_crop, color_transform
from jaxrl2.networks.constants import xavier_init
from jaxrl2.networks.encoders.networks import Encoder
from jaxrl2.networks.learned_std_normal_policy import LearnedStdTanhNormalPolicy
from jaxrl2.networks.values import StateActionEnsemble
from jaxrl2.utils.target_update import soft_target_update

from arli import rl

X_ACT_DIMS = 14  # meaningful ALOHA dims of the 32-padded pi0 action
N_FREQ = 8


@dataclasses.dataclass(frozen=True)
class SteerSpec:
    branch: str  # "B" stale | "C" fresh | "D" last-step (fresh)
    bound: float = 1.0  # |u| <= bound
    xfeat: int = 64

    @property
    def steps(self):
        return frozenset([9]) if self.branch == "D" else frozenset(range(1, 10))

    @property
    def fresh(self):
        return self.branch in ("C", "D")


def z_space(image_size=64):
    base = rl.observation_space("arli", image_size).spaces
    return gym.spaces.Dict({**base,
                            "x_noisy": gym.spaces.Box(-np.inf, np.inf, (50, 32), np.float32),
                            "td": gym.spaces.Box(-np.inf, np.inf, (2,), np.float32)})


W_KEYS = ("pixels", "prev_action", "state")


def time_features(td):
    """sinusoidal embedding of (tau, delta/delay); td: (..., 2) -> (..., 2 * 2 * N_FREQ)."""
    freqs = 2.0 ** jnp.arange(N_FREQ) * jnp.pi
    ang = td[..., :, None] * freqs
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], -1).reshape(*td.shape[:-1], -1)


class SteerMultiplexer(nn.Module):
    """jaxrl2 PixelMultiplexer + X encoder + time features.  Sorted-key flattening puts the ARLI inputs
    (pixels-latent, prev_action, state) first and the new inputs (zt, zx) last, so the ARLI first-layer
    weights can be copied and the new rows zero-initialised."""
    encoder: nn.Module
    network: nn.Module
    latent_dim: int = 50
    xfeat: int = 64

    @nn.compact
    def __call__(self, observations, actions=None, training=False):
        observations = frozen_dict.FrozenDict(observations)
        x = self.encoder(observations["pixels"], training)
        x = nn.Dense(self.latent_dim, kernel_init=xavier_init())(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        xn = observations["x_noisy"][..., :X_ACT_DIMS].reshape(*observations["x_noisy"].shape[:-2], -1)
        zx = nn.tanh(nn.LayerNorm(name="x_ln")(nn.Dense(self.xfeat, kernel_init=xavier_init(), name="x_enc")(xn)))
        feats = {"pixels": x, "prev_action": observations["prev_action"], "state": observations["state"],
                 "zt": time_features(observations["td"]), "zx": zx}
        if actions is None:
            return self.network(feats, training=training)
        return self.network(feats, actions, training=training)


def _encoder():
    k = rl.AGENT_KWARGS
    return Encoder(k["cnn_features"], k["cnn_strides"], k["cnn_padding"])


def make_steer_defs(spec: SteerSpec):
    k = rl.AGENT_KWARGS
    actor = SteerMultiplexer(_encoder(), LearnedStdTanhNormalPolicy(
        k["hidden_dims"], rl.NOISE_DIM, dropout_rate=k["dropout_rate"], low=-spec.bound, high=spec.bound),
        k["latent_dim"], spec.xfeat)
    critic = SteerMultiplexer(_encoder(), StateActionEnsemble(k["hidden_dims"], num_qs=k["num_qs"]),
                              k["latent_dim"], spec.xfeat)
    return actor, critic


def _pad_first_dense(new_kernel, old_kernel, row_map):
    """new_kernel[rows_new] = old_kernel[rows_old]; everything else zero."""
    out = jnp.zeros_like(new_kernel)
    for (a_new, b_new), (a_old, b_old) in row_map:
        out = out.at[a_new:b_new].set(old_kernel[a_old:b_old])
    return out


def warm_start_params(new, old, n_old_in: int, n_new_in: int, critic: bool, n_action: int = 32,
                      zero_action_rows: bool = True):
    """Copy ARLI encoder / bottleneck / MLP weights into the steering network.

    Input layout of the first MLP Dense: actor [base(n_old_in) | zt | zx];
    critic [actions(32) | base | zt | zx].  New rows (and, for the critic, the action rows) start at zero.
    Output heads of the actor are left at their fresh initialisation (different action semantics).
    """
    new = jax.tree_util.tree_map(lambda x: x, frozen_dict.unfreeze(new))
    old = frozen_dict.unfreeze(old)
    new["encoder"] = copy.deepcopy(old["encoder"])
    new["Dense_0"] = copy.deepcopy(old["Dense_0"])
    new["LayerNorm_0"] = copy.deepcopy(old["LayerNorm_0"])
    if critic:
        o_mlp = old["network"]["VmapStateActionValue_0"]["MLP_0"]
        n_mlp = new["network"]["VmapStateActionValue_0"]["MLP_0"]
        base = n_old_in - n_action
        rows = [((n_action, n_action + base), (n_action, n_action + base))]
        if not zero_action_rows:
            rows.append(((0, n_action), (0, n_action)))
        for name in o_mlp:
            if name == "Dense_0":
                n_mlp[name]["kernel"] = jax.vmap(lambda nk, ok: _pad_first_dense(nk, ok, rows))(
                    n_mlp[name]["kernel"], o_mlp[name]["kernel"])
                n_mlp[name]["bias"] = o_mlp[name]["bias"]
            else:
                n_mlp[name] = copy.deepcopy(o_mlp[name])
    else:
        o_mlp = old["network"]["MLP_0"]
        n_mlp = new["network"]["MLP_0"]
        for name in o_mlp:
            if name == "Dense_0":
                n_mlp[name]["kernel"] = _pad_first_dense(n_mlp[name]["kernel"], o_mlp[name]["kernel"],
                                                         [((0, n_old_in), (0, n_old_in))])
                n_mlp[name]["bias"] = o_mlp[name]["bias"]
            else:
                n_mlp[name] = copy.deepcopy(o_mlp[name])
    return frozen_dict.freeze(new)


def count_params(tree) -> int:
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(tree)))


# ------------------------------------------------------------------------------------------------
def _augment(rng, pixels):
    """jaxrl2 PixelSAC augmentation for a 2-camera (6-channel) stack: random crop, same color jitter per cam."""
    rng, k1, k2 = jax.random.split(rng, 3)
    p = batched_random_crop(k1, pixels)
    for i in range(2):
        p = p.at[:, :, :, i * 3:(i + 1) * 3].set(
            (color_transform(k2, p[:, :, :, i * 3:(i + 1) * 3].astype(jnp.float32) / 255.) * 255).astype(jnp.uint8))
    return rng, p


def _with_pixels(obs, p):
    return obs.copy(add_or_replace={"pixels": p})


def _wsub(obs):
    return frozen_dict.FrozenDict({k: obs[k] for k in W_KEYS})


@functools.partial(jax.jit, static_argnames=("has_w",))
def update_step(rng, Aw, Cw, Cw_t, Tw, Au, Cu, Cu_t, Tu, bw, bu, tau_polyak, target_entropy, has_w=True):
    info = {}
    # ---- augmentation ---------------------------------------------------------------------------
    rng, p = _augment(rng, bu["observations"]["pixels"])
    bu = bu.copy(add_or_replace={"observations": _with_pixels(bu["observations"], p)})
    rng, p = _augment(rng, bu["next_observations"]["pixels"])
    bu = bu.copy(add_or_replace={"next_observations": _with_pixels(bu["next_observations"], p)})
    if has_w:
        rng, p = _augment(rng, bw["observations"]["pixels"])
        bw = bw.copy(add_or_replace={"observations": _with_pixels(bw["observations"], p)})
        rng, p = _augment(rng, bw["next_observations"]["pixels"])
        bw = bw.copy(add_or_replace={"next_observations": _with_pixels(bw["next_observations"], p)})

    def q_next_u(key, nobs):
        a, _ = Au.apply_fn({"params": Au.params}, nobs).sample_and_log_prob(seed=key)
        return Cu.apply_fn({"params": Cu_t}, nobs, a).mean(0)

    def q_next_w(key, nobs):
        a, _ = Aw.apply_fn({"params": Aw.params}, nobs).sample_and_log_prob(seed=key)
        return Cw.apply_fn({"params": Cw_t}, nobs, a).mean(0)

    # ---- steering critic Q_u ----------------------------------------------------------------------
    rng, k1, k2 = jax.random.split(rng, 3)
    nq = jnp.where(bu["next_is_w"] > 0.5, q_next_w(k1, _wsub(bu["next_observations"])),
                   q_next_u(k2, bu["next_observations"]))
    target_u = bu["rewards"] + bu["discount"] * bu["masks"] * nq

    def loss_u(params):
        qs = Cu.apply_fn({"params": params}, bu["observations"], bu["actions"])
        return ((qs - target_u) ** 2).mean(), qs.mean()

    (lu, qu), g = jax.value_and_grad(loss_u, has_aux=True)(Cu.params)
    Cu = Cu.apply_gradients(grads=g)
    Cu_t = soft_target_update(Cu.params, Cu_t, tau_polyak)
    info.update(critic_u_loss=lu, q_u=qu, target_u=target_u.mean())

    # ---- noise critic Q_w (next state is always a steering state) --------------------------------
    if has_w:
        rng, k1 = jax.random.split(rng)
        target_w = bw["rewards"] + bw["discount"] * bw["masks"] * q_next_u(k1, bw["next_observations"])

        def loss_w(params):
            qs = Cw.apply_fn({"params": params}, bw["observations"], bw["actions"])
            return ((qs - target_w) ** 2).mean(), qs.mean()

        (lw, qw), g = jax.value_and_grad(loss_w, has_aux=True)(Cw.params)
        Cw = Cw.apply_gradients(grads=g)
        Cw_t = soft_target_update(Cw.params, Cw_t, tau_polyak)
        info.update(critic_w_loss=lw, q_w=qw, target_w=target_w.mean())

    # ---- actors + temperatures (jaxrl2 updaters) ---------------------------------------------------
    rng, k1, k2 = jax.random.split(rng, 3)
    Au, ai = update_actor(k1, Au, Cu, Tu, bu, critic_reduction="mean")
    Tu, ti = update_temperature(Tu, ai["entropy"], target_entropy)
    info.update(actor_u_loss=ai["actor_loss"], entropy_u=ai["entropy"], temp_u=ti["temperature"])
    if has_w:
        # jaxrl2's update_actor also evaluates the actor on next_observations (logging only); for W-batches those
        # are steering states, so hand it the current observations instead.
        bw_a = bw.copy(add_or_replace={"next_observations": bw["observations"]})
        Aw, ai = update_actor(k2, Aw, Cw, Tw, bw_a, critic_reduction="mean")
        Tw, ti = update_temperature(Tw, ai["entropy"], target_entropy)
        info.update(actor_w_loss=ai["actor_loss"], entropy_w=ai["entropy"], temp_w=ti["temperature"])
    return rng, Aw, Cw, Cw_t, Tw, Au, Cu, Cu_t, Tu, info


class SteeringLearner:
    """Holds (pi_w, Q_w) warm-started from an ARLI PixelSACLearner and the new shared (pi_u, Q_u)."""

    def __init__(self, arli_agent, spec: SteerSpec, seed: int, log_std_init: float = float(np.log(0.3))):
        self.spec = spec
        k = rl.AGENT_KWARGS
        self.Aw, self.Cw, self.Cw_t, self.Tw = (arli_agent._actor, arli_agent._critic,
                                                arli_agent._target_critic_params, arli_agent._temp)
        rng = jax.random.PRNGKey(seed + 7919)
        rng, ka, kc = jax.random.split(rng, 3)
        a_def, c_def = make_steer_defs(spec)
        sample = {kk: s.sample()[None] for kk, s in z_space().spaces.items()}
        act = np.zeros((1, rl.NOISE_DIM), np.float32)
        ap = a_def.init(ka, sample)["params"]
        cp = c_def.init(kc, sample, act)["params"]
        n_old_actor = 50 + 20 * 14 + 28
        ap = warm_start_params(ap, self.Aw.params, n_old_actor, None, critic=False)
        cp = warm_start_params(cp, self.Cw.params, rl.NOISE_DIM + n_old_actor, None, critic=True)
        ap = frozen_dict.unfreeze(ap)  # conservative exploration at start: log_std bias
        ap["network"]["Dense_1"]["bias"] = jnp.full_like(ap["network"]["Dense_1"]["bias"], log_std_init)
        ap = frozen_dict.freeze(ap)
        self.Au = TrainState.create(apply_fn=a_def.apply, params=ap, tx=optax.adam(k["actor_lr"]), batch_stats=None)
        self.Cu = TrainState.create(apply_fn=c_def.apply, params=cp, tx=optax.adam(k["critic_lr"]), batch_stats=None)
        self.Cu_t = cp
        temp_def = Temperature(1.0)
        tp = temp_def.init(jax.random.PRNGKey(0))["params"]
        tp = jax.tree_util.tree_map(lambda x: jnp.asarray(self.Tw.params["log_temp"]).reshape(x.shape), tp)
        self.Tu = TrainState.create(apply_fn=temp_def.apply, params=tp, tx=optax.adam(k["temp_lr"]), batch_stats=None)
        self.rng = rng
        self.target_entropy = -rl.NOISE_DIM / 2
        self.tau = k["tau"]
        self.n_params = dict(actor_u=count_params(self.Au.params), critic_u=count_params(self.Cu.params),
                             actor_w=count_params(self.Aw.params), critic_w=count_params(self.Cw.params))

    def update(self, bw, bu):
        (self.rng, self.Aw, self.Cw, self.Cw_t, self.Tw, self.Au, self.Cu, self.Cu_t, self.Tu, info) = update_step(
            self.rng, self.Aw, self.Cw, self.Cw_t, self.Tw, self.Au, self.Cu, self.Cu_t, self.Tu, bw, bu,
            self.tau, self.target_entropy, has_w=bw is not None)
        return info

    def sample_u(self, z: dict, key):
        return np.asarray(rl._sample(key, self.Au.apply_fn, self.Au.params, rl.batchify(z)))[0]

    def sample_w(self, obs: dict, key):
        return np.asarray(rl._sample(key, self.Aw.apply_fn, self.Aw.params, rl.batchify(obs)))[0]

    def state(self):
        return dict(Aw=self.Aw, Cw=self.Cw, Cw_t=self.Cw_t, Tw=self.Tw, Au=self.Au, Cu=self.Cu, Cu_t=self.Cu_t,
                    Tu=self.Tu, rng=self.rng)

    def load(self, sd):
        for k_, v in sd.items():
            setattr(self, k_, v)


# ------------------------------------------------------------------------------------------------
class SteeringController:
    """rollout controller: pi_w picks initial noise; pi_u corrects X at the branch's expert steps."""

    def __init__(self, learner: SteeringLearner, sched, key, record_all: bool = False):
        self.L, self.sched, self.key = learner, sched, key
        self.steering = self  # rollout looks for .steering
        self.steps = learner.spec.steps

    def _next_key(self):
        self.key, k = jax.random.split(self.key)
        return k

    def initial_noise(self, ctx):
        from arli.controllers import broadcast_noise, rl_obs_from_ctx
        obs = rl_obs_from_ctx("arli", ctx)
        w = np.reshape(self.L.sample_w(obs, self._next_key()), (1, rl.NOISE_DIM)).astype(np.float32)
        return broadcast_noise(w), dict(obs=obs, action=w)

    def correct(self, ctx, k, x, tau, latest):
        if self.L.spec.fresh:
            img, q, t_obs = latest["img"], latest["q"], latest["t"]
        else:  # B: hold the observation read at the initial-noise call
            img, q, t_obs = ctx["img_mid"], ctx["q_mid"], ctx["t_mid"]
        j = ctx["j"]
        t_now = latest["t"]
        delta = (self.sched.T(j) - t_now) / self.sched.delay if j > 0 else 0.0
        z = rl.make_rl_obs("arli", ctx["img_old"], ctx["q_old"], ctx["committed"], img, q)
        z["x_noisy"] = np.asarray(x, np.float32)[0]
        z["td"] = np.asarray([tau, delta], np.float32)
        u = np.reshape(self.L.sample_u(z, self._next_key()), (1, rl.NOISE_DIM)).astype(np.float32)
        x_tilde = np.asarray(x, np.float32) + tau * np.repeat(u[None], 50, axis=1)
        return x_tilde, dict(obs=z, action=u, t_obs=t_obs, t_now=t_now, fresh_frame=bool(t_obs > ctx["t_mid"]),
                             u_norm=float(np.linalg.norm(u)), dx_norm=float(np.linalg.norm(x_tilde - x)))


def steering_transitions(res, sched, gamma):
    """Returns (w_transitions, u_transitions) with the credit assignment described in the module docstring."""
    js = res["executed_decisions"]
    dec = res["decisions"]
    W, U = [], []
    for i, j in enumerate(js):
        last = i == len(js) - 1
        r, mask = (0.0, 0.0) if (last and res["success"]) else (-1.0, 1.0)
        steer = dec[j]["steer"]
        assert len(steer) > 0
        t_dec = [sched.m(j)] + [s["t_now"] for s in steer]
        w_obs = dec[j]["rec"]["obs"]
        W.append(dict(obs=w_obs, action=dec[j]["rec"]["action"], reward=0.0, mask=1.0,
                      discount=gamma ** (t_dec[1] - t_dec[0]), next_obs=steer[0]["obs"]))
        used = t_dec[-1] - t_dec[0]
        for i_s, s in enumerate(steer):
            if i_s + 1 < len(steer):
                U.append(dict(obs=s["obs"], action=s["action"], reward=0.0, mask=1.0,
                              discount=gamma ** (t_dec[i_s + 2] - t_dec[i_s + 1]), next_obs=steer[i_s + 1]["obs"],
                              next_is_w=0.0))
            else:
                nxt = dec[j + 1]["rec"]["obs"] if (j + 1) in dec else w_obs
                nz = dict(nxt, x_noisy=np.zeros((50, 32), np.float32), td=np.zeros(2, np.float32))
                U.append(dict(obs=s["obs"], action=s["action"], reward=r, mask=mask,
                              discount=gamma ** (sched.n_exec - used), next_obs=nz, next_is_w=1.0))
    return W, U
