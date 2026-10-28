"""Single-actor denoising-step steering (two peer methods), trained from RL step zero.

Methods (both keep pi0's VLM and action expert frozen; 10 Euler denoising iterations k = 0..9):

* ``shared_every_step``: ONE actor pi_theta (one parameter set, one output head) is called immediately before every
  iteration k = 0..9 of every chunk (10 calls / chunk).
    - k = 0: there is no partially denoised trajectory yet.  Input placeholder X = 0 (50 x 32) and the "first call"
      indicator f = 1 (with tau = 1).  The output a in [-1, 1]^32 SELECTS THE INITIAL NOISE:
      X_0 = broadcast_50(W0_SCALE * a), W0_SCALE = 2 (DSRL/ARLI noise bound b_w = 2).
    - k >= 1: input X = X_k (the actual current trajectory), f = 0.  The output is a RESIDUAL CORRECTION:
      X~_k = X_k + C_SCALE * broadcast_50(a), C_SCALE = 0.2 (constant in k).
* ``final_step_only``: ONE actor called only before iteration k = 9 (1 call / chunk).  The initial noise is the
  frozen pi0 sampling convention, X_0 ~ N(0, I) of shape (50, 32) i.i.d.; iterations 0..8 are unsteered; the actor
  applies the same correction X~_9 = X_9 + C_SCALE * broadcast_50(a).

Then X_{k+1} = X~_k + dt * V(X~_k, tau_k | c_old), dt = -0.1, tau_k = 1 - 0.1 k, with c_old the VLM context of the
inference-start observation (never recomputed).  Corresponding iterations use the identical correction
parameterisation, bound and scaling in both methods.

Actor/critic input z (same architecture for both methods):
  pixels   = cam_high(o_s) ++ cam_high(o_t)   64x64x6 (old observation at inference start s_j; newest frame at the call)
  state    = q(o_s) ++ q(o_t)                 28
  prev_action = committed actions [s_j, T_j)  20 x 14
  x_noisy  = X_k (zeros at the k = 0 call)    50 x 32 (first 14 dims encoded)
  td       = (tau_k, delta_k, f)              delta = (T_j - t) / t_delay (0 for the startup chunk), f = first-call
Call time t = m_j + k for chunk j >= 1 (m_j = T_j - 10, one expert iteration per env step); the startup chunk j = 0 is
computed with the clock frozen (all calls see o_0).  See DECISIONS.md D19-D22.

Learning objective (soft actor-critic on decisions, chunk-level bootstrap): every actor call d = (j, k) of an
executed chunk j is a transition with
    target(d) = r_j + gamma^(dt_d) * mask_j * mean_i Qbar_i(z', a'),   a' ~ pi_theta(z'),
where z' is the input of the FIRST actor call of chunk j+1 (k = 0 for shared_every_step, k = 9 for final_step_only),
r_j = -1 per executed chunk (0 with mask 0 for the chunk executing at success: DSRL convention), and dt_d = physical
env steps from call d to that next call, so the chunk-level discount stays gamma^25 as in ARLI.  Each physical
reward is counted once per chunk and shared by the chunk's decisions.  final_step_only has one decision per chunk,
so this is exactly ARLI's chunk-level target.  For shared_every_step it is an n-step-style target: the later
corrections of the same chunk that lie between d and z' come from the behaviour policy in the replay (off-policy
bias, accepted), and the actor improves each call against Q given those later corrections.  No Q -> Q hop inside a
chunk (the "chain" formulation diverged in the superseded S1 experiments).
"""

from __future__ import annotations

import dataclasses
import functools
import os
import pickle

import flax.serialization
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
from jaxrl2.networks.learned_std_normal_policy import LearnedStdTanhNormalPolicy
from jaxrl2.networks.values import StateActionEnsemble
from jaxrl2.utils.target_update import soft_target_update

from arli import rl
from arli.steering import SteerMultiplexer, _augment, _encoder, count_params

METHODS = ("shared_every_step", "final_step_only")
W0_SCALE = 2.0  # initial-noise scale for the k = 0 call (= DSRL/ARLI b_w)
C_SCALE = 0.2  # residual-correction scale for k >= 1 (fixed implementation choice; not validated)
K_STEPS = 10


def call_steps(method):
    return tuple(range(K_STEPS)) if method == "shared_every_step" else (K_STEPS - 1,)


def z_space(image_size=64):
    base = rl.observation_space("arli", image_size).spaces
    return gym.spaces.Dict({**base,
                            "x_noisy": gym.spaces.Box(-np.inf, np.inf, (50, 32), np.float32),
                            "td": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32)})


def make_z(ctx, latest_img, latest_q, x, tau, delta, first):
    z = rl.make_rl_obs("arli", ctx["img_old"], ctx["q_old"], ctx["committed"], latest_img, latest_q)
    z["x_noisy"] = np.zeros((50, 32), np.float32) if x is None else np.asarray(x, np.float32).reshape(50, 32)
    z["td"] = np.asarray([tau, delta, 1.0 if first else 0.0], np.float32)
    return z


def make_defs():
    k = rl.AGENT_KWARGS
    actor = SteerMultiplexer(_encoder(), LearnedStdTanhNormalPolicy(
        k["hidden_dims"], rl.NOISE_DIM, dropout_rate=k["dropout_rate"], low=-1.0, high=1.0), k["latent_dim"], 64)
    critic = SteerMultiplexer(_encoder(), StateActionEnsemble(k["hidden_dims"], num_qs=k["num_qs"]),
                              k["latent_dim"], 64)
    return actor, critic


# ------------------------------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnames=())
def update_step(rng, A, Cr, Ct, T, batch, tau_polyak, target_entropy):
    rng, p = _augment(rng, batch["observations"]["pixels"])
    batch = batch.copy(add_or_replace={"observations": batch["observations"].copy(add_or_replace={"pixels": p})})
    rng, p = _augment(rng, batch["next_observations"]["pixels"])
    batch = batch.copy(add_or_replace={"next_observations": batch["next_observations"].copy(add_or_replace={"pixels": p})})
    rng, k1, k2 = jax.random.split(rng, 3)
    a2, _ = A.apply_fn({"params": A.params}, batch["next_observations"]).sample_and_log_prob(seed=k1)
    nq = Cr.apply_fn({"params": Ct}, batch["next_observations"], a2).mean(0)
    target = batch["rewards"] + batch["discount"] * batch["masks"] * nq

    def loss(params):
        qs = Cr.apply_fn({"params": params}, batch["observations"], batch["actions"])
        return ((qs - target) ** 2).mean(), qs

    (lc, qs), g = jax.value_and_grad(loss, has_aux=True)(Cr.params)
    Cr = Cr.apply_gradients(grads=g)
    Ct = soft_target_update(Cr.params, Ct, tau_polyak)
    params_before = A.params
    A, ai = update_actor(k2, A, Cr, T, batch, critic_reduction="mean")
    T, ti = update_temperature(T, ai["entropy"], target_entropy)
    actor_step = optax.global_norm(jax.tree_util.tree_map(lambda a, b: a - b, A.params, params_before))
    info = dict(critic_loss=lc, q=qs.mean(), q_max=qs.max(), target_q=target.mean(), actor_loss=ai["actor_loss"],
                entropy=ai["entropy"], temperature=ti["temperature"], actor_param_step=actor_step)
    return rng, A, Cr, Ct, T, info


class SingleActorLearner:
    """Fresh actor / 10-critic ensemble / temperature (ARLI Table 3 hyperparameters); no warm start."""

    def __init__(self, seed: int):
        k = rl.AGENT_KWARGS
        rng = jax.random.PRNGKey(seed)
        rng, ka, kc = jax.random.split(rng, 3)
        a_def, c_def = make_defs()
        sample = {kk: s.sample()[None] for kk, s in z_space().spaces.items()}
        ap = a_def.init(ka, sample)["params"]
        cp = c_def.init(kc, sample, np.zeros((1, rl.NOISE_DIM), np.float32))["params"]
        self.A = TrainState.create(apply_fn=a_def.apply, params=ap, tx=optax.adam(k["actor_lr"]), batch_stats=None)
        self.C = TrainState.create(apply_fn=c_def.apply, params=cp, tx=optax.adam(k["critic_lr"]), batch_stats=None)
        self.Ct = cp
        tdef = Temperature(1.0)
        self.T = TrainState.create(apply_fn=tdef.apply, params=tdef.init(jax.random.PRNGKey(0))["params"],
                                   tx=optax.adam(k["temp_lr"]), batch_stats=None)
        self.rng = rng
        self.target_entropy = -rl.NOISE_DIM / 2
        self.tau = k["tau"]
        self.n_params = dict(actor=count_params(ap), critic_ensemble=count_params(cp))

    def update(self, batch):
        self.rng, self.A, self.C, self.Ct, self.T, info = update_step(
            self.rng, self.A, self.C, self.Ct, self.T, batch, self.tau, self.target_entropy)
        return info

    def sample(self, z, key):
        return np.asarray(rl._sample(key, self.A.apply_fn, self.A.params, rl.batchify(z)))[0]

    def state_bytes(self):
        return flax.serialization.to_bytes(dict(A=self.A, C=self.C, Ct=self.Ct, T=self.T, rng=self.rng))

    def load_bytes(self, blob):
        sd = flax.serialization.from_bytes(dict(A=self.A, C=self.C, Ct=self.Ct, T=self.T, rng=self.rng), blob)
        self.A, self.C, self.Ct, self.T, self.rng = sd["A"], sd["C"], sd["Ct"], sd["T"], sd["rng"]


# ------------------------------------------------------------------------------------------------
class Controller:
    """rollout controller for both methods.  mode: 'train' | 'eval' (actor sample) | 'prefill' (Gaussian action)."""

    def __init__(self, method, learner, sched, key, noise_rng, mode="train", prefill_std=0.3):
        assert method in METHODS
        self.method, self.L, self.sched, self.key, self.noise_rng = method, learner, sched, key, noise_rng
        self.mode, self.prefill_std = mode, prefill_std
        self.steering = self
        self.steps = frozenset(k for k in call_steps(method) if k >= 1)
        self.calls = []  # (j, k, t_obs, t_now) for every actor call, for audits

    def _act(self, z):
        if self.mode == "prefill":
            # ARLI Table 6 prefill: initial noise ~ N(0, 0.3^2) -> actor-output units a = w / W0_SCALE; the same
            # action-space distribution is used for every call so all calls share one prefill distribution.
            return (self.prefill_std / W0_SCALE * self.noise_rng.standard_normal((1, rl.NOISE_DIM))).astype(np.float32)
        self.key, k = jax.random.split(self.key)
        return np.reshape(self.L.sample(z, k), (1, rl.NOISE_DIM)).astype(np.float32)

    def _delta(self, j, t):
        return (self.sched.T(j) - t) / self.sched.delay if j > 0 else 0.0

    def initial_noise(self, ctx):
        j = ctx["j"]
        if self.method == "final_step_only":  # frozen pi0 convention, also during prefill and evaluation
            return self.noise_rng.standard_normal((1, 50, rl.NOISE_DIM)).astype(np.float32), dict(calls=[])
        z = make_z(ctx, ctx["img_mid"], ctx["q_mid"], None, 1.0, self._delta(j, ctx["t_mid"]), first=True)
        a = self._act(z)
        self.calls.append((j, 0, ctx["t_mid"], ctx["t_mid"]))
        rec = dict(calls=[dict(k=0, obs=z, action=a, t_obs=ctx["t_mid"], t_now=ctx["t_mid"])])
        return np.repeat((W0_SCALE * a)[None], 50, axis=1).astype(np.float32), rec

    def correct(self, ctx, k, x, tau, latest):
        j = ctx["j"]
        z = make_z(ctx, latest["img"], latest["q"], np.asarray(x)[0], tau, self._delta(j, latest["t"]), first=False)
        a = self._act(z)
        self.calls.append((j, k, latest["t"], latest["t"]))
        x_tilde = np.asarray(x, np.float32) + C_SCALE * np.repeat(a[None], 50, axis=1)
        return x_tilde, dict(obs=z, action=a, t_obs=latest["t"], t_now=latest["t"],
                             fresh_frame=bool(latest["t"] > ctx["t_mid"]), dx_norm=float(np.linalg.norm(x_tilde - x)))


def decision_list(dec_j):
    """All actor calls of one chunk in call order: [(k, obs, action, t_now), ...]."""
    out = [(c["k"], c["obs"], c["action"], c["t_now"]) for c in dec_j["rec"].get("calls", [])]
    out += [(s["k"], s["obs"], s["action"], s["t_now"]) for s in dec_j["steer"]]
    return sorted(out, key=lambda d: d[0])


def transitions(res, sched, gamma):
    """Chunk-level-bootstrap decision transitions (module docstring).  Returns list of dicts with 'next' = the
    first call of chunk j+1 (as an obs dict) or None if chunk j is terminal / no successor exists."""
    js = res["executed_decisions"]
    dec = res["decisions"]
    out = []
    for i, j in enumerate(js):
        last = i == len(js) - 1
        r, mask = (0.0, 0.0) if (last and res["success"]) else (-1.0, 1.0)
        calls = decision_list(dec[j])
        nxt = decision_list(dec[j + 1]) if (j + 1) in dec else []
        if mask == 1.0:
            assert nxt, f"non-terminal chunk {j} has no recorded successor call"
        nxt_obs, nxt_t = (nxt[0][1], nxt[0][3]) if nxt else (None, None)
        for k, z, a, t_now in calls:
            elapsed = (t_now - sched.m(j)) if j > 0 else 0
            out.append(dict(j=j, k=k, obs=z, action=a, reward=r, mask=mask,
                            discount=gamma ** (sched.n_exec - elapsed), next_obs=nxt_obs))
    return out


# ------------------------------------------------------------------------------------------------
class DecisionReplay:
    """Append-only decision store.  next_obs of every decision of chunk j is stored once (row of the successor's
    first call); checkpoints write only rows added since the previous checkpoint (part files)."""

    KEYS = ("pixels", "state", "prev_action", "x_noisy", "td")

    def __init__(self, capacity, seed):
        sp = z_space().spaces
        dt = dict(x_noisy=np.float16)
        self.obs = {k: np.empty((capacity, *sp[k].shape), dt.get(k, sp[k].dtype)) for k in self.KEYS}
        self.act = np.empty((capacity, rl.NOISE_DIM), np.float32)
        self.rew = np.empty((capacity,), np.float32)
        self.mask = np.empty((capacity,), np.float32)
        self.disc = np.empty((capacity,), np.float32)
        self.next = np.empty((capacity,), np.int64)  # row of next obs, -1 = none (terminal)
        self.trainable = np.zeros((capacity,), bool)
        self.size, self.saved, self.capacity = 0, 0, capacity
        self.rng = np.random.default_rng(seed)
        self._train_idx = []

    def _put(self, z, a=None, r=0.0, m=0.0, d=0.0, nxt=-1, trainable=False):
        i = self.size
        if i >= self.capacity:
            raise RuntimeError("replay full")
        for k in self.KEYS:
            self.obs[k][i] = z[k]
        self.act[i] = 0 if a is None else np.reshape(a, -1)
        self.rew[i], self.mask[i], self.disc[i], self.next[i], self.trainable[i] = r, m, d, nxt, trainable
        if trainable:
            self._train_idx.append(i)
        self.size += 1
        return i

    def add_episode(self, trans):
        """trans from ``transitions``; successor obs rows are added (non-trainable) where needed."""
        cache = {}
        for tr in trans:
            nid = -1
            if tr["next_obs"] is not None and tr["mask"] > 0:
                key = id(tr["next_obs"])
                if key not in cache:
                    cache[key] = self._put(tr["next_obs"])
                nid = cache[key]
            self._put(tr["obs"], tr["action"], tr["reward"], tr["mask"], tr["discount"], nid, trainable=True)
        return len(trans)

    def n_trainable(self):
        return len(self._train_idx)

    def sample(self, n):
        idx = np.asarray(self._train_idx)[self.rng.integers(len(self._train_idx), size=n)]
        nid = np.where(self.next[idx] >= 0, self.next[idx], idx)  # terminal: any obs (masked out)

        def get(rows):
            o = {k: self.obs[k][rows] for k in self.KEYS}
            o["x_noisy"] = o["x_noisy"].astype(np.float32)
            o["pixels"] = o["pixels"]
            return o
        return frozen_dict.freeze(dict(observations=get(idx), next_observations=get(nid), actions=self.act[idx],
                                       rewards=self.rew[idx], masks=self.mask[idx], discount=self.disc[idx]))

    # -- checkpointing (append-only parts) ----------------------------------------------------------
    def save_new(self, dirpath):
        os.makedirs(dirpath, exist_ok=True)
        a, b = self.saved, self.size
        if b > a:
            part = dict(a=a, b=b, obs={k: v[a:b] for k, v in self.obs.items()}, act=self.act[a:b], rew=self.rew[a:b],
                        mask=self.mask[a:b], disc=self.disc[a:b], next=self.next[a:b], trainable=self.trainable[a:b])
            tmp = f"{dirpath}/part_{a:09d}_{b:09d}.pkl.tmp"
            with open(tmp, "wb") as f:
                pickle.dump(part, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, tmp[:-4])
        self.saved = b
        return dict(size=b, rng=self.rng.bit_generator.state)

    def load(self, dirpath, meta):
        import glob
        n = meta["size"]
        for p in sorted(glob.glob(f"{dirpath}/part_*.pkl")):
            a, b = (int(x) for x in os.path.basename(p)[5:-4].split("_"))
            if a >= n:
                os.remove(p)  # written after the checkpoint we resume from (crash before the state file) -> drop
                continue
            with open(p, "rb") as f:
                part = pickle.load(f)
            b = min(b, n)
            m = b - a
            for k in self.KEYS:
                self.obs[k][a:b] = part["obs"][k][:m]
            for name in ("act", "rew", "mask", "disc", "next", "trainable"):
                getattr(self, name)[a:b] = part[name][:m]
        self.size = self.saved = n
        self._train_idx = list(np.nonzero(self.trainable[:n])[0])
        self.rng.bit_generator.state = meta["rng"]
