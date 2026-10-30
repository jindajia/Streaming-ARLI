"""Action-disagreement probe: how much does the queued (old) plan disagree with a fresh plan from the latest
observation, per control step and future offset?  Diagnostic only; nothing here is executed by the robot.

Stage `record` (GPU): regenerate the three fixed-baseline reference episodes (N = 10, d = 10, H = 50, K = 25, RTC on;
KEEP while q > 10, REPLAN(10,10) at q = 10; noise default_rng([seed, 1]) as in the held-out evaluation) with a
READ-ONLY logging proxy around the env.  At every env.step the proxy copies, from the executor's own state dict:
t, chunk, p, the queue before this step's pop, the installed plan (full normalised x0 + env-space actions + origin),
the pending request (launch/ready/handoff/R/N), the observation o_t (raw 480x640 image + agent_pos) and, after the
step, the contact stage (gym_aloha reward 0-4).  The proxy never writes executor state; the reference noise stream is
wrapped only to record its draws.

Stage `probe` (GPU, offline from the saved records): at every reference step t, fresh plans conditioned on o_t
(same instruction, N = 10) with two fixed probe-noise tensors per episode (default_rng([seed, 4242, k]), k = 0, 1;
independent of the reference stream; reused at every t; shared between variants):
  A  RTC off  -- unconstrained diagnostic plan
  B  RTC on   -- existing guidance with the earliest feasible handoff R = d = 10: reference = previous full plan
                 shifted by t - origin, committed prefix 10, exp soft mask until H - shift (exactly Executor._launch)
RTC-on probes exist only where the executor would allow a new request (no inference pending at decision time, i.e.
idle or the baseline's own launch step, and q >= d); elsewhere they are N/A.  Also probes the 22 exact saved
snapshots of r_sensitivity_probe (states with recorded branch outcomes).  Probe outputs are only saved.

    python action_disagreement_probe/adprobe.py --stage record|probe|snapshots [--out DIR] [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import socket
import subprocess
import sys
import time

import numpy as np

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [10_000_014, 10_000_016, 10_000_004]
N, D_LAT, H, K = 10, 10, 50, 25
RSENS = "/N/scratch/jindjia/arli/runs_scheduling/r_sensitivity_probe"


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def probe_noise(seed, k):
    return np.random.default_rng([seed, 4242, k]).standard_normal((H, 32)).astype(np.float32)


class RecordingRNG:
    """Delegates to the reference generator; only records the draws (the stream is unchanged)."""

    def __init__(self, rng):
        self.rng, self.draws = rng, []

    def standard_normal(self, *a, **k):
        x = self.rng.standard_normal(*a, **k)
        self.draws.append(np.asarray(x, np.float32).copy())
        return x


class LogProxy:
    """Read-only logger around the env.  Gets a reference to the executor's state dict from on_tick (step 0 is
    always an idle decision tick) and copies what it needs at every step, before the executor updates t/p."""

    def __init__(self, env):
        self._w, self.ex, self.rows, self.plans, self._plan_ids = env, None, [], [], {}

    def __getattr__(self, k):
        return getattr(self._w, k)

    def bind(self, ex, env_):
        self.ex = ex

    def _plan_id(self, d):
        key = id(d["x0"])
        if key not in self._plan_ids:
            self._plan_ids[key] = len(self.plans)
            self.plans.append(dict(x0=np.asarray(d["x0"], np.float32).copy(), actions=np.asarray(d["actions"], np.float32).copy(),
                                   origin=int(d["origin"]), n=int(d["n"])))
        return self._plan_ids[key]

    def reset(self, *a, **k):
        self.rows, self.plans, self._plan_ids = [], [], {}
        return self._w.reset(*a, **k)

    def step(self, a):
        ex = self.ex
        inf = ex["inflight"]
        queue_before = [np.asarray(a, np.float32)] + [np.asarray(x, np.float32) for x in ex["queue"]]
        row = dict(t=int(ex["t"]), chunk=int(ex["chunk"]), p=int(ex["p"]), q=len(queue_before),
                   queue=np.stack(queue_before), plan=self._plan_id(ex["prev"]),
                   img=ex["raw"]["pixels"]["top"].copy(), agent_pos=np.asarray(ex["raw"]["agent_pos"], np.float32).copy(),
                   pending=None if inf is None else dict(launch_t=int(inf["launch_t"]), ready_t=int(inf["ready_t"]),
                                                         handoff_t=int(inf["handoff_t"]), r=int(inf["r"]), n=int(inf["n"]),
                                                         plan=self._plan_id(inf)))
        out = self._w.step(a)
        row["stage"] = int(out[1])
        self.rows.append(row)
        return out


def provenance():
    from arli.pi0_policy import CHECKPOINT
    from sch01 import sampler as smp
    ns = f"{CHECKPOINT}/assets/lerobot/aloha_sim_transfer_cube_human/norm_stats.json"
    return dict(pi0_checkpoint=CHECKPOINT, norm_stats_sha256=hashlib.sha256(open(ns, "rb").read()).hexdigest(),
                instruction="'Transfer cube' (pi0_aloha_sim data-config default prompt; obs_to_pi0_input passes none)",
                N=N, d=D_LAT, H=H, K=K, rtc=dict(max_guidance=smp.MAX_GUIDANCE, mask="official exp prefix weights"),
                git=subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
                gpu=subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())


def build(smoke):
    if smoke:
        sys.path.insert(0, HERE)
        from fakes import FakeEnv, FakeSampler
        return FakeEnv(), FakeSampler(), None
    from arli.pi0_policy import configure_caches
    configure_caches()
    from arli import aloha_env
    from arli.pi0_policy import FrozenPi0
    from sch01.sampler import RTCSampler
    fp = FrozenPi0(10)
    return aloha_env.make_env(), RTCSampler(fp), fp


# ------------------------------------------------------------------------------------------- record
def stage_record(a):
    from sch02 import core
    from sch02.executor import Executor, reference_policy
    assert core.DELAY[N] == D_LAT and core.H == H and core.K_CAP == K and core.BASELINE_Q == 10
    env, sampler, _ = build(a.smoke)
    proxy = LogProxy(env)
    exe = Executor(proxy, sampler, gamma=0.999)
    os.makedirs(f"{a.out}/reference", exist_ok=True)
    prov = {} if a.smoke else provenance()
    for seed in (SEEDS[:1] if a.smoke else SEEDS):
        path = f"{a.out}/reference/ref_{seed}.npz"
        if os.path.exists(path):
            continue
        rr = RecordingRNG(np.random.default_rng([seed, 1]))
        t0 = time.time()
        res = exe.run(seed, reference_policy, rr, record=True, on_tick=lambda ex, e: proxy.bind(ex, e) if proxy.ex is not ex else None)
        rows, plans = proxy.rows, proxy.plans
        T = len(rows)
        assert T == res["t_end"] and [r["t"] for r in rows] == list(range(T))
        qpad = np.zeros((T, K, 14), np.float32)
        for i, r in enumerate(rows):
            qpad[i, :r["q"]] = r["queue"]
        pend = np.full((T, 6), -1, np.int64)
        for i, r in enumerate(rows):
            if r["pending"]:
                pn = r["pending"]
                pend[i] = [pn["launch_t"], pn["ready_t"], pn["handoff_t"], pn["r"], pn["n"], pn["plan"]]
        np.savez_compressed(path, t=np.arange(T), chunk=np.array([r["chunk"] for r in rows]), p=np.array([r["p"] for r in rows]),
                            q=np.array([r["q"] for r in rows]), queue=qpad, plan=np.array([r["plan"] for r in rows]),
                            pending=pend, stage=np.array([r["stage"] for r in rows]),
                            img=np.stack([r["img"] for r in rows]), agent_pos=np.stack([r["agent_pos"] for r in rows]),
                            plan_x0=np.stack([pl["x0"] for pl in plans]), plan_actions=np.stack([pl["actions"] for pl in plans]),
                            plan_origin=np.array([pl["origin"] for pl in plans]), plan_n=np.array([pl["n"] for pl in plans]),
                            ref_noise=np.stack(rr.draws))
        meta = dict(seed=seed, regenerated=True, success=bool(res["success"]), t_end=int(res["t_end"]), steps=T,
                    requests=[{k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in q.items()
                               if k in ("launch_t", "obs_t", "ready_t", "handoff_t", "r", "n", "shift", "committed", "overlap_end")}
                              | dict(abandoned=bool(q.get("abandoned", False))) for q in res["requests"]],
                    n_plans=len(plans), n_ref_noise_draws=len(rr.draws), wall_s=time.time() - t0,
                    note="REGENERATED reference episode (renderer not bit-exact): may differ from the original source runs.")
        json.dump(meta | dict(provenance=prov), open(f"{a.out}/reference/ref_{seed}.json", "w"), indent=1)
        log(f"reference {seed}: success {res['success']} t_end {res['t_end']} plans {len(plans)}")
        proxy.ex = None
    log("RECORD DONE")


# ------------------------------------------------------------------------------------------- probe helpers
class Prober:
    def __init__(self, sampler, fp):
        self.s, self.fp = sampler, fp

    def sample(self, obs_pi0, noise, y=None, w=None, st=None):
        """Same arithmetic as RTCSampler.sample, optionally reusing a precomputed VLM prefix state ``st``."""
        if self.fp is None:  # smoke: fake sampler
            return self.s.sample(obs_pi0, N, noise, y, w)[1]
        import jax.numpy as jnp
        st = st or self.fp.begin(obs_pi0)
        x = jnp.asarray(noise, jnp.float32)[None]
        dt = jnp.float32(-1.0 / N)
        for k in range(N):
            tau = jnp.float32(1.0 - k / N)
            if y is None:
                x = self.s._plain(self.s._state, st.observation, st.kv_cache, st.prefix_mask, x, tau, dt)
            else:
                x = self.s._rtc(self.s._state, st.observation, st.kv_cache, st.prefix_mask, x, tau, dt,
                                jnp.asarray(y, jnp.float32)[None], jnp.asarray(w, jnp.float32))
        return np.asarray(self.fp.finish(st, x), np.float32)


def eligible(q, pending_launch, t):
    """A new request with R = d is allowed iff no inference is pending at decision time (idle, or the baseline's own
    launch at this step) and at least d old actions remain."""
    return (pending_launch < 0 or pending_launch == t) and q >= D_LAT


def stage_probe(a):
    from arli.pi0_policy import obs_to_pi0_input
    from sch01.sampler import rtc_weights, shift_reference
    env, sampler, fp = build(a.smoke)
    pr = Prober(sampler, fp)
    os.makedirs(f"{a.out}/probes", exist_ok=True)
    deadline = T0 + a.deadline_min * 60
    checks = []
    for seed in (SEEDS[:1] if a.smoke else SEEDS):
        Z = dict(np.load(f"{a.out}/reference/ref_{seed}.npz"))
        T = len(Z["t"])
        noises = [probe_noise(seed, k) for k in (0, 1)]
        nh = [hashlib.sha256(x.tobytes()).hexdigest() for x in noises]
        part = f"{a.out}/probes/probe_{seed}.npz"
        if os.path.exists(part):
            continue
        acts = np.full((T, 2, 2, H, 14), np.nan, np.float32)  # [t, variant(off,on), sample, h, dim]
        align_err = np.zeros(T)
        elig = np.zeros(T, bool)
        shift = np.zeros(T, int)
        n_calls = 0
        for t in range(T):
            if time.time() > deadline:
                log("deadline reached; rerun to resume (completed episodes are skipped)")
                sys.exit(3)
            q, pl = int(Z["q"][t]), int(Z["plan"][t])
            origin = int(Z["plan_origin"][pl])
            sh = t - origin
            shift[t] = sh
            old = Z["plan_actions"][pl]
            # executor alignment: queue[h] must be the old plan at index t - origin + h
            align_err[t] = float(np.abs(Z["queue"][t, :q] - old[sh:sh + q]).max())
            elig[t] = eligible(q, int(Z["pending"][t, 0]), t)
            obs = obs_to_pi0_input({"pixels": {"top": Z["img"][t]}, "agent_pos": Z["agent_pos"][t]})
            st = fp.begin(obs) if fp is not None else None
            y = shift_reference(Z["plan_x0"][pl], sh)
            w = rtc_weights(D_LAT, H - sh)
            for k in (0, 1):
                acts[t, 0, k] = pr.sample(obs, noises[k], st=st)
                n_calls += 1
                if elig[t]:
                    acts[t, 1, k] = pr.sample(obs, noises[k], y, w, st=st)
                    n_calls += 1
            if t % 50 == 0:
                log(f"probe {seed} t={t}/{T} calls {n_calls}")
        # validation at the baseline's own launch steps: an RTC-on probe with the REFERENCE noise of that request
        # must reproduce the plan the executor actually installed (alignment + mask + reference check)
        launches = [int(x) for x in sorted(set(Z["pending"][:, 0])) if x >= 0][:3]
        draw_of_launch = {lt: i + 1 for i, lt in enumerate(sorted(set(int(x) for x in Z["pending"][:, 0] if x >= 0)))}
        for lt in launches:
            pl = int(Z["plan"][lt])
            sh = lt - int(Z["plan_origin"][pl])
            obs = obs_to_pi0_input({"pixels": {"top": Z["img"][lt]}, "agent_pos": Z["agent_pos"][lt]})
            newp = int(Z["pending"][lt, 5])
            rep = pr.sample(obs, Z["ref_noise"][draw_of_launch[lt]], shift_reference(Z["plan_x0"][pl], sh), rtc_weights(D_LAT, H - sh))
            rep2 = pr.sample(obs, Z["ref_noise"][draw_of_launch[lt]], shift_reference(Z["plan_x0"][pl], sh), rtc_weights(D_LAT, H - sh))
            n_calls += 2
            checks.append(dict(seed=seed, launch_t=lt, shift=sh, committed=D_LAT, overlap_end=H - sh,
                               reproduce_installed_plan_maxabs=float(np.abs(rep - Z["plan_actions"][newp]).max()),
                               repeat_identical=bool(np.array_equal(rep, rep2))))
        assert np.isfinite(acts[:, 0]).all(), "non-finite RTC-off probe"
        assert np.isfinite(acts[elig][:, 1]).all(), "non-finite RTC-on probe"
        np.savez_compressed(part, acts=acts, eligible=elig, shift=shift, align_err=align_err, noise=np.stack(noises))
        json.dump(dict(seed=seed, steps=T, inference_calls=n_calls, eligible_steps=int(elig.sum()), noise_sha256=nh,
                       noise_rule="numpy default_rng([seed, 4242, k]).standard_normal((50, 32)), k = 0, 1",
                       max_alignment_error=float(align_err.max()), checks=[c for c in checks if c["seed"] == seed],
                       wall_s=time.time() - T0), open(f"{a.out}/probes/probe_{seed}.json", "w"), indent=1)
        log(f"probe {seed} done: {n_calls} calls, eligible {elig.sum()}/{T}, max align err {align_err.max():.2e}, checks {checks[-len(launches):]}")
    log("PROBE DONE")


def stage_snapshots(a):
    """Probe the exact saved r_sensitivity_probe snapshots (same states as the recorded branch outcomes)."""
    from arli.pi0_policy import obs_to_pi0_input
    from sch01.sampler import rtc_weights, shift_reference
    env, sampler, fp = build(a.smoke)
    pr = Prober(sampler, fp)
    man = json.load(open(f"{HERE}/../r_sensitivity_probe/manifest.json"))
    out = {}
    for seed in SEEDS:
        snaps = {(int(s["chunk"]), int(s["p"])): s for s in pickle.load(open(f"{RSENS}/sources/snapshots_{seed}.pkl", "rb"))}
        noises = [probe_noise(seed, k) for k in (0, 1)]
        for m in man["snapshots"]:
            if m.get("missing") or m["source_seed"] != seed:
                continue
            s = snaps[(m["chunk"], m["p"])]
            prev = s["prev"]
            sh = int(s["t"]) - int(prev["origin"])
            q = len(s["queue"])
            Q = np.stack(s["queue"]).astype(np.float32)
            obs = obs_to_pi0_input(s["raw"])
            st = fp.begin(obs) if fp is not None else None
            y, w = shift_reference(prev["x0"], sh), rtc_weights(D_LAT, H - sh)
            acts = np.stack([np.stack([pr.sample(obs, nz, st=st), pr.sample(obs, nz, y, w, st=st)]) for nz in noises], 1)  # [2 var, 2 samp, H, 14]
            out[m["snapshot_id"]] = dict(acts=acts, queue=Q, old=np.asarray(prev["actions"], np.float32), shift=sh, q=q, t=int(s["t"]),
                                         align_err=float(np.abs(Q - np.asarray(prev["actions"])[sh:sh + q]).max()))
            log(f"snapshot {m['snapshot_id']}: shift {sh} q {q} align err {out[m['snapshot_id']]['align_err']:.1e}")
    pickle.dump(out, open(f"{a.out}/snapshot_probes.pkl", "wb"))
    log(f"SNAPSHOTS DONE ({4 * len(out)} calls)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["record", "probe", "snapshots", "all"])
    ap.add_argument("--out", default="/N/scratch/jindjia/arli/runs_scheduling/action_disagreement_probe")
    ap.add_argument("--deadline_min", type=float, default=52)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for stg in (["record", "probe", "snapshots"] if a.stage == "all" else [a.stage]):
        {"record": stage_record, "probe": stage_probe, "snapshots": stage_snapshots}[stg](a)
