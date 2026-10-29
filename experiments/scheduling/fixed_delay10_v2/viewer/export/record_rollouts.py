"""NEW frozen-checkpoint inspection rollouts + cross-checkpoint probes for the viewer (inspection only; no training,
no replay writes, the run directory is only read).

Phase 1 (record): for every curated seed (seeds.json) and every policy -- the fixed baseline and each ACTUAL saved
learner checkpoint ckpt/learner_<steps>.msgpack -- run one episode with the held-out evaluation protocol of
sch02.train: Executor(gamma=0.999), noise stream numpy default_rng([seed, 1]) (one draw per REPLAN request, as in
evaluation), deterministic masked argmax over the checkpoint's policy at the checkpoint's own training-time prior
strength (core.prior_strength(train_steps); 0 from 50k on).  Instrumentation is passive: a frame-capturing env proxy
(JPEG copies of the 'top' camera) and a policy wrapper that stores the probability vector / residual logits it
already computes; no extra RNG is consumed and decisions are unchanged.

Frame convention: frames[t] = observation o_t, i.e. the frame BEFORE the action of control step t (frames[0] is the
reset observation; frames[t_end] is the observation after the last executed action).  The scheduler decision at
an idle tick t is made on o_t.

Rendering is not bit-deterministic (+-1 intensity), so a regenerated rollout can diverge from the episode logged
during training even with the same seed and checkpoint; meta.json records both outcomes and whether they match.

Phase 2 (probe, one state at a time = the online policy's own forward pass): every recorded decision state (full scheduler input: current and source 64x64 images, queue +
mask, robot state, runtime features incl. plan age / source N / remaining time; same feasibility mask) is fed to
every checkpoint: residual logits (actor head, before the prior), Q1, Q2 (online critics).  Probabilities with or
without the prior are formed in the viewer from these exact values with the checkpoint's own prior strength.

Outputs (DATA_DIR/episodes/<policy>_s<seed>/): meta.json, frames.bin (+ offsets in meta), masks.bin uint8 [n,201],
rec.bin float32 [n,2,201] (recorded probabilities used for the action, residual logits), inputs.bin float32 [n,408]
(queue 25x14, qmask 25, robot 28, runtime 5), netimg.bin uint8 [n,64,64,3] (img_now; img_src is img_now of the
source decision), probes/<ck>.bin float32 [n,3,201] (residual, Q1, Q2).  ARCHIVE_DIR/<ep>.npz: full inputs.

    python viewer/export/record_rollouts.py [--fake] [--max_wall_hours 0.8]
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ARCHIVE_DIR, DATA_DIR, GAMMA, MAX_TICKS, RUN_DIR, SUCCESS_BONUS, TIME_PEN, UNDERRUN_PEN, \
    checkpoints, jl, sha256  # noqa: E402

FRAME_W, FRAME_H, JPEG_Q = 320, 240, 85
LEARNER_CFG = dict(actor_lr=1e-4, critic_lr=3e-4, alpha_lr=3e-4, alpha_init=0.001, gamma=0.999, tau=0.005, p0=0.999,
                   target_entropy_ratio=0.1)  # values of configs/online_rn_seed0.yaml (checked against launch json)


class FrameProxy:
    """Passes every call through to the wrapped env and keeps a JPEG copy of each returned 'top' frame."""

    def __init__(self, env):
        import cv2
        self._cv2, self._wrapped, self.frames = cv2, env, []

    def __getattr__(self, k):
        return getattr(self._wrapped, k)

    def _grab(self, obs):
        img = self._cv2.resize(obs["pixels"]["top"], (FRAME_W, FRAME_H), interpolation=self._cv2.INTER_AREA)
        ok, buf = self._cv2.imencode(".jpg", img[:, :, ::-1], [self._cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        assert ok
        self.frames.append(buf.tobytes())

    def reset(self, *a, **k):
        self.frames = []
        out = self._wrapped.reset(*a, **k)
        self._grab(out[0])
        return out

    def step(self, a):
        out = self._wrapped.step(a)
        self._grab(out[0])
        return out


def git_rev():
    here = os.path.dirname(os.path.abspath(__file__))
    return subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="CPU smoke test: fake env/sampler, 2 checkpoints, 1 seed")
    ap.add_argument("--max_wall_hours", type=float, default=0.8)
    ap.add_argument("--out", default=DATA_DIR)
    ap.add_argument("--archive", default=ARCHIVE_DIR)
    args = ap.parse_args()
    wall0 = time.time()
    out_root, arch = f"{args.out}/episodes", args.archive
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(arch, exist_ok=True)

    if not args.fake:
        from arli.pi0_policy import configure_caches
        configure_caches()
    import jax
    from sch01.learner import Learner, STATE_KEYS
    from sch02 import core
    from sch02.executor import Executor, reference_policy

    seeds = json.load(open(f"{args.out}/seeds.json"))["seeds"]
    cks = json.load(open(f"{args.out}/checkpoints.json"))
    launch = json.load(open(f"{RUN_DIR}/{sorted(f for f in os.listdir(RUN_DIR) if f.startswith('launch_'))[0]}"))
    for k, v in LEARNER_CFG.items():
        assert launch["config"][k] == v, (k, launch["config"][k], v)
    evlog = {e["train_steps"]: {x["seed"]: x for x in e["episodes"]} for e in
             sorted({e["tag"]: e for e in jl(f"{RUN_DIR}/evals.jsonl")}.values(), key=lambda e: e["train_steps"])}
    reflog = {x["seed"]: x for x in jl(f"{RUN_DIR}/reference_eval.jsonl")[-1]["episodes"]}
    if args.fake:
        cks, seeds = [cks[0], cks[-1]], seeds[:1]

    if args.fake:
        from fakes import FakeEnv, FakeSampler
        env, sampler, pi0_ckpt, ns_sha = FakeEnv(), FakeSampler(), "FAKE", "FAKE"
    else:
        from arli import aloha_env
        from arli.pi0_policy import CHECKPOINT, FrozenPi0
        from sch01.sampler import RTCSampler
        env, sampler = aloha_env.make_env(), RTCSampler(FrozenPi0(10))
        pi0_ckpt = CHECKPOINT
        ns_sha = sha256(f"{CHECKPOINT}/assets/lerobot/aloha_sim_transfer_cube_human/norm_stats.json")
        assert ns_sha == launch["norm_stats_sha256"], "pi0 normalisation stats differ from the training run"
    penv = FrameProxy(env)
    exe = Executor(penv, sampler, gamma=GAMMA)
    L = Learner(0, LEARNER_CFG)
    prov = dict(git_viewer=git_rev(), git_training_launch=launch["git"], host=socket.gethostname(),
                slurm_job=os.environ.get("SLURM_JOB_ID"), pi0_checkpoint=pi0_ckpt, norm_stats_sha256=ns_sha,
                delay_table={str(k): v for k, v in core.DELAY.items()}, jax_backend=jax.default_backend(),
                fake=args.fake)

    def load_ckpt(c):
        blob = open(c["file"] if os.path.isabs(c["file"]) else f"{RUN_DIR}/{c['file']}", "rb").read()
        if not args.fake:
            import hashlib
            assert hashlib.sha256(blob).hexdigest() == c["sha256"], f"checkpoint hash changed: {c['id']}"
        L.load_bytes(blob)

    def out_of_time():
        return (time.time() - wall0) / 3600 > args.max_wall_hours

    # ---------------- phase 1: record ----------------
    policies = [dict(id="baseline", kind="fixed_baseline")] + [dict(c, kind="learned") for c in cks]
    for pol_d in policies:
        todo = [s for s in seeds if not os.path.exists(f"{out_root}/{pol_d['id']}_s{s['seed']}/meta.json")]
        if not todo:
            continue
        if pol_d["kind"] == "learned":
            load_ckpt(pol_d)
            strength = core.prior_strength(pol_d["train_steps"])
            assert abs(strength - pol_d["prior_strength"]) < 1e-9
        for sd in todo:
            if out_of_time():
                print("wall budget reached during recording; rerun resumes", flush=True)
                sys.exit(3)
            seed = sd["seed"]
            rec = []
            if pol_d["kind"] == "learned":
                def pol(st, mask, t):  # identical to sch02.train make_policy("eval") + passive recording
                    ref, ref_kind = core.reference_action(st["q"], mask)
                    p, res = L.policy_probs(st, mask, ref, strength)
                    a = int(np.argmax(p))
                    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
                    rec.append(dict(t=t, p=p, res=res))
                    return a, dict(kind="learned", p=float(p[a]), ref=ref, ref_kind=ref_kind, departed=a != ref,
                                   entropy=ent, ref_prob=float(p[ref]))
            else:
                def pol(st, mask, t):
                    a, info = reference_policy(st, mask, t)
                    rec.append(dict(t=t, p=None, res=None))
                    return a, info
            t0 = time.time()
            res = exe.run(seed, pol, np.random.default_rng([seed, 1]), record=True)
            wall = time.time() - t0
            write_episode(out_root, arch, pol_d, sd, res, rec, penv.frames, prov, wall,
                          logged=(reflog if pol_d["kind"] == "fixed_baseline" else evlog[pol_d["train_steps"]]).get(seed),
                          core=core, STATE_KEYS=STATE_KEYS)
            print(f"[rec] {pol_d['id']} seed {seed}: success {res['success']} t_end {res['t_end']} "
                  f"requests {len(res['requests'])} decisions {len(res['decisions'])} {wall:.1f}s", flush=True)

    # ---------------- phase 2: probes ----------------

    # One state at a time through the SAME jitted actor function the policy used (Learner._act on a batch of 1):
    # batched evaluation changes GPU kernel choices and gave residual differences up to ~0.4 logits (one argmax flip
    # in 8,569 own-checkpoint states), so probes are computed exactly as the online policy computes them.
    crit = jax.jit(lambda cp_, s_: L.critic_def.apply({"params": cp_}, s_))

    eps = sorted(d for d in os.listdir(out_root) if os.path.exists(f"{out_root}/{d}/meta.json"))
    want = [f"{p['id']}_s{s['seed']}" for p in policies for s in seeds]
    assert set(want) <= set(eps), "recording incomplete"
    eps = want
    report = []
    for c in cks:
        todo = [e for e in eps if not os.path.exists(f"{out_root}/{e}/probes/{c['id']}.bin")]
        if not todo:
            continue
        load_ckpt(c)
        for e in todo:
            if out_of_time():
                print("wall budget reached during probing; rerun resumes", flush=True)
                sys.exit(3)
            z = np.load(f"{arch}/{e}.npz")
            n = len(z["t"])
            zs = {k: z[k] for k in STATE_KEYS}
            r = np.zeros((n, core.NUM_ACTIONS), np.float32)
            q = np.zeros((2, n, core.NUM_ACTIONS), np.float32)
            for i in range(n):
                si = {k: zs[k][i:i + 1] for k in STATE_KEYS}
                r[i] = np.asarray(L._act(L.A.params, si))[0]
                q[:, i] = np.asarray(crit(L.C.params, si))[:, 0]
            arr = np.stack([r, q[0], q[1]], 1).astype(np.float32)  # [n, 3, 201]
            os.makedirs(f"{out_root}/{e}/probes", exist_ok=True)
            arr.tofile(f"{out_root}/{e}/probes/{c['id']}.bin.tmp")
            os.replace(f"{out_root}/{e}/probes/{c['id']}.bin.tmp", f"{out_root}/{e}/probes/{c['id']}.bin")
            if e.startswith(c["id"] + "_"):  # own checkpoint: batched probe must reproduce the recorded policy
                meta = json.load(open(f"{out_root}/{e}/meta.json"))
                recb = np.fromfile(f"{out_root}/{e}/rec.bin", np.float32).reshape(n, 2, core.NUM_ACTIONS)
                masks = np.fromfile(f"{out_root}/{e}/masks.bin", np.uint8).reshape(n, core.NUM_ACTIONS).astype(bool)
                dres, agree, maxdp = 0.0, 0, 0.0
                for i in range(n):
                    if not masks[i].any():
                        continue
                    lg = r[i].astype(np.float64) + core.prior_logits(meta["decisions"]["ref"][i], masks[i],
                                                                     meta["policy"]["prior_strength"]).astype(np.float64)
                    lg[~masks[i]] = -np.inf
                    pp = np.exp(lg - lg.max())
                    pp /= pp.sum()
                    agree += int(np.argmax(pp) == meta["decisions"]["action"][i])
                    dres = max(dres, float(np.abs(r[i] - recb[i, 1])[masks[i]].max()))
                    maxdp = max(maxdp, float(np.abs(pp - recb[i, 0]).max()))
                nv = int(masks.any(1).sum())
                report.append(dict(episode=e, decisions=nv, argmax_agree=agree, max_abs_residual_diff=dres,
                                   max_abs_prob_diff=maxdp))
                print(f"[probe-check] {e}: argmax agree {agree}/{nv}, max|dres| {dres:.2e}, max|dp| {maxdp:.2e}",
                      flush=True)
        print(f"[probe] {c['id']} done ({len(todo)} episodes)", flush=True)
    if report:
        old = json.load(open(f"{args.out}/probe_check.json")) if os.path.exists(f"{args.out}/probe_check.json") else []
        json.dump(old + report, open(f"{args.out}/probe_check.json", "w"), indent=1)
    print("RECORD+PROBE DONE", flush=True)


def write_episode(out_root, arch, pol_d, sd, res, rec, frames, prov, wall, logged, core, STATE_KEYS):
    seed = sd["seed"]
    eid = f"{pol_d['id']}_s{seed}"
    d = f"{out_root}/{eid}"
    os.makedirs(d, exist_ok=True)
    dec = res["decisions"]
    n = len(dec)
    # decisions -> recorded probabilities (same order: one policy call per non-forced decision)
    assert len(rec) == n and all(r["t"] == x["t"] for r, x in zip(rec, dec)), "decision/record misalignment"
    t_end = res["t_end"]
    assert len(frames) == t_end + 1, (len(frames), t_end)
    masks = np.stack([x["mask"] for x in dec]).astype(np.uint8)
    recb = np.zeros((n, 2, core.NUM_ACTIONS), np.float32)
    for i, r in enumerate(rec):
        if r["p"] is not None:
            recb[i, 0], recb[i, 1] = r["p"], r["res"]
    inputs = np.stack([np.concatenate([x["state"]["queue"].ravel(), x["state"]["qmask"], x["state"]["robot"],
                                       x["state"]["runtime"]]) for x in dec]).astype(np.float32)
    netimg = np.stack([x["state"]["img_now"] for x in dec]).astype(np.uint8)
    np.savez_compressed(f"{arch}/{eid}.npz", t=np.asarray([x["t"] for x in dec]), mask=masks.astype(bool),
                        **{k: np.stack([x["state"][k] for x in dec]) for k in STATE_KEYS})
    masks.tofile(f"{d}/masks.bin")
    recb.tofile(f"{d}/rec.bin")
    inputs.tofile(f"{d}/inputs.bin")
    netimg.tofile(f"{d}/netimg.bin")
    offs = np.cumsum([0] + [len(f) for f in frames]).tolist()
    with open(f"{d}/frames.bin", "wb") as f:
        for b in frames:
            f.write(b)
    ticks = res["ticks"]
    rewards = []
    for k, tk in enumerate(ticks):
        u = tk[5]
        rewards.append(-TIME_PEN - UNDERRUN_PEN * u + (SUCCESS_BONUS if (res["success"] and k == len(ticks) - 1) else 0.0))
    assert abs(sum(rewards) - res["ret"]) < 1e-9
    refs = []
    for x in dec:
        refs.append(core.reference_index(x["state"]["q"], x["mask"]) if x["mask"].any() else -1)
    ref_kinds = [x["info"].get("ref_kind", x["info"].get("kind")) for x in dec]
    decs = dict(t=[x["t"] for x in dec], action=[int(x["action"]) for x in dec], q=[int(x["state"]["q"]) for x in dec],
                p=[int(x["state"]["p"]) for x in dec], chunk=[int(x["state"]["chunk"]) for x in dec],
                ref=[int(r) for r in refs], ref_kind=ref_kinds, kind=[x["info"].get("kind") for x in dec],
                nfeas=[int(x["mask"].sum()) for x in dec],
                plan_age=[int(round(float(x["state"]["runtime"][1]) * core.H)) for x in dec],
                plan_n=[int(round(float(x["state"]["runtime"][2]) * core.N_MAX)) for x in dec],
                ticks_left=[MAX_TICKS - x["t"] for x in dec], G=[float(x["G"]) for x in dec],
                D=[int(x["D"]) for x in dec], terminal=[bool(x["terminal"]) for x in dec])
    pol_meta = dict(id=pol_d["id"], kind=pol_d["kind"])
    if pol_d["kind"] == "learned":
        pol_meta.update({k: pol_d[k] for k in ("train_steps", "grad_steps", "prior_strength", "sha256", "file",
                                               "eval_tag")})
        pol_meta["eval_mode"] = "deterministic masked argmax, checkpoint's own training-time prior strength"
    else:
        pol_meta.update(eval_mode="fixed baseline rule: KEEP while q > 10, REPLAN(10,10) at q = 10; recovery "
                                  "REPLAN(q, largest N) for q < 10", prior_strength=None)
    meta = dict(
        id=eid, policy=pol_meta, seed=seed, seed_category=sd["category"], noise_rng="numpy default_rng([seed, 1])",
        new_recording=True,
        recording_note="NEW recording made after training with the frozen checkpoint; not a replay of a training-time "
                       "episode. Rendering is not bit-deterministic, so this rollout may differ from the episode "
                       "logged during training.",
        success=bool(res["success"]), t_end=t_end, ret=res["ret"], discounted_return=res["discounted_return"],
        underrun_ticks=res["underrun_ticks"], n_calls=res["n_calls"], modeled_ms=res["modeled_ms"],
        logged=None if logged is None else dict(success=bool(logged["success"]), t_end=logged["t_end"],
                                                ret=logged["ret"]),
        outcome_matches_log=None if logged is None else (bool(logged["success"]) == bool(res["success"])
                                                         and logged["t_end"] == t_end),
        frame_convention="frames[t] = observation o_t BEFORE the action of control step t; frames[0] = reset "
                         "observation; frames[t_end] = after the last action",
        frame_size=[FRAME_W, FRAME_H], jpeg_quality=JPEG_Q, frame_offsets=offs,
        ticks=dict(t=[tk[0] for tk in ticks], chunk=[tk[1] for tk in ticks], p=[tk[2] for tk in ticks],
                   qlen_after=[tk[3] for tk in ticks], inflight=[tk[4] for tk in ticks],
                   underrun=[tk[5] for tk in ticks], reward=rewards),
        requests=jsonable(res["requests"]), startup=jsonable(res["startup"]), decisions=decs,
        n_decisions=n, wall_s=wall, provenance=prov,
        binary=dict(masks="uint8 [n,201]", rec="float32 [n,2,201] (recorded probs, residual logits)",
                    inputs="float32 [n,408] queue(25x14) qmask(25) robot(28) runtime(5)",
                    netimg="uint8 [n,64,64,3] img_now", probes="float32 [n,3,201] residual, Q1, Q2"))
    with open(f"{d}/meta.json.tmp", "w") as f:
        json.dump(jsonable(meta), f, separators=(",", ":"))
    os.replace(f"{d}/meta.json.tmp", f"{d}/meta.json")  # written last: marks the episode complete


if __name__ == "__main__":
    main()
