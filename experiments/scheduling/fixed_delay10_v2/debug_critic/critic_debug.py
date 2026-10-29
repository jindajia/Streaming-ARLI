"""SCH-01 fixed_delay10_v2 / rn_seed0_d10: time-bounded critic diagnostics (read-only on the run directory).

Phases, in priority order (each saves its outputs as soon as it finishes; a global deadline stops cleanly):
  0. data/bookkeeping audit of the whole replay (CPU numpy): masks, durations, rewards, terminal flags,
     state/next-state chaining, finiteness and input ranges.
  1. critic-target decomposition on FIXED replay transitions (ordinary + stratified sets) for checkpoints
     100,247 / 225,337 / 300,155:  y = G + gamma^D (1-term) E_pi[min Qt] + gamma^D (1-term) alpha H(pi)
     (exactly sch01.learner._update), no augmentation; plus a pass over every replay transition.
  2. same-state branching pilot with the frozen 100,247 actor (2 snapshots x 3 first actions x 3 paired noise seeds
     + restoration / identical-action replay checks).
  3. critic-only ablation from 100,247: A = original target, B = future-entropy term removed; actor, alpha, replay
     rows (< 100k) and batch/augmentation sequence frozen and identical; <= 300 updates per branch.

    python debug_critic/critic_debug.py --out <dir> --deadline_min 17.5 [--cpu_smoke]

Never calls Replay.load (it deletes part files beyond a requested size); parts are read with pickle only.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import socket
import subprocess
import sys
import time

import numpy as np

RUN = "/N/scratch/jindjia/arli/runs_scheduling/fixed_delay10_v2/online/rn_seed0_d10"
CKPTS = [100247, 225337, 300155]
GAMMA, P0, TAU = 0.999, 0.999, 0.005
C_TIME, C_UNDER = 0.1 / 400, 0.5 / 400
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def qs(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {}
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(np.median(x)), p95=float(np.quantile(x, .95)),
                p99=float(np.quantile(x, .99)), max=float(x.max()), min=float(x.min()))


# ----------------------------------------------------------------------------------------------- replay (read-only)
def load_replay(rd, max_parts=None):
    parts = sorted(glob.glob(f"{rd}/replay/part_*.pkl"))[:max_parts]
    P = [pickle.load(open(p, "rb")) for p in parts]
    for a, b in zip(P, P[1:]):
        for k in "fsd":
            assert a[k][1] == b[k][0], f"non-contiguous replay parts ({k})"
    fr = np.concatenate([p["f"][2] for p in P])
    st = {k: np.concatenate([p["s"][2][k] for p in P]) for k in P[0]["s"][2]}
    dc = {k: np.concatenate([p["d"][2][k] for p in P]) for k in P[0]["d"][2]}
    assert P[0]["f"][0] == P[0]["s"][0] == P[0]["d"][0] == 0
    return fr, st, dc, [os.path.basename(p) for p in parts]


def make_batch(fr, st, dc, idx):
    out = {}
    for pre, rows in (("s_", dc["s"][idx]), ("n_", dc["n"][idx])):
        out[pre + "img_now"] = fr[st["f_now"][rows]]
        out[pre + "img_src"] = fr[st["f_src"][rows]]
        for k in ("queue", "qmask", "robot", "runtime", "mask", "ref", "nfeas"):
            out[pre + k] = st[k][rows]
    for k in ("action", "G", "D", "terminal"):
        out[k] = dc[k][idx]
    return out


# ----------------------------------------------------------------------------------------------- phase 0
def audit(fr, st, dc, core, episodes):
    A = core.NUM_ACTIONS
    nd, ns = len(dc["action"]), len(st["ref"])
    res, fails = {}, {}

    def chk(name, bad_mask, detail=None):
        nb = int(np.sum(bad_mask))
        res[name] = dict(violations=nb)
        if nb:
            ix = np.flatnonzero(bad_mask)[:5].tolist()
            res[name]["first_rows"] = ix
            if detail is not None:
                res[name]["detail"] = detail(ix)
            fails[name] = nb

    q = st["qmask"].sum(1).round().astype(int)
    rem = st["runtime"][:, 3] * 400
    remi = np.round(rem).astype(int)
    chk("runtime_q_matches_qmask", np.abs(st["runtime"][:, 0] * core.K_CAP - q) > 1e-4)
    chk("runtime_remaining_integer", np.abs(rem - remi) > 1e-3)
    chk("runtime_const_one", st["runtime"][:, 4] != 1.0)
    table = {(qq, tl): core.feasible_mask(qq, tl) for qq in range(0, 26) for tl in (1, 2)}
    exp_mask = np.stack([table[(int(a), 2 if b > 1 else 1)] for a, b in zip(q, remi)])
    # terminal next-states are stored with mask(q, max(1, ticks_left)); identical rule
    chk("state_mask_equals_delay_table", np.any(exp_mask != st["mask"], 1))
    chk("nfeas_equals_mask_sum", st["nfeas"] != st["mask"].sum(1))
    refs = np.array([core.reference_index(int(a), m) if m.any() else 0 for a, m in zip(q, st["mask"])])
    chk("ref_equals_rule", refs != st["ref"])
    s, n, a = dc["s"], dc["n"], dc["action"]
    chk("action_feasible_in_state", ~st["mask"][s, a])
    keep = a == 0
    R = np.where(keep, 0, (a - 1) // 8 + 1)
    term = dc["terminal"] > 0.5
    D = dc["D"]
    chk("KEEP_duration_1", keep & (D != 1))
    chk("REPLAN_duration_equals_R_nonterminal", (~keep) & (~term) & (D != R),
        lambda ix: [dict(row=i, R=int(R[i]), D=float(D[i])) for i in ix])
    chk("REPLAN_duration_le_R_terminal", (~keep) & term & (D > R))
    chk("duration_positive", D < 1)
    # rewards: no underruns occurred in training (episodes.jsonl), so G is determined by D and success
    base = -C_TIME * (1 - GAMMA ** D) / (1 - GAMMA)
    succ_term = term & (dc["G"] > 0.5)
    expG = base + np.where(succ_term, GAMMA ** (D - 1), 0.0)
    chk("G_matches_reward_model", np.abs(dc["G"] - expG) > 2e-6,
        lambda ix: [dict(row=i, G=float(dc["G"][i]), expected=float(expG[i]), D=float(D[i]), term=bool(term[i])) for i in ix])
    # next-state remaining time = remaining - D (except terminal success ends early: also exact)
    chk("next_remaining_equals_rem_minus_D", np.abs(remi[n] - (remi[s] - D)) > 0.5)
    # chaining: next state of a non-terminal decision is the state of the following decision
    nt = np.flatnonzero(~term[:-1])
    chk("next_state_is_following_state", n[nt] != s[nt + 1])
    chk("terminal_count_equals_episodes", np.array([term.sum() != len(episodes)]))
    chk("nonterminal_next_mask_nonempty", (~term) & (st["nfeas"][n] < 1))
    chk("timeout_terminal_at_400", term & ~succ_term & (remi[n] != 0))
    for k in ("queue", "robot", "runtime"):
        chk(f"finite_{k}", ~np.isfinite(st[k]).all(axis=tuple(range(1, st[k].ndim))))
    chk("finite_G_D", ~(np.isfinite(dc["G"]) & np.isfinite(dc["D"])))
    # input scales (outliers): robust z of every scalar feature
    feats = np.concatenate([st["queue"].reshape(ns, -1), st["robot"], st["runtime"]], 1)
    mu, sd_ = feats.mean(0), np.maximum(feats.std(0), 1e-2)  # std floor: near-constant features are not outliers
    zmax = (np.abs(feats - mu) / sd_).max(1)
    res["input_ranges"] = dict(queue_absmax=float(np.abs(st["queue"]).max()), robot_absmax=float(np.abs(st["robot"]).max()),
                               runtime_min=float(st["runtime"].min()), runtime_max=float(st["runtime"].max()),
                               frame_min=int(fr.min()), frame_max=int(fr.max()), frames=int(len(fr)),
                               states_feature_z_gt_8=int((zmax > 8).sum()), states_feature_z_gt_15=int((zmax > 15).sum()), states=int(ns))
    res["replay_counts"] = dict(decisions=int(nd), states=int(ns), keep_frac=float(keep.mean()), terminal=int(term.sum()),
                                terminal_success=int(succ_term.sum()), D_hist={int(k): int(v) for k, v in zip(*np.unique(D, return_counts=True))})
    return res, fails, dict(q=q, remi=remi, keep=keep, R=R, term=term, succ_term=succ_term, zmax=zmax)


# ----------------------------------------------------------------------------------------------- phase 1 helpers
def build_fwd(L, core):
    import jax
    import jax.numpy as jnp
    from sch01.learner import STATE_KEYS, masked_policy

    @jax.jit
    def fwd(Ap, Cp, Ctp, log_alpha, strength, b):
        s = {k: b["s_" + k] for k in STATE_KEYS}
        s2 = {k: b["n_" + k] for k in STATE_KEYS}
        a = b["action"]
        q = L.critic_def.apply({"params": Cp}, s)                       # [2,B,A] online critics
        qa = jnp.take_along_axis(q, a[None, :, None], -1)[..., 0]
        qt = L.critic_def.apply({"params": Ctp}, s)
        qta = jnp.take_along_axis(qt, a[None, :, None], -1)[..., 0]
        res2 = L.actor_def.apply({"params": Ap}, s2)
        p2, logp2 = masked_policy(res2, b["n_ref"], b["n_nfeas"], b["n_mask"], strength, P0)
        q2 = L.critic_def.apply({"params": Ctp}, s2).min(0)              # target critics at s'
        alpha = jnp.exp(log_alpha)
        eq = jnp.sum(p2 * jnp.where(b["n_mask"], q2, 0.0), -1)
        h = -jnp.sum(p2 * logp2, -1)
        disc = (GAMMA ** b["D"]) * (1.0 - b["terminal"])
        fut, ent = disc * eq, disc * alpha * h
        y = b["G"] + fut + ent
        # learner's own expression (sum p (q - alpha logp)) as a cross-check of the decomposition
        y_impl = b["G"] + disc * jnp.sum(p2 * jnp.where(b["n_mask"], q2 - alpha * logp2, 0.0), -1)
        q2max = jnp.where(b["n_mask"], q2, -jnp.inf).max(-1)
        qmin_s = q.min(0)
        gap = jnp.where(b["s_mask"], qmin_s, -jnp.inf).max(-1) - jnp.where(b["s_mask"], qmin_s, jnp.inf).min(-1)
        return dict(Q1=qa[0], Q2=qa[1], Qt1=qta[0], Qt2=qta[1], G=b["G"], fut=fut, ent=ent, y=y, y_impl=y_impl,
                    H_next=h, Eq_next=eq, q2max_next=q2max, alpha=alpha + 0 * h, gap_s=gap)
    return fwd


def run_fwd(fwd, params, fr, st, dc, idx, bs=512):
    import jax.numpy as jnp
    out = {}
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        pad = bs - len(j)
        jj = np.concatenate([j, np.repeat(j[:1], pad)]) if pad else j
        b = make_batch(fr, st, dc, jj)
        r = fwd(*params, {k: jnp.asarray(v) for k, v in b.items()})
        for k, v in r.items():
            out.setdefault(k, []).append(np.asarray(v)[:len(j)])
    return {k: np.concatenate(v) for k, v in out.items()}


def bound_hi(alpha, rem, mmax):
    """Upper bound on the soft value with fixed alpha: task return <= 1 (success bonus once, costs <= 0) plus at most
    one decision per control step, each adding alpha * H <= alpha * log(mmax) discounted per step."""
    return 1.0 + alpha * np.log(mmax) * (1 - GAMMA ** np.maximum(rem, 0)) / (1 - GAMMA)


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline_min", type=float, default=17.5)
    ap.add_argument("--cpu_smoke", action="store_true", help="small replay subset, fake env/sampler, few updates")
    ap.add_argument("--abl_updates", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    deadline = T0 + args.deadline_min * 60
    left = lambda: (deadline - time.time()) / 60  # noqa: E731
    status = dict(started=time.strftime("%F %T"), host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
                  git=subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(), phases={}, skipped=[])

    def save_status():
        status["elapsed_s"] = time.time() - T0
        json.dump(status, open(f"{args.out}/status.json", "w"), indent=1)

    if not args.cpu_smoke:
        from arli.pi0_policy import configure_caches
        configure_caches()
    import flax.serialization
    import jax
    import jax.numpy as jnp
    import optax
    from sch01.learner import Learner, STATE_KEYS, augment, masked_policy
    from sch02 import core
    status["jax_backend"] = jax.default_backend()
    log("backend", jax.default_backend())

    episodes = [json.loads(l) for l in open(f"{RUN}/episodes.jsonl")]
    fr, st, dc, parts = load_replay(RUN, max_parts=3 if args.cpu_smoke else None)
    nd = len(dc["action"])
    if args.cpu_smoke:
        episodes = episodes[:int((dc["terminal"] > 0.5).sum())]
    log(f"replay loaded read-only: {nd} decisions, {len(st['ref'])} states, {len(fr)} frames, {len(parts)} parts")
    nd_at = {}
    for c in CKPTS:
        e = [x for x in episodes if x["steps"] <= c]
        nd_at[c] = min(nd, e[-1]["replay"]) if e else nd
    # decision row -> training step at the end of its episode
    ep_end = np.array([e["replay"] for e in episodes])
    ep_steps = np.array([e["steps"] for e in episodes])
    row_steps = ep_steps[np.minimum(np.searchsorted(ep_end, np.arange(nd), side="right"), len(ep_steps) - 1)]
    status["replay"] = dict(decisions=nd, parts=parts, decisions_at_ckpt={str(k): int(v) for k, v in nd_at.items()},
                            grad_steps_total=episodes[-1]["grad_steps"], batch=256)
    save_status()

    # ---------------- phase 0: audit
    t = time.time()
    aud, fails, F = audit(fr, st, dc, core, episodes)
    json.dump(dict(checks=aud, failed=fails), open(f"{args.out}/audit.json", "w"), indent=1)
    status["phases"]["audit"] = dict(seconds=time.time() - t, failed=fails)
    save_status()
    log("audit", "OK" if not fails else fails)

    # ---------------- checkpoints
    L = Learner(0, dict(actor_lr=1e-4, critic_lr=3e-4, alpha_lr=3e-4, alpha_init=0.001, gamma=GAMMA, tau=TAU, p0=P0,
                        target_entropy_ratio=0.1))
    tmpl = dict(A=L.A, C=L.C, Ct=L.Ct, T=L.T, rng=L.rng)
    SD = {}
    for c in CKPTS:
        blob = open(f"{RUN}/ckpt/learner_{c:07d}.msgpack", "rb").read()
        import hashlib
        SD[c] = dict(sd=flax.serialization.from_bytes(tmpl, blob), sha=hashlib.sha256(blob).hexdigest(),
                     strength=core.prior_strength(c))
        SD[c]["alpha"] = float(np.exp(np.asarray(SD[c]["sd"]["T"].params["log_alpha"])))
    status["checkpoints"] = {str(c): dict(file=f"ckpt/learner_{c:07d}.msgpack", sha256=v["sha"], alpha=v["alpha"],
                                          prior_strength=v["strength"]) for c, v in SD.items()}
    save_status()
    mmax = int(st["nfeas"].max())

    # ---------------- phase 1: decomposition on fixed sets
    t = time.time()
    rng = np.random.default_rng(20261029)
    keep, term, remi = F["keep"], F["term"], F["remi"]
    rem_s = remi[dc["s"]]

    def pick(pool, k):
        pool = np.flatnonzero(pool)
        return np.sort(rng.choice(pool, size=min(k, len(pool)), replace=False)) if len(pool) else pool
    k0 = 256 if args.cpu_smoke else 2048
    sets = {"ordinary_all": pick(np.ones(nd, bool), k0),
            "ordinary_pre100k": pick(np.arange(nd) < nd_at[100247], k0 // 2),
            "KEEP": pick(keep, k0 // 4), "REPLAN": pick(~keep, k0 // 4),
            "terminal_success": pick(term & F["succ_term"], k0 // 4), "terminal_timeout": pick(term & ~F["succ_term"], k0 // 4),
            "rem_0_50": pick(rem_s < 50, k0 // 4), "rem_50_150": pick((rem_s >= 50) & (rem_s < 150), k0 // 4),
            "rem_150_300": pick((rem_s >= 150) & (rem_s < 300), k0 // 4), "rem_300_400": pick(rem_s >= 300, k0 // 4)}
    json.dump({k: v.tolist() for k, v in sets.items()}, open(f"{args.out}/fixed_sets.json", "w"))
    fwd = build_fwd(L, core)
    summ = {}
    with open(f"{args.out}/decomp_fixed.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["ckpt", "set", "row", "row_train_steps", "action", "R", "N", "D", "terminal", "success_terminal", "rem_steps",
                "q", "G", "fut", "ent", "y", "Q1", "Q2", "Qdis", "td1_sq", "td2_sq", "Qt1", "Qt2", "alpha", "H_next",
                "Eq_next", "q2max_next", "gap_s", "bound_hi", "max_input_z"]
        w.writerow(cols)
        for c in CKPTS:
            sd = SD[c]["sd"]
            params = (sd["A"].params, sd["C"].params, sd["Ct"], sd["T"].params["log_alpha"], jnp.float32(SD[c]["strength"]))
            summ[c] = {}
            for name, idx in sets.items():
                if not len(idx):
                    continue
                r = run_fwd(fwd, params, fr, st, dc, idx)
                assert np.allclose(r["y"], r["y_impl"], atol=1e-4 + 1e-5 * np.abs(r["y"]).max()), "decomposition != learner"
                act = dc["action"][idx]
                bh = bound_hi(SD[c]["alpha"], rem_s[idx], mmax)
                qmin = np.minimum(r["Q1"], r["Q2"])
                summ[c][name] = {k: qs(v) for k, v in dict(Q1=r["Q1"], Q2=r["Q2"], Qdis=np.abs(r["Q1"] - r["Q2"]), y=r["y"],
                                                            G=r["G"], fut=r["fut"], ent=r["ent"],
                                                            td_sq=(r["Q1"] - r["y"]) ** 2, H_next=r["H_next"]).items()}
                summ[c][name]["frac_Qmin_above_bound"] = float(np.mean(qmin > bh))
                summ[c][name]["frac_y_above_bound"] = float(np.mean(r["y"] > bh))
                summ[c][name]["ent_share_of_y_abs"] = float(np.abs(r["ent"]).sum() / max(1e-12, np.abs(r["y"]).sum()))
                summ[c][name]["n_nan"] = int(sum((~np.isfinite(v)).sum() for v in r.values()))
                for i, row in enumerate(idx):
                    a = int(act[i])
                    w.writerow([c, name, int(row), int(row_steps[row]), a, int(F["R"][row]), "" if a == 0 else 3 + (a - 1) % 8,
                                float(dc["D"][row]), int(term[row]), int(F["succ_term"][row]), int(rem_s[row]), int(F["q"][dc["s"][row]]),
                                *[f"{float(r[k][i]):.6g}" for k in ("G", "fut", "ent", "y", "Q1", "Q2")],
                                f"{abs(float(r['Q1'][i] - r['Q2'][i])):.6g}", f"{float((r['Q1'][i] - r['y'][i]) ** 2):.6g}",
                                f"{float((r['Q2'][i] - r['y'][i]) ** 2):.6g}", f"{float(r['Qt1'][i]):.6g}", f"{float(r['Qt2'][i]):.6g}",
                                f"{SD[c]['alpha']:.6g}", f"{float(r['H_next'][i]):.6g}", f"{float(r['Eq_next'][i]):.6g}",
                                f"{float(r['q2max_next'][i]):.6g}", f"{float(r['gap_s'][i]):.6g}", f"{float(bh[i]):.6g}",
                                f"{float(F['zmax'][dc['s'][row]]):.3g}"])
            log(f"decomp ckpt {c} done")
    json.dump({str(k): v for k, v in summ.items()}, open(f"{args.out}/decomp_fixed_summary.json", "w"), indent=1)
    status["phases"]["decomposition_fixed"] = dict(seconds=time.time() - t, set_sizes={k: int(len(v)) for k, v in sets.items()})
    save_status()

    # full-replay pass (aggregate statistics, kept separate from the stratified sets)
    t = time.time()
    full = {}
    for c in CKPTS:
        if left() < 11.5 and not args.cpu_smoke:
            status["skipped"].append(f"full-replay pass for {c} (time)")
            continue
        sd = SD[c]["sd"]
        params = (sd["A"].params, sd["C"].params, sd["Ct"], sd["T"].params["log_alpha"], jnp.float32(SD[c]["strength"]))
        r = run_fwd(fwd, params, fr, st, dc, np.arange(nd), bs=1024)
        np.savez_compressed(f"{args.out}/full_replay_{c}.npz", **{k: r[k].astype(np.float32) for k in
                                                                 ("Q1", "Q2", "G", "fut", "ent", "y", "H_next", "q2max_next", "gap_s")})
        qmin = np.minimum(r["Q1"], r["Q2"])
        bh = bound_hi(SD[c]["alpha"], rem_s, mmax)
        td = (r["Q1"] - r["y"]) ** 2
        top = np.argsort(-td)[:30]
        seen = np.arange(nd) < nd_at[c]
        full[c] = dict(all={k: qs(v) for k, v in dict(Qmin=qmin, y=r["y"], fut=r["fut"], ent=r["ent"], td_sq=td,
                                                       Qdis=np.abs(r["Q1"] - r["Q2"])).items()},
                       seen_by_ckpt=qs(qmin[seen]), unseen_by_ckpt=qs(qmin[~seen]) if (~seen).any() else None,
                       frac_Qmin_above_bound=float(np.mean(qmin > bh)), frac_y_above_bound=float(np.mean(r["y"] > bh)),
                       td_share_top1pct=float(np.sort(td)[-max(1, nd // 100):].sum() / td.sum()),
                       by_action={"KEEP": qs(qmin[keep]), "REPLAN": qs(qmin[~keep])},
                       top_td=[dict(row=int(i), td_sq=float(td[i]), Q1=float(r["Q1"][i]), Q2=float(r["Q2"][i]), y=float(r["y"][i]),
                                    G=float(r["G"][i]), fut=float(r["fut"][i]), ent=float(r["ent"][i]), action=int(dc["action"][i]),
                                    D=float(dc["D"][i]), terminal=bool(term[i]), rem=int(rem_s[i]), q=int(F["q"][dc["s"][i]]),
                                    row_train_steps=int(row_steps[i]), input_z=float(F["zmax"][dc["s"][i]]),
                                    img_mean=float(fr[st["f_now"][dc["s"][i]]].mean())) for i in top])
        log(f"full replay {c}: Qmin median {full[c]['all']['Qmin']['median']:.3g} p99 {full[c]['all']['Qmin']['p99']:.3g} "
            f"max {full[c]['all']['Qmin']['max']:.3g}")
    json.dump({str(k): v for k, v in full.items()}, open(f"{args.out}/full_replay_summary.json", "w"), indent=1)
    status["phases"]["full_replay"] = dict(seconds=time.time() - t, ckpts=[int(k) for k in full])
    save_status()

    # ---------------- phase 2: branching pilot (frozen 100k actor)
    if left() < 6.5 and not args.cpu_smoke:
        status["skipped"].append("branching pilot (time)")
    else:
        t = time.time()
        try:
            pilot(args, L, SD, core, status, save_status, left)
        except Exception as e:  # keep later phases and partial outputs
            import traceback
            status["phases"]["pilot_error"] = traceback.format_exc()
            log("PILOT ERROR", e)
        status["phases"].setdefault("pilot", {})["seconds"] = time.time() - t
        save_status()

    # ---------------- phase 3: critic-only ablation
    if left() < 2.0 and not args.cpu_smoke:
        status["skipped"].append("critic-only ablation (time)")
    else:
        t = time.time()
        ablation(args, L, SD, fr, st, dc, nd_at, core, status, save_status, left, jax, jnp, optax, augment, masked_policy,
                 STATE_KEYS, fwd)
        status["phases"].setdefault("ablation", {})["seconds"] = time.time() - t
    status["finished"] = time.strftime("%F %T")
    save_status()
    log("ALL DONE", json.dumps(status["skipped"]))


def pilot(args, L, SD, core, status, save_status, left):
    from sch02.executor import Executor, restore_snapshot, take_snapshot
    if args.cpu_smoke:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from fakes import FakeEnv, FakeSampler
        env, sampler = FakeEnv(), FakeSampler()
    else:
        from arli import aloha_env
        from arli.pi0_policy import FrozenPi0
        from sch01.sampler import RTCSampler
        env, sampler = aloha_env.make_env(), RTCSampler(FrozenPi0(10))
    exe = Executor(env, sampler, gamma=GAMMA)
    ck = 100247
    L.load_bytes(open(f"{RUN}/ckpt/learner_{ck:07d}.msgpack", "rb").read())
    strength = core.prior_strength(ck)

    def actor(st_, mask, t):
        ref, _ = core.reference_action(st_["q"], mask)
        p, _ = L.policy_probs(st_, mask, ref, strength)
        return int(np.argmax(p)), dict(kind="learned", p=float(p[int(np.argmax(p))]))

    seed = 10_000_000  # held-out eval seed (always solved in the logged evaluations; viewer category 'always_success')
    targets = [("early_movement", 40), ("grasp_contact", 130)]
    snaps = []

    def grab(ex, env_):
        if len(snaps) >= len(targets):
            return
        name, t_min = targets[len(snaps)]
        q, t = len(ex["queue"]), ex["t"]
        if t >= t_min and q - 1 >= core.MIN_DELAY and 400 - t > 1:
            st_ = exe._state(ex)
            snaps.append(dict(name=name, snap=take_snapshot(ex, env_, extra=dict(env_seed=seed)), state=st_,
                              mask=core.feasible_mask(q, 400 - t)))
    src = exe.run(seed, actor, np.random.default_rng([seed, 1]), on_tick=grab)
    out = dict(checkpoint=ck, seed=seed, source=dict(success=src["success"], t_end=src["t_end"], requests=len(src["requests"])),
               snapshots=[], branches=[], replay_checks=[])
    # critic scores of the candidates under each checkpoint (soft Q incl. entropy; not a task-return estimate)
    from sch01.learner import STATE_KEYS
    for sn in snaps:
        st_, mask = sn["state"], sn["mask"]
        ref, _ = core.reference_action(st_["q"], mask)
        p, _ = L.policy_probs(st_, mask, ref, strength)
        order = np.argsort(-p)
        pref = int(next(a for a in order if a != 0 and mask[a]))
        alt = core.encode(10, 10) if (mask[core.encode(10, 10)] and pref != core.encode(10, 10)) else core.encode(min(st_["q"], 15), 5)
        cands = [("KEEP", 0), ("actor_preferred_REPLAN", pref), ("alternative_REPLAN", int(alt))]
        assert all(mask[a] for _, a in cands)
        s1 = {k: np.asarray(st_[k])[None] for k in STATE_KEYS}
        scores = {}
        for c, v in SD.items():
            qq = np.asarray(L.critic_def.apply({"params": v["sd"]["C"].params}, s1))[:, 0]
            scores[str(c)] = {nm: dict(Q1=float(qq[0, a]), Q2=float(qq[1, a])) for nm, a in cands}
        sn["cands"] = cands
        out["snapshots"].append(dict(name=sn["name"], t=int(sn["snap"]["t"]), chunk=int(sn["snap"]["chunk"]), p=int(sn["snap"]["p"]),
                                     q=int(st_["q"]), actor_argmax=core.decode(int(order[0])), candidates=[
                                         dict(name=nm, action=int(a), decoded=core.decode(a), prob_100k=float(p[a])) for nm, a in cands],
                                     critic_scores=scores))
    json.dump(out, open(f"{args.out}/pilot.json", "w"), indent=1, default=str)
    # restoration check: physics state after restore, and one identical env step twice
    for sn in snaps:
        s = sn["snap"]
        ex = restore_snapshot(env, s)
        ph = env._env.physics
        d_state = float(np.abs(ph.get_state() - s["physics_state"]).max())
        a0 = ex["queue"][0]
        o1 = env.step(a0)[0]["pixels"]["top"].astype(int)
        st1 = env._env.physics.get_state().copy()
        restore_snapshot(env, s)
        o2 = env.step(a0)[0]["pixels"]["top"].astype(int)
        st2 = env._env.physics.get_state().copy()
        out["replay_checks"].append(dict(snapshot=sn["name"], physics_state_maxdiff_after_restore=d_state,
                                         one_step_physics_maxdiff=float(np.abs(st1 - st2).max()),
                                         one_step_image_maxdiff=int(np.abs(o1 - o2).max()),
                                         one_step_image_pixels_differing=int((o1 != o2).any(-1).sum())))
    reps = 3
    plan = [(sn, nm, a, rep) for rep in range(reps) for sn in snaps for nm, a in sn["cands"]]
    plan += [(snaps[k], snaps[k]["cands"][0][0], snaps[k]["cands"][0][1], 0) for k in range(len(snaps))]  # duplicates
    done_keys = set()
    for sn, nm, a, rep in plan:
        if left() < 2.5 and not args.cpu_smoke:
            status["skipped"].append(f"pilot branch {sn['name']}/{nm}/rep{rep} and later (time)")
            break
        s = sn["snap"]
        key = (sn["name"], nm, rep)
        dup = key in done_keys
        done_keys.add(key)
        nr = np.random.default_rng([seed, s["t"], rep, 101])
        t0 = time.time()
        r = exe.run(seed, actor, nr, snapshot=s, forced_first=a, continuation=actor)
        first = r["requests"][0] if r["requests"] else None
        out["branches"].append(dict(snapshot=sn["name"], t_snap=int(s["t"]), candidate=nm, action=int(a), rep=rep,
                                    duplicate_run=dup, success=bool(r["success"]), t_end=int(r["t_end"]),
                                    steps_to_termination=int(r["t_end"] - s["t"]), task_return=float(r["ret"]),
                                    discounted_task_return=float(r["discounted_return"]), inference_calls=len(r["requests"]),
                                    underrun=int(r["underrun_ticks"]), first_request=None if first is None else
                                    {k: first[k] for k in ("obs_t", "r", "n", "handoff_t", "discarded")},
                                    request_signature=[(int(q["obs_t"]), int(q["r"]), int(q["n"])) for q in r["requests"]][:80],
                                    wall_s=time.time() - t0))
        json.dump(out, open(f"{args.out}/pilot.json", "w"), indent=1, default=str)
        log(f"pilot {sn['name']} {nm} rep{rep}{' DUP' if dup else ''}: success {r['success']} t_end {r['t_end']} calls {len(r['requests'])}")
    status["phases"]["pilot"] = dict(branch_rollouts=len(out["branches"]), source_rollouts=1)


def ablation(args, L, SD, fr, st, dc, nd_at, core, status, save_status, left, jax, jnp, optax, augment, masked_policy,
             STATE_KEYS, fwd):
    ck = 100247
    sd = SD[ck]["sd"]
    alpha = jnp.float32(SD[ck]["alpha"])
    Ap = sd["A"].params
    strength = jnp.float32(SD[ck]["strength"])
    n_upd = 10 if args.cpu_smoke else args.abl_updates
    bsz = 256
    pool = nd_at[ck]
    rng = np.random.default_rng(424242)
    batches = [rng.integers(pool, size=bsz) for _ in range(n_upd)]
    keys = jax.random.split(jax.random.PRNGKey(7), n_upd)
    ev_rng = np.random.default_rng(99)
    evals = {"replay_pre100k": np.sort(ev_rng.choice(pool, size=min(2048, pool), replace=False))}
    if len(dc["action"]) > pool:
        evals["replay_post100k_unseen"] = np.sort(ev_rng.choice(np.arange(pool, len(dc["action"])), size=min(2048, len(dc["action"]) - pool), replace=False))

    def mk_update(backup_entropy):
        @jax.jit
        def upd(C, Ct, b, key):
            k1, k2 = jax.random.split(key)
            s = augment(k1, {k: b["s_" + k] for k in STATE_KEYS})
            s2 = augment(k2, {k: b["n_" + k] for k in STATE_KEYS})
            res2 = L.actor_def.apply({"params": Ap}, s2)
            p2, logp2 = masked_policy(res2, b["n_ref"], b["n_nfeas"], b["n_mask"], strength, P0)
            q2 = C.apply_fn({"params": Ct}, s2).min(0)
            ent_term = alpha * logp2 if backup_entropy else 0.0 * logp2
            v2 = jnp.sum(p2 * jnp.where(b["n_mask"], q2 - ent_term, 0.0), -1)
            y = jax.lax.stop_gradient(b["G"] + (GAMMA ** b["D"]) * (1.0 - b["terminal"]) * v2)
            a = b["action"]

            def loss_fn(cp):
                qs_ = C.apply_fn({"params": cp}, s)
                qa = jnp.take_along_axis(qs_, a[None, :, None], -1)[..., 0]
                return ((qa - y[None]) ** 2).mean(), qa
            (lc, qa), g = jax.value_and_grad(loss_fn, has_aux=True)(C.params)
            gn = optax.global_norm(g)
            C2 = C.apply_gradients(grads=g)
            Ct2 = optax.incremental_update(C2.params, Ct, TAU)
            return C2, Ct2, dict(loss=lc, q=qa.mean(), qmax=qa.max(), y=y.mean(), gnorm=gn)
        return upd

    def evaluate(C, Ct):
        o = {}
        for name, idx in evals.items():
            r = run_fwd(fwd, (Ap, C.params, Ct, jnp.log(alpha), strength), fr, st, dc, idx)
            y_noent = r["G"] + r["fut"]
            qmin = np.minimum(r["Q1"], r["Q2"])
            o[name] = dict(Qmin=qs(qmin), y_with_entropy=qs(r["y"]), y_without_entropy=qs(y_noent), ent=qs(r["ent"]),
                           td_sq_vs_y_with=qs((r["Q1"] - r["y"]) ** 2), td_sq_vs_y_without=qs((r["Q1"] - y_noent) ** 2),
                           Qdis=qs(np.abs(r["Q1"] - r["Q2"])))
        return o

    res = dict(checkpoint=ck, updates_per_branch=n_upd, batch=bsz, replay_rows_used=int(pool), alpha_frozen=float(alpha),
               actor_frozen=True, branches={})
    res["before"] = evaluate(sd["C"], sd["Ct"])
    total = 0
    for name, be in (("A_original_with_entropy", True), ("B_no_future_entropy", False)):
        upd = mk_update(be)
        C, Ct = sd["C"], sd["Ct"]  # identical starting critic, target critic and Adam state (immutable pytrees)
        trace = []
        for i in range(n_upd):
            if left() < 0.6 and not args.cpu_smoke:
                status["skipped"].append(f"ablation {name} stopped at update {i} (time)")
                break
            b = make_batch(fr, st, dc, batches[i])
            C, Ct, info = upd(C, Ct, {k: jnp.asarray(v) for k, v in b.items()}, keys[i])
            total += 1
            trace.append({k: float(v) for k, v in info.items()} | dict(update=i + 1))
        res["branches"][name] = dict(updates=len(trace), trace=trace, after=evaluate(C, Ct))
        json.dump(res, open(f"{args.out}/ablation.json", "w"), indent=1)
        log(f"ablation {name}: {len(trace)} updates, final loss {trace[-1]['loss'] if trace else None}")
    status["phases"]["ablation"] = dict(total_critic_updates=total)
    save_status()


if __name__ == "__main__":
    main()
