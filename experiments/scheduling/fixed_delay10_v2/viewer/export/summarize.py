"""Quantities quoted in viewer/OBSERVATIONS.md, computed from the exported viewer data (CPU).

    python viewer/export/summarize.py      # writes DATA_DIR/summary.json and prints a markdown digest
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_DIR, GAMMA  # noqa: E402
from validate import NA, probs  # noqa: E402


def main():
    root = f"{DATA_DIR}/episodes"
    cks = json.load(open(f"{DATA_DIR}/checkpoints.json"))
    seeds = [s["seed"] for s in json.load(open(f"{DATA_DIR}/seeds.json"))["seeds"]]
    pols = ["baseline"] + [c["id"] for c in cks]
    E = {}
    for p in pols:
        for s in seeds:
            e = f"{p}_s{s}"
            M = json.load(open(f"{root}/{e}/meta.json"))
            n = M["n_decisions"]
            E[e] = dict(M=M, n=n, masks=np.fromfile(f"{root}/{e}/masks.bin", np.uint8).reshape(n, NA).astype(bool),
                        rec=np.fromfile(f"{root}/{e}/rec.bin", np.float32).reshape(n, 2, NA))
    out = {}
    # 1. rollouts per policy
    roll = []
    for p in pols:
        ms = [E[f"{p}_s{s}"]["M"] for s in seeds]
        R = [q for m in ms for q in m["requests"]]
        dec = [(m, i) for m in ms for i in range(m["n_decisions"])]
        keep_feas = [(m, i) for m, i in dec if m["decisions"]["ticks_left"][i] > 1 and m["decisions"]["q"][i] - 1 >= 6]
        roll.append(dict(policy=p, successes=sum(m["success"] for m in ms), logged_successes=sum(m["logged"]["success"] for m in ms),
                         matches_log=sum(bool(m["outcome_matches_log"]) for m in ms),
                         mean_t_end_success=float(np.mean([m["t_end"] for m in ms if m["success"]])) if any(m["success"] for m in ms) else None,
                         requests_per_ep=len(R) / len(ms), mean_N=float(np.mean([q["n"] for q in R])),
                         mean_R=float(np.mean([q["r"] for q in R])), mean_launch_p=float(np.mean([q["p_at_launch"] for q in R])),
                         mean_discarded=float(np.mean([q["discarded"] for q in R])),
                         keep_rate_when_feasible=float(np.mean([m["decisions"]["action"][i] == 0 for m, i in keep_feas])) if keep_feas else None,
                         launch_p0_frac=float(np.mean([q["p_at_launch"] == 0 for q in R]))))
    out["rollouts"] = roll
    # 2. locked probes: all decision states of the 6 fixed-baseline rollouts, every checkpoint
    rows = []
    for c in cks:
        st = dict(ck=c["id"], steps=c["train_steps"], strength=c["prior_strength"], states=0, argmax_keep=0, argmax_ref=0,
                  argmax_keep_off=0, pmax=[], EN=[], ER=[], adv=[], dis=[], dis_all=[], keep_states=0)
        for s in seeds:
            e = f"baseline_s{s}"
            d = E[e]
            M = d["M"]
            pr = np.fromfile(f"{root}/{e}/probes/{c['id']}.bin", np.float32).reshape(d["n"], 3, NA)
            for i in range(d["n"]):
                m = d["masks"][i]
                ref = M["decisions"]["ref"][i]
                p_on = probs(pr[i, 0], m, ref, c["prior_strength"])
                p_off = probs(pr[i, 0], m, ref, 0.0)
                a = int(np.argmax(p_on))
                qm = np.minimum(pr[i, 1], pr[i, 2]).astype(np.float64)
                dis = np.abs(pr[i, 1] - pr[i, 2]).astype(np.float64)
                st["states"] += 1
                st["keep_states"] += int(m[0])
                st["argmax_keep"] += int(a == 0)
                st["argmax_keep_off"] += int(np.argmax(p_off) == 0)
                st["argmax_ref"] += int(a == ref)
                st["pmax"].append(p_on[a])
                rep = p_on[1:].sum()
                if rep > 1e-6:
                    nn = np.array([3 + (k - 1) % 8 for k in range(1, NA)])
                    rr = np.array([(k - 1) // 8 + 1 for k in range(1, NA)])
                    st["EN"].append(float((p_on[1:] * nn).sum() / rep))
                    st["ER"].append(float((p_on[1:] * rr).sum() / rep))
                st["adv"].append(qm[a] - qm[ref])
                st["dis"].append(dis[a])
                st["dis_all"].append(float(dis[m].mean()))
        rows.append(dict(ck=st["ck"], steps=st["steps"], prior_strength=st["strength"], states=st["states"],
                         keep_feasible_states=st["keep_states"],
                         argmax_keep_frac=st["argmax_keep"] / st["states"],
                         argmax_keep_frac_prior_off=st["argmax_keep_off"] / st["states"],
                         argmax_equals_baseline_frac=st["argmax_ref"] / st["states"],
                         median_pmax=float(np.median(st["pmax"])),
                         mean_EN_given_replan=float(np.mean(st["EN"])) if st["EN"] else None,
                         mean_ER_given_replan=float(np.mean(st["ER"])) if st["ER"] else None,
                         median_q_adv_argmax_vs_baseline=float(np.median(st["adv"])),
                         median_q_disagreement_at_argmax=float(np.median(st["dis"])),
                         median_q_disagreement_mean_feasible=float(np.median(st["dis_all"]))))
    out["probes_on_baseline_states"] = rows
    # 3. predicted Q (own checkpoint, selected action) vs observed discounted return-to-go at the same tick
    qv = []
    for c in cks:
        pred, obs = [], []
        for s in seeds:
            e = f"{c['id']}_s{s}"
            d = E[e]
            M = d["M"]
            rw = np.asarray(M["ticks"]["reward"])
            rtg = np.zeros(len(rw) + 1)
            for t in range(len(rw) - 1, -1, -1):
                rtg[t] = rw[t] + GAMMA * rtg[t + 1]
            pr = np.fromfile(f"{root}/{e}/probes/{c['id']}.bin", np.float32).reshape(d["n"], 3, NA)
            for i in range(d["n"]):
                a = M["decisions"]["action"][i]
                pred.append(float(min(pr[i, 1, a], pr[i, 2, a])))
                obs.append(float(rtg[M["decisions"]["t"][i]]))
        pred, obs = np.asarray(pred), np.asarray(obs)
        qv.append(dict(ck=c["id"], steps=c["train_steps"], n=len(pred), mean_pred=float(pred.mean()), mean_obs=float(obs.mean()),
                       corr=float(np.corrcoef(pred, obs)[0, 1]) if pred.std() > 0 and obs.std() > 0 else None))
    out["q_vs_observed"] = qv
    json.dump(out, open(f"{DATA_DIR}/summary.json", "w"), indent=1)
    print("| policy | successes (new / logged) | outcome+t_end match log | requests/ep | mean N | mean R | launch p | p=0 launches | discarded | KEEP rate when feasible |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in roll:
        print(f"| {r['policy']} | {r['successes']}/6 / {r['logged_successes']}/6 | {r['matches_log']}/6 | {r['requests_per_ep']:.1f} | "
              f"{r['mean_N']:.2f} | {r['mean_R']:.2f} | {r['mean_launch_p']:.2f} | {r['launch_p0_frac']:.2f} | {r['mean_discarded']:.2f} | "
              f"{r['keep_rate_when_feasible'] if r['keep_rate_when_feasible'] is None else round(r['keep_rate_when_feasible'], 3)} |")
    print()
    print("| checkpoint | prior | argmax KEEP (incl. prior / net only) | argmax = baseline | median max prob | E[N|REPLAN] | E[R|REPLAN] | median Q adv (argmax − baseline) | median |Q1−Q2| at argmax |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['steps']:,} | {r['prior_strength']:.2f} | {r['argmax_keep_frac']:.2f} / {r['argmax_keep_frac_prior_off']:.2f} | "
              f"{r['argmax_equals_baseline_frac']:.2f} | {r['median_pmax']:.3f} | {r['mean_EN_given_replan']:.2f} | {r['mean_ER_given_replan']:.2f} | "
              f"{r['median_q_adv_argmax_vs_baseline']:.4f} | {r['median_q_disagreement_at_argmax']:.4f} |")
    print(f"\n(locked-probe states: {rows[0]['states']} decision states from the 6 fixed-baseline rollouts; "
          f"{rows[0]['keep_feasible_states']} with KEEP feasible)\n")
    print("| checkpoint | decisions | mean predicted Q(selected) | mean observed return-to-go | corr |")
    print("|---|---|---|---|---|")
    for r in qv:
        print(f"| {r['steps']:,} | {r['n']} | {r['mean_pred']:.3f} | {r['mean_obs']:.3f} | {r['corr'] if r['corr'] is None else round(r['corr'], 2)} |")


if __name__ == "__main__":
    main()
