"""CPU validation of the exported viewer data (run after record_rollouts.py; writes DATA_DIR/validation.json).

Checks per recorded episode:
  masks     every stored feasibility mask == core.feasible_mask(q, 400 - t) (delay table + queue)
  probs     recorded probabilities: zero on infeasible actions, P(KEEP) + sum P(REPLAN) == 1
  recompute recorded probabilities == softmax(residual + prior(ref, mask, checkpoint strength)) (the viewer's formula)
  actions   learned: recorded action == argmax of the recorded joint distribution; baseline: == reference rule
  timeline  ready = obs + d(N); handoff = obs + R; d(N) <= R <= q at launch; discarded = q - R; old executed = p + R;
            actual handoffs (chunk id changes) == planned handoffs of non-abandoned requests; abandoned iff the
            planned handoff is at/after the episode end; decisions only at ticks with no request in flight
  frames    t_end + 1 frames, every JPEG decodes at the stated size
  rewards   per-tick rewards sum to the return; discounted sum == executor's discounted return == closed form
  inputs    served inputs/net images == archived full scheduler inputs; probe files present for every checkpoint
  prior     checkpoints with strength 0: prior-on == prior-off distributions

    python viewer/export/validate.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ARCHIVE_DIR, DATA_DIR, GAMMA, MAX_TICKS, reconstructed_discounted_return  # noqa: E402

DELAY = {3: 6, 4: 7, 5: 7, 6: 8, 7: 9, 8: 9, 9: 10, 10: 10}
NA, P0 = 201, 0.999


def enc(r, n):
    return 1 + (r - 1) * 8 + (n - 3)


def feasible_mask(q, ticks_left):
    m = np.zeros(NA, bool)
    for n in DELAY:
        for r in range(DELAY[n], min(q, 25) + 1):
            if r + 25 <= 50:
                m[enc(r, n)] = True
    if q - 1 >= 6 and ticks_left > 1:
        m[0] = True
    return m


def reference(q, mask):
    if q > 10 and mask[0]:
        return 0
    if q == 10 and mask[enc(10, 10)]:
        return enc(10, 10)
    r = min(q, 25)
    for n in sorted(DELAY, reverse=True):
        if DELAY[n] <= r and mask[enc(r, n)]:
            return enc(r, n)
    return int(np.flatnonzero(mask)[0])


def probs(res, mask, ref, s):
    lg = res.astype(np.float64).copy()
    m = int(mask.sum())
    if m > 1:
        lg[ref] += s * np.log(P0 * (m - 1) / (1 - P0))
    lg[~mask] = -np.inf
    p = np.exp(lg - lg.max())
    return p / p.sum()


def main(data=DATA_DIR, arch=ARCHIVE_DIR):
    import cv2
    root = f"{data}/episodes"
    cks = json.load(open(f"{data}/checkpoints.json"))
    eps = sorted(d for d in os.listdir(root) if os.path.exists(f"{root}/{d}/meta.json"))
    tot = dict(episodes=0, decisions=0, frames=0, max_sum_err=0.0, max_recompute_err=0.0, prior_equal_checked=0)
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    for e in eps:
        M = json.load(open(f"{root}/{e}/meta.json"))
        n, T = M["n_decisions"], M["t_end"]
        D, R, tk = M["decisions"], M["requests"], M["ticks"]
        masks = np.fromfile(f"{root}/{e}/masks.bin", np.uint8).reshape(n, NA).astype(bool)
        rec = np.fromfile(f"{root}/{e}/rec.bin", np.float32).reshape(n, 2, NA)
        learned = M["policy"]["kind"] == "learned"
        for i in range(n):
            t, q = D["t"][i], D["q"][i]
            check(np.array_equal(masks[i], feasible_mask(q, MAX_TICKS - t)), f"{e} mask {i}")
            check(D["ref"][i] == reference(q, masks[i]), f"{e} ref {i}")
            a = D["action"][i]
            check(masks[i][a], f"{e} infeasible action {i}")
            if learned:
                p = rec[i, 0].astype(np.float64)
                check(np.all(p[~masks[i]] == 0), f"{e} mass on infeasible {i}")
                tot["max_sum_err"] = max(tot["max_sum_err"], abs(p[0] + p[1:].sum() - 1))
                check(int(np.argmax(p)) == a, f"{e} action != argmax {i}")
                pr = probs(rec[i, 1], masks[i], D["ref"][i], M["policy"]["prior_strength"])
                tot["max_recompute_err"] = max(tot["max_recompute_err"], float(np.abs(pr - p).max()))
            else:
                check(a == D["ref"][i], f"{e} baseline action != rule {i}")
        # timeline
        for j, r in enumerate(R):
            check(r["ready_t"] == r["obs_t"] + DELAY[r["n"]] and r["handoff_t"] == r["obs_t"] + r["r"], f"{e} req {j} timing")
            check(DELAY[r["n"]] <= r["r"] <= r["q_at_launch"] and r["r"] + 25 <= 50, f"{e} req {j} feasibility")
            check(r["discarded"] == r["q_at_launch"] - r["r"] and r["old_executed"] == r["p_at_launch"] + r["r"], f"{e} req {j} counts")
            check(bool(r.get("abandoned")) == (r["handoff_t"] >= T), f"{e} req {j} abandoned flag")
            check(r["obs_t"] in D["t"], f"{e} req {j} launched without a decision")
        actual = [t for t in range(1, len(tk["t"])) if tk["chunk"][t] != tk["chunk"][t - 1]]
        planned = [r["handoff_t"] for r in R if not r.get("abandoned")]
        check(actual == planned, f"{e} handoffs actual {actual[:5]} != planned {planned[:5]}")
        busy = set()
        for r in R:
            busy.update(range(r["obs_t"] + 1, min(r["handoff_t"], T + 1)))
        dec_t = set(D["t"])
        idle = set(range(0, T)) - busy
        check(dec_t - {T} <= idle and idle <= dec_t, f"{e} decision ticks != idle ticks")
        # frames
        offs = M["frame_offsets"]
        check(len(offs) == T + 2, f"{e} frame count")
        buf = open(f"{root}/{e}/frames.bin", "rb").read()
        check(len(buf) == offs[-1], f"{e} frames.bin size")
        for t in (0, T // 2, T):
            img = cv2.imdecode(np.frombuffer(buf[offs[t]:offs[t + 1]], np.uint8), cv2.IMREAD_COLOR)
            check(img is not None and img.shape[:2] == (M["frame_size"][1], M["frame_size"][0]), f"{e} frame {t} decode")
        # rewards
        rw = np.asarray(tk["reward"])
        check(len(rw) == T and abs(rw.sum() - M["ret"]) < 1e-9, f"{e} reward sum")
        disc = float(sum(GAMMA ** k * r for k, r in enumerate(rw)))
        check(abs(disc - M["discounted_return"]) < 1e-9, f"{e} discounted return")
        if M["underrun_ticks"] == 0:
            check(abs(reconstructed_discounted_return(T, M["success"]) - disc) < 1e-9, f"{e} closed-form discounted return")
        # inputs vs archive, probes
        z = np.load(f"{arch}/{e}.npz")
        inp = np.fromfile(f"{root}/{e}/inputs.bin", np.float32).reshape(n, 408)
        ref_inp = np.concatenate([z["queue"].reshape(n, -1), z["qmask"], z["robot"], z["runtime"]], 1)
        check(np.array_equal(inp, ref_inp.astype(np.float32)), f"{e} inputs != archive")
        ni = np.fromfile(f"{root}/{e}/netimg.bin", np.uint8).reshape(n, 64, 64, 3)
        check(np.array_equal(ni, z["img_now"]), f"{e} netimg != archive")
        check(np.array_equal(z["mask"], masks), f"{e} archive mask")
        for c in cks:
            pf = f"{root}/{e}/probes/{c['id']}.bin"
            ok = os.path.exists(pf) and os.path.getsize(pf) == n * 3 * NA * 4
            check(ok, f"{e} probe {c['id']}")
            if ok and c["prior_strength"] == 0 and n:
                pa = np.fromfile(pf, np.float32).reshape(n, 3, NA)
                i = n // 2
                check(np.array_equal(probs(pa[i, 0], masks[i], D["ref"][i], 0.0),
                                     probs(pa[i, 0], masks[i], D["ref"][i], c["prior_strength"])), f"{e} prior0")
                tot["prior_equal_checked"] += 1
        tot["episodes"] += 1
        tot["decisions"] += n
        tot["frames"] += T + 1
    pc = json.load(open(f"{data}/probe_check.json")) if os.path.exists(f"{data}/probe_check.json") else []
    out = dict(totals=tot, n_failures=len(fails), failures=fails[:200],
               probe_check=dict(episodes=len(pc), argmax_agree=sum(r["argmax_agree"] for r in pc),
                                decisions=sum(r["decisions"] for r in pc),
                                max_abs_residual_diff=max([r["max_abs_residual_diff"] for r in pc], default=None),
                                max_abs_prob_diff=max([r["max_abs_prob_diff"] for r in pc], default=None)))
    json.dump(out, open(f"{data}/validation.json", "w"), indent=1)
    print(json.dumps(dict(totals=tot, n_failures=len(fails), first=fails[:10], probe_check=out["probe_check"]), indent=1))
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*a) if a else main()
