"""Metrics, figures and videos for the action-disagreement probe (CPU).

Q_t[h] = queued old action for physical time t+h (h < q); beyond q the old full plan at index t - origin + h is
prediction-only context (valid while t - origin + h < 50).  B_t[h] = fresh plan action for time t+h.
Regions: committed prefix 0 <= h < min(d, q); replaceable execution window d <= h < q (empty -> N/A if q <= d);
prediction-only q <= h < 50 - (t - origin).
Arm error: RMS over the 12 arm joints of (old - fresh) / std_j, std_j = the pi0_aloha_sim action norm-stat std
(z-score normalisation is the policy's own; aloha->pi joint flips only change signs).  Grippers (dims 6, 13): absolute
difference in env action units (normalised gripper position), reported separately.  Window scalars are RMS over the
window's (offset, joint) entries; discrepancies are computed per generated sample, then summarised.

    python action_disagreement_probe/analyze.py [--src DIR]
"""
import argparse
import csv
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [10_000_014, 10_000_016, 10_000_004]
LABEL = {10_000_014: "seed 10000014 (typical success)", 10_000_016: "seed 10000016 (marginal / late success)",
         10_000_004: "seed 10000004 (failure)"}
D, H = 10, 50
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
LEFT, RIGHT, GRIP = [0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12], [6, 13]
NORM = "/N/scratch/jindjia/arli/checkpoints/pi0_aloha_sim/assets/lerobot/aloha_sim_transfer_cube_human/norm_stats.json"
VAR = ["RTC off", "RTC on"]


def per_offset(old, new, std):
    """old, new: [..., H, 14] -> dict of [..., H] discrepancies (NaN propagates)."""
    z = (old - new) / std
    return dict(arm=np.sqrt(np.mean(z[..., ARM] ** 2, -1)), left=np.sqrt(np.mean(z[..., LEFT] ** 2, -1)),
                right=np.sqrt(np.mean(z[..., RIGHT] ** 2, -1)), arm_rad=np.sqrt(np.mean((old - new)[..., ARM] ** 2, -1)),
                grip_l=np.abs(old - new)[..., 6], grip_r=np.abs(old - new)[..., 13])


def wrms(x, lo, hi):
    if hi <= lo:
        return np.nan
    v = x[lo:hi]
    return float(np.sqrt(np.nanmean(v ** 2))) if np.isfinite(v).any() else np.nan


def episode(src, seed, std):
    Z = np.load(f"{src}/reference/ref_{seed}.npz")
    P = np.load(f"{src}/probes/probe_{seed}.npz")
    meta = json.load(open(f"{src}/reference/ref_{seed}.json"))
    T = len(Z["t"])
    q, plan, origin = Z["q"], Z["plan"], Z["plan_origin"]
    acts = P["acts"]  # [T, 2, 2, H, 14]
    old = np.full((T, H, 14), np.nan, np.float32)
    for t in range(T):
        sh = t - origin[plan[t]]
        old[t, :H - sh] = Z["plan_actions"][plan[t]][sh:]
        old[t, :q[t]] = Z["queue"][t, :q[t]]
    oldf = per_offset(old[:, None, None], acts, std)              # [T, 2 var, 2 samp, H]
    ff = per_offset(acts[:, :, 0], acts[:, :, 1], std)              # [T, 2 var, H]
    valid_old = np.isfinite(old[..., 0])                            # [T, H]
    for k in ff:  # compare fresh-fresh on the same offsets as old-fresh
        ff[k] = np.where(valid_old[:, None], ff[k], np.nan)
    rows = []
    for t in range(T):
        sh = int(t - origin[plan[t]])
        qt, lim = int(q[t]), H - sh
        r = dict(t=t, chunk=int(Z["chunk"][t]), p=int(Z["p"][t]), q=qt, obs_age_of_executing_plan=sh,
                 pending=int(Z["pending"][t, 0] >= 0 and Z["pending"][t, 0] != t), eligible=int(P["eligible"][t]),
                 stage=int(Z["stage"][t]), n_committed=min(D, qt), n_replaceable=max(0, qt - D), n_prediction_only=max(0, lim - qt))
        for v in (0, 1):
            vn = "off" if v == 0 else "on"
            for s in (0, 1):
                for m in ("arm", "left", "right", "grip_l", "grip_r"):
                    x = oldf[m][t, v, s]
                    r[f"{vn}_s{s + 1}_{m}_replaceable"] = wrms(x, D, qt)
                r[f"{vn}_s{s + 1}_arm_first5"] = wrms(oldf["arm"][t, v, s], D, D + 5) if qt >= D + 5 else np.nan
                r[f"{vn}_s{s + 1}_arm_committed"] = wrms(oldf["arm"][t, v, s], 0, min(D, qt))
                r[f"{vn}_s{s + 1}_arm_prediction_only"] = wrms(oldf["arm"][t, v, s], qt, lim)
            for m in ("arm", "grip_l", "grip_r"):
                r[f"{vn}_freshfresh_{m}_replaceable"] = wrms(ff[m][t, v], D, qt)
            r[f"{vn}_freshfresh_arm_first5"] = wrms(ff["arm"][t, v], D, D + 5) if qt >= D + 5 else np.nan
        rows.append(r)
    return dict(Z=Z, P=P, meta=meta, T=T, old=old, oldf=oldf, ff=ff, rows=rows)


def events(Z):
    st = Z["stage"]
    return {k: (int(np.flatnonzero(st >= k)[0] + 1) if (st >= k).any() else None) for k in (1, 2, 3, 4)}


def col(rows, key):
    return np.array([r[key] for r in rows], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/N/scratch/jindjia/arli/runs_scheduling/action_disagreement_probe")
    ap.add_argument("--no_video", action="store_true")
    a = ap.parse_args()
    out = f"{HERE}/results"
    os.makedirs(out, exist_ok=True)
    std = np.asarray(json.load(open(NORM))["norm_stats"]["actions"]["std"][:14], np.float32)
    E = {s: episode(a.src, s, std) for s in SEEDS if os.path.exists(f"{a.src}/probes/probe_{s}.npz")}
    # consistent colour scale across episodes and variants
    allv = np.concatenate([np.nanmean(e["oldf"]["arm"], 2)[np.isfinite(np.nanmean(e["oldf"]["arm"], 2))] for e in E.values()])
    vmax = float(np.quantile(allv, 0.99))
    clip = float(np.mean(allv > vmax))
    summary = dict(color_scale=dict(metric="arm RMS of (old - fresh)/std, mean of the two samples", vmin=0, vmax=vmax,
                                    clipped_fraction=clip), arm_std=std[ARM].tolist(), episodes={})
    for seed, e in E.items():
        Z, rows, T = e["Z"], e["rows"], e["T"]
        with open(f"{out}/timeseries_{seed}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
        np.savez_compressed(f"{a.src}/discrepancy_{seed}.npz", **{f"oldfresh_{k}": v.astype(np.float32) for k, v in e["oldf"].items()},
                            **{f"freshfresh_{k}": v.astype(np.float32) for k, v in e["ff"].items()})
        ev = events(Z)
        # segment medians (eligible steps with a non-empty replaceable window)
        t1, t2, t3, t4 = ev[1], ev[2], ev[3], ev[4]
        segs = [("free motion", 0, (t1 or T) - 25), ("approach", (t1 or T) - 25, t1 or T)]
        if t1:
            segs.append(("grasp / lift", t1, (t2 or t1) + 25))
            if t3:
                segs += [("carry", (t2 or t1) + 25, t3 - 30), ("transfer approach", t3 - 30, T)]
            else:
                segs.append(("after grasp (no transfer)", (t2 or t1) + 25, T))
        segsum = []
        for name, lo, hi in segs:
            lo, hi = max(0, lo), min(T, hi)
            idx = [i for i in range(lo, hi) if rows[i]["eligible"] and rows[i]["n_replaceable"] > 0]
            if not idx:
                continue
            d = {}
            for vn in ("off", "on"):
                oldv = np.nanmean([[rows[i][f"{vn}_s{s}_arm_replaceable"] for s in (1, 2)] for i in idx], 1)
                d[vn] = dict(median_oldfresh=float(np.nanmedian(oldv)), median_freshfresh=float(np.nanmedian([rows[i][f"{vn}_freshfresh_arm_replaceable"] for i in idx])),
                             median_grip_l=float(np.nanmedian([np.nanmean([rows[i][f"{vn}_s{s}_grip_l_replaceable"] for s in (1, 2)]) for i in idx])),
                             median_grip_r=float(np.nanmedian([np.nanmean([rows[i][f"{vn}_s{s}_grip_r_replaceable"] for s in (1, 2)]) for i in idx])))
            segsum.append(dict(segment=name, t=[lo, hi], n_steps=len(idx), **d))
        # peaks (RTC off, mean of samples, eligible, >= 5 replaceable actions), separated by >= 25 steps
        rep = np.nanmean([col(rows, "off_s1_arm_replaceable"), col(rows, "off_s2_arm_replaceable")], 0)
        ok = (col(rows, "eligible") > 0) & (col(rows, "n_replaceable") >= 5)
        order = np.argsort(-np.where(ok, rep, -1))
        peaks = []
        for i in order:
            if not ok[i] or len(peaks) >= 4:
                break
            if all(abs(i - j) >= 25 for j in peaks):
                peaks.append(int(i))
        summary["episodes"][str(seed)] = dict(meta={k: e["meta"][k] for k in ("success", "t_end", "steps", "regenerated")}, events=ev,
                                              segments=segsum, peaks=[dict(t=p, chunk=rows[p]["chunk"], p=rows[p]["p"], q=rows[p]["q"], stage=rows[p]["stage"],
                                                                           off_arm_replaceable=float(rep[p]),
                                                                           on_arm_replaceable=float(np.nanmean([rows[p]["on_s1_arm_replaceable"], rows[p]["on_s2_arm_replaceable"]])),
                                                                           off_freshfresh=rows[p]["off_freshfresh_arm_replaceable"]) for p in peaks],
                                              overall=dict(
                                                  median_off_replaceable=float(np.nanmedian(rep[ok])),
                                                  median_on_replaceable=float(np.nanmedian(np.nanmean([col(rows, "on_s1_arm_replaceable"), col(rows, "on_s2_arm_replaceable")], 0)[ok])),
                                                  median_off_freshfresh=float(np.nanmedian(col(rows, "off_freshfresh_arm_replaceable")[ok])),
                                                  median_on_freshfresh=float(np.nanmedian(col(rows, "on_freshfresh_arm_replaceable")[ok])),
                                                  median_off_committed=float(np.nanmedian(np.nanmean([col(rows, "off_s1_arm_committed"), col(rows, "off_s2_arm_committed")], 0)[ok])),
                                                  median_on_committed=float(np.nanmedian(np.nanmean([col(rows, "on_s1_arm_committed"), col(rows, "on_s2_arm_committed")], 0)[ok])),
                                                  median_off_prediction_only=float(np.nanmedian(np.nanmean([col(rows, "off_s1_arm_prediction_only"), col(rows, "off_s2_arm_prediction_only")], 0)[ok])),
                                                  median_on_prediction_only=float(np.nanmedian(np.nanmean([col(rows, "on_s1_arm_prediction_only"), col(rows, "on_s2_arm_prediction_only")], 0)[ok])))
                                              )
        figure(seed, e, vmax, clip, ev, peaks, out)
        for k, pk in enumerate(peaks[:3]):
            zoom(seed, e, vmax, pk, k, out, std)
        if not a.no_video:
            video(seed, e, a.src)
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    snapshot_table(a.src, std, out)
    print(json.dumps(summary, indent=1)[:4000])


def _overlays(ax, e, ev, T):
    Z = e["Z"]
    q = Z["q"]
    tt = np.arange(T)
    ax.step(tt, q, where="mid", color="white", lw=1.2)
    lim = H - (tt - Z["plan_origin"][Z["plan"]])
    ax.step(tt, lim, where="mid", color="#bbbbbb", lw=0.8, ls=":")
    ax.axhline(D, color="#7fd4ff", lw=1.2, ls="--")
    hand = [t for t in range(1, T) if Z["chunk"][t] != Z["chunk"][t - 1]]
    for t in hand:
        ax.axvline(t, color="white", lw=0.5, alpha=0.5)
    launches = sorted(set(int(x) for x in Z["pending"][:, 0] if x >= 0))
    ax.scatter(launches, [H - 1] * len(launches), marker="v", color="#ffcc00", s=14, zorder=5)
    for k, c in ((1, "#00e5ff"), (2, "#7cff4f"), (3, "#ff7ce5"), (4, "#ffffff")):
        if ev[k]:
            ax.axvline(ev[k], color=c, lw=1.4, ls="-.")


def figure(seed, e, vmax, clip, ev, peaks, out):
    Z, rows, T = e["Z"], e["rows"], e["T"]
    tt = np.arange(T)
    elig = col(rows, "eligible") > 0
    fig = plt.figure(figsize=(15, 17))
    gs = fig.add_gridspec(6, 1, height_ratios=[1.1, 2.2, 2.2, 1.3, 1.3, 1.3], hspace=0.32, bottom=0.11)
    # 1 timeline + frames
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    nthumb = 9
    for k, t in enumerate(np.linspace(0, T - 1, nthumb).astype(int)):
        im = cv2.cvtColor(cv2.resize(Z["img"][t], (160, 120)), cv2.COLOR_BGR2RGB)[:, :, ::-1]
        x0 = t - T / nthumb / 2.2
        ax.imshow(im, extent=(x0, x0 + T / nthumb / 1.1, 0.25, 1.0), aspect="auto", zorder=1)
        ax.text(t, 0.17, f"t={t}", ha="center", fontsize=7)
    ax.step(tt, Z["stage"] / 4 * 0.15, where="post", color="#d9632b", lw=1)
    ax.text(0, 0.02, "contact stage (0-4)", fontsize=7, color="#d9632b")
    ax.set_title(f"{LABEL[seed]} — regenerated reference: {'success' if e['meta']['success'] else 'no success'} at step {e['meta']['t_end']}. "
                 f"Events: right-finger contact {ev[1]}, lifted {ev[2]}, left-finger contact {ev[3]}, success {ev[4]}", fontsize=9, loc="left")
    ax.set_aspect("auto")
    # 2-3 heatmaps
    cmap = matplotlib.colormaps["magma"].copy()
    cmap.set_bad("#d8d8d8")
    for i, v in enumerate((0, 1)):
        ax = fig.add_subplot(gs[1 + i])
        M = np.nanmean(e["oldf"]["arm"][:, v], 1).T  # [H, T] mean of the two samples' discrepancies
        im = ax.imshow(np.ma.masked_invalid(M), origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                       extent=(-0.5, T - 0.5, -0.5, H - 0.5), interpolation="nearest")
        _overlays(ax, e, ev, T)
        # ineligible strip
        for t in np.flatnonzero(~elig):
            ax.add_patch(plt.Rectangle((t - 0.5, H - 0.5), 1, 2.5, color="#555555", clip_on=False, lw=0))
        ax.set_ylim(-0.5, H + 2)
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylabel("future offset h (steps)")
        ax.set_title(f"Action Disagreement — {VAR[v]}: arm RMS of (queued/old − fresh)/std, mean over the two probe samples"
                     + ("" if v == 0 else "  (N/A = grey: no new request allowed)"), fontsize=9, loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label(f"normalised RMS (clipped above {vmax:.2f}: {100 * clip:.1f}% of cells)", fontsize=7)
    # 4 arm curves
    ax = fig.add_subplot(gs[3])
    for vn, c in (("off", "#d9632b"), ("on", "#2a63c4")):
        for s, ls in ((1, "-"), (2, "--")):
            ax.plot(tt, col(rows, f"{vn}_s{s}_arm_replaceable"), color=c, ls=ls, lw=1, label=f"old vs fresh {s}, RTC {vn}")
    ax.set_ylabel("arm RMS\n(replaceable window)")
    _shade(ax, elig, T)
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_title("Summary: arm disagreement over the replaceable execution window d ≤ h < q (N/A when q ≤ d or no request allowed)", fontsize=9, loc="left")
    # 5 gripper curves
    ax = fig.add_subplot(gs[4])
    for vn, c in (("off", "#d9632b"), ("on", "#2a63c4")):
        ax.plot(tt, np.nanmean([col(rows, f"{vn}_s{s}_grip_l_replaceable") for s in (1, 2)], 0), color=c, lw=1, label=f"left gripper, RTC {vn}")
        ax.plot(tt, np.nanmean([col(rows, f"{vn}_s{s}_grip_r_replaceable") for s in (1, 2)], 0), color=c, lw=1, ls=":", label=f"right gripper, RTC {vn}")
    ax.set_ylabel("gripper |Δ|\n(env units, RMS)")
    _shade(ax, elig, T)
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_title("Summary: gripper disagreement over the replaceable window (mean of the two samples)", fontsize=9, loc="left")
    # 6 fresh-fresh
    ax = fig.add_subplot(gs[5])
    for vn, c in (("off", "#d9632b"), ("on", "#2a63c4")):
        ax.plot(tt, np.nanmean([col(rows, f"{vn}_s{s}_arm_replaceable") for s in (1, 2)], 0), color=c, lw=1.2, label=f"old vs fresh (mean of 2), RTC {vn}")
        ax.plot(tt, col(rows, f"{vn}_freshfresh_arm_replaceable"), color=c, lw=1, ls="--", label=f"fresh 1 vs fresh 2, RTC {vn}")
    ax.set_ylabel("arm RMS\n(replaceable window)")
    ax.set_xlabel("global episode control step t (50 Hz)")
    _shade(ax, elig, T)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_title("Sampling variation: fresh–fresh discrepancy at the same observation (two samples only; not a confidence interval)", fontsize=9, loc="left")
    fig.text(0.125, 0.085, "Heatmaps: white step = q(t) (above it: prediction-only tail, not scheduled from the current chunk); cyan dashed = h = d = 10 "
             "(below: committed prefix; between d and q: replaceable execution window); dotted grey = end of the old plan's prediction; grey = missing "
             "(no old prediction, or RTC-on probe not allowed); dark strip on top = new request not allowed (inference pending or q < d); "
             "yellow ▼ = baseline launch; thin white vertical = handoff; dash-dot vertical = first contact events (cyan right-finger touch, green lifted, "
             "pink left-finger touch, white success). Grey bands in the curve panels = new request not allowed.", fontsize=7.5, wrap=True, va="top")
    fig.savefig(f"{out}/action_disagreement_{seed}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def _shade(ax, elig, T):
    for t in np.flatnonzero(~elig):
        ax.axvspan(t - 0.5, t + 0.5, color="#eeeeee", lw=0, zorder=0)
    ax.grid(alpha=0.3)


def zoom(seed, e, vmax, pk, k, out, std):
    Z, rows, T = e["Z"], e["rows"], e["T"]
    lo, hi = max(0, pk - 30), min(T, pk + 31)
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.3, 1.0], hspace=0.35, wspace=0.25)
    ev = events(Z)
    cmap = matplotlib.colormaps["magma"].copy()
    cmap.set_bad("#d8d8d8")
    for v in (0, 1):
        ax = fig.add_subplot(gs[0, v])
        M = np.nanmean(e["oldf"]["arm"][lo:hi, v], 1).T
        ax.imshow(np.ma.masked_invalid(M), origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=vmax, extent=(lo - 0.5, hi - 0.5, -0.5, H - 0.5),
                  interpolation="nearest")
        tt = np.arange(lo, hi)
        ax.step(tt, Z["q"][lo:hi], where="mid", color="white")
        ax.axhline(D, color="#7fd4ff", ls="--")
        ax.axvline(pk, color="#00ff88", lw=1)
        ax.set_title(f"{VAR[v]} (t = {lo}…{hi - 1}); green = peak t={pk}", fontsize=9)
        ax.set_xlabel("t")
        ax.set_ylabel("h")
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(cv2.cvtColor(Z["img"][pk], cv2.COLOR_BGR2RGB)[:, :, ::-1])
    r = rows[pk]
    ax.set_title(f"o_t at t={pk}: chunk {r['chunk']}, p={r['p']}, q={r['q']}, stage {r['stage']}", fontsize=9)
    ax.axis("off")
    # per-offset trajectories of the 2 most discrepant arm joints + grippers at the peak
    old = e["old"][pk]
    acts = e["P"]["acts"][pk]
    zd = np.nanmean(np.abs((old[None, None] - acts) / std)[0, :, D:r["q"]], (0, 1))
    js = [ARM[i] for i in np.argsort(-zd[ARM])[:2]] + [13]
    names = {j: (f"left arm joint {j}" if j < 6 else f"right arm joint {j - 7}") for j in ARM}
    names[13], names[6] = "right gripper", "left gripper"
    for c, j in enumerate(js):
        ax = fig.add_subplot(gs[1, c])
        h = np.arange(H)
        ax.axvspan(-0.5, min(D, r["q"]) - 0.5, color="#dddddd", alpha=0.6, lw=0, label="committed prefix")
        if r["q"] > D:
            ax.axvspan(D - 0.5, r["q"] - 0.5, color="#ffe9b3", alpha=0.8, lw=0, label="replaceable window")
        ax.plot(h, old[:, j], color="k", lw=2, label="queued / old plan")
        for v, cc in ((0, "#d9632b"), (1, "#2a63c4")):
            for s, ls in ((0, "-"), (1, "--")):
                if np.isfinite(acts[v, s, 0, j]):
                    ax.plot(h, acts[v, s, :, j], color=cc, ls=ls, lw=1, label=f"fresh {s + 1}, {VAR[v]}")
        ax.set_title(f"{names[j]} at t={pk}", fontsize=9)
        ax.set_xlabel("future offset h")
        if c == 0:
            ax.legend(fontsize=6.5)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{LABEL[seed]}: disagreement peak {k + 1} at t={pk} (RTC-off replaceable-window arm RMS {np.nanmean([r['off_s1_arm_replaceable'], r['off_s2_arm_replaceable']]):.2f}; "
                 f"fresh–fresh {r['off_freshfresh_arm_replaceable']:.2f}). A peak does not show that replanning would help.", fontsize=9)
    fig.savefig(f"{out}/zoom_{seed}_peak{k + 1}_t{pk}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def video(seed, e, src):
    Z, rows, T = e["Z"], e["rows"], e["T"]
    os.makedirs(f"{src}/videos", exist_ok=True)
    vw = cv2.VideoWriter(f"{src}/videos/reference_{seed}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 480))
    for t in range(T):
        im = Z["img"][t][:, :, ::-1].copy()
        r = rows[t]
        txt = [f"seed {seed}  t={t} ({t / 50:.2f} s)  chunk {r['chunk']}  p={r['p']}  q={r['q']}  stage {r['stage']}",
               ("inference pending" if r["pending"] else "no inference pending") + ("  | new request allowed" if r["eligible"] else "  | new request NOT allowed"),
               f"replaceable-window arm RMS: RTC off {np.nanmean([r['off_s1_arm_replaceable'], r['off_s2_arm_replaceable']]):.2f}   "
               f"RTC on {np.nanmean([r['on_s1_arm_replaceable'], r['on_s2_arm_replaceable']]):.2f}   fresh-fresh off {r['off_freshfresh_arm_replaceable']:.2f}"]
        for i, s in enumerate(txt):
            cv2.putText(im, s, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        vw.write(im)
    vw.release()


def snapshot_table(src, std, out):
    """Exact saved r_sensitivity snapshots: disagreement on the same states whose branch outcomes were recorded."""
    p = f"{src}/snapshot_probes.pkl"
    if not os.path.exists(p):
        return
    import pickle
    S = pickle.load(open(p, "rb"))
    cells = list(csv.DictReader(open(f"{HERE}/../r_sensitivity_probe/results/cells.csv")))
    with open(f"{out}/rsens_snapshots_disagreement.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["snapshot_id", "t", "q", "shift", "align_err", "off_arm_replaceable_s1", "off_arm_replaceable_s2", "on_arm_replaceable_s1",
                    "on_arm_replaceable_s2", "off_freshfresh", "on_freshfresh", "off_grip_r_replaceable", "on_grip_r_replaceable",
                    "rsens_successes_by_choice", "rsens_success_spread"])
        for sid, s in S.items():
            old = np.full((H, 14), np.nan, np.float32)
            old[:H - s["shift"]] = s["old"][s["shift"]:]
            old[:s["q"]] = s["queue"]
            o = per_offset(old[None, None], s["acts"], std)
            f = per_offset(s["acts"][:, 0], s["acts"][:, 1], std)
            succ = {c["choice"]: int(c["successes"]) for c in cells if c["snapshot_id"] == sid}
            w.writerow([sid, s["t"], s["q"], s["shift"], f"{s['align_err']:.1e}"] + [f"{wrms(o['arm'][v, k], D, s['q']):.4f}" for v in (0, 1) for k in (0, 1)]
                       + [f"{wrms(f['arm'][v], D, s['q']):.4f}" for v in (0, 1)]
                       + [f"{np.nanmean([wrms(o['grip_r'][v, k], D, s['q']) for k in (0, 1)]):.4f}" for v in (0, 1)]
                       + [" ".join(f"{k}:{v}/3" for k, v in succ.items()), max(succ.values()) - min(succ.values()) if succ else ""])


if __name__ == "__main__":
    main()
