"""R-sensitivity probe: same-state branching with a single forced handoff delay R (N = 10, d = 10, RTC on).

Stage `sources` (GPU): run the 3 source episodes with the full fixed baseline (KEEP while q > 10, REPLAN(10,10) at
q = 10; noise default_rng([seed, 1]) exactly as the held-out evaluation), capture snapshots of the UNMODIFIED source
trajectory at p = 5 and p = 10 of every chunk >= 2 (no inference in flight), log the simulator's contact stage per
step (gym_aloha TransferCube reward: 1 right-finger contact, 2 lifted, 3 left-finger contact, 4 transfer success),
save JPEG frames; then restoration checks (state equality, re-rendered input, and identical intervention twice on
two snapshots = 4 control rollouts).

Stage `matrix` (GPU): for every snapshot in the FROZEN manifest (manifest.json, written before any branch outcome is
seen): choices {forced R in R_SET[p]} + full-baseline control, continuation seeds {0, 1, 2}.  A forced branch launches
REPLAN(R, N=10) on the snapshot observation, ready at t+10, hands off at t+R installing new[R:R+25]; afterwards the
fixed baseline rule resumes (it next requests at q = 10 of the new chunk).  The control continues the baseline from
the snapshot (its next request at p = 15).  Policy noise: one standard-normal [50, 32] draw per inference request from
numpy default_rng([source_seed, t_snap, cont_seed, 303]); the k-th request after the snapshot uses the k-th draw in
every choice (pairing by request ordinal).  The environment has no randomness after reset (physics is deterministic;
the renderer is not bit-exact).  Results append to branches.jsonl; rerunning skips finished rows.

    python r_sensitivity_probe/rsens.py --stage sources|matrix --out <bulky dir> [--deadline_min 55]
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
SOURCE_SEEDS = [10_000_014, 10_000_016, 10_000_004]  # rule in README: median success, latest success, first failure
R_SET = {5: (10, 15, 20), 10: (10, 12, 15)}
CONT_SEEDS = (0, 1, 2)
N_FIXED = 10
GAMMA = 0.999


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


class StageProxy:
    """Passes every call to the env; records the env reward (contact stage) per step and optionally JPEG frames."""

    def __init__(self, env, frames=False):
        import cv2
        self._w, self._cv2, self.frames_on = env, cv2, frames
        self.stages, self.frames = [], []

    def __getattr__(self, k):
        return getattr(self._w, k)

    def _grab(self, obs):
        if self.frames_on:
            im = self._cv2.resize(obs["pixels"]["top"], (320, 240), interpolation=self._cv2.INTER_AREA)
            self.frames.append(self._cv2.imencode(".jpg", im[:, :, ::-1], [self._cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())

    def reset(self, *a, **k):
        out = self._w.reset(*a, **k)
        self.stages, self.frames = [], []
        self._grab(out[0])
        return out

    def step(self, a):
        out = self._w.step(a)
        self.stages.append(int(out[1]))
        self._grab(out[0])
        return out

    def restart_capture(self, raw_obs):
        self.stages, self.frames = [], []
        self._grab(raw_obs)

    def fix_first_frame(self, raw_obs):
        """Executor.run(snapshot=...) calls env.reset() before restoring physics; replace the reset frame that the
        proxy captured with the snapshot observation so frames[0] = o_t_snap."""
        if self.frames_on and self.frames:
            first = self.frames
            self.frames = []
            self._grab(raw_obs)
            self.frames += first[1:]


def setup():
    from arli.pi0_policy import configure_caches
    configure_caches()
    from arli import aloha_env
    from arli.pi0_policy import CHECKPOINT, FrozenPi0
    from sch01 import sampler as smp
    from sch02 import core
    from sch02.executor import Executor
    env = StageProxy(aloha_env.make_env())
    exe = Executor(env, smp.RTCSampler(FrozenPi0(10)), gamma=GAMMA)
    ns = f"{CHECKPOINT}/assets/lerobot/aloha_sim_transfer_cube_human/norm_stats.json"
    params = f"{CHECKPOINT}/params/_METADATA"
    prov = dict(pi0_checkpoint=CHECKPOINT, norm_stats_sha256=hashlib.sha256(open(ns, "rb").read()).hexdigest(),
                params_metadata_sha256=hashlib.sha256(open(params, "rb").read()).hexdigest() if os.path.exists(params) else None,
                delay_table={str(k): v for k, v in core.DELAY.items()}, N=N_FIXED, d=core.DELAY[N_FIXED], H=core.H, K=core.K_CAP,
                rtc=dict(enabled=True, mask="official exp soft mask: weight 1 on new[0:R] (committed), decaying until H - shift",
                         max_guidance=smp.MAX_GUIDANCE, reference="previous full normalised prediction shifted by t - its observation tick"),
                git=subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
                gpu=subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
    assert core.DELAY[N_FIXED] == 10 and core.H == 50 and core.K_CAP == 25 and core.BASELINE_Q == 10
    return env, exe, core, prov


def summarize(res, t_snap, stages, core, choice_r):
    reqs = res["requests"]
    first = reqs[0] if reqs else None
    T = res["t_end"]
    st = np.asarray(stages, int)
    ev = {f"first_stage{k}": (int(t_snap + np.flatnonzero(st >= k)[0] + 1) if (st >= k).any() else None) for k in (1, 2, 3, 4)}
    out = dict(success=bool(res["success"]), t_end=int(T), termination=("success" if res["success"] else
                                                                         ("timeout" if T >= 400 else "other")),
               steps_from_snapshot=int(T - t_snap), steps_to_success=int(T - t_snap) if res["success"] else None,
               steps_to_timeout=int(T - t_snap) if (not res["success"] and T >= 400) else None,
               task_return=float(res["ret"]), discounted_task_return=float(res["discounted_return"]),
               inference_calls=len(reqs), underrun_ticks=int(res["underrun_ticks"]), max_stage=int(st.max()) if len(st) else None,
               **ev)
    if first is not None:
        out["first_request"] = {k: (int(first[k]) if isinstance(first[k], (int, np.integer)) else first[k]) for k in
                                ("obs_t", "ready_t", "handoff_t", "r", "n", "d", "shift", "committed", "overlap_end",
                                 "p_at_launch", "q_at_launch", "old_executed", "discarded", "decision_kind")}
        out["first_request"]["abandoned"] = bool(first.get("abandoned", False))
        out["first_request"]["installed_slice"] = f"new[{first['r']}:{first['r'] + core.K_CAP}]"
        out["terminated_before_first_handoff"] = bool(first.get("abandoned", False))
    else:
        out["first_request"], out["terminated_before_first_handoff"] = None, None
    out["request_signature"] = [[int(q["obs_t"]), int(q["r"]), int(q["n"])] for q in reqs]
    anomalies = []
    if choice_r is not None and first is not None and (first["obs_t"] != t_snap or first["r"] != choice_r or first["n"] != N_FIXED):
        anomalies.append("first request does not match the forced choice")
    if res["underrun_ticks"]:
        anomalies.append("queue underrun")
    out["anomalies"] = anomalies
    return out


def stage_sources(a):
    from sch02.executor import reference_policy, restore_snapshot, take_snapshot
    env, exe, core, prov = setup()
    os.makedirs(f"{a.out}/sources", exist_ok=True)
    env.frames_on = True
    meta = dict(provenance=prov, sources=[])
    for seed in SOURCE_SEEDS:
        snaps = []

        def grab(ex, env_, seed=seed):
            if ex["chunk"] >= 2 and ex["p"] in (5, 10) and ex["inflight"] is None and len(ex["queue"]) == core.K_CAP - ex["p"]:
                snaps.append(take_snapshot(ex, env_._w, extra=dict(env_seed=seed)))
        t0 = time.time()
        res = exe.run(seed, reference_policy, np.random.default_rng([seed, 1]), record=True, on_tick=grab)
        st = env.stages
        info = dict(seed=seed, success=bool(res["success"]), t_end=int(res["t_end"]), wall_s=time.time() - t0,
                    requests=[[int(q["obs_t"]), int(q["handoff_t"])] for q in res["requests"]],
                    first_stage={k: (int(np.flatnonzero(np.asarray(st) >= k)[0] + 1) if (np.asarray(st) >= k).any() else None)
                                 for k in (1, 2, 3, 4)},
                    stage_per_step=st, chunk_per_step=[int(x[1]) for x in res["ticks"]], p_per_step=[int(x[2]) for x in res["ticks"]],
                    n_snapshots=len(snaps), snapshot_keys=[[int(s["chunk"]), int(s["p"]), int(s["t"])] for s in snaps])
        offs = np.cumsum([0] + [len(f) for f in env.frames]).tolist()
        with open(f"{a.out}/sources/frames_{seed}.bin", "wb") as fh:
            for f in env.frames:
                fh.write(f)
        info["frame_offsets"] = offs
        pickle.dump(snaps, open(f"{a.out}/sources/snapshots_{seed}.pkl", "wb"))
        meta["sources"].append(info)
        log(f"source {seed}: success {res['success']} t_end {res['t_end']} stages first {info['first_stage']} snapshots {len(snaps)}")
    json.dump(meta, open(f"{a.out}/sources/sources.json", "w"))
    env.frames_on = False
    # restoration checks on two snapshots (fixed choice: source 1 chunk 6 p=5, source 2 chunk 8 p=10)
    checks = []
    for seed, chunk, p, r in ((SOURCE_SEEDS[0], 6, 5, 15), (SOURCE_SEEDS[1], 8, 10, 12)):
        snaps = pickle.load(open(f"{a.out}/sources/snapshots_{seed}.pkl", "rb"))
        s = next(x for x in snaps if x["chunk"] == chunk and x["p"] == p)
        ex = restore_snapshot(env._w, s)
        ph = env._w._env.physics
        rendered = ph.render(height=480, width=640, camera_id="top").astype(int)
        c = dict(seed=seed, chunk=chunk, p=p, t=int(s["t"]), forced_R=r,
                 physics_maxdiff=float(np.abs(ph.get_state() - s["physics_state"]).max()),
                 ctrl_maxdiff=float(np.abs(ph.data.ctrl - s["ctrl"]).max()),
                 queue_equal=bool(len(ex["queue"]) == len(s["queue"]) and all(np.array_equal(u, v) for u, v in zip(ex["queue"], s["queue"]))),
                 prev_plan_equal=bool(np.array_equal(ex["prev"]["x0"], s["prev"]["x0"]) and ex["prev"]["origin"] == s["prev"]["origin"]),
                 counters_equal=bool((ex["t"], ex["p"], ex["chunk"], ex["n_calls"]) == (s["t"], s["p"], s["chunk"], s["n_calls"])),
                 rerendered_vs_saved_input_maxdiff=int(np.abs(rendered - s["raw"]["pixels"]["top"].astype(int)).max()),
                 rerendered_pixels_differing=int((rendered != s["raw"]["pixels"]["top"]).any(-1).sum()), repeats=[])
        for rep in range(2):
            env.restart_capture(s["raw"])
            nr = np.random.default_rng([seed, s["t"], 0, 303])
            res = exe.run(seed, reference_policy, nr, snapshot=s, forced_first=core.encode(r, N_FIXED), continuation=reference_policy)
            c["repeats"].append(summarize(res, s["t"], env.stages, core, r))
        a0, a1 = c["repeats"]
        c["identical_outcome"] = (a0["success"], a0["t_end"], a0["request_signature"]) == (a1["success"], a1["t_end"], a1["request_signature"])
        checks.append(c)
        log(f"restoration check {seed} chunk {chunk} p{p}: physics diff {c['physics_maxdiff']}, render diff {c['rerendered_vs_saved_input_maxdiff']}, "
            f"repeat t_end {a0['t_end']} vs {a1['t_end']}, identical {c['identical_outcome']}")
    json.dump(dict(checks=checks, control_rollouts=4, provenance=prov), open(f"{a.out}/restoration_checks.json", "w"), indent=1)
    log("SOURCES DONE")


def stage_matrix(a):
    from sch02.executor import reference_policy
    man = json.load(open(f"{HERE}/manifest.json"))
    env, exe, core, prov = setup()
    deadline = T0 + a.deadline_min * 60
    out = f"{a.out}/branches.jsonl"
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.add((r["snapshot_id"], r["choice"], r["cont_seed"]))
    snaps = {}
    for seed in SOURCE_SEEDS:
        for s in pickle.load(open(f"{a.out}/sources/snapshots_{seed}.pkl", "rb")):
            snaps[(seed, int(s["chunk"]), int(s["p"]))] = s
    video_ids = set(man.get("video_snapshots", []))
    os.makedirs(f"{a.out}/video_frames", exist_ok=True)
    fh = open(out, "a", buffering=1)
    n_new = 0
    for cs in CONT_SEEDS:  # one full pass per continuation seed: partial results stay balanced across snapshots
        for m in man["snapshots"]:
            if m.get("missing"):
                continue
            s = snaps[(m["source_seed"], m["chunk"], m["p"])]
            assert int(s["t"]) == m["t"] and len(s["queue"]) == m["q"]
            for choice in [f"R{r}" for r in R_SET[m["p"]]] + ["baseline"]:
                key = (m["snapshot_id"], choice, cs)
                if key in done:
                    continue
                if time.time() > deadline:
                    log("deadline reached; resubmit to continue (finished rows are skipped)")
                    sys.exit(3)
                r = None if choice == "baseline" else int(choice[1:])
                vid = m["snapshot_id"] in video_ids and cs == 0
                env.frames_on = vid
                env.restart_capture(s["raw"])
                nr = np.random.default_rng([m["source_seed"], m["t"], cs, 303])
                t0 = time.time()
                if r is None:
                    res = exe.run(m["source_seed"], reference_policy, nr, snapshot=s)
                else:
                    res = exe.run(m["source_seed"], reference_policy, nr, snapshot=s, forced_first=core.encode(r, N_FIXED),
                                  continuation=reference_policy)
                env.fix_first_frame(s["raw"])
                row = dict(snapshot_id=m["snapshot_id"], phase=m["phase"], source_seed=m["source_seed"], cont_seed=cs,
                           choice=choice, t=m["t"], p=m["p"], q=m["q"], chunk=m["chunk"], N=N_FIXED, d=core.DELAY[N_FIXED],
                           requested_R=r, noise_stream=f"default_rng([{m['source_seed']}, {m['t']}, {cs}, 303])",
                           wall_s=time.time() - t0, **summarize(res, m["t"], env.stages, core, r))
                fh.write(json.dumps(row) + "\n")
                n_new += 1
                if vid:
                    offs = np.cumsum([0] + [len(f) for f in env.frames]).tolist()
                    with open(f"{a.out}/video_frames/{m['snapshot_id']}_{choice}_c{cs}.bin", "wb") as vf:
                        for f in env.frames:
                            vf.write(f)
                    json.dump(dict(t0=m["t"], offsets=offs), open(f"{a.out}/video_frames/{m['snapshot_id']}_{choice}_c{cs}.json", "w"))
                log(f"{m['snapshot_id']} {choice} c{cs}: success {row['success']} t_end {row['t_end']} calls {row['inference_calls']}")
    json.dump(dict(provenance=prov, new_rows=n_new, finished=time.strftime("%F %T")), open(f"{a.out}/matrix_run_{os.environ.get('SLURM_JOB_ID', 'local')}.json", "w"), indent=1)
    log("MATRIX DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["sources", "matrix"])
    ap.add_argument("--out", default="/N/scratch/jindjia/arli/runs_scheduling/r_sensitivity_probe")
    ap.add_argument("--deadline_min", type=float, default=54)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"sources": stage_sources, "matrix": stage_matrix}[a.stage](a)
