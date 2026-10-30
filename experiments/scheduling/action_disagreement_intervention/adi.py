"""Intervention test at the t=175 disagreement peak of REGENERATED episode 10000004 (action_disagreement_probe).

Snapshot: the probe's record (ref_10000004.npz) has observations, queue, plans and contact stages but no physics
state, so t = 175 is RECONSTRUCTED by resetting the env with seed 10000004 and replaying the recorded executed
actions a_t = queue[t, 0] for t = 0..174 (MuJoCo physics is deterministic given the actions; only rendering is not
bit-exact).  The replay is validated step by step against the record (robot state, contact stage, images).  The
executor state at t = 175 is built from the record: queue[175, :25], the installed plan (plan 7, origin 165, N = 10),
no pending request, chunk 8, p = 0, n_calls 8, last action a_174, observation o_175 as recorded.

Branches (N = 10, d = 10, RTC on, frozen pi0, 400-step horizon), all from the same snapshot:
  A  fixed baseline (KEEP until q = 10: request at 190, handoff 200, ...)
  B  one forced REPLAN(R=10, N=10) at 175 (handoff 185), then the fixed baseline
  C  forced REPLAN(10, 10) at the first three decisions (175, 185, 195; handoffs 185, 195, 205), then the baseline
Paired continuation noise: seed s -> 40 tensors from default_rng([10000004, 175, s, 505]); the k-th inference request
after the snapshot uses tensor k in every branch (pairs request ORDER, not observation time).
Controls: (1) branch A driven by the reference's own recorded noise draws (should reproduce the saved trajectory),
(2) branch A, seed 0, run twice.

    python action_disagreement_intervention/adi.py --out DIR [--deadline_min 50]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
SEED, T_SNAP = 10_000_004, 175
REF = "/N/scratch/jindjia/arli/runs_scheduling/action_disagreement_probe/reference/ref_10000004.npz"
N, D = 10, 10
N_SEEDS = 10
FORCED = {"A": 0, "B": 1, "C": 3}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


class SeqNoise:
    """Returns pre-generated tensors in request order (the executor draws one per REPLAN)."""

    def __init__(self, tensors):
        self.tensors, self.k = tensors, 0

    def standard_normal(self, *a, **k):
        x = self.tensors[self.k]
        self.k += 1
        return x.copy()


def contacts(physics):
    pairs = set()
    for i in range(physics.data.ncon):
        c = physics.data.contact[i]
        pairs.add((physics.model.id2name(c.geom1, "geom"), physics.model.id2name(c.geom2, "geom")))
    return dict(right=("red_box", "vx300s_right/10_right_gripper_finger") in pairs,
                left=("red_box", "vx300s_left/10_left_gripper_finger") in pairs,
                table=("red_box", "table") in pairs)


class StepLog:
    """Read-only per-step logger (binds the executor's state dict through on_tick)."""

    def __init__(self, env, frames=True):
        import cv2
        self._w, self._cv2, self.ex, self.frames_on = env, cv2, None, frames
        self.rows, self.frames = [], []

    def __getattr__(self, k):
        return getattr(self._w, k)

    def bind(self, ex, env_):
        self.ex = ex

    def reset(self, *a, **k):
        self.rows, self.frames = [], []
        return self._w.reset(*a, **k)

    def step(self, a):
        ex = self.ex
        inf = ex["inflight"]
        if self.frames_on:
            im = self._cv2.resize(ex["raw"]["pixels"]["top"], (320, 240), interpolation=self._cv2.INTER_AREA)
            self.frames.append(self._cv2.imencode(".jpg", im[:, :, ::-1], [self._cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
        row = dict(t=int(ex["t"]), chunk=int(ex["chunk"]), p=int(ex["p"]), q=len(ex["queue"]) + 1,
                   cmd=np.asarray(a, np.float32).tolist(), grip_state_l=float(ex["raw"]["agent_pos"][6]),
                   grip_state_r=float(ex["raw"]["agent_pos"][13]),
                   pending=None if inf is None else [int(inf["launch_t"]), int(inf["ready_t"]), int(inf["handoff_t"])])
        out = self._w.step(a)
        ph = self._w._env.physics
        row.update(stage=int(out[1]), cube=np.asarray(ph.named.data.qpos[-7:-4]).astype(float).tolist(), **contacts(ph))
        self.rows.append(row)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/N/scratch/jindjia/arli/runs_scheduling/action_disagreement_intervention")
    ap.add_argument("--deadline_min", type=float, default=50)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/frames", exist_ok=True)
    deadline = T0 + a.deadline_min * 60
    from arli.pi0_policy import configure_caches
    configure_caches()
    from arli import aloha_env
    from arli.pi0_policy import CHECKPOINT, FrozenPi0
    from sch01 import sampler as smp
    from sch02 import core
    from sch02.executor import Executor, reference_policy, take_snapshot
    assert core.DELAY[N] == D and core.H == 50 and core.K_CAP == 25 and core.BASELINE_Q == 10
    Z = dict(np.load(REF))
    ns = f"{CHECKPOINT}/assets/lerobot/aloha_sim_transfer_cube_human/norm_stats.json"
    prov = dict(pi0_checkpoint=CHECKPOINT, norm_stats_sha256=hashlib.sha256(open(ns, "rb").read()).hexdigest(),
                reference_record=REF, reference_record_sha256=hashlib.sha256(open(REF, "rb").read()).hexdigest(),
                git=subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
                gpu=subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
                N=N, d=D, H=50, K=25, rtc=dict(enabled=True, max_guidance=smp.MAX_GUIDANCE), horizon=400)

    # ---------------- 1. reconstruct t = 175 by replaying recorded actions
    raw_env = aloha_env.make_env()
    obs, _ = raw_env.reset(seed=SEED)
    val = dict(agent_pos_maxdiff=[], image_maxdiff=[], stage_mismatch=[])
    val["image_maxdiff"].append(int(np.abs(obs["pixels"]["top"].astype(int) - Z["img"][0].astype(int)).max()))
    for t in range(T_SNAP):
        assert Z["q"][t] >= 1
        obs, r, term, _, _ = raw_env.step(Z["queue"][t, 0])
        val["agent_pos_maxdiff"].append(float(np.abs(np.asarray(obs["agent_pos"]) - Z["agent_pos"][t + 1]).max()))
        val["image_maxdiff"].append(int(np.abs(obs["pixels"]["top"].astype(int) - Z["img"][t + 1].astype(int)).max()))
        if int(r) != int(Z["stage"][t]):
            val["stage_mismatch"].append(t)
        assert not term
    pl = int(Z["plan"][T_SNAP])
    snap_check = dict(chunk=int(Z["chunk"][T_SNAP]), p=int(Z["p"][T_SNAP]), q=int(Z["q"][T_SNAP]), plan=pl,
                      plan_origin=int(Z["plan_origin"][pl]), plan_n=int(Z["plan_n"][pl]), pending=Z["pending"][T_SNAP].tolist(),
                      queue_equals_plan_slice=bool(np.array_equal(Z["queue"][T_SNAP, :25], Z["plan_actions"][pl][10:35])))
    assert (snap_check["chunk"], snap_check["p"], snap_check["q"], snap_check["plan_origin"]) == (8, 0, 25, 165)
    assert snap_check["pending"][0] == -1 and snap_check["queue_equals_plan_slice"]
    raw175 = {"pixels": {"top": Z["img"][T_SNAP].copy()}, "agent_pos": Z["agent_pos"][T_SNAP].copy()}
    prev_raw = {"pixels": {"top": Z["img"][165]}, "agent_pos": Z["agent_pos"][165]}
    ex = dict(t=T_SNAP, raw=raw175, img=aloha_env.rl_image(raw175), qpos=aloha_env.qpos(raw175),
              queue=[Z["queue"][T_SNAP, h].copy() for h in range(int(Z["q"][T_SNAP]))], last_a=Z["queue"][T_SNAP - 1, 0].copy(),
              inflight=None, success=False, n_calls=pl + 1, p=0, chunk=8,
              prev=dict(x0=Z["plan_x0"][pl].copy(), actions=Z["plan_actions"][pl].copy(), origin=165, n=int(Z["plan_n"][pl]),
                        img=aloha_env.rl_image(prev_raw), qpos=aloha_env.qpos(prev_raw)))
    snap = take_snapshot(ex, raw_env, extra=dict(env_seed=SEED))
    validation = dict(replayed_steps=T_SNAP, max_agent_pos_diff=max(val["agent_pos_maxdiff"]), max_image_diff=max(val["image_maxdiff"]),
                      steps_with_image_diff=int(sum(x > 0 for x in val["image_maxdiff"])), stage_mismatches=val["stage_mismatch"],
                      snapshot=snap_check, observation_used="recorded o_175 (exact pixels the reference saw)",
                      note="physics reconstructed by replaying recorded actions from reset; robot state and contact stage are "
                           "compared with the record; the cube pose itself was not recorded by the probe")
    log("reconstruction:", json.dumps({k: v for k, v in validation.items() if k != "snapshot"}))
    json.dump(validation | dict(provenance=prov), open(f"{a.out}/snapshot_validation.json", "w"), indent=1)

    # ---------------- 2. branches
    env = StepLog(raw_env)
    exe = Executor(env, smp.RTCSampler(FrozenPi0(10)), gamma=0.999)

    def make_policy(k_forced):
        state = dict(n=0)

        def pol(st, mask, t):
            if state["n"] < k_forced:
                state["n"] += 1
                act = core.encode(D, N)
                assert mask[act]
                return act, dict(kind="forced_early_replan")
            return reference_policy(st, mask, t)
        return pol

    def run(branch, noise_tensors, tag):
        t0 = time.time()
        res = exe.run(SEED, make_policy(FORCED[branch]), SeqNoise(noise_tensors), snapshot=snap,
                      on_tick=lambda e, _: env.bind(e, _) if env.ex is not e else None)
        rows, frames = env.rows, env.frames
        env.ex = None
        reqs = [{k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in q.items() if k in (
            "launch_t", "obs_t", "ready_t", "handoff_t", "r", "n", "shift", "committed", "overlap_end", "obs_age_at_handoff",
            "p_at_launch", "q_at_launch", "decision_kind")} | dict(abandoned=bool(q.get("abandoned", False))) for q in res["requests"]]
        exp = {"A": [(190, 200)], "B": [(175, 185), (200, 210)], "C": [(175, 185), (185, 195), (195, 205), (220, 230)]}[branch]
        got = [(q["obs_t"], q["handoff_t"]) for q in reqs][:len(exp)]
        sched_ok = got == exp[:len(got)] and all(q["ready_t"] == q["obs_t"] + D and q["r"] == D for q in reqs)
        T = res["t_end"]
        st = np.array([r["stage"] for r in rows])
        right = np.array([r["right"] for r in rows])
        cube_z = np.array([r["cube"][2] for r in rows])
        first_loss = next((rows[i]["t"] for i in range(1, len(rows)) if right[i - 1] and not right[i]), None)
        out = dict(tag=tag, branch=branch, success=bool(res["success"]), t_end=int(T), steps_from_snapshot=int(T - T_SNAP),
                   steps_to_success=int(T - T_SNAP) if res["success"] else None, termination="success" if res["success"] else "timeout",
                   discounted_task_return=float(res["discounted_return"]), task_return=float(res["ret"]),
                   inference_calls=len(reqs), requests=reqs, schedule_as_expected=bool(sched_ok),
                   first_right_contact_loss=first_loss, right_contact_at_185=bool(right[185 - T_SNAP]) if T > 185 else None,
                   max_stage=int(st.max()), max_cube_height=float(cube_z.max()),
                   regained_right_contact_after_185=bool(right[185 - T_SNAP:].any()) if T > 185 else None,
                   first_left_contact=next((r["t"] for r in rows if r["left"]), None), wall_s=time.time() - t0)
        np.savez_compressed(f"{a.out}/trace_{tag}.npz", t=np.array([r["t"] for r in rows]), chunk=np.array([r["chunk"] for r in rows]),
                            p=np.array([r["p"] for r in rows]), q=np.array([r["q"] for r in rows]), cmd=np.array([r["cmd"] for r in rows]),
                            grip_state_l=np.array([r["grip_state_l"] for r in rows]), grip_state_r=np.array([r["grip_state_r"] for r in rows]),
                            stage=st, cube=np.array([r["cube"] for r in rows]), right=right, left=np.array([r["left"] for r in rows]),
                            table=np.array([r["table"] for r in rows]),
                            pending=np.array([r["pending"] or [-1, -1, -1] for r in rows]))
        offs = np.cumsum([0] + [len(f) for f in frames]).tolist()
        with open(f"{a.out}/frames/{tag}.bin", "wb") as fh:
            for f in frames:
                fh.write(f)
        json.dump(dict(offsets=offs, t0=T_SNAP), open(f"{a.out}/frames/{tag}.json", "w"))
        log(f"{tag}: success {out['success']} t_end {T} calls {len(reqs)} schedule_ok {sched_ok} loss {first_loss} "
            f"contact@185 {out['right_contact_at_185']} regained {out['regained_right_contact_after_185']}")
        return out

    results = f"{a.out}/rollouts.jsonl"
    done = {json.loads(l)["tag"] for l in open(results)} if os.path.exists(results) else set()
    fh = open(results, "a", buffering=1)
    noises = {}
    for s in range(N_SEEDS):  # 40 sequential draws from one generator per seed
        g = np.random.default_rng([SEED, T_SNAP, s, 505])
        noises[s] = [g.standard_normal((50, 32)).astype(np.float32) for _ in range(40)]
    json.dump({str(s): dict(rule="numpy default_rng([10000004, 175, s, 505]).standard_normal((50, 32)) drawn 40 times in order",
                            sha256_first3=[hashlib.sha256(x.tobytes()).hexdigest()[:16] for x in noises[s][:3]]) for s in range(N_SEEDS)},
              open(f"{a.out}/noise_manifest.json", "w"), indent=1)
    # controls first
    plan = [("ctrl_A_refnoise", "A", [Z["ref_noise"][i] for i in range(pl + 1, len(Z["ref_noise"]))]),
            ("ctrl_A_s0_rep1", "A", noises[0]), ("ctrl_A_s0_rep2", "A", noises[0])]
    plan += [(f"{b}_s{s}", b, noises[s]) for s in range(N_SEEDS) for b in ("A", "B", "C")]
    for tag, b, nz in plan:
        if tag in done:
            continue
        if time.time() > deadline:
            log("deadline reached; rerun to resume missing rollouts")
            sys.exit(3)
        fh.write(json.dumps(run(b, nz, tag)) + "\n")
    json.dump(dict(provenance=prov, finished=time.strftime("%F %T")), open(f"{a.out}/run_{os.environ.get('SLURM_JOB_ID', 'local')}.json", "w"), indent=1)
    log("ALL DONE")


if __name__ == "__main__":
    main()
