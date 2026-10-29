"""Shared paths and helpers for the fixed_delay10_v2 viewer exporters (no GPU, no JAX imports here)."""
import hashlib
import json
import os

RUN_ROOT = "/N/scratch/jindjia/arli/runs_scheduling/fixed_delay10_v2"
RUN_ID = "rn_seed0_d10"
RUN_DIR = f"{RUN_ROOT}/online/{RUN_ID}"
VIEWER_ROOT = f"{RUN_ROOT}/viewer"              # everything the viewer exports lives here (never inside RUN_DIR)
SITE_DIR = f"{VIEWER_ROOT}/site"                # served directory: web files + data/
DATA_DIR = f"{SITE_DIR}/data"
ARCHIVE_DIR = f"{VIEWER_ROOT}/archive"          # full scheduler inputs (npz), not served
GAMMA = 0.999
TIME_PEN, UNDERRUN_PEN, SUCCESS_BONUS = 0.1 / 400, 0.5 / 400, 1.0
MAX_TICKS = 400


def jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def evals_by_tag():
    """Deduplicated learned evaluations (last row per tag), sorted by training steps."""
    return sorted({e["tag"]: e for e in jl(f"{RUN_DIR}/evals.jsonl")}.values(), key=lambda e: e["train_steps"])


def checkpoints():
    """Actual saved learner checkpoints (ckpt/learner_<steps>.msgpack) joined with the evaluation that produced each
    one.  The checkpoint file name is the training-step counter at the evaluation, so it matches evals.train_steps."""
    ev = {e["train_steps"]: e for e in evals_by_tag()}
    out = []
    for f in sorted(os.listdir(f"{RUN_DIR}/ckpt")):
        if not (f.startswith("learner_") and f.endswith(".msgpack")):
            continue
        steps = int(f[len("learner_"):-len(".msgpack")])
        e = ev.get(steps)
        assert e is not None, f"checkpoint {f} has no evaluation row"
        out.append(dict(id=f"ck{steps:07d}", file=f"{RUN_DIR}/ckpt/{f}", train_steps=steps, grad_steps=e["grad_steps"],
                        eval_tag=e["tag"], prior_strength=e["prior_strength"], eval_success=e["n_success"],
                        eval_episodes=e["n_episodes"]))
    return out


def sha256(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def reconstructed_discounted_return(t_end, success, underrun_ticks=0):
    """Discounted return of one episode from logged summaries, gamma = 0.999 per control step.
    Reward at tick k: -TIME_PEN (every tick) - UNDERRUN_PEN * u_k + SUCCESS_BONUS at the success tick (the last tick
    of a successful episode).  Exact only when underrun_ticks == 0 (the positions of underrun ticks are not logged);
    returns None otherwise."""
    if underrun_ticks:
        return None
    g = -TIME_PEN * (1 - GAMMA ** t_end) / (1 - GAMMA)
    if success:
        g += SUCCESS_BONUS * GAMMA ** (t_end - 1)
    return g
