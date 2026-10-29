"""SCH-01 real-stack checks (one gpu-debug job): RTC sampler on frozen pi0 for N = 3..10, snapshot restore, reference
episodes, step-0 learned-policy parity, throughput.  Writes results/checks/gpu_checks_<job>.json.

    python scripts/gpu_checks.py
"""
import json
import os
import time

import numpy as np

from arli.pi0_policy import configure_caches

configure_caches()
from arli import aloha_env  # noqa: E402
from arli.pi0_policy import FrozenPi0, obs_to_pi0_input  # noqa: E402
from sch01 import core  # noqa: E402
from sch01.executor import Executor, reference_policy, restore_snapshot, take_snapshot  # noqa: E402
from sch01.learner import Learner  # noqa: E402
from sch01.sampler import RTCSampler, rtc_weights, shift_reference  # noqa: E402

OUT = {}
env = aloha_env.make_env()
fp = FrozenPi0(10)
S = RTCSampler(fp)
obs, _ = env.reset(seed=10_000_000)
inp = obs_to_pi0_input(obs)
rng = np.random.default_rng(0)
noise = rng.standard_normal((50, 32)).astype(np.float32)

# A. sampler: full interval for every N; zero-weight RTC == plain; RTC pulls the committed prefix toward y
prev, prev_act, _ = S.sample(inp, 10, noise)
OUT["sampler"] = {}
for n in core.N_VALUES:
    x_plain, a_plain, taus = S.sample(inp, n, noise)
    assert abs(taus[0] - 1.0) < 1e-6 and abs(taus[-1] - 1.0 / n) < 1e-6 and len(taus) == n
    x_zero, _, _ = S.sample(inp, n, noise, np.zeros((50, 32), np.float32), np.zeros(50, np.float32))
    y = shift_reference(prev, 18)
    w = rtc_weights(7, 32)
    noise2 = rng.standard_normal((50, 32)).astype(np.float32)
    x_free, _, _ = S.sample(inp, n, noise2)
    t0 = time.perf_counter()
    x_rtc, _, _ = S.sample(inp, n, noise2, y, w)
    wall = time.perf_counter() - t0
    e_free = float(np.abs(x_free[:7] - y[:7]).mean())
    e_rtc = float(np.abs(x_rtc[:7] - y[:7]).mean())
    OUT["sampler"][n] = dict(zero_weight_max_diff=float(np.abs(x_zero - x_plain).max()), prefix_err_free=e_free,
                             prefix_err_rtc=e_rtc, rtc_wall_s=wall, taus=[round(t, 4) for t in taus])
    print(f"N={n}: zero-weight RTC vs plain max|d| {OUT['sampler'][n]['zero_weight_max_diff']:.2e}; committed-prefix "
          f"error free {e_free:.3f} -> RTC {e_rtc:.3f}; RTC sample wall {wall * 1e3:.0f} ms", flush=True)

# B. snapshot restore: identical action sequence from a restored snapshot -> identical physics
exe = Executor(env, S)
snaps = []


def grab(ex, env_):
    if ex["t"] == 50:
        snaps.append(take_snapshot(ex, env_, extra=dict(env_seed=10_000_001)))


res = exe.run(10_000_001, reference_policy, np.random.default_rng(1), np.random.default_rng(2), on_snapshot=grab)
snap = snaps[0]
acts = [a.copy() for a in snap["queue"][:20]]
runs = []
for _ in range(2):
    restore_snapshot(env, snap)
    qs, ims = [], []
    for a in acts:
        o = env.step(a)[0]
        qs.append(np.asarray(o["agent_pos"]).copy())
        ims.append(o["pixels"]["top"].astype(int))
    runs.append((np.stack(qs), np.stack(ims)))
OUT["snapshot_restore"] = dict(qpos_identical=bool(np.array_equal(runs[0][0], runs[1][0])),
                               max_image_diff=int(np.abs(runs[0][1] - runs[1][1]).max()),
                               restored_t=snap["t"], queue_len=len(snap["queue"]))
print("snapshot restore:", OUT["snapshot_restore"], flush=True)
assert OUT["snapshot_restore"]["qpos_identical"]

# C. reference episodes + D. step-0 learned policy parity
L = Learner(0, dict(actor_lr=1e-4, critic_lr=3e-4, alpha_lr=3e-4, alpha_init=0.01, gamma=0.999, tau=0.005, p0=0.999,
                    target_entropy_ratio=0.1))
parity = []


def learned(st, mask, t):
    ref = core.reference_action(st["q"], mask)
    p, _ = L.policy_probs(st, mask, ref, 1.0)
    a = int(np.argmax(p))
    parity.append(a == ref)
    return a, {}


OUT["episodes"] = []
for e in range(3):
    seed = 10_000_000 + e
    t0 = time.time()
    rr = exe.run(seed, reference_policy, np.random.default_rng([seed, 1]), np.random.default_rng([seed, 2]))
    t1 = time.time()
    rl_ = exe.run(seed, learned, np.random.default_rng([seed, 1]), np.random.default_rng([seed, 2]))
    OUT["episodes"].append(dict(seed=seed, ref_success=rr["success"], ref_t_end=rr["t_end"], ref_wall_s=t1 - t0,
                                learned_success=rl_["success"], learned_t_end=rl_["t_end"],
                                same_request_schedule=[(q["launch_t"], q["r"], q["n"]) for q in rr["requests"]] ==
                                [(q["launch_t"], q["r"], q["n"]) for q in rl_["requests"]]))
    print("episode", OUT["episodes"][-1], flush=True)
OUT["step0_parity"] = float(np.mean(parity))
assert OUT["step0_parity"] == 1.0
os.makedirs("results/checks", exist_ok=True)
with open(f"results/checks/gpu_checks_{os.environ.get('SLURM_JOB_ID', 'local')}.json", "w") as f:
    json.dump(OUT, f, indent=1, default=float)
print("GPU CHECKS OK; step-0 parity", OUT["step0_parity"])
