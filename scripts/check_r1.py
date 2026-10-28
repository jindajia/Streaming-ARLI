"""R1 checks on one GPU: env render patch, pi0 load, step-wise sampler == openpi sampler, timings,
one rendered asynchronous frozen-policy episode with an event trace.

    python scripts/check_r1.py --out /N/scratch/jindjia/arli/r1_checks
"""
import argparse
import json
import os
import time

import numpy as np

from arli.pi0_policy import configure_caches

configure_caches()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from arli import aloha_env  # noqa: E402
from arli.controllers import FrozenPolicy  # noqa: E402
from arli.pi0_policy import FrozenPi0, obs_to_pi0_input, time_call  # noqa: E402
from arli.rollout import run_episode  # noqa: E402
from arli.schedule import Schedule  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/N/scratch/jindjia/arli/r1_checks")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
R = {"jax_devices": [str(d) for d in jax.devices()]}

# 1) render patch leaves the 'top' image unchanged
from gym_aloha.env import AlohaEnv  # noqa: E402  (unpatched first)
e0 = AlohaEnv(task="transfer_cube", obs_type="pixels_agent_pos", render_mode="rgb_array")
o0, _ = e0.reset(seed=123)
a = o0["agent_pos"].copy()
o0b = e0.step(a)[0]
env = aloha_env.make_env()  # patched
o1, _ = env.reset(seed=123)
o1b = env.step(a)[0]
R["render_patch_identical"] = bool(np.array_equal(o0["pixels"]["top"], o1["pixels"]["top"])
                                   and np.array_equal(o0b["pixels"]["top"], o1b["pixels"]["top"]))
t0 = time.perf_counter(); [e0.step(a) for _ in range(20)]; R["sim_step_ms_unpatched"] = (time.perf_counter() - t0) / 20 * 1e3
t0 = time.perf_counter(); [env.step(a) for _ in range(20)]; R["sim_step_ms_patched"] = (time.perf_counter() - t0) / 20 * 1e3

# 2) pi0
t0 = time.time(); pi0 = FrozenPi0(10); R["pi0_load_s"] = time.time() - t0
R["checkpoint_dir"] = str(pi0.checkpoint_dir)
o, _ = env.reset(seed=7)
inp = obs_to_pi0_input(o)
rng = np.random.default_rng(0)
diffs = []
for i in range(3):
    noise = rng.standard_normal((1, 50, 32)).astype(np.float32)
    a_ref = pi0.infer(inp, noise)
    a_step = pi0.sample_stepwise(inp, noise)
    a_zero = pi0.sample_stepwise(inp, noise, hook=lambda k, x: x + 0.0 * x)  # residual-off hook path
    diffs.append(dict(max_abs_stepwise=float(np.abs(a_ref - a_step).max()),
                      max_abs_zero_hook=float(np.abs(a_ref - a_zero).max()),
                      action_scale=float(np.abs(a_ref).max())))
R["sampler_equivalence"] = diffs
R["action_shape"] = list(a_ref.shape)
noise = jnp.asarray(rng.standard_normal((1, 50, 32)), jnp.float32)
R["infer_ms"] = time_call(lambda: pi0.policy.infer(inp, noise=noise)["actions"]) * 1e3
st = pi0.begin(inp)
R["prefix_ms"] = time_call(lambda: pi0.begin(inp).kv_cache) * 1e3
R["expert_step_ms"] = time_call(lambda: pi0.step(st, noise, 3)[0]) * 1e3
R["taus"] = [float(t) for t in pi0.taus]

# 3) one rendered async frozen episode + trace
sched = Schedule()
res = run_episode(env, pi0, sched, 10_000_000, FrozenPolicy(np.random.default_rng(1)), record=True, video=True)
R["episode"] = dict(success=res["success"], t_end=res["t_end"], max_reward=res["max_reward"], timing=res["timing"],
                    executed_decisions=res["executed_decisions"], first_trace=res["trace"][:3] + res["trace"][23:28])
import imageio  # noqa: E402
imageio.mimsave(f"{args.out}/frozen_async_seed10000000.mp4", res["frames"], fps=50)
with open(f"{args.out}/trace_seed10000000.json", "w") as f:
    json.dump(res["trace"], f)
print(json.dumps(R, indent=1, default=float))
with open(f"{args.out}/r1_checks.json", "w") as f:
    json.dump(R, f, indent=1, default=float)
