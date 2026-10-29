"""Real-stack v2 checks: baseline timeline on the real simulator, mid-chunk snapshot restore, REPLAN(7,5) example."""
import json, os
import numpy as np
from arli.pi0_policy import configure_caches
configure_caches()
from arli import aloha_env  # noqa: E402
from arli.pi0_policy import FrozenPi0  # noqa: E402
from sch01.sampler import RTCSampler  # noqa: E402
from sch02 import core  # noqa: E402
from sch02.executor import Executor, reference_policy, take_snapshot  # noqa: E402
env = aloha_env.make_env(); exe = Executor(env, RTCSampler(FrozenPi0(10))); out = {}
snaps = []
def grab(ex, e):
    if ex["chunk"] == 2 and ex["p"] == 5 and not snaps:
        snaps.append(take_snapshot(ex, e, extra=dict(env_seed=10_000_003)))
r = exe.run(10_000_003, reference_policy, np.random.default_rng(3), on_tick=grab)
reqs = r["requests"]
out["baseline"] = dict(success=r["success"], t_end=r["t_end"], obs_t=[q["obs_t"] for q in reqs][:6],
                       kinds=sorted({q["decision_kind"] for q in reqs}), exec_len=sorted({q["old_executed"] for q in reqs if not q.get("abandoned")}),
                       wall_ms_N10=float(np.mean([q["wall_s"] for q in reqs]) * 1e3))
assert out["baseline"]["obs_t"][:3] == [15, 40, 65] and out["baseline"]["kinds"] == ["baseline_replan"]
s = snaps[0]
rb = exe.run(10_000_003, reference_policy, np.random.default_rng(9), snapshot=s)
ra = exe.run(10_000_003, reference_policy, np.random.default_rng(9), snapshot=s, forced_first=core.encode(7, 5), continuation=reference_policy)
b0, a0 = rb["requests"][0], ra["requests"][0]
out["snapshot"] = dict(t=s["t"], p=s["p"], q=len(s["queue"]),
                       baseline_first=dict(obs_t=b0["obs_t"], handoff_t=b0["handoff_t"], old_executed=b0["old_executed"]),
                       replan_7_5=dict(obs_t=a0["obs_t"], ready_t=a0["ready_t"], handoff_t=a0["handoff_t"], old_executed=a0["old_executed"],
                                       discarded=a0["discarded"], shift=a0["shift"], committed=a0["committed"]),
                       next_after_early=dict(obs_t=ra["requests"][1]["obs_t"], shift=ra["requests"][1]["shift"]),
                       baseline_success=rb["success"], replan_success=ra["success"])
assert (a0["obs_t"], a0["handoff_t"], a0["old_executed"], a0["discarded"]) == (30, 37, 12, 13)
assert (b0["obs_t"], b0["handoff_t"]) == (40, 50)
os.makedirs("results/checks", exist_ok=True)
json.dump(out, open(f"results/checks/gpu_checks_v2_{os.environ.get('SLURM_JOB_ID', 'local')}.json", "w"), indent=1)
print(json.dumps(out, indent=1)); print("GPU V2 CHECKS OK")
