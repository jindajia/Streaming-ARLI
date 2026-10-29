"""Learning-diagnostics export (CPU, read-only on the run directory): $DATA_DIR/learning.json + checkpoints.json.

Sources (rn_seed0_d10 only; legacy v1 logs are never read):
  episodes.jsonl          one row per stochastic TRAINING episode (behaviour policy, sampling, current prior)
  updates.jsonl           means over the last 500 gradient updates (every 25th update sampled)
  evals.jsonl             held-out evaluation, 50 seeds, deterministic argmax, checkpoint's own prior strength
  reference_eval.jsonl    the fixed baseline on the same 50 seeds
  evals_supplemental.jsonl  25k checkpoint with the prior disabled

Discounted return is NOT logged per episode.  It is reconstructed exactly from (t_end, success) because every
training episode has underrun_ticks == 0 (checked here): G = -c (1 - g^T)/(1 - g) + 1[success] g^(T-1),
c = 0.1/400, g = 0.999 (see common.reconstructed_discounted_return).  The logged 'ret' is the UNDISCOUNTED sum.

    python viewer/export/export_logs.py
"""
import json
import math
import os

from common import DATA_DIR, RUN_DIR, checkpoints, evals_by_tag, jl, reconstructed_discounted_return, sha256

EP_KEYS = ["steps", "success", "t_end", "ret", "grad_steps", "prior_strength_start", "prior_strength_end", "decisions",
           "replans", "departures", "recovery_states", "entropy", "mean_N", "mean_R", "mean_launch_p", "mean_exec_len",
           "mean_discarded", "mean_obs_age_handoff", "calls_per_1k", "modeled_ms", "modeled_ms_per_1k", "underrun_ticks",
           "wall_infer_s"]
EVAL_KEYS = ["tag", "policy", "train_steps", "grad_steps", "prior_strength", "n_success", "n_episodes", "success_rate",
             "mean_N", "mean_R", "mean_launch_p", "mean_exec_len", "mean_obs_age_handoff", "calls_per_1k",
             "modeled_ms_per_1k", "departures", "recovery_states", "underrun_ticks", "reference_parity"]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def eval_row(e):
    r = {k: e.get(k) for k in EVAL_KEYS}
    r["wilson95"] = wilson(e["n_success"], e["n_episodes"])
    r["episodes"] = dict(seed=[x["seed"] for x in e["episodes"]], success=[int(x["success"]) for x in e["episodes"]],
                         t_end=[x["t_end"] for x in e["episodes"]], ret=[x["ret"] for x in e["episodes"]],
                         discounted_return_reconstructed=[reconstructed_discounted_return(
                             x["t_end"], x["success"], x["underrun_ticks"]) for x in e["episodes"]])
    return r


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    eps = jl(f"{RUN_DIR}/episodes.jsonl")
    upd = jl(f"{RUN_DIR}/updates.jsonl")
    assert sum(e["underrun_ticks"] for e in eps) == 0, "discounted-return reconstruction needs zero underruns"
    ep_cols = {k: [e.get(k) for e in eps] for k in EP_KEYS}
    ep_cols["discounted_return_reconstructed"] = [reconstructed_discounted_return(e["t_end"], e["success"],
                                                                                  e["underrun_ticks"]) for e in eps]
    upd_keys = sorted({k for u in upd for k in u})
    upd_cols = {k: [u.get(k) for u in upd] for k in upd_keys}
    ck = checkpoints()
    for c in ck:
        c["sha256"] = sha256(c["file"])
        c["file"] = os.path.relpath(c["file"], RUN_DIR)
    launches = sorted(f for f in os.listdir(RUN_DIR) if f.startswith("launch_"))
    l0 = json.load(open(f"{RUN_DIR}/{launches[0]}"))
    cfg = l0["config"]
    out = dict(
        run=dict(run_id=cfg["run_id"], run_dir=RUN_DIR, git_at_launch=sorted({json.load(open(f"{RUN_DIR}/{f}"))["git"]
                                                                             for f in launches}),
                 pi0_checkpoint=l0["pi0_checkpoint"], norm_stats_sha256=l0["norm_stats_sha256"],
                 delay_table=l0["delay_table"], timing_label=l0["timing_label"], config=cfg,
                 done=open(f"{RUN_DIR}/DONE").read().strip()),
        annotations=[dict(steps=cfg["warmup_steps"], label="end of 5k warm-up (updates start; prior held at 1)"),
                     dict(steps=cfg["prior_zero_at"], label="prior fully removed (50k)")],
        reward=dict(formula="1[first success] - 0.00025 per control step - 0.00125 per underrun step",
                    gamma=cfg["gamma"], gpu_cost_in_reward=False),
        episodes=ep_cols,
        updates=upd_cols,
        evals=[eval_row(e) for e in evals_by_tag()],
        reference_eval=[eval_row(e) for e in jl(f"{RUN_DIR}/reference_eval.jsonl")],
        supplemental=[eval_row(e) for e in jl(f"{RUN_DIR}/evals_supplemental.jsonl")],
        checkpoints=ck,
        notes=dict(
            training_episodes="Stochastic behaviour-policy episodes (sampled actions, current prior strength). Not "
                              "comparable to held-out evaluation without that caveat.",
            evals="Held-out: 50 seeds 10,000,000..10,000,049, deterministic masked argmax, each checkpoint's own "
                  "training-time prior strength (0 from 50k on). One training seed; intervals are Wilson 95% over "
                  "episodes, not across seeds.",
            discounted_return="Reconstructed (not logged) from t_end and success; exact because all training "
                              "episodes have zero underrun steps. 'ret' is the undiscounted sum.",
            updates="Each row: mean over the last 500 gradient updates (sampled every 25th update). 'target' = mean "
                    "critic target y; 'q_taken' = mean Q(s,a) for replay actions; 'q_gap_feasible' = mean over the "
                    "batch of max-min of min(Q1,Q2) over feasible actions; 'q_disagreement' = mean |Q1-Q2| at the "
                    "replay action.",
            q_values="Q is a learned soft return estimate under gamma=0.999 per control step (including the "
                     "entropy bonus of the target policy), not a success probability; no calibration is claimed."),
        missing=[
            "per-episode discounted return (reconstructed instead, see notes)",
            "per-update critic loss for individual critics (only the mean of both is logged)",
            "held-out evaluation returns are logged only as undiscounted 'ret' per episode (discounted reconstructed)",
            "per-decision Q values / action probabilities of training episodes (not logged; the viewer's rollout "
            "data are new recordings)",
            "updates before the first 500 gradient steps (logging starts at grad step 500)",
        ],
    )
    json.dump(out, open(f"{DATA_DIR}/learning.json", "w"), separators=(",", ":"))
    json.dump(ck, open(f"{DATA_DIR}/checkpoints.json", "w"), indent=1)
    print(f"learning.json: {len(eps)} training episodes, {len(upd)} update rows, {len(out['evals'])} evals, "
          f"{len(ck)} checkpoints; {os.path.getsize(f'{DATA_DIR}/learning.json') / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
