"""Curated inspection seeds for the viewer (a curated inspection set, NOT an unbiased new benchmark).

Source: the per-episode outcomes already logged for the 50 held-out evaluation seeds (10,000,000 + e) by the fixed
baseline (reference_eval.jsonl) and by every learned checkpoint (evals.jsonl).  Rule (deterministic; first = lowest
seed in the category, categories filled in this order, a seed is used once):

  1. always_success  - succeeded under the fixed baseline and every learned checkpoint
  2. always_fail     - failed under the fixed baseline and every learned checkpoint
  3. baseline_success_final_fail - fixed baseline succeeded, final (300k) checkpoint failed
  4. baseline_fail_final_success - fixed baseline failed, final checkpoint succeeded
  5. most_flips      - most outcome changes between consecutive checkpoints (ties: lowest seed)
  6. random          - one remaining seed drawn with numpy default_rng(0) (an un-curated control)

    python viewer/export/select_seeds.py      # writes $DATA_DIR/seeds.json
"""
import json
import os

import numpy as np

from common import DATA_DIR, RUN_DIR, evals_by_tag, jl


def main():
    ref = jl(f"{RUN_DIR}/reference_eval.jsonl")[-1]
    ev = evals_by_tag()
    seeds = [x["seed"] for x in ref["episodes"]]
    base = {x["seed"]: bool(x["success"]) for x in ref["episodes"]}
    learned = [{x["seed"]: bool(x["success"]) for x in e["episodes"]} for e in ev]
    final = learned[-1]
    n_succ = {s: base[s] + sum(m[s] for m in learned) for s in seeds}
    flips = {s: sum(a[s] != b[s] for a, b in zip(learned, learned[1:])) for s in seeds}
    n_pol = 1 + len(learned)
    cats = [("always_success", lambda s: n_succ[s] == n_pol),
            ("always_fail", lambda s: n_succ[s] == 0),
            ("baseline_success_final_fail", lambda s: base[s] and not final[s]),
            ("baseline_fail_final_success", lambda s: (not base[s]) and final[s])]
    chosen = []
    for name, pred in cats:
        c = [s for s in seeds if pred(s) and s not in [x["seed"] for x in chosen]]
        if c:
            chosen.append(dict(seed=c[0], category=name, n_candidates=len(c)))
    rest = [s for s in seeds if s not in [x["seed"] for x in chosen]]
    mf = max(flips[s] for s in rest)
    chosen.append(dict(seed=[s for s in rest if flips[s] == mf][0], category="most_flips", n_candidates=1))
    rest = [s for s in seeds if s not in [x["seed"] for x in chosen]]
    chosen.append(dict(seed=int(np.random.default_rng(0).choice(rest)), category="random", n_candidates=len(rest)))
    for c in chosen:
        s = c["seed"]
        c.update(logged_baseline_success=base[s], logged_learned_successes=int(sum(m[s] for m in learned)),
                 logged_outcome_flips=flips[s],
                 logged_success_by_checkpoint={str(e["train_steps"]): m[s] for e, m in zip(ev, learned)})
    os.makedirs(DATA_DIR, exist_ok=True)
    out = dict(note="Curated inspection set, NOT an unbiased new benchmark. Seeds are held-out evaluation seeds whose "
                    "outcomes were already logged during training; categories chosen to cover successful, failed and "
                    "changed-outcome cases.", rule=__doc__.split("Rule")[1].split("python")[0].strip(), seeds=chosen)
    json.dump(out, open(f"{DATA_DIR}/seeds.json", "w"), indent=1)
    for c in chosen:
        print(c["category"], c["seed"], "baseline", c["logged_baseline_success"], "learned succ",
              c["logged_learned_successes"], "/", len(learned), "flips", c["logged_outcome_flips"])


if __name__ == "__main__":
    main()
