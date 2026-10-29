"""SCH-01 v2 (fixed_delay10_v2) core: THE single shared delay table, action space, masks, fixed baseline, prior.

Simulated inference delay d(N) in 50 Hz control steps (deterministic; no jitter; no margin; a simulation setting that
keeps the shape of the earlier A100 profile while calibrating N = 10 to 10 control steps -- not a GPU measurement):

    N    : 3  4  5  6  7  8  9  10
    d(N) : 6  7  7  8  9  9  10 10

Every script (heatmap diagnostic, online training, evaluation, tests) imports DELAY from here.

Action space (unchanged from legacy): 0 = KEEP; 1 + (R - 1) * 8 + (N - 3) = REPLAN(R, N), R = 1..25, N = 3..10.
"""

from __future__ import annotations

import math

import numpy as np

CONTROL_MS = 20.0
H = 50
K_CAP = 25
N_MIN, N_MAX = 3, 10
R_MAX = 25
N_VALUES = tuple(range(N_MIN, N_MAX + 1))
NUM_ACTIONS = 1 + R_MAX * len(N_VALUES)
KEEP = 0
REF_N = 10
DELAY = {3: 6, 4: 7, 5: 7, 6: 8, 7: 9, 8: 9, 9: 10, 10: 10}  # authoritative lookup table
MIN_DELAY = min(DELAY.values())  # 6
BASELINE_Q = DELAY[REF_N]  # fixed baseline requests when q == 10 (after 15 executed actions)
assert tuple(sorted(DELAY)) == N_VALUES


def delay_steps(n: int) -> int:
    return DELAY[n]


def modeled_inference_ms(n: int) -> float:
    """Modeled request duration implied by the delay table (d(N) * 20 ms); used only for compute accounting."""
    return DELAY[n] * CONTROL_MS


def encode(r: int, n: int) -> int:
    assert 1 <= r <= R_MAX and N_MIN <= n <= N_MAX
    return 1 + (r - 1) * len(N_VALUES) + (n - N_MIN)


def decode(a: int):
    if a == KEEP:
        return "KEEP", None, None
    r, k = divmod(a - 1, len(N_VALUES))
    return "REPLAN", r + 1, N_MIN + k


def feasible_mask(q: int, ticks_left: int) -> np.ndarray:
    """REPLAN(R, N): d(N) <= R <= q and R + K_CAP <= H.  KEEP: only if after one more tick a request is still feasible
    (q - 1 >= MIN_DELAY = 6) and more than one tick remains."""
    m = np.zeros(NUM_ACTIONS, bool)
    for n in N_VALUES:
        for r in range(DELAY[n], min(q, R_MAX) + 1):
            if r + K_CAP <= H:
                m[encode(r, n)] = True
    if q - 1 >= MIN_DELAY and ticks_left > 1:
        m[KEEP] = True
    return m


def reference_action(q: int, mask: np.ndarray):
    """New fixed baseline as a state rule: q > 10 -> KEEP; q == 10 -> REPLAN(10, 10).
    Off-baseline recovery (q < 10, only reachable after exploration): REPLAN(q, largest N with d(N) <= q).
    Returns (action, kind) with kind in {'keep', 'baseline_replan', 'recovery', 'fallback'}."""
    if q > BASELINE_Q and mask[KEEP]:
        return KEEP, "keep"
    r = min(q, R_MAX)
    if q == BASELINE_Q and mask[encode(BASELINE_Q, REF_N)]:
        return encode(BASELINE_Q, REF_N), "baseline_replan"
    for n in sorted(N_VALUES, reverse=True):
        if DELAY[n] <= r and mask[encode(r, n)]:
            return encode(r, n), "recovery"
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        raise RuntimeError("no feasible action")
    return int(idx[0]), "fallback"


def reference_index(q: int, mask: np.ndarray) -> int:
    return reference_action(q, mask)[0]


def prior_strength(step: int, hold_until: int = 5_000, zero_at: int = 50_000) -> float:
    if step <= hold_until:
        return 1.0
    if step >= zero_at:
        return 0.0
    return 1.0 - (step - hold_until) / (zero_at - hold_until)


def prior_logits(ref: int, mask: np.ndarray, strength: float, p0: float = 0.999) -> np.ndarray:
    out = np.where(mask, 0.0, -np.inf).astype(np.float32)
    m = int(mask.sum())
    if m > 1:
        out[ref] = strength * math.log(p0 * (m - 1) / (1 - p0))
    return out
