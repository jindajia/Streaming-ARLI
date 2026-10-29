"""SCH-01 core definitions: proxy-derived timing model, action space, feasibility masks, reference rule and prior.

Timing (proxy-derived simulation model, NOT measured native ALOHA RTC latency; see DECISIONS.md S2):
    estimated_latency_ms(N) = 42.5 + 8.17 N            (A100 pi0+RTC profile, linear in N)
    jitter_ms               = clip(Normal(0, 5), -10, 10)   (synthetic)
    actual_delay_steps      = ceil((estimated_latency_ms(N) + jitter_ms) / 20)
    budget_steps(N)         = ceil((estimated_latency_ms(N) + 10) / 20)
at 50 Hz (20 ms per control step).  The scheduler never sees the jitter draw.

Action space: index 0 = KEEP; index 1 + (R - 1) * 8 + (N - 3) = REPLAN(R, N), R = 1..25, N = 3..10 -> 201 actions.
"""

from __future__ import annotations

import math

import numpy as np

CONTROL_MS = 20.0
H = 50  # prediction horizon
K_CAP = 25  # installed executable queue cap
N_MIN, N_MAX = 3, 10
R_MAX = 25
N_VALUES = tuple(range(N_MIN, N_MAX + 1))
NUM_ACTIONS = 1 + R_MAX * len(N_VALUES)
KEEP = 0
LAT_A, LAT_B, JITTER_STD, JITTER_CLIP = 42.5, 8.17, 5.0, 10.0
REF_N = 10


def estimated_latency_ms(n: int) -> float:
    return LAT_A + LAT_B * n


def budget_steps(n: int) -> int:
    return math.ceil((estimated_latency_ms(n) + JITTER_CLIP) / CONTROL_MS - 1e-9)


def sample_jitter_ms(rng: np.random.Generator) -> float:
    return float(np.clip(rng.normal(0.0, JITTER_STD), -JITTER_CLIP, JITTER_CLIP))


def actual_delay_steps(n: int, jitter_ms: float) -> int:
    return math.ceil((estimated_latency_ms(n) + jitter_ms) / CONTROL_MS - 1e-9)


BUDGET = {n: budget_steps(n) for n in N_VALUES}
MIN_BUDGET = min(BUDGET.values())  # 4 (N = 3)


def encode(r: int, n: int) -> int:
    assert 1 <= r <= R_MAX and N_MIN <= n <= N_MAX
    return 1 + (r - 1) * len(N_VALUES) + (n - N_MIN)


def decode(a: int):
    """-> ('KEEP', None, None) or ('REPLAN', R, N)."""
    if a == KEEP:
        return "KEEP", None, None
    r, k = divmod(a - 1, len(N_VALUES))
    return "REPLAN", r + 1, N_MIN + k


def feasible_mask(q: int, ticks_left: int) -> np.ndarray:
    """Deterministic feasibility for an idle worker with q selected old actions remaining.

    REPLAN(R, N): budget(N) <= R <= q (result is ready before the handoff, which is inside the old queue's coverage),
                  R + K_CAP <= H (valid generated suffix: install new[R:R+K_CAP]); the irrevocable prefix c = 1 is
                  implied by R >= budget >= 4.
    KEEP: allowed only if after one more tick (q - 1 actions) some REPLAN is still feasible (q - 1 >= MIN_BUDGET),
          and the episode has more than one tick left.
    """
    m = np.zeros(NUM_ACTIONS, bool)
    for n in N_VALUES:
        for r in range(BUDGET[n], min(q, R_MAX) + 1):
            if r + K_CAP <= H:
                m[encode(r, n)] = True
    if q - 1 >= MIN_BUDGET and ticks_left > 1:
        m[KEEP] = True
    return m


def reference_action(q: int, mask: np.ndarray) -> int:
    """Matched fixed scheduler as a state-dependent rule (ONLINE_TRAINING-eng.md):
    q > budget(10) -> KEEP; q == budget(10) -> REPLAN(q, 10); q < budget(10) -> recovery REPLAN(q, largest N whose
    budget fits).  Falls back to the first feasible action if the rule's choice is masked (e.g. end of episode)."""
    b10 = BUDGET[REF_N]
    if q > b10 and mask[KEEP]:
        return KEEP
    r = min(q, R_MAX)
    for n in sorted(N_VALUES, reverse=True):
        if BUDGET[n] <= r and mask[encode(r, n)]:
            return encode(r, n)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        raise RuntimeError("no feasible action")
    return int(idx[0])


def prior_strength(step: int, hold_until: int = 5_000, zero_at: int = 50_000) -> float:
    if step <= hold_until:
        return 1.0
    if step >= zero_at:
        return 0.0
    return 1.0 - (step - hold_until) / (zero_at - hold_until)


def prior_logits(ref: int, mask: np.ndarray, strength: float, p0: float = 0.999) -> np.ndarray:
    """Reference prior: with M feasible actions the reference logit is strength * log(p0 (M-1) / (1-p0)), all other
    feasible logits 0, infeasible -inf.  At strength 1 and zero residual logits the reference has probability p0."""
    out = np.where(mask, 0.0, -np.inf).astype(np.float32)
    m = int(mask.sum())
    if m > 1:
        out[ref] = strength * math.log(p0 * (m - 1) / (1 - p0))
    return out
