"""Causal asynchronous chunk schedule (ARLI §3-4, no RTC), in environment steps.

Clock: ``t`` counts ``env.step`` calls; ``o_t`` is the observation after ``t`` steps (``o_0`` = reset).
"At time t" means before ``env.step`` number ``t`` is issued, so ``o_t`` is the newest frame.

Cycle ``j`` (chunk ``j``):
  * ``T_j = n_exec * j``                 chunk j starts executing (deadline for inference j)
  * ``s_j = T_j - d``  (d = t_delay=20)  inference j starts; VLM reads ``o_{s_j}``
  * ``m_j = T_j - d_rl`` (d_rl = 10)     RL actor picks initial noise; ARLI reads ``o_{m_j}``;
                                         action expert runs denoising steps k=0..K-1 over [m_j, T_j)
  * chunk index ``i`` corresponds to physical time ``s_j + i``; execution starts at index
    ``e_j = T_j - s_j = d`` and runs ``n_exec`` actions: indices [d, d + n_exec).
  * committed (intermediate) actions of decision j: actions executed in [s_j, T_j), all from chunk j-1.

The remaining delay d_rl is *part of* d (it is not added on top of it).

Startup (j = 0, our documented choice; the paper does not specify it): the robot is at rest after reset,
the episode clock starts when the first chunk is ready, so s_0 = m_0 = T_0 = 0, e_0 = 0, and decision 0 sees
(o_0, hold-pose placeholder for the committed actions, o_0).  From j = 1 on, the steady state above applies
with no special cases: j=1 -> s=5, m=15, T=25, executes chunk-0 indices 5..24 as its committed actions.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Schedule:
    horizon: int = 50  # chunk length H
    n_exec: int = 25  # actions executed per chunk
    delay: int = 20  # t_delay: inference start -> new chunk available
    delay_rl: int = 10  # t_delay^rl: RL actor + action expert (remaining part of `delay`)
    num_denoise: int = 10  # flow-matching integration steps
    max_steps: int = 400  # episode length (gym_aloha TimeLimit)

    def __post_init__(self):
        assert 0 <= self.delay_rl <= self.delay, "remaining delay must be part of the total delay"
        assert self.delay + self.n_exec <= self.horizon, "executed slice must fit inside the chunk"
        assert self.delay < self.n_exec + self.delay, "paper assumes t_delay < k * dt"

    # --- cycle times -------------------------------------------------------------------------
    def T(self, j: int) -> int:
        return self.n_exec * j

    def s(self, j: int) -> int:
        return 0 if j == 0 else self.T(j) - self.delay

    def m(self, j: int) -> int:
        return 0 if j == 0 else self.T(j) - self.delay_rl

    def exec_start_index(self, j: int) -> int:
        return self.T(j) - self.s(j)

    def chunk_index(self, j: int, t: int) -> int:
        """Index into chunk j of the action executed at time t (T_j <= t < T_j + n_exec)."""
        assert self.T(j) <= t < self.T(j) + self.n_exec, (j, t)
        return t - self.s(j)

    def executing_chunk(self, t: int) -> int:
        return t // self.n_exec

    def committed_window(self, j: int) -> tuple[int, int]:
        """Physical times [a, b) of the committed actions seen by decision j (j >= 1)."""
        return self.s(j), self.T(j)

    def denoise_step_time(self, j: int, k: int) -> int:
        """Physical time at which expert step k (k=0..num_denoise-1) of inference j runs.

        Our clock for the steering extension: the (actor + expert) interval [m_j, T_j) of length d_rl is
        split evenly over the denoising steps.  With d_rl = num_denoise = 10, step k runs at m_j + k and the
        latest causally available frame is o_{m_j + k}.  For j = 0 (synchronous startup) every step is at 0.
        """
        if j == 0:
            return 0
        return self.m(j) + (k * self.delay_rl) // self.num_denoise

    def num_cycles(self) -> int:
        return -(-self.max_steps // self.n_exec)

    def trace(self, n_cycles: int = 4) -> list[dict]:
        """Human-readable event trace used by tests and the R0 timing audit."""
        rows = []
        for j in range(n_cycles):
            a, b = (None, None) if j == 0 else self.committed_window(j)
            rows.append(dict(
                j=j, s=self.s(j), m=self.m(j), T=self.T(j),
                exec_indices=(self.exec_start_index(j), self.exec_start_index(j) + self.n_exec),
                exec_times=(self.T(j), self.T(j) + self.n_exec),
                committed_times=(a, b),
                committed_from_prev_chunk_indices=None if j == 0 else (a - self.s(j - 1), b - self.s(j - 1)),
                denoise_times=[self.denoise_step_time(j, k) for k in range(self.num_denoise)],
            ))
        return rows
