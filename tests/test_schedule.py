"""Timing audit for the asynchronous schedule (runs on CPU, no simulator).  python tests/test_schedule.py"""
import json
import sys

sys.path.insert(0, "src")
from arli.schedule import Schedule  # noqa: E402

S = Schedule()  # H=50, n=25, d=20, d_rl=10, K=10, 400 steps


def test_steady_state():
    for j in range(1, S.num_cycles()):
        assert S.T(j) - S.s(j) == 20 and S.T(j) - S.m(j) == 10  # remaining delay 10 is inside the 20
        assert S.exec_start_index(j) == 20
        a, b = S.committed_window(j)
        assert b - a == 20
        # committed actions come from chunk j-1 and were all known when inference j started
        assert S.T(j - 1) <= a, "committed actions must already be available at inference start"
        lo, hi = a - S.s(j - 1), b - S.s(j - 1)
        assert (lo, hi) == ((5, 25) if j == 1 else (25, 45))
        assert hi <= S.exec_start_index(j - 1) + S.n_exec  # never beyond what chunk j-1 executes
        assert S.denoise_step_time(j, 0) == S.m(j) and S.denoise_step_time(j, 9) == S.T(j) - 1


def test_every_time_has_exactly_one_action():
    seen = []
    for t in range(S.max_steps):
        j = S.executing_chunk(t)
        i = S.chunk_index(j, t)
        assert S.s(j) + i == t or j == 0  # index i <-> physical time s_j + i
        assert 0 <= i < S.horizon
        seen.append((t, j, i))
    assert len(seen) == 400 and seen[25] == (25, 1, 20) and seen[24] == (24, 0, 24) and seen[399] == (399, 15, 44)


def test_events_are_causal():
    for j in range(1, S.num_cycles()):
        assert S.s(j) < S.m(j) < S.T(j)
        assert S.s(j) > S.T(j - 1) - 1  # inference j starts after chunk j-1 began executing


if __name__ == "__main__":
    test_steady_state(); test_every_time_has_exactly_one_action(); test_events_are_causal()
    for r in S.trace(4):
        print(json.dumps(r))
    print("schedule OK")
