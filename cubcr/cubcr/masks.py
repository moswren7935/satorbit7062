from __future__ import annotations

import numpy as np

MAX_AGENTS = 64


def agent_mask(agents) -> int:

    m = 0
    for a in agents:
        if not (0 <= a < MAX_AGENTS):
            raise ValueError(f"agent id {a} out of range")
        m |= 1 << int(a)
    return m


def popcount(x: int) -> int:
    return bin(int(x)).count("1")


def mask_agents(mask: int, n: int):
    return [i for i in range(n) if (mask >> i) & 1]


def player_index(i: int, t: int, k: int) -> int:
    return i * k + t


def player_of(p: int, k: int):
    return divmod(p, k)


def step_mask_matrix(players, n: int, k: int) -> np.ndarray:


    cols = np.zeros(k, dtype=np.uint64)
    for p in players:
        i, t = divmod(int(p), k)
        if not (0 <= i < n and 0 <= t < k):
            raise ValueError(f"player {p} out of range for n={n}, k={k}")
        cols[t] |= np.uint64(1) << np.uint64(i)
    return cols


def all_agent_masks(n: int) -> np.ndarray:

    if n > 30:
        raise ValueError("enumerating 2^n masks for n>30 is infeasible in memory")
    return np.arange(1 << n, dtype=np.uint64)


def popcount_array(masks: np.ndarray) -> np.ndarray:

    m = masks.astype(np.uint64, copy=True)
    cnt = np.zeros(m.shape, dtype=np.int64)
    while True:
        nz = m != 0
        if not nz.any():
            break
        cnt += nz
        m = m & (m - np.uint64(1))
    return cnt
