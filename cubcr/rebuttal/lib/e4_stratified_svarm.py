from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np


def paper_size_distribution(n: int) -> list:

    dist = [0 for _ in range(n + 1)]
    if n % 2 == 0:
        nlogn = n * math.log(n)
        H = sum([1 / s for s in range(1, int(n / 2))])
        nominator = nlogn - 1
        denominator = 2 * nlogn * (H - 1)
        frac = nominator / denominator
        for s in range(2, int(n / 2)):
            dist[s] = frac / s
            dist[n - s] = frac / s
        dist[int(n / 2)] = 1 / nlogn
    else:
        H = sum([1 / s for s in range(1, int((n - 1) / 2 + 1))])
        frac = 1 / (2 * (H - 1))
        for s in range(2, int((n - 1) / 2 + 1)):
            dist[s] = frac / s
            dist[n - s] = frac / s
    return dist


def warmup_cost(n: int, count_empty: bool = True) -> int:

    base = 2 * n + 1 + 2 * sum(-(-n // s) for s in range(2, n - 1))
    return base + (1 if count_empty else 0)


def min_budget(n: int) -> int:

    return warmup_cost(n, count_empty=True)


class _Strata:


    def __init__(self, n: int):
        self.n = n
        self.sum_p = np.zeros((n, n)); self.cnt_p = np.zeros((n, n))
        self.sum_m = np.zeros((n, n)); self.cnt_m = np.zeros((n, n))

    def set_p(self, i, l, v):
        self.sum_p[i, l] = v; self.cnt_p[i, l] = 1

    def set_m(self, i, l, v):
        self.sum_m[i, l] = v; self.cnt_m[i, l] = 1

    def estimate(self) -> np.ndarray:
        mp = np.divide(self.sum_p, self.cnt_p, out=np.zeros_like(self.sum_p), where=self.cnt_p > 0)
        mm = np.divide(self.sum_m, self.cnt_m, out=np.zeros_like(self.sum_m), where=self.cnt_m > 0)
        nz_p = np.sum(self.cnt_p > 0, axis=1); nz_m = np.sum(self.cnt_m > 0, axis=1)
        return mp.sum(axis=1) / nz_p - mm.sum(axis=1) / nz_m


def stratified_svarm(value_of_masks: Callable[[np.ndarray], np.ndarray], n: int, budget: int, seed: int,
                     mode: str = "fast", count_empty: bool = True) -> dict:


    if n < 4:
        raise ValueError("Stratified SVARM needs n >= 4")
    T = int(budget) - (1 if count_empty else 0)
    wu = warmup_cost(n, count_empty=False)
    if T < wu:
        raise ValueError(f"budget {budget} below the Stratified SVARM warm-up cost {wu + int(count_empty)}")
    if count_empty:
        value_of_masks(np.zeros(1, dtype=np.int64))
    allm = (1 << n) - 1
    bit = (np.int64(1) << np.arange(n, dtype=np.int64))
    st = _Strata(n)
    if mode == "reference":
        rs = np.random.RandomState(seed)
        choice = rs.choice
    elif mode == "fast":
        rng = np.random.default_rng(seed)
        choice = rng.choice
    else:
        raise ValueError(mode)
    players = list(range(n))

    def mask_of(A):
        return int(np.sum(bit[np.asarray(A, dtype=np.int64)])) if len(A) else 0


    for i in range(n):
        st.set_m(i, 0, 0.0)
    grand = float(value_of_masks(np.array([allm], dtype=np.int64))[0])
    for i in range(n):
        st.set_p(i, n - 1, grand)
    single = value_of_masks(bit.copy())
    compl = value_of_masks(allm ^ bit)

    for i in range(n):
        st.set_p(i, 0, float(single[i]))
        st.set_m(i, n - 1, float(compl[i]))

    ssum, csum = float(np.sum(single)), float(np.sum(compl))
    for j in range(n):
        st.sum_m[j, 1] = ssum - single[j]; st.cnt_m[j, 1] = n - 1
        st.sum_p[j, n - 2] = csum - compl[j]; st.cnt_p[j, n - 2] = n - 1
    used = 1 + 2 * n


    def blocks(s):
        pi = choice(players, n, replace=False)
        out = []
        for k in range(0, int(n / s)):
            out.append((list(pi[k * s:(k + 1) * s]), None))
        if n % s != 0:
            A = list(pi[n - (n % s):n])
            rest = [p for p in players if p not in A]
            B = list(choice(rest, s - (n % s), replace=False))
            out.append((A, B))
        return out

    for s in range(2, n - 1):
        for A, B in blocks(s):
            S = A if B is None else list(set(list(A) + list(B)))
            v = float(value_of_masks(np.array([mask_of(S)], dtype=np.int64))[0])
            used += 1
            for i in A:
                st.set_p(int(i), s - 1, v)
    for s in range(2, n - 1):
        for A, B in blocks(s):
            S = A if B is None else list(set(list(A) + list(B)))
            v = float(value_of_masks(np.array([allm ^ mask_of(S)], dtype=np.int64))[0])
            used += 1
            for i in A:
                st.set_m(int(i), n - s, v)
    assert used == wu, (used, wu)


    M = T - wu
    probs = paper_size_distribution(n)
    if M > 0:
        if mode == "reference":
            masks = np.empty(M, dtype=np.int64)
            sizes = np.empty(M, dtype=np.int64)
            for t in range(M):
                s = rs.choice(range(0, n + 1), 1, p=probs)
                A = rs.choice(players, s, replace=False)
                masks[t] = mask_of(A); sizes[t] = len(A)
        else:
            sizes = rng.choice(n + 1, size=M, p=np.asarray(probs, dtype=np.float64)).astype(np.int64)
            keys = rng.random((M, n))
            ranks = np.argsort(np.argsort(keys, axis=1), axis=1)
            member = ranks < sizes[:, None]
            masks = (member.astype(np.int64) * bit[None, :]).sum(axis=1)
        vals = value_of_masks(masks)
        member = ((masks[:, None] >> np.arange(n, dtype=np.int64)[None, :]) & 1).astype(bool)

        rows, cols = np.nonzero(member)
        np.add.at(st.sum_p, (cols, sizes[rows] - 1), vals[rows])
        np.add.at(st.cnt_p, (cols, sizes[rows] - 1), 1.0)
        rows, cols = np.nonzero(~member)
        np.add.at(st.sum_m, (cols, sizes[rows]), vals[rows])
        np.add.at(st.cnt_m, (cols, sizes[rows]), 1.0)
    return {"phi": st.estimate(), "n_main": int(M), "warmup": int(wu + (1 if count_empty else 0))}
