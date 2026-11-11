from __future__ import annotations

import itertools
import math
from typing import Callable, Dict, Sequence

import numpy as np

from .masks import popcount_array


def shapley_weights(n: int) -> np.ndarray:

    return np.asarray([math.factorial(j) * math.factorial(n - j - 1) / math.factorial(n)
                       for j in range(n)], dtype=np.float64)


def shapley_from_table(v: np.ndarray, n: int) -> np.ndarray:


    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1 or v.shape[0] != (1 << n):
        raise ValueError("value table must be a 1-D array with 2^n entries")
    masks = np.arange(1 << n, dtype=np.uint64)
    pc = popcount_array(masks)
    w = shapley_weights(n)
    phi = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bit = np.uint64(1) << np.uint64(i)
        without = (masks & bit) == 0
        S = masks[without]
        phi[i] = np.sum(w[pc[without]] * (v[(S | bit).astype(np.int64)] - v[S.astype(np.int64)]))
    return phi


def shapley_from_oracle(vfun: Callable[[int], float], n: int) -> np.ndarray:

    v = np.asarray([vfun(m) for m in range(1 << n)], dtype=np.float64)
    return shapley_from_table(v, n)


def shapley_by_permutations(vfun: Callable[[int], float], n: int) -> np.ndarray:

    phi = np.zeros(n)
    cnt = 0
    for perm in itertools.permutations(range(n)):
        m = 0
        prev = vfun(0)
        for i in perm:
            m |= 1 << i
            cur = vfun(m)
            phi[i] += cur - prev
            prev = cur
        cnt += 1
    return phi / cnt


def attributable_value(v: np.ndarray, n: int) -> float:
    return float(v[(1 << n) - 1] - v[0])
