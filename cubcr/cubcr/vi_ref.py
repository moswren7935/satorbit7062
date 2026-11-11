from __future__ import annotations

import itertools
import math
from typing import Sequence, Union

import numpy as np

from .model import CSG

MaskLike = Union[int, Sequence[int], np.ndarray]


def _mask_cols(mask: MaskLike, k: int):
    if isinstance(mask, (int, np.integer)):
        return [int(mask)] * k
    cols = [int(x) for x in mask]
    if len(cols) != k:
        raise ValueError(f"step-level mask must have k={k} columns, got {len(cols)}")
    return cols


def sweep(csg: CSG, V: np.ndarray, mask_t: int) -> np.ndarray:

    S = csg.n_states
    fail = set(csg.fail_states)
    avoid = set(csg.avoid_states)
    Vn = np.zeros(S, dtype=np.float64)
    for s in range(S):
        if s in fail:
            Vn[s] = 1.0
            continue
        if s in avoid:
            Vn[s] = 0.0
            continue
        d = csg.state_defs[s]
        slots = list(range(len(d.active)))
        free = [i for i in slots if not ((mask_t >> d.active[i]) & 1)]
        pinned = [i for i in slots if (mask_t >> d.active[i]) & 1]
        best = math.inf
        for fa in itertools.product(*[range(d.n_actions[i]) for i in free]):
            exp = 0.0
            for pa in itertools.product(*[range(d.n_actions[i]) for i in pinned]):
                joint = [0] * len(slots)
                for i, a in zip(free, fa):
                    joint[i] = a
                w = 1.0
                for i, a in zip(pinned, pa):
                    joint[i] = a
                    w *= float(csg.observed_policy[(s, d.active[i])][a])
                idx = 0
                for i in slots:
                    idx = idx * d.n_actions[i] + joint[i]
                _, succ = d.rows[idx]
                q = 0.0
                for sp, p in succ:
                    q += p * V[sp]
                exp += w * q
            best = min(best, exp)
        Vn[s] = best
    return Vn


def solve(csg: CSG, mask: MaskLike, k: int | None = None, return_vector: bool = False):

    if k is None:
        k = csg.horizon_k
    cols = _mask_cols(mask, k)
    S = csg.n_states
    V = np.zeros(S, dtype=np.float64)
    V[list(csg.fail_states)] = 1.0
    for t in reversed(range(k)):
        V = sweep(csg, V, cols[t])
    return V if return_vector else float(V[csg.init_state])


def solve_many(csg: CSG, masks, k: int | None = None) -> np.ndarray:
    return np.asarray([solve(csg, m, k) for m in masks], dtype=np.float64)
