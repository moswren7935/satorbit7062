from __future__ import annotations

import math
import warnings

import numpy as np

try:
    import shapiq
    _GameBase = shapiq.Game
except Exception:
    shapiq = None
    _GameBase = object


def n_from_table(table) -> int:
    n = int(round(math.log2(np.asarray(table).size)))
    if (1 << n) != np.asarray(table).size:
        raise ValueError("table size is not a power of two")
    return n


def bits_to_masks(coalitions, d: int) -> np.ndarray:

    C = np.asarray(coalitions).reshape(-1, d).astype(bool)
    w = (np.int64(1) << np.arange(d, dtype=np.int64))
    return (C.astype(np.int64) * w[None, :]).sum(axis=1)


def masks_to_bits(masks, d: int) -> np.ndarray:
    m = np.asarray(masks, dtype=np.int64).reshape(-1)
    return ((m[:, None] >> np.arange(d, dtype=np.int64)[None, :]) & 1).astype(bool)


class EvalCounter:


    def __init__(self, d: int):
        self.d = d
        self._seen = np.zeros(1 << d, dtype=bool)
        self.n_total = 0
        self.n_calls = 0

    def add(self, masks: np.ndarray):
        self.n_total += int(masks.size)
        self.n_calls += 1
        self._seen[masks] = True

    @property
    def n_unique(self) -> int:
        return int(np.count_nonzero(self._seen))


class TableGame(_GameBase):


    def __init__(self, table, normalize: bool = True):
        table = np.ascontiguousarray(np.asarray(table, dtype=np.float64))
        d = n_from_table(table)
        self.table = table
        self.d = d
        self.v_empty = float(table[0])
        self.v_full = float(table[-1])
        self.upsilon = self.v_full - self.v_empty
        if shapiq is None:
            raise ImportError("shapiq is required for TableGame")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super().__init__(n_players=d, normalize=normalize,
                             normalization_value=self.v_empty if normalize else None)
        if not normalize:
            self.normalization_value = 0.0
        self.counter = EvalCounter(d)


    def reset_counts(self):
        self.counter = EvalCounter(self.d)

    @property
    def n_total(self) -> int:
        return self.counter.n_total

    @property
    def n_unique(self) -> int:
        return self.counter.n_unique


    def value_function(self, coalitions: np.ndarray) -> np.ndarray:

        m = bits_to_masks(coalitions, self.d)
        self.counter.add(m)
        return self.table[m]

    def values_of_masks(self, masks, normalized: bool = True) -> np.ndarray:
        m = np.asarray(masks, dtype=np.int64).reshape(-1)
        self.counter.add(m)
        v = self.table[m]
        return v - self.v_empty if normalized else v.copy()


def exact_shapley(table) -> np.ndarray:

    from cubcr.shapley_fast import shapley_from_table_fast
    t = np.asarray(table, dtype=np.float64)
    return np.asarray(shapley_from_table_fast(t, n_from_table(t)), dtype=np.float64)


def monotone_violation_along_chains(table, n_chains: int = 1000, seed: int = 0) -> float:

    t = np.asarray(table, dtype=np.float64)
    d = n_from_table(t)
    rng = np.random.default_rng(seed)
    worst = -np.inf
    for _ in range(n_chains):
        perm = rng.permutation(d)
        masks = np.concatenate([[0], np.cumsum(np.int64(1) << perm.astype(np.int64))])
        v = t[masks]
        worst = max(worst, float(np.max(v[:-1] - v[1:])))
    return worst
