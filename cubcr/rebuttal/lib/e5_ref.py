from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _digits(radices) -> np.ndarray:
    R = int(np.prod(radices)) if len(radices) else 1
    dig = np.zeros((R, len(radices)), dtype=np.int64)
    rr = np.arange(R, dtype=np.int64)
    for l in reversed(range(len(radices))):
        dig[:, l] = rr % radices[l]
        rr //= radices[l]
    return dig


class NumpyRef:


    def __init__(self, gm):
        self.gm = gm
        self.S = int(gm.n_states)
        self.kind = np.asarray(gm.state_kind)
        self.srs = np.asarray(gm.state_row_start, dtype=np.int64)
        self.rp = np.asarray(gm.row_ptr, dtype=np.int64)
        self.col = np.asarray(gm.col, dtype=np.int64)
        self.val = np.asarray(gm.val, dtype=np.float64)
        self.agent = np.asarray(gm.state_agent)
        self.radix = np.asarray(gm.state_radix).astype(np.int64)
        self.weights = np.asarray(gm.weights, dtype=np.float64)
        nt = np.nonzero(self.kind == 0)[0]
        d = (self.agent[nt] >= 0).sum(axis=1)

        self.struct: Dict[Tuple, np.ndarray] = {}
        keys = [tuple([int(dd)] + self.radix[s, :dd].tolist()) for s, dd in zip(nt, d)]
        buckets: Dict[Tuple, list] = {}
        for s, kk in zip(nt, keys):
            buckets.setdefault(kk, []).append(s)
        for kk, ss in buckets.items():
            self.struct[kk] = np.asarray(ss, dtype=np.int64)
        self.empty_rows = np.diff(self.rp) == 0

    def _q(self, V):
        prod = self.val * V[self.col]
        q = np.add.reduceat(prod, self.rp[:-1]) if prod.size else np.zeros(len(self.rp) - 1)
        q[self.empty_rows] = 0.0
        return q

    def solve(self, mask: int, k: int) -> float:
        V = (self.kind == 1).astype(np.float64)
        mask = int(mask)
        plan = []
        for kk, states in self.struct.items():
            d = kk[0]
            radices = list(kk[1:])
            dig = _digits(radices)
            R = dig.shape[0]
            ag = self.agent[states, :d].astype(np.int64)
            pinned = ((mask >> ag) & 1).astype(bool)
            pats = (pinned * (1 << np.arange(d))[None, :]).sum(axis=1) if d else np.zeros(len(states), dtype=np.int64)
            for p in np.unique(pats):
                st = states[pats == p]
                pin = [l for l in range(d) if (p >> l) & 1]
                free = [l for l in range(d) if not (p >> l) & 1]
                fkey = np.zeros(R, dtype=np.int64)
                for l in free:
                    fkey = fkey * radices[l] + dig[:, l]
                order = np.argsort(fkey, kind="stable")
                F = int(np.prod([radices[l] for l in free])) if free else 1
                P = R // F
                W = np.ones((len(st), R))
                for l in pin:
                    W *= self.weights[st][:, l, :][:, dig[:, l]]
                rows = self.srs[st][:, None] + np.arange(R, dtype=np.int64)[None, :]
                plan.append((st, rows[:, order], W[:, order], F, P))
        for _ in range(int(k)):
            q = self._q(V)
            Vn = np.zeros_like(V)
            Vn[self.kind == 1] = 1.0
            for st, rows, W, F, P in plan:
                Qs = q[rows] * W
                Vn[st] = Qs.reshape(len(st), F, P).sum(axis=2).min(axis=1)
            V = Vn
        return float(V[self.gm.init_state])

    def solve_many(self, masks, k: int) -> np.ndarray:
        return np.array([self.solve(int(m), k) for m in np.asarray(masks, dtype=np.uint64)], dtype=np.float64)
