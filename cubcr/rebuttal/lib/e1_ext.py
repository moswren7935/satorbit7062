from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DMAX = 3


def deff_of(comp) -> np.ndarray:

    used = np.asarray(comp.state_agent) >= 0
    slot1 = np.arange(1, used.shape[1] + 1)[None, :]
    return np.max(np.where(used, slot1, 0), axis=1).astype(np.int64)


@dataclass
class _PatternBlock:
    row_ids: np.ndarray
    factors: np.ndarray
    group_starts: np.ndarray
    n_groups: int


@dataclass
class _Template:
    d: int
    radix: Tuple[int, int, int]
    states: np.ndarray
    agents: np.ndarray
    patterns: Dict[int, _PatternBlock]


class TemplateStructure:


    def __init__(self, comp):
        self.comp = comp
        self.S = int(comp.n_states)
        self.kind = np.asarray(comp.state_kind).astype(np.int64)
        self.init = int(comp.init_state)
        self.row_ptr = np.asarray(comp.row_ptr, dtype=np.int64)
        self.col = np.asarray(comp.col, dtype=np.int64)
        self.val = np.asarray(comp.val, dtype=np.float64)
        self.n_rows = int(self.row_ptr.shape[0] - 1)
        nnz_row = np.diff(self.row_ptr)
        self.empty_rows = np.flatnonzero(nnz_row == 0)
        srs = np.asarray(comp.state_row_start, dtype=np.int64)
        R_s = np.diff(srs)
        deff = deff_of(comp)
        radix = np.asarray(comp.state_radix).astype(np.int64)
        agents = np.asarray(comp.state_agent).astype(np.int64)
        W = np.asarray(comp.weights, dtype=np.float64)
        normal = np.flatnonzero(self.kind == 0)

        rad_eff = np.where(np.arange(DMAX)[None, :] < deff[:, None], radix, 1)
        keys = np.concatenate([deff[:, None], rad_eff], axis=1)
        self.templates: List[_Template] = []
        if normal.size:
            uk, inv = np.unique(keys[normal], axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            for ti in range(uk.shape[0]):
                d, r0, r1, r2 = (int(x) for x in uk[ti])
                st = normal[inv == ti]
                R = r0 * r1 * r2
                if np.any(R_s[st] != R):
                    raise ValueError(f"template {uk[ti]}: rows per state != product of radices")
                rr = np.arange(R)
                dg = np.stack([rr // (r1 * r2), (rr // r2) % r1, rr % r2], axis=1)
                stride = np.array([r1 * r2, r2, 1])
                pats: Dict[int, _PatternBlock] = {}
                for p in range(1 << d):
                    pinned = np.array([(l < d) and bool((p >> l) & 1) for l in range(DMAX)])
                    free = np.array([(l < d) and not bool((p >> l) & 1) for l in range(DMAX)])
                    key = (dg * stride * free[None, :]).sum(axis=1)
                    order = np.argsort(key, kind="stable")
                    ks = key[order]
                    gstart_local = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
                    G = int(gstart_local.size)
                    T = st.size
                    row_ids = (srs[st][:, None] + order[None, :]).reshape(-1)
                    fac = np.ones((T, R, DMAX), dtype=np.float64)
                    for l in range(DMAX):
                        if pinned[l]:
                            fac[:, :, l] = W[st][:, l, :][:, dg[order, l]]
                    group_starts = (np.arange(T)[:, None] * R + gstart_local[None, :]).reshape(-1)
                    pats[p] = _PatternBlock(row_ids=row_ids, factors=fac.reshape(T * R, DMAX),
                                            group_starts=group_starts, n_groups=G)
                self.templates.append(_Template(d=d, radix=(r0, r1, r2), states=st, agents=agents[st], patterns=pats))

        self._depth = None


    def depth(self) -> np.ndarray:
        if self._depth is None:
            S = self.S
            depth = np.full(S, -1, dtype=np.int64)
            depth[self.init] = 0
            srs = np.asarray(self.comp.state_row_start, dtype=np.int64)
            p0 = self.row_ptr[srs[:-1]]
            p1 = self.row_ptr[srs[1:]]
            frontier = np.array([self.init], dtype=np.int64)
            dd = 0
            while frontier.size:
                dd += 1
                fr = frontier[self.kind[frontier] == 0]
                if fr.size == 0:
                    break
                idx = np.concatenate([np.arange(a, b) for a, b in zip(p0[fr], p1[fr])])
                succ = np.unique(self.col[idx])
                new = succ[depth[succ] < 0]
                depth[new] = dd
                frontier = new
            self._depth = depth
        return self._depth


def _patterns_for(tpl: _Template, mask_row: np.ndarray) -> np.ndarray:

    T = tpl.states.size
    pat = np.zeros((T, mask_row.size), dtype=np.int64)
    for l in range(tpl.d):
        ag = tpl.agents[:, l]
        bit = ((mask_row[None, :] >> ag[:, None].astype(np.uint64)) & np.uint64(1)).astype(np.int64)
        pat |= bit << l
    return pat


def values_numpy(struct: TemplateStructure, masks, k: int, dtype=np.longdouble, chunk: int = 64,
                 return_V0: bool = False) -> np.ndarray:


    masks = np.asarray(masks, dtype=np.uint64)
    step = masks.ndim == 2
    if step and masks.shape[0] != k:
        raise ValueError("step-level masks must have shape (k, B)")
    B = masks.shape[-1]
    out = np.empty(B, dtype=dtype)
    val = struct.val.astype(dtype)
    one, zero = dtype(1), dtype(0)
    fixed1 = np.flatnonzero(struct.kind == 1)
    fixed2 = np.flatnonzero(struct.kind == 2)

    wts = {}
    for ti, tpl in enumerate(struct.templates):
        for p, blk in tpl.patterns.items():
            f = blk.factors.astype(dtype)
            wts[(ti, p)] = (f[:, 0] * f[:, 1]) * f[:, 2]
    for c0 in range(0, B, chunk):
        c1 = min(B, c0 + chunk)
        Bc = c1 - c0
        V = np.zeros((struct.S, Bc), dtype=dtype)
        V[fixed1, :] = one
        for t in range(k - 1, -1, -1):
            mrow = masks[t, c0:c1] if step else masks[c0:c1]
            prod = val[:, None] * V[struct.col, :]
            q = np.add.reduceat(prod, struct.row_ptr[:-1], axis=0) if prod.shape[0] else np.zeros((struct.n_rows, Bc), dtype)
            if struct.empty_rows.size:
                q[struct.empty_rows, :] = zero
            Vn = np.empty_like(V)
            Vn[fixed1, :] = one
            Vn[fixed2, :] = zero
            for ti, tpl in enumerate(struct.templates):
                T = tpl.states.size
                pat = _patterns_for(tpl, mrow) if tpl.d else np.zeros((T, Bc), dtype=np.int64)
                res = np.empty((T, Bc), dtype=dtype)
                for p, blk in tpl.patterns.items():
                    sel = pat == p
                    if not sel.any():
                        continue
                    contrib = wts[(ti, p)][:, None] * q[blk.row_ids, :]
                    gs = np.add.reduceat(contrib, blk.group_starts, axis=0)
                    vmin = gs.reshape(T, blk.n_groups, Bc).min(axis=1)
                    res = np.where(sel, vmin, res)
                Vn[tpl.states, :] = res
            V = Vn
        out[c0:c1] = V[struct.init, :]
    return out


class MpReference:


    def __init__(self, struct: TemplateStructure, dps: int = 50):
        import mpmath
        self.mpmath = mpmath
        self.dps = int(dps)
        self.struct = struct
        mpf = mpmath.mpf
        with mpmath.workdps(self.dps):
            self.val = [mpf(float(x)) for x in struct.val]
            self.wts = {}
            for ti, tpl in enumerate(struct.templates):
                for p, blk in tpl.patterns.items():
                    self.wts[(ti, p)] = [mpf(float(a)) * mpf(float(b)) * mpf(float(c)) for a, b, c in blk.factors]
        self.rp = struct.row_ptr.tolist()
        self.col = struct.col.tolist()
        depth = struct.depth()
        self.depth = depth

        self.tpl_of = {}
        for ti, tpl in enumerate(struct.templates):
            for j, s in enumerate(tpl.states.tolist()):
                self.tpl_of[s] = (ti, j)

    def value(self, mask, k: int):

        mpmath = self.mpmath
        st = self.struct
        if isinstance(mask, (int, np.integer)):
            cols = [int(mask)] * k
        else:
            cols = [int(x) for x in mask]
            if len(cols) != k:
                raise ValueError("step mask must have k entries")
        with mpmath.workdps(self.dps):
            mpf = mpmath.mpf
            one, zero = mpf(1), mpf(0)
            S = st.S
            kind = st.kind
            V = [one if kind[s] == 1 else zero for s in range(S)]
            depth = self.depth
            need_by_t = {}
            reach = np.flatnonzero(depth >= 0)
            order = reach[np.argsort(depth[reach], kind="stable")]
            dsorted = depth[order]
            for t in range(k - 1, -1, -1):
                mk = cols[t]
                upto = int(np.searchsorted(dsorted, t, side="right"))
                Vn = list(V)
                for s in order[:upto].tolist():
                    ks = kind[s]
                    if ks == 1:
                        Vn[s] = one
                        continue
                    if ks == 2:
                        Vn[s] = zero
                        continue
                    ti, j = self.tpl_of[s]
                    tpl = st.templates[ti]
                    pat = 0
                    for l in range(tpl.d):
                        if (mk >> int(tpl.agents[j, l])) & 1:
                            pat |= 1 << l
                    blk = tpl.patterns[pat]
                    R = blk.row_ids.size // tpl.states.size
                    G = blk.n_groups
                    base = j * R
                    rows = blk.row_ids[base:base + R].tolist()
                    w = self.wts[(ti, pat)][base:base + R]
                    gst = (blk.group_starts[j * G:(j + 1) * G] - base).tolist() + [R]
                    best = None
                    for g in range(G):
                        acc = zero
                        for i in range(gst[g], gst[g + 1]):
                            r = rows[i]
                            q = zero
                            for e in range(self.rp[r], self.rp[r + 1]):
                                q += self.val[e] * V[self.col[e]]
                            acc += w[i] * q
                        if best is None or acc < best:
                            best = acc
                    Vn[s] = best
                V = Vn
            return V[st.init]


def ulp(x: float) -> float:
    return float(np.spacing(np.float64(abs(x))))
