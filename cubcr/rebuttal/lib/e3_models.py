from __future__ import annotations

import dataclasses
import time
from typing import List, Tuple

import numpy as np

import rb
from cubcr.model import AMAX, DMAX, CompiledCSG

PAPER_PFAIL = 0.05
DT_SPEC = dict(n=12, states=100000, k=25, seed=20260829)
OC_K = 10


def _tb_triplets(lay):

    from generators.ticketbench import ADV_E, ADV_H, FAIL, H, E, STAY_E, STAY_H, ABORT
    P, R, C, n_stage, S = lay.P, lay.R, lay.C, lay.n_stage, lay.n_states
    FAILID = lay.fail_state
    rr = np.arange(R, dtype=np.int64)
    nxt = (rr + 1) % R
    cpow = P.rho ** np.arange(C)
    pinj = np.clip(P.pfail * lay.stage_mult[None, :, None] * lay.req_mult[:, None, None] * cpow[None, None, :], 0.0, 1.0)
    ntpl = len(lay.templates)
    tpl_rows = np.array([len(t.rows) for t in lay.templates], dtype=np.int64)
    state_tpl = np.empty(S, dtype=np.int32)
    state_tpl[:S - 1] = np.tile(np.arange(ntpl, dtype=np.int32), R)
    state_tpl[FAILID] = -1
    rows_per_state = np.empty(S, dtype=np.int64)
    rows_per_state[:S - 1] = tpl_rows[state_tpl[:S - 1]]
    rows_per_state[FAILID] = 1
    state_row_start = np.zeros(S + 1, dtype=np.int64)
    np.cumsum(rows_per_state, out=state_row_start[1:])
    row_ids: List[np.ndarray] = []
    succ_ids: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    for ti, t in enumerate(lay.templates):
        j, c, h = t.j, t.c, t.h
        s = lay.sid(rr, j, c, h)
        rstart = state_row_start[s]
        p = pinj[:, j, c]
        targets = {}
        if j < n_stage - 1:
            targets[ADV_H] = lay.sid(rr, j + 1, 0, H)
            targets[ADV_E] = lay.sid(rr, j + 1, 0, E)
        else:
            targets[ADV_H] = lay.sid(nxt, 0, 0, H)
        if c < C - 1:
            targets[STAY_H] = lay.sid(rr, j, c + 1, H)
            targets[STAY_E] = lay.sid(rr, j, c + 1, E)
        targets[ABORT] = lay.sid(nxt, 0, 0, H)
        targets[FAIL] = np.full(R, FAILID, dtype=np.int64)
        for i, d in enumerate(t.rows):
            for sym, (al, be) in d.items():
                if sym not in targets:
                    raise RuntimeError(f"template ({j},{c},{h}) row {i}: symbol {sym} not mappable")
                pr = al + be * p if be != 0.0 else np.full(R, al)
                row_ids.append(rstart + i)
                succ_ids.append(targets[sym])
                probs.append(pr)
    row_ids.append(np.array([state_row_start[FAILID]], dtype=np.int64))
    succ_ids.append(np.array([FAILID], dtype=np.int64))
    probs.append(np.array([1.0]))
    return dict(row_ids=row_ids, succ_ids=succ_ids, probs=probs, state_row_start=state_row_start,
                state_tpl=state_tpl, rows_per_state=rows_per_state)


def _tb_csr(lay, tr):

    from generators.ticketbench import Arrays, NA
    rid = np.concatenate(tr["row_ids"])
    sid = np.concatenate(tr["succ_ids"])
    pr = np.concatenate(tr["probs"])
    keep = pr != 0.0
    rid, sid, pr = rid[keep], sid[keep], pr[keep]
    order = np.lexsort((sid, rid))
    rid, sid, pr = rid[order], sid[order], pr[order]
    key_new = np.empty(rid.shape[0], dtype=bool)
    key_new[0] = True
    key_new[1:] = (rid[1:] != rid[:-1]) | (sid[1:] != sid[:-1])
    starts = np.flatnonzero(key_new)
    val = np.add.reduceat(pr, starts)
    col = sid[starts].astype(np.int32)
    rid_u = rid[starts]
    state_row_start = tr["state_row_start"]
    n_rows = int(state_row_start[-1])
    counts = np.bincount(rid_u, minlength=n_rows)
    if np.any(counts == 0):
        raise RuntimeError("internal: a row without successors")
    row_ptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(counts, out=row_ptr[1:])
    if not np.all(np.isfinite(val)) or float(val.min()) < 0.0 or float(val.max()) > 1.0 + 1e-12:
        raise RuntimeError("internal: transition probabilities outside [0,1]")
    row_sum = np.add.reduceat(val, row_ptr[:-1])
    if float(np.max(np.abs(row_sum - 1.0))) > 1e-9:
        raise RuntimeError("internal: a row's transition probabilities do not sum to 1")
    for key, w in lay.weights.items():
        if w.shape != (NA,) or not np.all(np.isfinite(w)) or float(w.min()) < 0.0 or abs(float(w.sum()) - 1.0) > 1e-9:
            raise RuntimeError(f"internal: observed policy for agent {key[0]} (h={key[1]}) is not a distribution")
    arr = Arrays(lay, state_row_start, tr["state_tpl"], row_ptr, col, val, {})
    return arr, rid_u


def _tb_locality(lay, arr, rid_u, rows_per_state):

    S, C, n_stage = lay.n_states, lay.C, lay.n_stage
    FAILID = lay.fail_state
    col = arr.col
    src_state = np.repeat(np.arange(S, dtype=np.int64), rows_per_state)
    src_of_entry = src_state[rid_u]
    jump = np.abs(col.astype(np.int64) - src_of_entry)
    nonfail = col != FAILID
    win = 2 * C * n_stage
    return {"window_states": int(win),
            "frac_edges_to_fail": float(np.mean(~nonfail)),
            "frac_nonfail_edges_within_window": float(np.mean(jump[nonfail] <= win)) if nonfail.any() else 1.0,
            "median_abs_jump_nonfail": float(np.median(jump[nonfail])) if nonfail.any() else 0.0}


def tb_build_phased(n: int, states: int, k: int, seed: int, pfail: float = PAPER_PFAIL, pattern: str = "none"):


    from generators import ticketbench as tbm
    t = {}
    c0 = time.perf_counter()
    P = tbm._spec(n, states, k, pfail, pattern, seed)
    lay = tbm.make_layout(P)
    c1 = time.perf_counter()
    tr = _tb_triplets(lay)
    c2 = time.perf_counter()
    arr, rid_u = _tb_csr(lay, tr)
    c3 = time.perf_counter()
    comp = tbm._compiled_from_arrays(lay, arr)
    c4 = time.perf_counter()
    arr.locality = _tb_locality(lay, arr, rid_u, tr["rows_per_state"])
    c5 = time.perf_counter()
    meta = tbm._meta(lay, arr, tbm._reach_stats(lay, comp))
    c6 = time.perf_counter()
    t.update(p0_layout_s=c1 - c0, p0_triplets_s=c2 - c1, p1_csr_s=c3 - c2, p1_compile_s=c4 - c3,
             p0m_locality_s=c5 - c4, p0m_meta_s=c6 - c5)
    t["p0_s"] = t["p0_layout_s"] + t["p0_triplets_s"]
    t["p0m_s"] = t["p0m_locality_s"] + t["p0m_meta_s"]
    t["p1_s"] = t["p1_csr_s"] + t["p1_compile_s"]
    return comp, meta, t


def overcooked_import():

    import pandas
    from generators import overcooked_csg as oc
    oc._import_mdp()
    return oc


def overcooked_build_phased(k: int = OC_K):
    oc = overcooked_import()
    t = {}
    c0 = time.perf_counter()
    mdp, keys, succ, layers = oc.enumerate_layers("cramped_room", k, verbose=False)
    c1 = time.perf_counter()
    pol, level, pmeta = oc.fit_human_policy(mdp, "cramped_room", keys, 3)
    c2 = time.perf_counter()
    comp, index, meta = oc.build_layered_compiled(keys, succ, layers, pol, k)
    c3 = time.perf_counter()
    t.update(p0_enumerate_s=c1 - c0, p0_policy_fit_s=c2 - c1, p1_compile_s=c3 - c2, p0m_s=0.0)
    t["p0_s"] = t["p0_enumerate_s"] + t["p0_policy_fit_s"]
    t["p1_s"] = t["p1_compile_s"]
    meta = dict(meta, instance_id=f"overcooked_cramped_room_k{k}", policy_level_counts=pmeta.get("level_counts"))
    return comp, meta, t


def validate_compiled(c: CompiledCSG, tol: float = 1e-9) -> dict:

    rp = np.asarray(c.row_ptr)
    val = np.asarray(c.val)
    if np.any(np.diff(rp) <= 0):
        raise ValueError("empty row")
    rs = np.add.reduceat(val, rp[:-1])
    dev_rows = float(np.max(np.abs(rs - 1.0)))
    if not np.all(np.isfinite(val)) or val.min() < 0.0 or val.max() > 1.0 + tol or dev_rows > tol:
        raise ValueError(f"invalid transition rows (max |sum-1| = {dev_rows})")
    act = (c.state_agent >= 0) & (c.state_kind[:, None] == 0)
    s, l = np.nonzero(act)
    W = c.weights[s, l, :]
    m = c.state_radix[s, l].astype(np.int64)
    legal = np.arange(AMAX)[None, :] < m[:, None]
    if np.any(W[~legal] != 0.0):
        raise ValueError("policy mass on an illegal action")
    ps = W.sum(axis=1)
    dev_pol = float(np.max(np.abs(ps - 1.0))) if ps.size else 0.0
    if W.min(initial=0.0) < 0.0 or dev_pol > tol:
        raise ValueError(f"invalid policy (max |sum-1| = {dev_pol})")
    if np.any(c.col < 0) or np.any(c.col >= c.n_states):
        raise ValueError("successor out of range")
    return {"max_row_dev": dev_rows, "max_policy_dev": dev_pol}


def revise_policy(comp: CompiledCSG, seed: int, sigma: float = 0.5, agent=None) -> Tuple[CompiledCSG, dict]:
    rng = np.random.default_rng([int(seed), 0xE3, 1])
    if agent is None:
        agent = int(rng.integers(comp.n_agents))
    sa, rad, kind = comp.state_agent, comp.state_radix, comp.state_kind
    s, l = np.nonzero((sa == agent) & (kind[:, None] == 0))
    W = comp.weights.copy()
    w = W[s, l, :]
    legal = np.arange(AMAX)[None, :] < rad[s, l].astype(np.int64)[:, None]
    z = rng.standard_normal(w.shape)
    w2 = np.where(legal, w * np.exp(sigma * z), 0.0)
    w2 = w2 / w2.sum(axis=1, keepdims=True)
    W[s, l, :] = w2
    new = dataclasses.replace(comp, weights=W)
    info = {"revision": "R1", "agent": agent, "n_states_changed": int(np.unique(s).size),
            "n_slots_changed": int(s.size), "max_abs_dpi": float(np.max(np.abs(w2 - w))) if s.size else 0.0,
            "sigma": sigma, "seed": int(seed)}
    return new, info


def _row_of_nz(rp):
    return np.repeat(np.arange(rp.shape[0] - 1, dtype=np.int64), np.diff(rp))


def _select_states(comp, rng, frac):
    cand = np.flatnonzero(comp.state_kind == 0)
    nsel = max(1, int(round(frac * cand.size)))
    return np.sort(rng.choice(cand, size=nsel, replace=False))


def _rows_of_states(comp, states):
    srs = comp.state_row_start
    from cubcr.model import _ragged_gather
    return _ragged_gather(srs[states], srs[states + 1] - srs[states])


def revise_values(comp: CompiledCSG, seed: int, frac: float = 0.1, sigma: float = 0.5) -> Tuple[CompiledCSG, dict]:
    rng = np.random.default_rng([int(seed), 0xE3, 2])
    sel = _select_states(comp, rng, frac)
    rows = _rows_of_states(comp, sel)
    rp = comp.row_ptr
    cnt = np.diff(rp)
    row_sel = np.zeros(cnt.size, dtype=bool)
    row_sel[rows] = True
    row_sel &= cnt >= 2
    rnz = _row_of_nz(rp)
    nzsel = row_sel[rnz]
    val = comp.val.copy()
    val[nzsel] = val[nzsel] * np.exp(sigma * rng.standard_normal(int(nzsel.sum())))
    sums = np.add.reduceat(val, rp[:-1])
    val[nzsel] = val[nzsel] / sums[rnz[nzsel]]
    new = dataclasses.replace(comp, val=val)
    info = {"revision": "R2", "n_states_selected": int(sel.size), "n_rows_changed": int(row_sel.sum()),
            "n_nnz_changed": int(nzsel.sum()), "max_abs_dp": float(np.max(np.abs(val - comp.val))) if nzsel.any() else 0.0,
            "degenerate": bool(not nzsel.any()), "sigma": sigma, "frac": frac, "seed": int(seed)}
    return new, info


def revise_structure_triplets(comp: CompiledCSG, seed: int, frac: float = 0.1, delta: float = 0.1):


    rng = np.random.default_rng([int(seed), 0xE3, 3])
    sel = _select_states(comp, rng, frac)
    ops = np.where(rng.random(sel.size) < 0.5, "add", "remove")
    srs, rp, col, val = comp.state_row_start, comp.row_ptr, comp.col, comp.val
    rnz = _row_of_nz(rp)
    keep = np.ones(col.size, dtype=bool)
    scale = np.ones(col.size, dtype=np.float64)
    add_rows, add_cols, add_vals = [], [], []
    n_add = n_rem = n_fallback = 0
    renorm_rows = []
    for s, op in zip(sel.tolist(), ops.tolist()):
        r0, r1 = int(srs[s]), int(srs[s + 1])
        R = r1 - r0
        if op == "remove":
            cand_rows = [r for r in range(r0, r1) if rp[r + 1] - rp[r] >= 2]
            if not cand_rows:
                n_fallback += 1
            else:
                for r in cand_rows:
                    a, b = int(rp[r]), int(rp[r + 1])
                    v = val[a:b]
                    j = a + int(np.flatnonzero(v == v.min())[-1])
                    keep[j] = False
                    renorm_rows.append(r)
                    n_rem += 1
                continue
        for r in range(r0, r1):
            a, b = int(rp[r]), int(rp[r + 1])
            have = set(col[a:b].tolist())
            new_c = None
            for off in range(1, R):
                rn = r0 + (r - r0 + off) % R
                for cc in sorted(col[int(rp[rn]):int(rp[rn + 1])].tolist()):
                    if cc not in have:
                        new_c = cc
                        break
                if new_c is not None:
                    break
            if new_c is None:
                continue
            scale[a:b] = 1.0 - delta
            add_rows.append(r); add_cols.append(new_c); add_vals.append(delta)
            n_add += 1
    rows = np.concatenate([rnz[keep], np.asarray(add_rows, dtype=np.int64)])
    cols = np.concatenate([col[keep].astype(np.int64), np.asarray(add_cols, dtype=np.int64)])
    vals = np.concatenate([(val * scale)[keep], np.asarray(add_vals, dtype=np.float64)])
    if renorm_rows:
        rsum = np.bincount(rows, weights=vals, minlength=comp.n_rows)
        rsel = np.zeros(comp.n_rows, dtype=bool)
        rsel[np.asarray(renorm_rows, dtype=np.int64)] = True
        m = rsel[rows]
        vals[m] = vals[m] / rsum[rows[m]]
    info = {"revision": "R3", "n_states_selected": int(sel.size), "n_add_states": int((ops == "add").sum()) + n_fallback,
            "n_entries_added": n_add, "n_entries_removed": n_rem, "n_remove_fallback_to_add": n_fallback,
            "nnz_before": int(comp.nnz), "nnz_after": int(rows.size), "delta": delta, "frac": frac, "seed": int(seed),
            "degenerate": bool(n_add + n_rem == 0)}
    return {"rows": rows, "cols": cols, "vals": vals}, info


def assemble_csr(base: CompiledCSG, rows, cols, vals, tol: float = 1e-9) -> CompiledCSG:


    rows = np.asarray(rows, dtype=np.int64); cols = np.asarray(cols, dtype=np.int64); vals = np.asarray(vals, dtype=np.float64)
    keep = vals != 0.0
    rows, cols, vals = rows[keep], cols[keep], vals[keep]
    order = np.lexsort((cols, rows))
    rows, cols, vals = rows[order], cols[order], vals[order]
    key_new = np.empty(rows.shape[0], dtype=bool)
    key_new[0] = True
    key_new[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
    starts = np.flatnonzero(key_new)
    val = np.add.reduceat(vals, starts)
    col = cols[starts].astype(np.int32)
    rid = rows[starts]
    n_rows = int(base.n_rows)
    counts = np.bincount(rid, minlength=n_rows)
    if counts.shape[0] != n_rows or np.any(counts == 0):
        raise RuntimeError("assemble_csr: a row without successors (or a row id out of range)")
    row_ptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(counts, out=row_ptr[1:])
    if not np.all(np.isfinite(val)) or float(val.min()) < 0.0 or float(val.max()) > 1.0 + 1e-12:
        raise RuntimeError("assemble_csr: transition probabilities outside [0,1]")
    row_sum = np.add.reduceat(val, row_ptr[:-1])
    if float(np.max(np.abs(row_sum - 1.0))) > tol:
        raise RuntimeError("assemble_csr: a row's transition probabilities do not sum to 1")
    if np.any(col < 0) or np.any(col >= base.n_states):
        raise RuntimeError("assemble_csr: successor out of range")
    return dataclasses.replace(base, row_ptr=row_ptr, col=col, val=val)


def revise_structure(comp: CompiledCSG, seed: int, frac: float = 0.1, delta: float = 0.1) -> Tuple[CompiledCSG, dict]:
    tr, info = revise_structure_triplets(comp, seed, frac, delta)
    return assemble_csr(comp, tr["rows"], tr["cols"], tr["vals"]), info


REVISERS = {"R1": revise_policy, "R2": revise_values, "R3": revise_structure}
