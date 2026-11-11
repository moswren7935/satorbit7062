#!/usr/bin/env python3


from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cubcr.model import (CSG, CompiledCSG, StateDef, DMAX, AMAX, save_csg,
                         reachable_within_k, permute_states)

GENERATOR_VERSION = "1.1.0"
PERMUTATIONS = ("none", "random")
_PERM_STREAM = 0x7E4B
PATTERNS = ("none", "dummy", "symmetric", "redundant", "joint")
ACTIONS = ["proceed", "retry", "reject", "compensate"]
PROCEED, RETRY, REJECT, COMPENSATE = 0, 1, 2, 3
NA = 4


ADV_H, ADV_E, STAY_H, STAY_E, ABORT, FAIL = range(6)
H, E = 0, 1

FLAG_H, FLAG_E, FLAG_VOID = 0, 1, 2

_MIDDLE_ROLES = ["inventory", "pricing", "seatmap", "fraud", "loyalty", "tax", "notify",
                 "ledger", "cache", "search", "auth", "quota"]


@dataclass
class Params:
    n: int
    states: int
    k: int
    pfail: float
    pattern: str = "none"
    seed: int = 20260829
    retries: int = 3
    rho: float = 0.5
    prec: float = 0.5
    pcomp: float = 0.75
    pleak: float = 0.05
    q: float = 0.5
    concentration: float = 60.0
    hazard: float = 0.0
    permute: str = "none"

    def as_dict(self) -> dict:
        return {"n": self.n, "states": self.states, "k": self.k, "pfail": self.pfail,
                "pattern": self.pattern, "seed": self.seed, "retries": self.retries,
                "rho": self.rho, "prec": self.prec, "pcomp": self.pcomp, "pleak": self.pleak,
                "q": self.q, "concentration": self.concentration,
                "hazard": self.hazard, "permute": self.permute}


Aff = Dict[int, Tuple[float, float]]


def _add(d: Aff, sym: int, a: float, b: float = 0.0) -> None:
    if sym in d:
        d[sym] = (d[sym][0] + a, d[sym][1] + b)
    else:
        d[sym] = (a, b)


def _single_action(a: int, h: int, exhausted: bool, P: Params) -> Aff:

    d: Aff = {}
    if h == H:
        if a == PROCEED or exhausted:
            _add(d, ADV_H, 1.0, -1.0)
            _add(d, ADV_E, 0.0, 1.0)
        else:


            _add(d, STAY_H, 1.0)
    else:
        if a == PROCEED:
            _add(d, ADV_E, 1.0)
        elif a == REJECT:
            _add(d, ABORT, 1.0 - P.pleak)
            _add(d, ADV_E, P.pleak)
        elif exhausted:
            _add(d, ADV_E, 1.0)
        elif a == RETRY:
            _add(d, STAY_H, P.prec)
            _add(d, STAY_E, 1.0 - P.prec)
        else:
            _add(d, STAY_H, P.pcomp)
            _add(d, STAY_E, 1.0 - P.pcomp)
    return d


def _pay_step(a: int, h: int, exhausted: bool, P: Params) -> Aff:

    d: Aff = {}
    if h == H:
        if a == PROCEED or exhausted:
            _add(d, FLAG_H, 1.0, -1.0)
            _add(d, FLAG_E, 0.0, 1.0)
        elif a == REJECT:
            _add(d, FLAG_VOID, 1.0)
        else:
            _add(d, FLAG_H, 1.0)
    else:
        if a == PROCEED:
            _add(d, FLAG_E, 1.0)
        elif a == REJECT:
            _add(d, FLAG_VOID, 1.0)
        elif exhausted:
            _add(d, FLAG_E, 1.0)
        elif a == RETRY:
            _add(d, FLAG_H, P.prec)
            _add(d, FLAG_E, 1.0 - P.prec)
        else:
            _add(d, FLAG_H, P.pcomp)
            _add(d, FLAG_E, 1.0 - P.pcomp)
    return d


def _iss_step(a: int, flag: int, exhausted: bool, P: Params) -> Dict[int, float]:

    if flag == FLAG_VOID:
        if a == PROCEED:
            return {ABORT: 1.0 - P.pleak, FAIL: P.pleak}
        return {ABORT: 1.0}
    if flag == FLAG_H:
        if a == PROCEED or exhausted:
            return {ADV_H: 1.0}
        if a == REJECT:
            return {ABORT: 1.0 - P.pleak, ADV_H: P.pleak}
        return {STAY_H: 1.0}

    if a == PROCEED or exhausted:
        return {FAIL: 1.0}
    if a == REJECT:
        return {ABORT: 1.0 - P.pleak, FAIL: P.pleak}
    if a == RETRY:
        return {STAY_H: P.prec, STAY_E: 1.0 - P.prec}
    return {STAY_H: P.pcomp, STAY_E: 1.0 - P.pcomp}


def _commit_row(a_pay: int, a_iss: int, h: int, exhausted: bool, P: Params) -> Aff:
    d: Aff = {}
    for flag, (al, be) in _pay_step(a_pay, h, exhausted, P).items():
        for sym, pr in _iss_step(a_iss, flag, exhausted, P).items():
            _add(d, sym, al * pr, be * pr)
    return d


def _pair_row(a_a: int, a_b: int, h: int, rule: str, P: Params) -> Aff:

    bad_a, bad_b = (a_a == PROCEED), (a_b == PROCEED)
    qh = P.q if h == H else 1.0
    if rule == "and":
        pf = qh if (bad_a and bad_b) else 0.0
    elif rule == "or":
        pf = qh if (bad_a or bad_b) else 0.0
    elif rule == "add":
        pf = qh * (float(bad_a) + float(bad_b)) / 2.0
    else:
        raise ValueError(rule)
    d: Aff = {}
    if pf > 0.0:
        _add(d, FAIL, pf)
    if 1.0 - pf > 0.0:
        _add(d, ADV_H, 1.0 - pf)
    return d


@dataclass
class Stage:
    agents: List[int]
    kind: str
    rule: str = ""


@dataclass
class Template:
    j: int
    c: int
    h: int
    agents: List[int]
    radices: List[int]
    joints: List[Tuple[int, ...]]
    rows: List[Aff]


@dataclass
class Layout:
    P: Params
    n_stage: int
    R: int
    C: int
    stages: List[Stage]
    roles: List[str]
    templates: List[Template]
    weights: Dict[Tuple[int, int], np.ndarray]
    stage_mult: np.ndarray
    req_mult: np.ndarray
    pattern_info: dict
    n_states: int = 0
    fail_state: int = 0

    def sid(self, r, j, c, h):
        return ((r * self.n_stage + j) * self.C + c) * 2 + h


def _roles(n_stage: int, pattern: str, n: int) -> List[str]:
    if pattern in ("symmetric", "redundant", "joint"):
        single = n_stage - 1
        names = ["web"] + [_MIDDLE_ROLES[i % len(_MIDDLE_ROLES)] + ("" if i < len(_MIDDLE_ROLES) else str(i // len(_MIDDLE_ROLES)))
                           for i in range(max(0, single - 1))]
        names = names[:single]
        return names + ["issuer_a", "issuer_b"]
    names = ["web"]
    mid = n_stage - 3
    names += [_MIDDLE_ROLES[i % len(_MIDDLE_ROLES)] + ("" if i < len(_MIDDLE_ROLES) else str(i // len(_MIDDLE_ROLES)))
              for i in range(max(0, mid))]
    if n_stage >= 3:
        names += ["payment"]
    names += ["issuer"]
    names = names[:n_stage]
    if pattern == "dummy":
        names = names + ["audit"]
    return names


def make_layout(P: Params) -> Layout:
    if P.pattern not in PATTERNS:
        raise ValueError(f"unknown pattern {P.pattern!r}; choose from {PATTERNS}")
    n = P.n
    if P.pattern == "none":
        if n < 2:
            raise ValueError("pattern 'none' needs n >= 2")
        n_stage = n
        stages = [Stage([j], "single") for j in range(n - 1)] + [Stage([n - 2, n - 1], "commit")]
    elif P.pattern == "dummy":
        if n < 3:
            raise ValueError("pattern 'dummy' needs n >= 3")
        n_stage = n - 1
        stages = [Stage([0, n - 1], "single_dummy")] + [Stage([j], "single") for j in range(1, n_stage - 1)] \
            + [Stage([n_stage - 2, n_stage - 1], "commit")]
    else:
        if n < 3:
            raise ValueError(f"pattern {P.pattern!r} needs n >= 3")
        n_stage = n - 1
        rule = {"symmetric": "add", "redundant": "or", "joint": "and"}[P.pattern]
        stages = [Stage([j], "single") for j in range(n_stage - 1)] + [Stage([n - 2, n - 1], "pair", rule)]
    C = int(P.retries)
    if C < 1:
        raise ValueError("retries must be >= 1")
    if P.k < 0:
        raise ValueError("k must be >= 0")
    if P.states < 1:
        raise ValueError("states must be >= 1")
    for name in ("pfail", "prec", "pcomp", "pleak", "q", "hazard"):
        x = float(getattr(P, name))
        if not (0.0 <= x <= 1.0) or x != x:
            raise ValueError(f"{name} must be in [0,1], got {x}")
    if P.permute not in PERMUTATIONS:
        raise ValueError(f"unknown permute {P.permute!r}; choose from {PERMUTATIONS}")
    if not (P.rho >= 0.0):
        raise ValueError(f"rho must be >= 0, got {P.rho}")
    if not (P.concentration > 0.0):
        raise ValueError(f"concentration must be > 0, got {P.concentration}")
    per_req = 2 * C * n_stage
    R = max(1, int(round((P.states - 1) / per_req)))
    S = per_req * R + 1

    rng = np.random.default_rng(P.seed)
    stage_mult = rng.uniform(0.6, 1.4, size=n_stage)
    req_mult = rng.uniform(0.7, 1.3, size=R)
    baseH = np.array([0.80, 0.12, 0.04, 0.04])
    baseE = np.array([0.55, 0.25, 0.10, 0.10])
    weights: Dict[Tuple[int, int], np.ndarray] = {}
    for i in range(n):
        weights[(i, H)] = rng.dirichlet(P.concentration * baseH)
        weights[(i, E)] = rng.dirichlet(P.concentration * baseE)
    pattern_info: dict = {"pattern": P.pattern}
    if P.pattern == "dummy":
        d = n - 1
        weights[(d, H)] = rng.dirichlet(np.ones(NA) * 4.0)
        weights[(d, E)] = rng.dirichlet(np.ones(NA) * 4.0)
        pattern_info.update({"dummy_agents": [d], "co_active_stage": 0,
                             "expect": "phi[dummy] == 0 and v(S|dummy) == v(S) for all S"})
    elif P.pattern in ("symmetric", "redundant", "joint"):
        a, b = n - 2, n - 1
        if P.pattern == "redundant":
            det = np.array([1.0, 0.0, 0.0, 0.0])
            weights[(a, H)] = weights[(a, E)] = det
            weights[(b, H)] = weights[(b, E)] = det.copy()
            pattern_info.update({"pair": [a, b], "rule": "or", "observed_policy": "deterministic proceed",
                                 "expect": "v(S|a) == v(S|b) == v(S|a|b) for all S without a,b; "
                                           "v(S|a) > v(S) (each alone sufficient)"})
        else:
            weights[(b, H)] = weights[(a, H)].copy()
            weights[(b, E)] = weights[(a, E)].copy()
            if P.pattern == "joint":
                pattern_info.update({"pair": [a, b], "rule": "and",
                                     "expect": "v(S|a) == v(S) and v(S|b) == v(S) whenever the partner is free; "
                                               "v(S|a|b) > v(S|a) (only the pair deviating causes failure)"})
            else:
                pattern_info.update({"pair": [a, b], "rule": "add",
                                     "expect": "phi[a] == phi[b] (interchangeable replicas)"})
    else:
        pattern_info.update({"expect": "none"})


    templates: List[Template] = []
    last = n_stage - 1
    for j, st in enumerate(stages):
        for c in range(C):
            exhausted = (c == C - 1)
            for h in (H, E):
                if st.kind == "single":
                    rows = [_single_action(a, h, exhausted, P) for a in range(NA)]
                    joints = [(a,) for a in range(NA)]
                    radices = [NA]
                elif st.kind == "single_dummy":
                    base = [_single_action(a, h, exhausted, P) for a in range(NA)]
                    rows = [dict(base[a0]) for a0 in range(NA) for _ in range(NA)]
                    joints = [(a0, a1) for a0 in range(NA) for a1 in range(NA)]
                    radices = [NA, NA]
                elif st.kind == "commit":
                    rows = [_commit_row(a0, a1, h, exhausted, P) for a0 in range(NA) for a1 in range(NA)]
                    joints = [(a0, a1) for a0 in range(NA) for a1 in range(NA)]
                    radices = [NA, NA]
                else:
                    rows = [_pair_row(a0, a1, h, st.rule, P) for a0 in range(NA) for a1 in range(NA)]
                    joints = [(a0, a1) for a0 in range(NA) for a1 in range(NA)]
                    radices = [NA, NA]
                if j == last:

                    for d in rows:
                        if ADV_E in d:
                            al, be = d.pop(ADV_E)
                            _add(d, FAIL, al, be)
                    if P.hazard > 0.0:


                        hz = float(P.hazard)
                        for d in rows:
                            for sym in list(d):
                                al, be = d[sym]
                                d[sym] = ((1.0 - hz) * al, (1.0 - hz) * be)
                            _add(d, FAIL, hz)
                templates.append(Template(j, c, h, list(st.agents), radices, joints, rows))
    lay = Layout(P=P, n_stage=n_stage, R=R, C=C, stages=stages, roles=_roles(n_stage, P.pattern, n),
                 templates=templates, weights=weights, stage_mult=stage_mult, req_mult=req_mult,
                 pattern_info=pattern_info, n_states=S, fail_state=S - 1)
    return lay


@dataclass
class Arrays:
    lay: Layout
    state_row_start: np.ndarray
    state_tpl: np.ndarray
    row_ptr: np.ndarray
    col: np.ndarray
    val: np.ndarray
    locality: dict = field(default_factory=dict)


def build_arrays(lay: Layout) -> Arrays:

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
    rid = np.concatenate(row_ids)
    sid = np.concatenate(succ_ids)
    pr = np.concatenate(probs)
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

    src_state = np.repeat(np.arange(S, dtype=np.int64), rows_per_state)
    src_of_entry = src_state[rid_u]
    jump = np.abs(col.astype(np.int64) - src_of_entry)
    nonfail = col != FAILID
    win = 2 * C * n_stage
    locality = {"window_states": int(win),
                "frac_edges_to_fail": float(np.mean(~nonfail)),
                "frac_nonfail_edges_within_window": float(np.mean(jump[nonfail] <= win)) if nonfail.any() else 1.0,
                "median_abs_jump_nonfail": float(np.median(jump[nonfail])) if nonfail.any() else 0.0}
    return Arrays(lay, state_row_start, state_tpl, row_ptr, col, val, locality)


def _instance_id(lay: Layout) -> str:
    P = lay.P
    iid = f"tb_{P.pattern}_n{P.n}_S{lay.n_states}_k{P.k}_pf{P.pfail:g}_C{lay.C}_s{P.seed}"
    if P.hazard > 0.0:
        iid += f"_hz{P.hazard:g}"
    if P.permute != "none":
        iid += f"_perm{P.permute}"
    return iid


def _reach_stats(lay: Layout, comp: CompiledCSG) -> dict:

    n_reach, frac, mask = reachable_within_k(comp, lay.P.k)
    fail_id = int(np.flatnonzero(comp.state_kind == 1)[0])
    fail_reachable = bool(mask[fail_id])


    if fail_reachable and lay.P.k < lay.n_stage:
        raise RuntimeError("internal: BFS reaches FAIL in fewer than min_steps_to_fail = n_stage steps")
    if (not fail_reachable) and lay.P.k >= lay.n_stage and lay.P.pfail > 0.0:
        raise RuntimeError("internal: FAIL unreachable within k >= n_stage although pfail > 0")
    return {"n_reachable_k": int(n_reach), "frac_reachable_k": float(frac), "fail_reachable_within_k": fail_reachable}


def _meta(lay: Layout, arr: Arrays, reach: Optional[dict] = None) -> dict:
    P = lay.P
    rows_per_state = np.diff(arr.state_row_start)
    m = {"instance_id": _instance_id(lay), "seed": P.seed, "params": P.as_dict(),
         "generator": "TicketBench-MAS", "generator_version": GENERATOR_VERSION,
         "n_agents": P.n, "horizon_k": P.k, "states": lay.n_states, "states_requested": P.states,
         "n_states": lay.n_states,
         "n_requests": lay.R, "retry_budget": lay.C, "n_stages": lay.n_stage,
         "stage_agents": [s.agents for s in lay.stages], "stage_kinds": [s.kind for s in lay.stages],
         "agent_roles": lay.roles, "actions": ACTIONS, "fail_state": lay.fail_state,
         "min_steps_to_fail": lay.n_stage, "fail_reachable_within_k": bool(P.k >= lay.n_stage),
         "state_encoding": "((r*n_stages + j)*C + c)*2 + h ; h: 0=healthy 1=inconsistent ; FAIL = S-1"
                           + (" ; then renumbered by a seeded random permutation (permute=random)"
                              if P.permute != "none" else ""),
         "n_rows": int(arr.row_ptr.shape[0] - 1), "nnz": int(arr.col.shape[0]),
         "max_rows_per_state": int(rows_per_state.max()),
         "max_active_agents": 2, "max_actions": NA, "locality": arr.locality,
         "hazard": float(P.hazard), "permute": P.permute,
         "n_reachable_k": None, "frac_reachable_k": None,
         "reachability_note": "states reachable from init within k transitions (cubcr.model.reachable_within_k); "
                              "the engines sweep all n_states states unpruned",
         "v_stats": None,
         "pattern_info": lay.pattern_info}
    if reach is not None:
        m.update(reach)
    if P.permute != "none":
        m["locality"] = dict(arr.locality, note="statistics of the stage-major numbering BEFORE the permutation; "
                                                 "after permute=random successor ids are uniformly scattered")
        m["fail_state"] = None
    return m


def _spec(n, states, k, pfail, pattern, seed, **kw) -> Params:
    return Params(n=int(n), states=int(states), k=int(k), pfail=float(pfail), pattern=str(pattern),
                  seed=int(seed), **kw)


def build_instance(n: int, states: int, k: int, pfail: float, pattern: str = "none",
                   seed: int = 20260829, **kw) -> CSG:


    P = _spec(n, states, k, pfail, pattern, seed, **kw)
    if P.permute != "none":
        raise ValueError("permute is a compiled-level option: use build_compiled(..., permute='random')")
    lay = make_layout(P)
    arr = build_arrays(lay)
    reach = _reach_stats(lay, _compiled_from_arrays(lay, arr))
    return _csg_from_arrays(lay, arr, _meta(lay, arr, reach))


def build_compiled(n: int, states: int, k: int, pfail: float, pattern: str = "none",
                   seed: int = 20260829, **kw) -> Tuple[CompiledCSG, dict]:


    P = _spec(n, states, k, pfail, pattern, seed, **kw)
    lay = make_layout(P)
    arr = build_arrays(lay)
    comp = _compiled_from_arrays(lay, arr)
    if P.permute == "random":
        perm = np.random.default_rng([int(P.seed), _PERM_STREAM]).permutation(lay.n_states)
        comp = permute_states(comp, perm)
    return comp, _meta(lay, arr, _reach_stats(lay, comp))


def _csg_from_arrays(lay: Layout, arr: Arrays, meta: Optional[dict] = None) -> CSG:
    P = lay.P
    if meta is None:
        meta = _meta(lay, arr, _reach_stats(lay, _compiled_from_arrays(lay, arr)))
    S = lay.n_states
    srs = arr.state_row_start.tolist()
    rp = arr.row_ptr.tolist()
    cols = arr.col.tolist()
    vals = arr.val.tolist()
    stpl = arr.state_tpl.tolist()
    tpls = lay.templates
    tpl_active = [t.agents for t in tpls]
    tpl_rad = [t.radices for t in tpls]
    tpl_joints = [t.joints for t in tpls]
    tpl_labels = [[ACTIONS] * len(t.agents) for t in tpls]
    sds: List[StateDef] = [None] * S
    obs: Dict[Tuple[int, int], np.ndarray] = {}
    W = lay.weights
    for s in range(S - 1):
        ti = stpl[s]
        r0 = srs[s]
        J = tpl_joints[ti]
        rows = []
        for i in range(len(J)):
            a, b = rp[r0 + i], rp[r0 + i + 1]
            rows.append((J[i], list(zip(cols[a:b], vals[a:b]))))
        sds[s] = StateDef(active=tpl_active[ti], n_actions=tpl_rad[ti], rows=rows, action_labels=tpl_labels[ti])
        h = tpls[ti].h
        for ag in tpl_active[ti]:
            obs[(s, ag)] = W[(ag, h)]
    f = lay.fail_state
    sds[f] = StateDef(active=[], n_actions=[], rows=[((), [(f, 1.0)])])
    csg = CSG(n_agents=P.n, horizon_k=P.k, init_state=0, fail_states=[f], state_defs=sds,
              observed_policy=obs, avoid_states=[], agent_names=list(lay.roles), meta=meta)
    return csg


def _compiled_from_arrays(lay: Layout, arr: Arrays) -> CompiledCSG:
    P = lay.P
    S = lay.n_states
    kind = np.zeros(S, dtype=np.uint8)
    kind[lay.fail_state] = 1
    agent = np.full((S, DMAX), -1, dtype=np.int16)
    radix = np.ones((S, DMAX), dtype=np.uint8)
    weights = np.zeros((S, DMAX, AMAX), dtype=np.float64)
    weights[:, :, 0] = 1.0
    stpl = arr.state_tpl
    for ti, t in enumerate(lay.templates):
        sel = np.flatnonzero(stpl == ti)
        for slot, (ag, m) in enumerate(zip(t.agents, t.radices)):
            agent[sel, slot] = ag
            radix[sel, slot] = m
            weights[sel, slot, :] = 0.0
            weights[sel, slot, :m] = lay.weights[(ag, t.h)]
    return CompiledCSG(n_agents=P.n, n_states=S, init_state=0, horizon_k=P.k, state_kind=kind,
                       state_row_start=arr.state_row_start.copy(), state_agent=agent, state_radix=radix,
                       weights=weights, row_ptr=arr.row_ptr.copy(), col=arr.col.copy(), val=arr.val.copy())


def save_compiled_npz(c: CompiledCSG, meta: dict, path: str) -> None:
    np.savez(path, n_agents=c.n_agents, n_states=c.n_states, init_state=c.init_state, horizon_k=c.horizon_k,
             state_kind=c.state_kind, state_row_start=c.state_row_start, state_agent=c.state_agent,
             state_radix=c.state_radix, weights=c.weights, row_ptr=c.row_ptr, col=c.col, val=c.val,
             meta=json.dumps(meta))


def load_compiled_npz(path: str) -> Tuple[CompiledCSG, dict]:
    z = np.load(path, allow_pickle=False)
    c = CompiledCSG(n_agents=int(z["n_agents"]), n_states=int(z["n_states"]), init_state=int(z["init_state"]),
                    horizon_k=int(z["horizon_k"]), state_kind=z["state_kind"], state_row_start=z["state_row_start"],
                    state_agent=z["state_agent"], state_radix=z["state_radix"], weights=z["weights"],
                    row_ptr=z["row_ptr"], col=z["col"], val=z["val"])
    return c, json.loads(str(z["meta"]))


def emit_prism(csg: CSG, out_dir: str, instance_id: str, masks: Optional[List[int]] = None) -> List[str]:


    try:
        from cubcr.io_prism import emit_coalition_model
    except ImportError as e:
        print(f"[ticketbench] PRISM emission skipped: cubcr.io_prism not available ({e})")
        return []
    if masks is None:
        masks = [0, (1 << csg.n_agents) - 1]
    written: List[str] = []
    for m in masks:
        m = int(m)
        try:
            sig = inspect.signature(emit_coalition_model)
            kw = {}
            if "k" in sig.parameters:
                kw["k"] = csg.horizon_k
            if "comment" in sig.parameters:
                kw["comment"] = f"TicketBench-MAS {instance_id} mask={m}"
            res = emit_coalition_model(csg, m, **kw)
        except Exception as e:
            print(f"[ticketbench] PRISM emission failed for mask={m}: {type(e).__name__}: {e}")
            return written
        if not (isinstance(res, tuple) and len(res) == 2 and all(isinstance(x, str) for x in res)):
            print(f"[ticketbench] PRISM emission skipped: unexpected return type {type(res).__name__}")
            return written
        base = os.path.join(out_dir, f"{instance_id}.mask{m}")
        with open(base + ".prism", "w") as f:
            f.write(res[0])
        with open(base + ".props", "w") as f:
            f.write(res[1])
        written += [base + ".prism", base + ".props"]
        print(f"[ticketbench] wrote {base}.prism / .props")
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="TicketBench-MAS CSG generator")
    ap.add_argument("--n", type=int, default=8, help="number of agents (total, incl. pattern agents)")
    ap.add_argument("--states", type=int, default=1000, help="target |S| (exact count reported in meta)")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--pfail", type=float, default=0.05)
    ap.add_argument("--pattern", choices=PATTERNS, default="none")
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--out", default=os.path.join(_ROOT, "datasets", "ticketbench"))
    ap.add_argument("--emit", default="json", help="comma list of json,prism,npz or none")
    ap.add_argument("--retries", type=int, default=3, help="retry budget C per stage")
    ap.add_argument("--pleak", type=float, default=0.05)
    ap.add_argument("--prec", type=float, default=0.5)
    ap.add_argument("--pcomp", type=float, default=0.75)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--q", type=float, default=0.5)
    ap.add_argument("--hazard", type=float, default=0.0,
                    help="unavoidable failure probability per commit-stage step (0 < v(empty) < v(N) when > 0)")
    ap.add_argument("--permute", choices=PERMUTATIONS, default="none",
                    help="'random': seeded random state renumbering (locality-hostile; npz/none emission only)")
    ap.add_argument("--prism-masks", default="0,all",
                    help="comma list of coalition masks (ints, or 'all' = 2^n-1) for --emit prism")
    a = ap.parse_args(argv)
    emits = set() if a.emit.strip().lower() == "none" else {e.strip().lower() for e in a.emit.split(",") if e.strip()}
    unknown = emits - {"json", "prism", "npz"}
    if unknown:
        ap.error(f"unknown --emit targets {sorted(unknown)}")
    kw = dict(retries=a.retries, pleak=a.pleak, prec=a.prec, pcomp=a.pcomp, rho=a.rho, q=a.q,
              hazard=a.hazard, permute=a.permute)
    if a.permute != "none" and ("json" in emits or "prism" in emits):
        ap.error("--permute random is a compiled-level option; use --emit npz or none")
    t0 = time.perf_counter()
    P = _spec(a.n, a.states, a.k, a.pfail, a.pattern, a.seed, **kw)
    lay = make_layout(P)
    arr = build_arrays(lay)
    t_arrays = time.perf_counter() - t0
    c = _compiled_from_arrays(lay, arr)
    if P.permute == "random":
        c = permute_states(c, np.random.default_rng([int(P.seed), _PERM_STREAM]).permutation(lay.n_states))
    meta = _meta(lay, arr, _reach_stats(lay, c))
    print(f"[ticketbench] {meta['instance_id']}: S={meta['states']} (requested {a.states}) rows={meta['n_rows']} "
          f"nnz={meta['nnz']} R={lay.R} C={lay.C} stages={lay.n_stage} arrays={t_arrays:.2f}s "
          f"reachable<=k={meta['n_reachable_k']} ({100.0 * meta['frac_reachable_k']:.3f}%) "
          f"hazard={P.hazard:g} permute={P.permute} build={time.perf_counter() - t0:.2f}s")
    if not meta["fail_reachable_within_k"]:
        print(f"[ticketbench] WARNING: k={a.k} < min_steps_to_fail={lay.n_stage}: v == 0 for every coalition")
    os.makedirs(a.out, exist_ok=True)
    if "json" in emits or "prism" in emits:
        t1 = time.perf_counter()
        csg = _csg_from_arrays(lay, arr, meta)
        print(f"[ticketbench] CSG object built in {time.perf_counter() - t1:.2f}s")
        if "json" in emits:
            if lay.n_states > 100_000:
                print("[ticketbench] note: JSON export above 1e5 states is large; consider --emit npz")
            path = os.path.join(a.out, meta["instance_id"] + ".json")
            save_csg(csg, path)
            print(f"[ticketbench] wrote {path}")
        if "prism" in emits:
            masks = [(1 << a.n) - 1 if tok.strip().lower() == "all" else int(tok)
                     for tok in a.prism_masks.split(",") if tok.strip()]
            emit_prism(csg, a.out, meta["instance_id"], masks)
    if "npz" in emits:
        path = os.path.join(a.out, meta["instance_id"] + ".npz")
        save_compiled_npz(c, meta, path)
        print(f"[ticketbench] wrote {path}")
    print(f"[ticketbench] total {time.perf_counter() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
