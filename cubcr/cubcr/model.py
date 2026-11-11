from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

FORMAT = "cubcr-csg-v1"
DMAX = 3
AMAX = 8
RMAX = 64


@dataclass
class StateDef:
    active: List[int]
    n_actions: List[int]
    rows: List[Tuple[Tuple[int, ...], List[Tuple[int, float]]]]
    action_labels: Optional[List[List[str]]] = None


@dataclass
class CSG:
    n_agents: int
    horizon_k: int
    init_state: int
    fail_states: List[int]
    state_defs: List[StateDef]
    observed_policy: Dict[Tuple[int, int], np.ndarray]
    avoid_states: List[int] = field(default_factory=list)
    agent_names: Optional[List[str]] = None
    meta: Dict = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return len(self.state_defs)


    def to_json(self) -> dict:
        sd = []
        for d in self.state_defs:
            e = {"active": list(map(int, d.active)), "n_actions": list(map(int, d.n_actions)),
                 "rows": [{"joint": list(map(int, j)), "succ": [[int(s), float(p)] for s, p in succ]}
                          for j, succ in d.rows]}
            if d.action_labels is not None:
                e["action_labels"] = d.action_labels
            sd.append(e)
        obs = [{"state": int(s), "agent": int(a), "dist": [float(x) for x in dist]}
               for (s, a), dist in sorted(self.observed_policy.items())]
        out = {"format": FORMAT, "n_agents": self.n_agents, "horizon_k": self.horizon_k,
               "states": self.n_states, "init_state": self.init_state,
               "fail_states": list(map(int, self.fail_states)),
               "avoid_states": list(map(int, self.avoid_states)),
               "state_defs": sd, "observed_policy": obs}
        if self.agent_names is not None:
            out["agent_names"] = self.agent_names
        if self.meta:
            out["meta"] = self.meta
        return out

    @staticmethod
    def from_json(obj: dict) -> "CSG":
        if obj.get("format", FORMAT) != FORMAT:
            raise ValueError(f"unknown format {obj.get('format')}")
        sds = []
        for e in obj["state_defs"]:
            rows = [(tuple(int(x) for x in r["joint"]), [(int(s), float(p)) for s, p in r["succ"]])
                    for r in e["rows"]]
            sds.append(StateDef(active=[int(a) for a in e.get("active", [])],
                                n_actions=[int(m) for m in e.get("n_actions", [])],
                                rows=rows, action_labels=e.get("action_labels")))
        obs = {(int(o["state"]), int(o["agent"])): np.asarray(o["dist"], dtype=np.float64)
               for o in obj.get("observed_policy", [])}
        csg = CSG(n_agents=int(obj["n_agents"]), horizon_k=int(obj["horizon_k"]),
                  init_state=int(obj["init_state"]), fail_states=[int(s) for s in obj["fail_states"]],
                  state_defs=sds, observed_policy=obs,
                  avoid_states=[int(s) for s in obj.get("avoid_states", [])],
                  agent_names=obj.get("agent_names"), meta=obj.get("meta", {}))
        if "states" in obj and int(obj["states"]) != csg.n_states:
            raise ValueError("states count mismatch")
        csg.validate()
        return csg


    def validate(self, tol: float = 1e-9) -> None:

        n, S = self.n_agents, self.n_states
        if not (1 <= n <= 64):
            raise ValueError("n_agents must be in 1..64")
        if not (0 <= self.init_state < S):
            raise ValueError("init_state out of range")
        for s in self.fail_states + self.avoid_states:
            if not (0 <= s < S):
                raise ValueError(f"fail/avoid state {s} out of range")
        if set(self.fail_states) & set(self.avoid_states):
            raise ValueError("fail and avoid sets overlap")
        if self.horizon_k < 0:
            raise ValueError("horizon_k must be >= 0")
        special = set(self.fail_states) | set(self.avoid_states)
        for s, d in enumerate(self.state_defs):
            if len(d.active) != len(d.n_actions):
                raise ValueError(f"state {s}: active/n_actions length mismatch")
            if len(d.active) > DMAX:
                raise ValueError(f"state {s}: more than DMAX={DMAX} active agents")
            if len(set(d.active)) != len(d.active):
                raise ValueError(f"state {s}: duplicate active agent")
            for a, m in zip(d.active, d.n_actions):
                if not (0 <= a < n):
                    raise ValueError(f"state {s}: agent {a} out of range")
                if not (1 <= m <= AMAX):
                    raise ValueError(f"state {s}: n_actions {m} not in 1..{AMAX}")
            nrows = int(np.prod(d.n_actions)) if d.n_actions else 1
            if nrows > RMAX:
                raise ValueError(f"state {s}: {nrows} joint-action rows exceed the engine limit RMAX={RMAX}")
            if len(d.rows) != nrows:
                raise ValueError(f"state {s}: expected {nrows} rows, got {len(d.rows)}")
            for r, (joint, succ) in enumerate(d.rows):
                if len(joint) != len(d.active):
                    raise ValueError(f"state {s} row {r}: joint length mismatch")
                idx = 0
                for a_i, m in zip(joint, d.n_actions):
                    if not (0 <= a_i < m):
                        raise ValueError(f"state {s} row {r}: action index out of range")
                    idx = idx * m + a_i
                if idx != r:
                    raise ValueError(f"state {s} row {r}: rows not in mixed-radix order")
                if not succ:
                    raise ValueError(f"state {s} row {r}: empty successor list")
                tot = 0.0
                for sp, p in succ:
                    if not (0 <= sp < S):
                        raise ValueError(f"state {s} row {r}: successor {sp} out of range")
                    if not np.isfinite(p) or p < -tol or p > 1.0 + tol:
                        raise ValueError(f"state {s} row {r}: probability {p} not in [0,1]")
                    tot += p
                if not abs(tot - 1.0) <= tol:
                    raise ValueError(f"state {s} row {r}: probabilities sum to {tot}")
            if s not in special:
                for a, m in zip(d.active, d.n_actions):
                    dist = self.observed_policy.get((s, a))
                    if dist is None:
                        raise ValueError(f"missing observed policy for state {s} agent {a}")
                    if len(dist) != m:
                        raise ValueError(f"observed policy for state {s} agent {a} has wrong length")
                    if (not np.all(np.isfinite(dist))) or np.any(dist < -tol) or not abs(float(dist.sum()) - 1.0) <= tol:
                        raise ValueError(f"observed policy for state {s} agent {a} not a distribution")


    def compile(self) -> "CompiledCSG":
        return CompiledCSG.from_csg(self)


def load_csg(path: str) -> CSG:
    with open(path) as f:
        return CSG.from_json(json.load(f))


def save_csg(csg: CSG, path: str) -> None:
    with open(path, "w") as f:
        json.dump(csg.to_json(), f)


@dataclass
class CompiledCSG:
    n_agents: int
    n_states: int
    init_state: int
    horizon_k: int
    state_kind: np.ndarray
    state_row_start: np.ndarray
    state_agent: np.ndarray
    state_radix: np.ndarray
    weights: np.ndarray
    row_ptr: np.ndarray
    col: np.ndarray
    val: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.row_ptr.shape[0] - 1)

    @property
    def nnz(self) -> int:
        return int(self.col.shape[0])

    @property
    def max_rows_per_state(self) -> int:
        return int(np.max(np.diff(self.state_row_start)))

    @staticmethod
    def from_csg(csg: CSG) -> "CompiledCSG":
        csg.validate()
        S = csg.n_states
        kind = np.zeros(S, dtype=np.uint8)
        kind[list(csg.fail_states)] = 1
        kind[list(csg.avoid_states)] = 2
        row_start = np.zeros(S + 1, dtype=np.int64)
        agent = np.full((S, DMAX), -1, dtype=np.int16)
        radix = np.ones((S, DMAX), dtype=np.uint8)
        weights = np.zeros((S, DMAX, AMAX), dtype=np.float64)
        weights[:, :, 0] = 1.0
        row_ptr = [0]
        cols: List[np.ndarray] = []
        vals: List[np.ndarray] = []
        for s, d in enumerate(csg.state_defs):
            row_start[s + 1] = row_start[s] + len(d.rows)
            for slot, (a, m) in enumerate(zip(d.active, d.n_actions)):
                agent[s, slot] = a
                radix[s, slot] = m
                weights[s, slot, :] = 0.0
                if kind[s] == 0:
                    weights[s, slot, :m] = csg.observed_policy[(s, a)]
                else:
                    weights[s, slot, 0] = 1.0
            for joint, succ in d.rows:

                acc: Dict[int, float] = {}
                for sp, p in succ:
                    acc[sp] = acc.get(sp, 0.0) + p
                keys = sorted(acc)
                cols.append(np.asarray(keys, dtype=np.int32))
                vals.append(np.asarray([acc[kk] for kk in keys], dtype=np.float64))
                row_ptr.append(row_ptr[-1] + len(keys))
        col = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int32)
        val = np.concatenate(vals) if vals else np.zeros(0, dtype=np.float64)
        return CompiledCSG(n_agents=csg.n_agents, n_states=S, init_state=csg.init_state,
                           horizon_k=csg.horizon_k, state_kind=kind, state_row_start=row_start,
                           state_agent=agent, state_radix=radix, weights=weights,
                           row_ptr=np.asarray(row_ptr, dtype=np.int64), col=col, val=val)

    def fingerprint(self) -> str:
        import hashlib
        h = hashlib.sha256()
        for arr in (self.state_kind, self.state_row_start, self.state_agent, self.state_radix,
                    self.weights, self.row_ptr, self.col, self.val):
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(f"{self.n_agents},{self.n_states},{self.init_state}".encode())
        return h.hexdigest()[:16]


def _ragged_gather(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:

    starts = np.asarray(starts, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offs = np.zeros(lengths.shape[0], dtype=np.int64)
    np.cumsum(lengths[:-1], out=offs[1:])
    return np.repeat(starts - offs, lengths) + np.arange(total, dtype=np.int64)


def reachable_within_k(compiled: "CompiledCSG", k: int) -> Tuple[int, float, np.ndarray]:


    S = int(compiled.n_states)
    k = int(k)
    if k < 0:
        raise ValueError("k must be >= 0")
    srs = np.asarray(compiled.state_row_start, dtype=np.int64)
    rp = np.asarray(compiled.row_ptr, dtype=np.int64)
    col = np.asarray(compiled.col)
    kind = np.asarray(compiled.state_kind)

    st_nz0 = rp[srs[:-1]]
    st_nz1 = rp[srs[1:]]
    visited = np.zeros(S, dtype=bool)
    init = int(compiled.init_state)
    visited[init] = True
    frontier = np.array([init], dtype=np.int64)
    for _ in range(k):
        if frontier.shape[0] == 0:
            break
        exp = frontier[kind[frontier] == 0]
        if exp.shape[0] == 0:
            break
        idx = _ragged_gather(st_nz0[exp], st_nz1[exp] - st_nz0[exp])
        cand = col[idx].astype(np.int64, copy=False)
        cand = cand[~visited[cand]]
        if cand.shape[0] == 0:
            break
        frontier = np.unique(cand)
        visited[frontier] = True
    n_reach = int(visited.sum())
    return n_reach, n_reach / S, visited


def permute_states(compiled: "CompiledCSG", perm) -> "CompiledCSG":


    S = int(compiled.n_states)
    perm = np.asarray(perm, dtype=np.int64)
    if perm.shape != (S,):
        raise ValueError(f"perm must have shape ({S},), got {perm.shape}")
    if S > 0 and (int(perm.min()) < 0 or int(perm.max()) >= S):
        raise ValueError("perm is not a permutation of range(S): entries out of range")
    inv = np.empty(S, dtype=np.int64)
    inv[perm] = np.arange(S, dtype=np.int64)
    if not np.array_equal(perm[inv], np.arange(S)):
        raise ValueError("perm is not a permutation of range(S)")
    srs = np.asarray(compiled.state_row_start, dtype=np.int64)
    rp = np.asarray(compiled.row_ptr, dtype=np.int64)
    rows_per_state = np.diff(srs)

    new_rows_per_state = rows_per_state[inv]
    new_srs = np.zeros(S + 1, dtype=np.int64)
    np.cumsum(new_rows_per_state, out=new_srs[1:])
    old_row_order = _ragged_gather(srs[inv], new_rows_per_state)
    nnz_per_row = np.diff(rp)
    new_nnz_per_row = nnz_per_row[old_row_order]
    new_rp = np.zeros(old_row_order.shape[0] + 1, dtype=np.int64)
    np.cumsum(new_nnz_per_row, out=new_rp[1:])
    old_nz_order = _ragged_gather(rp[old_row_order], new_nnz_per_row)
    new_col = perm[np.asarray(compiled.col, dtype=np.int64)[old_nz_order]].astype(np.int32)
    new_val = np.asarray(compiled.val, dtype=np.float64)[old_nz_order]
    return CompiledCSG(n_agents=compiled.n_agents, n_states=S, init_state=int(perm[compiled.init_state]),
                       horizon_k=compiled.horizon_k,
                       state_kind=np.asarray(compiled.state_kind)[inv].copy(),
                       state_row_start=new_srs,
                       state_agent=np.asarray(compiled.state_agent)[inv].copy(),
                       state_radix=np.asarray(compiled.state_radix)[inv].copy(),
                       weights=np.asarray(compiled.weights)[inv].copy(),
                       row_ptr=new_rp, col=new_col, val=new_val)


def restrict_to_horizon(comp: "CompiledCSG", k: int) -> "CompiledCSG":


    import numpy as np
    S = comp.n_states
    rs = comp.state_row_start
    rp, col = comp.row_ptr, comp.col

    s_p0 = rp[rs[:-1]]
    s_p1 = rp[rs[1:]]
    depth = np.full(S, -1, dtype=np.int64)
    depth[comp.init_state] = 0
    frontier = np.array([comp.init_state], dtype=np.int64)
    for d in range(1, k + 1):
        if frontier.size == 0:
            break

        fr = frontier[comp.state_kind[frontier] == 0]
        if fr.size == 0:
            break
        idx = np.concatenate([np.arange(a, b) for a, b in zip(s_p0[fr], s_p1[fr])]) if fr.size else np.zeros(0, dtype=np.int64)
        succ = np.unique(col[idx])
        new = succ[depth[succ] < 0]
        depth[new] = d
        frontier = new
    keep = np.nonzero(depth >= 0)[0]
    new_id = np.full(S, -1, dtype=np.int64)
    new_id[keep] = np.arange(keep.size)
    terminal = (depth[keep] == k) | (comp.state_kind[keep] != 0)

    row_start = [0]
    agent = np.full((keep.size, DMAX), -1, dtype=np.int16)
    radix = np.ones((keep.size, DMAX), dtype=np.uint8)
    weights = np.zeros((keep.size, DMAX, AMAX), dtype=np.float64)
    weights[:, :, 0] = 1.0
    row_ptr = [0]
    cols, vals = [], []
    for j, s in enumerate(keep):
        if terminal[j]:
            row_start.append(row_start[-1] + 1)
            row_ptr.append(row_ptr[-1] + 1)
            cols.append(np.array([j], dtype=np.int32)); vals.append(np.array([1.0]))
            continue
        agent[j] = comp.state_agent[s]; radix[j] = comp.state_radix[s]; weights[j] = comp.weights[s]
        r0, r1 = rs[s], rs[s + 1]
        row_start.append(row_start[-1] + (r1 - r0))
        for r in range(r0, r1):
            p0, p1 = rp[r], rp[r + 1]
            c = new_id[col[p0:p1]]
            if np.any(c < 0):
                raise RuntimeError("internal: successor of a depth<k state not kept")
            cols.append(c.astype(np.int32)); vals.append(comp.val[p0:p1])
            row_ptr.append(row_ptr[-1] + (p1 - p0))
    out = CompiledCSG(n_agents=comp.n_agents, n_states=int(keep.size), init_state=int(new_id[comp.init_state]),
                      horizon_k=k, state_kind=comp.state_kind[keep].copy(), state_row_start=np.asarray(row_start, dtype=np.int64),
                      state_agent=agent, state_radix=radix, weights=weights, row_ptr=np.asarray(row_ptr, dtype=np.int64),
                      col=np.concatenate(cols) if cols else np.zeros(0, np.int32), val=np.concatenate(vals) if vals else np.zeros(0))
    return out
