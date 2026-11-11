from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

GEN_VERSION = "e5-cap-1.0"
D4 = 4
AMAX = 8
N_STAGE_DEFAULT = None


@dataclass
class GenModel:

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
    meta: Dict = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return int(self.row_ptr.shape[0] - 1)

    @property
    def nnz(self) -> int:
        return int(self.col.shape[0])

    @property
    def max_rows_per_state(self) -> int:
        return int(np.max(np.diff(self.state_row_start)))

    @property
    def n_slots(self) -> int:
        return int(self.state_agent.shape[1])

    @property
    def z_model(self) -> int:

        return self.nnz + self.n_rows

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for arr in (self.state_kind, self.state_row_start, self.state_agent, self.state_radix,
                    self.weights, self.row_ptr, self.col, self.val):
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(f"{self.n_agents},{self.n_states},{self.init_state}".encode())
        return h.hexdigest()[:16]


def instance_id(n, a, m, R, k, seed, eta=0.02, gamma=2.0, stage_config=None) -> str:
    base = f"cap_n{n}_a{a}_m{m}_R{R}_S{n * R + 1}_k{k}_eta{eta:g}_g{gamma:g}_s{seed}"
    if stage_config is not None:
        base += "_mixed" + hashlib.sha256(repr(list(stage_config)).encode()).hexdigest()[:6]
    return base


def build(n: int = 16, a: int = 2, m: int = 4, R: int = 62, k: int = 50, seed: int = 20260829,
          eta: float = 0.02, gamma: float = 2.0, kappa_max: float = 6.0,
          stage_config: Optional[Sequence[Tuple[int, int]]] = None) -> GenModel:

    n_stage = n
    if stage_config is None:
        stage_config = [(a, m)] * n_stage
    stage_config = [(int(x), int(y)) for x, y in stage_config]
    if len(stage_config) != n_stage:
        raise ValueError("stage_config must have one (a, m) per stage")
    for aj, mj in stage_config:
        if not (1 <= aj <= min(D4, n)):
            raise ValueError(f"a={aj} must be in 1..min({D4}, n)")
        if not (2 <= mj <= AMAX):
            raise ValueError(f"m={mj} must be in 2..{AMAX}")
    if not (0.0 < eta and eta * 1.5 * 1.1 * (1.0 + gamma) < 1.0):
        raise ValueError("eta/gamma give failure probabilities >= 1")
    if R < 1 or n < 1 or n > 64:
        raise ValueError("bad R / n")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xE5CA9]))
    S_nt = n_stage * R
    S = S_nt + 1
    FAIL = S - 1

    kappa = np.linspace(-kappa_max, kappa_max, n)[rng.permutation(n)]
    rows_stage = np.array([mj ** aj for aj, mj in stage_config], dtype=np.int64)
    rows_per_state = np.empty(S, dtype=np.int64)
    rows_per_state[:S_nt] = np.tile(rows_stage, R)
    rows_per_state[FAIL] = 1
    srs = np.zeros(S + 1, dtype=np.int64)
    np.cumsum(rows_per_state, out=srs[1:])
    n_rows = int(srs[-1])
    nnz = 2 * (n_rows - 1) + 1
    row_ptr = np.empty(n_rows + 1, dtype=np.int64)
    row_ptr[:n_rows] = 2 * np.arange(n_rows, dtype=np.int64)
    row_ptr[n_rows] = nnz
    col = np.empty(nnz, dtype=np.int32)
    val = np.empty(nnz, dtype=np.float64)
    kind = np.zeros(S, dtype=np.uint8)
    kind[FAIL] = 1
    agent = np.full((S, D4), -1, dtype=np.int16)
    radix = np.ones((S, D4), dtype=np.uint8)
    weights = np.zeros((S, D4, AMAX), dtype=np.float64)
    weights[:, :, 0] = 1.0
    eta_s = eta * (0.5 + rng.random(S_nt))
    r_idx = np.arange(R, dtype=np.int64)
    for j, (aj, mj) in enumerate(stage_config):
        states = r_idx * n_stage + j
        act = np.sort(np.array([(j + l) % n for l in range(aj)], dtype=np.int64))
        u = rng.random((R, aj, mj))
        logits = -kappa[act][None, :, None] * u
        logits -= logits.max(axis=2, keepdims=True)
        pi = np.exp(logits)
        pi /= pi.sum(axis=2, keepdims=True)
        agent[states[:, None], np.arange(aj)[None, :]] = act[None, :].astype(np.int16)
        radix[states, :aj] = mj
        weights[states, :aj, :] = 0.0
        weights[states[:, None, None], np.arange(aj)[None, :, None], np.arange(mj)[None, None, :]] = pi

        Rj = mj ** aj
        dig = np.zeros((Rj, aj), dtype=np.int64)
        rr = np.arange(Rj, dtype=np.int64)
        for l in reversed(range(aj)):
            dig[:, l] = rr % mj
            rr //= mj
        ux = u[:, np.arange(aj)[None, :], dig]
        mean_u = ux.mean(axis=2)
        conflict = (np.all(dig == dig[:, :1], axis=1) if aj >= 2 else np.zeros(Rj, dtype=bool)).astype(np.float64)
        p = eta_s[states][:, None] * (0.1 + mean_u) * (1.0 + gamma * conflict[None, :])
        abort = (dig.sum(axis=1) % mj) == 0
        nxt_req = ((r_idx + 1) % R) * n_stage
        cont = (r_idx * n_stage + j + 1) if j + 1 < n_stage else nxt_req
        nxt = np.where(abort[None, :], nxt_req[:, None], cont[:, None] if np.ndim(cont) else cont)
        rows = srs[states][:, None] + np.arange(Rj, dtype=np.int64)[None, :]
        col[2 * rows] = nxt.astype(np.int32)
        col[2 * rows + 1] = FAIL
        val[2 * rows] = 1.0 - p
        val[2 * rows + 1] = p
    col[nnz - 1] = FAIL
    val[nnz - 1] = 1.0
    gm = GenModel(n_agents=n, n_states=S, init_state=0, horizon_k=int(k), state_kind=kind, state_row_start=srs,
                  state_agent=agent, state_radix=radix, weights=weights, row_ptr=row_ptr, col=col, val=val)
    a_list = sorted({x for x, _ in stage_config})
    m_list = sorted({y for _, y in stage_config})
    gm.meta = {"generator": "e5 co-active pipeline (CAP)", "generator_version": GEN_VERSION,
               "instance_id": instance_id(n, a, m, R, k, seed, eta, gamma,
                                          None if len(set(stage_config)) == 1 else stage_config),
               "n_agents": n, "n_stage": n_stage, "n_requests": R, "states": S, "horizon_k": int(k),
               "seed": int(seed), "eta": eta, "gamma": gamma, "kappa_max": kappa_max,
               "kappa": kappa.tolist(), "stage_config": stage_config,
               "a": a_list[0] if len(a_list) == 1 else a_list, "m": m_list[0] if len(m_list) == 1 else m_list,
               "rows_per_state_nonterminal": sorted(set(rows_stage.tolist())),
               "n_rows": n_rows, "nnz": nnz, "z_model": n_rows + nnz, "fail_state": FAIL,
               "succ_entries_per_row": 2}
    return gm


def validate(gm: GenModel, tol: float = 1e-12) -> dict:


    S = gm.n_states
    rp, col, val = gm.row_ptr, gm.col, gm.val
    if np.any(~np.isfinite(val)) or np.any(~np.isfinite(gm.weights)):
        raise ValueError("NaN/Inf in inputs")
    if np.any(val < 0) or np.any(gm.weights < 0):
        raise ValueError("negative probability")
    if np.any(col < 0) or np.any(col >= S):
        raise ValueError("successor out of range")
    row_sum = np.add.reduceat(val, rp[:-1])
    dev_row = float(np.max(np.abs(row_sum - 1.0)))
    if dev_row > tol:
        raise ValueError(f"row sums deviate from 1 by {dev_row}")
    fail = np.nonzero(gm.state_kind == 1)[0]
    for f in fail:
        r0, r1 = gm.state_row_start[f], gm.state_row_start[f + 1]
        for r in range(r0, r1):
            c = col[rp[r]:rp[r + 1]]; v = val[rp[r]:rp[r + 1]]
            if not (np.all(c == f) and abs(v.sum() - 1.0) <= tol):
                raise ValueError("FAIL state not absorbing")
    nt = gm.state_kind == 0
    used = gm.state_agent >= 0
    radix = gm.state_radix.astype(np.int64)
    rows_expected = np.prod(np.where(used, radix, 1), axis=1)
    rows = np.diff(gm.state_row_start)
    if not np.array_equal(rows[nt], rows_expected[nt]):
        raise ValueError("rows per state != product of radices")
    wsum = gm.weights.sum(axis=2)
    dev_pol = float(np.max(np.abs(wsum[used & nt[:, None]] - 1.0))) if np.any(used & nt[:, None]) else 0.0
    if dev_pol > tol:
        raise ValueError(f"policy sums deviate from 1 by {dev_pol}")

    a_idx = np.arange(gm.weights.shape[2])[None, None, :]
    pad = gm.weights * (a_idx >= radix[:, :, None])
    if np.any(pad != 0):
        raise ValueError("nonzero policy padding")
    if not np.all(gm.weights[~used][:, 0] == 1.0):
        raise ValueError("unused slot is not the unit distribution")
    act_counts = used.sum(axis=1)
    return {"max_row_sum_dev": dev_row, "max_policy_sum_dev": dev_pol,
            "active_agents_per_nonterminal_state": sorted(set(act_counts[nt].tolist())),
            "actions_per_active_agent": sorted(set(radix[used & nt[:, None]].tolist())),
            "rows_per_nonterminal_state": sorted(set(rows[nt].tolist())), "fail_absorbing": True,
            "min_val": float(val.min())}


def to_compiled3(gm: GenModel):


    from cubcr.model import CompiledCSG, DMAX
    if np.any(gm.state_agent[:, DMAX:] >= 0):
        raise ValueError("model has > 3 active agents in some state: not representable in the paper layout")
    return CompiledCSG(n_agents=gm.n_agents, n_states=gm.n_states, init_state=gm.init_state,
                       horizon_k=gm.horizon_k, state_kind=gm.state_kind.copy(),
                       state_row_start=gm.state_row_start.copy(),
                       state_agent=np.ascontiguousarray(gm.state_agent[:, :DMAX]),
                       state_radix=np.ascontiguousarray(gm.state_radix[:, :DMAX]),
                       weights=np.ascontiguousarray(gm.weights[:, :DMAX, :]),
                       row_ptr=gm.row_ptr.copy(), col=gm.col.copy(), val=gm.val.copy())


def to_csg(gm: GenModel):


    from cubcr.model import CSG, StateDef
    if gm.n_rows > 200_000:
        raise ValueError("to_csg is meant for tiny models")
    S = gm.n_states
    sds, obs = [], {}
    for s in range(S):
        slots = [l for l in range(gm.n_slots) if gm.state_agent[s, l] >= 0]
        active = [int(gm.state_agent[s, l]) for l in slots]
        nact = [int(gm.state_radix[s, l]) for l in slots]
        r0, r1 = int(gm.state_row_start[s]), int(gm.state_row_start[s + 1])
        rows = []
        for r in range(r0, r1):
            rl = r - r0
            joint = []
            for mm in reversed(nact):
                joint.append(rl % mm)
                rl //= mm
            joint = tuple(reversed(joint))
            succ = [(int(c), float(v)) for c, v in zip(gm.col[gm.row_ptr[r]:gm.row_ptr[r + 1]],
                                                       gm.val[gm.row_ptr[r]:gm.row_ptr[r + 1]])]
            rows.append((joint, succ))
        sds.append(StateDef(active=active, n_actions=nact, rows=rows))
        if gm.state_kind[s] == 0:
            for l, a_ in zip(slots, active):
                obs[(s, a_)] = gm.weights[s, l, :gm.state_radix[s, l]].copy()
    return CSG(n_agents=gm.n_agents, horizon_k=gm.horizon_k, init_state=gm.init_state,
               fail_states=[int(x) for x in np.nonzero(gm.state_kind == 1)[0]], state_defs=sds,
               observed_policy=obs, avoid_states=[int(x) for x in np.nonzero(gm.state_kind == 2)[0]],
               meta=dict(gm.meta))


def reachable_frac(gm: GenModel, k: Optional[int] = None) -> Tuple[int, float]:
    from cubcr.model import reachable_within_k
    n_reach, frac, _ = reachable_within_k(gm, gm.horizon_k if k is None else k)
    return int(n_reach), float(frac)


def agents_active_in_reach(gm: GenModel, k: Optional[int] = None) -> List[int]:
    from cubcr.model import reachable_within_k
    _, _, mask = reachable_within_k(gm, gm.horizon_k if k is None else k)
    nt = mask & (gm.state_kind == 0)
    ag = gm.state_agent[nt]
    return sorted(set(int(x) for x in ag[ag >= 0].ravel()))


E5_N = 16
E5_K = 50
E5_GEN_SEED = 20260829
E5_SIZES = {"1e3": 62, "1e4": 625}
E5_WITHIN = [(1, 8), (2, 4), (2, 8), (3, 2), (3, 4)]
E5_BEYOND = [(3, 8), (4, 4)]


def e5_model(a: int, m: int, size: str) -> GenModel:
    return build(n=E5_N, a=a, m=m, R=E5_SIZES[size], k=E5_K, seed=E5_GEN_SEED)


def e5_batch_masks(n: int = E5_N, seed: int = E5_GEN_SEED) -> np.ndarray:


    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xE5BA7]))
    return rng.permutation(np.arange(1 << n, dtype=np.uint64))
