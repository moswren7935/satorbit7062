from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from .masks import step_mask_matrix

DEFAULT_SEED = 20260829


def empirical_bernstein_half_width(var_bessel: np.ndarray, m: int, alpha: float,
                                   lo: float = -1.0, hi: float = 1.0,
                                   two_sided: bool = False) -> np.ndarray:


    var_bessel = np.asarray(var_bessel, dtype=np.float64)
    if m < 2:
        return np.full(var_bessel.shape, np.inf)
    rng_ = float(hi - lo)
    var_y = np.maximum(var_bessel, 0.0) / (rng_ * rng_)
    L = math.log((4.0 if two_sided else 2.0) / alpha)
    hw_y = np.sqrt(2.0 * var_y * L / m) + 7.0 * L / (3.0 * (m - 1))
    return rng_ * hw_y


def look_schedule_size(max_perms: int, perms_per_batch: int) -> int:


    if perms_per_batch < 1:
        raise ValueError("perms_per_batch must be >= 1")
    L, cover = 1, int(perms_per_batch)
    while cover < max_perms:
        cover *= 2
        L += 1
    return L


class CoalitionOracle:


    def __init__(self, engine, n_players: int, kind: str = "agent", n: Optional[int] = None,
                 k: Optional[int] = None, use_cache: bool = True, pad_rows: Optional[bool] = None):


        if kind not in ("agent", "step"):
            raise ValueError("kind must be 'agent' or 'step'")
        self.engine = engine
        self.kind = kind
        if pad_rows is None:
            pad_rows = False
        self.pad_rows = bool(pad_rows)
        self.n_padded_rows = 0
        self.graph_captures = None
        self._captures0 = None
        comp = getattr(engine, "compiled", None)
        if n is None:
            n = getattr(comp, "n_agents", None)
        if k is None:
            k = getattr(comp, "horizon_k", None)
        if kind == "agent":
            if n is None:
                n = n_players
            if n_players != n:
                raise ValueError("agent-level: n_players must equal n")
        else:
            if n is None or k is None:
                raise ValueError("step-level: n and k required")
            if n_players != n * k:
                raise ValueError("step-level: n_players must equal n*k")
        if n > 64:
            raise ValueError("at most 64 agents (uint64 masks)")
        self.n, self.k, self.n_players = int(n), int(k) if k is not None else None, int(n_players)
        self.kc = 1 if kind == "agent" else self.k

        self.onehot = np.zeros((self.n_players, self.kc), dtype=np.uint64)
        for p in range(self.n_players):
            if kind == "agent":
                self.onehot[p, 0] = np.uint64(1) << np.uint64(p)
            else:
                self.onehot[p] = step_mask_matrix([p], self.n, self.k)
        self.use_cache = use_cache
        self.cache: dict = {}
        self.n_solves = 0
        self.n_requested = 0
        self.n_engine_calls = 0
        self.cache_hits = 0
        self.engine_time = 0.0


    def prefix_masks(self, perm: np.ndarray) -> np.ndarray:

        acc = np.bitwise_or.accumulate(self.onehot[perm], axis=0)
        return np.concatenate([np.zeros((1, self.kc), dtype=np.uint64), acc], axis=0)

    def masks_from_bitsets(self, bitsets) -> np.ndarray:

        bitsets = [int(b) for b in bitsets]
        out = np.zeros((len(bitsets), self.kc), dtype=np.uint64)
        for p in range(self.n_players):
            sel = np.fromiter(((b >> p) & 1 for b in bitsets), dtype=bool, count=len(bitsets))
            if sel.any():
                out[sel] |= self.onehot[p]
        return out

    def _engine_input(self, rows: np.ndarray):
        rows = np.ascontiguousarray(rows, dtype=np.uint64)
        if self.kind == "agent":
            return rows[:, 0].copy()
        return np.ascontiguousarray(rows.T)


    @staticmethod
    def bucket_rows(B: int) -> int:

        if B <= 16384:
            return -(-B // 1024) * 1024
        return 1 << (B - 1).bit_length()

    def _solve(self, rows: np.ndarray) -> np.ndarray:

        B = rows.shape[0]
        Bp = self.bucket_rows(B) if self.pad_rows else B
        if Bp > B:
            rows = np.concatenate([rows, np.zeros((Bp - B, self.kc), dtype=np.uint64)], axis=0)
            self.n_padded_rows += Bp - B
        t0 = time.perf_counter()
        vals = self.engine.solve(self._engine_input(rows), k=self.k)
        self.engine_time += time.perf_counter() - t0
        self.n_solves += B
        self.n_engine_calls += 1
        st = getattr(self.engine, "stats", None)
        if isinstance(st, dict) and "graph_captures" in st:
            if self._captures0 is None:
                self._captures0 = int(st["graph_captures"]) - 1
            self.graph_captures = int(st["graph_captures"]) - self._captures0
        return np.asarray(vals, dtype=np.float64)[:B]

    def values(self, rows: np.ndarray) -> np.ndarray:


        rows = np.ascontiguousarray(rows, dtype=np.uint64)
        if rows.ndim == 1:
            rows = rows.reshape(-1, self.kc)
        B = rows.shape[0]
        self.n_requested += B
        if not self.use_cache or (self.kind == "step" and self.n_players > 24):
            return self._solve(rows)
        if self.kc == 1:
            uniq, inv = np.unique(rows[:, 0], return_inverse=True)
            uniq = uniq.reshape(-1, 1)
        else:
            uniq, inv = np.unique(rows, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        U = uniq.shape[0]
        keys = [r.tobytes() for r in uniq]
        vu = np.empty(U, dtype=np.float64)
        todo = []
        for i, key in enumerate(keys):
            val = self.cache.get(key)
            if val is None:
                todo.append(i)
            else:
                vu[i] = val
        self.cache_hits += B - len(todo)
        if todo:
            idx = np.asarray(todo, dtype=np.int64)
            vals = self._solve(uniq[idx])
            for i, v in zip(todo, vals):
                self.cache[keys[i]] = float(v)
                vu[i] = float(v)
        return vu[inv]

    def value(self, row) -> float:
        return float(self.values(np.asarray(row, dtype=np.uint64).reshape(1, self.kc))[0])

    @property
    def empty(self) -> np.ndarray:
        return np.zeros(self.kc, dtype=np.uint64)

    @property
    def grand(self) -> np.ndarray:
        return np.bitwise_or.reduce(self.onehot, axis=0)


def exact_player_table(engine, n_players: int, kind: str = "agent", n: Optional[int] = None,
                       k: Optional[int] = None, chunk: int = 1 << 16) -> np.ndarray:


    if n_players > 24:
        raise ValueError("exact table limited to n_players <= 24")
    orc = CoalitionOracle(engine, n_players, kind, n, k, use_cache=False)
    total = 1 << n_players
    out = np.empty(total, dtype=np.float64)
    for start in range(0, total, chunk):
        bits = np.arange(start, min(total, start + chunk), dtype=np.uint64)
        rows = np.zeros((len(bits), orc.kc), dtype=np.uint64)
        for p in range(n_players):
            sel = ((bits >> np.uint64(p)) & np.uint64(1)).astype(bool)
            if sel.any():
                rows[sel] |= orc.onehot[p]
        out[start:start + len(bits)] = orc.values(rows)
    return out


@dataclass
class SampledShapleyResult:
    estimates: np.ndarray
    half_widths: np.ndarray
    n_samples: int
    n_perms: int
    n_solves: int
    n_coalitions_requested: int
    n_engine_calls: int
    cache_hits: int
    wall_time: float
    engine_time: float
    status: str
    certified: bool
    kind: str
    n: int
    k: Optional[int]
    n_players: int
    alpha: float
    eps: Optional[float]
    top_k: Optional[int]
    v_empty: float
    v_full: float
    antithetic: bool
    stratify: bool
    sample_var: np.ndarray = field(default=None)
    top_k_players: Optional[np.ndarray] = None
    seed: Optional[int] = None

    two_sided: bool = False
    bonferroni: bool = False
    sequential: bool = False
    alpha_player: float = None
    alpha_look: float = None
    alpha_eff: float = None
    n_looks_max: int = 1
    n_looks_used: int = 0
    look_m: list = field(default_factory=list)
    certificate: Optional[dict] = None
    max_perms: Optional[int] = None
    batch_perms: Optional[int] = None
    perms_per_unit: int = 1
    time_budget: Optional[float] = None
    budget_overshoot_s: float = 0.0
    n_batches: int = 0
    value_range: Optional[tuple] = None
    n_padded_rows: int = 0
    graph_captures: Optional[int] = None
    pad_rows: bool = False

    ci_method: str = "eb_looks"
    cs_center: Optional[np.ndarray] = None
    cs_lower: Optional[np.ndarray] = None
    cs_upper: Optional[np.ndarray] = None
    cs_anomaly: bool = False
    fp_error: Optional[dict] = None
    fp_term: float = 0.0
    eps_upsilon: float = 0.0
    batch_schedule: str = "fixed"
    batch_sizes: list = field(default_factory=list)
    withheld_certificate: Optional[dict] = None

    @property
    def lower(self) -> np.ndarray:
        return self.estimates - self.half_widths

    @property
    def upper(self) -> np.ndarray:
        return self.estimates + self.half_widths

    @property
    def anytime_valid(self) -> bool:


        if self.ci_method == "prpl_eb_cs":
            return True
        return bool(self.two_sided and self.bonferroni and self.sequential)

    def heatmap(self) -> np.ndarray:
        return heatmap(self.estimates, self.n, self.k, self.kind)

    def half_width_heatmap(self) -> np.ndarray:
        return heatmap(self.half_widths, self.n, self.k, self.kind)

    def ranking(self) -> np.ndarray:

        return np.argsort(-self.estimates, kind="stable")

    def alpha_allocation(self) -> dict:
        return {"alpha": self.alpha, "alpha_player": self.alpha_player, "alpha_look": self.alpha_look,
                "alpha_eff": self.alpha_eff, "two_sided": self.two_sided, "bonferroni": self.bonferroni,
                "sequential": self.sequential, "n_looks_max": self.n_looks_max, "n_looks_used": self.n_looks_used,
                "look_m": list(self.look_m), "n_players": self.n_players, "anytime_valid": self.anytime_valid,
                "ci_method": self.ci_method, "fp_term": self.fp_term, "eps_upsilon": self.eps_upsilon,
                "batch_schedule": self.batch_schedule, "cs_anomaly": self.cs_anomaly}

    def summary(self) -> dict:
        return {"status": self.status, "certified": self.certified, "anytime_valid": self.anytime_valid,
                "n_perms": self.n_perms, "n_samples": self.n_samples, "n_solves": self.n_solves,
                "n_coalitions_requested": self.n_coalitions_requested, "cache_hits": self.cache_hits,
                "wall_time": self.wall_time, "engine_time": self.engine_time,
                "max_half_width": float(np.max(self.half_widths)) if len(self.half_widths) else 0.0,
                "sum_estimates": float(self.estimates.sum()),
                "attributable": self.v_full - self.v_empty,
                "alpha": self.alpha, "alpha_player": self.alpha_player, "alpha_look": self.alpha_look,
                "n_looks_max": self.n_looks_max, "n_looks_used": self.n_looks_used,
                "budget_overshoot_s": self.budget_overshoot_s,
                "n_padded_rows": self.n_padded_rows, "graph_captures": self.graph_captures,
                "ci_method": self.ci_method, "fp_term": self.fp_term, "batch_schedule": self.batch_schedule,
                "n_batches": self.n_batches, "cs_anomaly": self.cs_anomaly,
                "certificate_kind": None if self.certificate is None else self.certificate["kind"]}


def heatmap(values: np.ndarray, n: int, k: Optional[int], kind: str = "step") -> np.ndarray:

    values = np.asarray(values, dtype=np.float64)
    if kind == "agent" or k is None:
        return values.reshape(n, 1)
    return values.reshape(n, k)


def topk_certified(est: np.ndarray, hw: np.ndarray, top_k: int):

    N = len(est)
    if top_k <= 0 or top_k >= N:
        return True, np.argsort(-est, kind="stable")[:top_k]
    if not np.all(np.isfinite(hw)):
        return False, None
    order = np.argsort(-est, kind="stable")
    top, rest = order[:top_k], order[top_k:]
    if np.min(est[top] - hw[top]) > np.max(est[rest] + hw[rest]):
        return True, top
    return False, None


def _perm_marginals(orc: CoalitionOracle, perms: np.ndarray):


    P, N = perms.shape
    rows = np.concatenate([orc.prefix_masks(perms[j]) for j in range(P)], axis=0)
    vals = orc.values(rows).reshape(P, N + 1)
    diffs = np.diff(vals, axis=1)
    marg = np.empty((P, N), dtype=np.float64)
    rows_idx = np.arange(P)[:, None]
    marg[rows_idx, perms] = diffs
    return marg, vals[:, 0], vals[:, -1]


CI_METHODS = ("eb_looks", "prpl_eb_cs")
BATCH_SCHEDULES = ("fixed", "geometric")
_U64 = 2.0 ** -53


def _psi_e(lam: np.ndarray) -> np.ndarray:

    return (-np.log1p(-lam) - lam) / 4.0


def _gamma_u(n: float, u: float = _U64) -> float:
    return n * u / (1.0 - n * u)


class PrPlEBSequence:


    def __init__(self, N: int, alpha_p: float, lam_max: float = 0.5):
        if not (0.0 < alpha_p < 1.0):
            raise ValueError("alpha_p must be in (0, 1)")
        if N < 1:
            raise ValueError("N must be >= 1")
        self.N = int(N)
        self.log_term = math.log(2.0 / alpha_p)
        self.lam_max = float(lam_max)
        self.m = 0
        self.S1 = np.zeros(N)
        self.Q = np.zeros(N)
        self.SlamY = np.zeros(N)
        self.Slam = np.zeros(N)
        self.Spen = np.zeros(N)
        self.L = np.full(N, -np.inf)
        self.U = np.full(N, np.inf)
        self.center = np.full(N, np.nan)
        self.width = np.full(N, np.inf)
        self.anomaly = False
        self.lam_last = None
        self.trace: list = []

    def next_lambda(self) -> np.ndarray:
        i = self.m + 1
        sig2 = (0.25 + self.Q) / (self.m + 1)
        lam = np.sqrt(2.0 * self.log_term / (sig2 * i * math.log1p(i)))
        lam = np.minimum(lam, self.lam_max)
        if not np.all(lam > 0.0):
            raise RuntimeError("lambda must be positive (theorem domain (0, 1))")
        return lam

    def update(self, Y: np.ndarray):
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim != 2 or Y.shape[1] != self.N:
            raise ValueError("Y must have shape (P, N)")
        if not np.all(np.isfinite(Y)) or Y.min() < 0.0 or Y.max() > 1.0:
            raise RuntimeError("CS samples must lie in [0, 1]")
        P = Y.shape[0]
        lam = self.next_lambda()
        self.lam_last = lam
        i_glob = (self.m + np.arange(1, P + 1))[:, None].astype(np.float64)
        cum = np.cumsum(Y, axis=0)
        excl = cum - Y
        mu_prev = (0.5 + self.S1[None, :] + excl) / i_glob
        v = 4.0 * (Y - mu_prev) ** 2
        mu_incl = (0.5 + self.S1[None, :] + cum) / (i_glob + 1.0)
        self.Q += ((Y - mu_incl) ** 2).sum(axis=0)
        sy = Y.sum(axis=0)
        self.S1 += sy
        self.SlamY += lam * sy
        self.Slam += lam * P
        self.Spen += _psi_e(lam) * v.sum(axis=0)
        self.m += P
        center = self.SlamY / self.Slam
        width = (self.log_term + self.Spen) / self.Slam
        slack = 4.0 * _gamma_u(self.m + 4) * (1.0 + width)
        L, U = center - width - slack, center + width + slack
        self.L = np.maximum(self.L, L)
        self.U = np.minimum(self.U, U)
        self.center, self.width = center, width
        if np.any(self.L > self.U):
            self.anomaly = True
        self.trace.append({"m": int(self.m), "P": int(P), "lambda": lam.copy(), "center": center.copy(),
                           "width": width.copy(), "slack": float(slack.max())})
        return self.L.copy(), self.U.copy()


def sample_shapley(engine, n_players: Optional[int] = None, kind: str = "agent",
                   n: Optional[int] = None, k: Optional[int] = None,
                   rng: Union[None, int, np.random.Generator] = None, seed: Optional[int] = None,
                   batch_perms: int = 32, eps: Optional[float] = None, alpha: float = 0.05,
                   top_k: Optional[int] = None, max_perms: int = 10_000,
                   time_budget: Optional[float] = None, antithetic: bool = True,
                   stratify: bool = False, use_cache: bool = True, bonferroni: Optional[bool] = None,
                   min_samples: int = 2, callback=None, two_sided: Optional[bool] = None,
                   sequential: Optional[bool] = None, value_range=None,
                   pad_rows: Optional[bool] = None, ci: str = "eb_looks", batch_schedule: str = "fixed",
                   batch_max: Optional[int] = None, fp_bound=None) -> SampledShapleyResult:


    if rng is None:
        rng = np.random.default_rng(DEFAULT_SEED if seed is None else seed)
    elif isinstance(rng, (int, np.integer)):
        seed = int(rng)
        rng = np.random.default_rng(seed)
    if batch_perms is None or int(batch_perms) < 1:
        raise ValueError("batch_perms must be >= 1")
    batch_perms = int(batch_perms)
    if max_perms is None or int(max_perms) < 1:
        raise ValueError("max_perms must be >= 1")
    max_perms = int(max_perms)
    if ci not in CI_METHODS:
        raise ValueError(f"ci must be one of {CI_METHODS}")
    if batch_schedule not in BATCH_SCHEDULES:
        raise ValueError(f"batch_schedule must be one of {BATCH_SCHEDULES}")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    use_cs = ci == "prpl_eb_cs"
    if eps is not None and (eps < 0.0 or (use_cs and not eps > 0.0)):
        raise ValueError("eps must be > 0 (legacy path: >= 0)")
    if batch_schedule == "geometric" and not use_cs:
        raise ValueError("batch_schedule='geometric' (data-dependent batch sizes) requires ci='prpl_eb_cs'")
    if use_cs and value_range is None:
        raise ValueError("ci='prpl_eb_cs' requires value_range ('monotone' or an explicit a.s. range)")
    if use_cs and any(f is False for f in (two_sided, bonferroni, sequential)):
        raise ValueError("two_sided/bonferroni/sequential cannot be disabled under ci='prpl_eb_cs'")
    comp = getattr(engine, "compiled", None)
    if n is None:
        n = getattr(comp, "n_agents", None)
    if k is None:
        k = getattr(comp, "horizon_k", None)
    if n_players is None:
        n_players = n if kind == "agent" else n * k
    orc = CoalitionOracle(engine, n_players, kind, n, k, use_cache=use_cache, pad_rows=pad_rows)
    N = orc.n_players
    if use_cs and top_k is not None and not (1 <= int(top_k) < N):
        raise ValueError("top_k must satisfy 1 <= top_k < n_players")


    certifying = (eps is not None) or (top_k is not None)
    two_sided = (certifying or use_cs) if two_sided is None else bool(two_sided)
    bonferroni = (certifying or use_cs) if bonferroni is None else bool(bonferroni)
    sequential = (certifying or use_cs) if sequential is None else bool(sequential)
    do_strat = stratify and N > 1
    perms_per_unit = (N if do_strat else 1) * (2 if antithetic else 1)
    if use_cs:
        n_looks_max = None
        alpha_player = alpha / N
        alpha_look = alpha_player
        alpha_eff = alpha_player
    else:
        n_looks_max = look_schedule_size(max_perms, batch_perms * perms_per_unit) if sequential else 1
        alpha_player = alpha / N if bonferroni else alpha
        alpha_look = alpha_player / n_looks_max if sequential else alpha_player
        alpha_eff = alpha_look
    if batch_max is None:
        batch_max = 16 * batch_perms
    batch_max = max(int(batch_max), batch_perms)


    fp = fp_bound
    fp_available = True
    if use_cs and fp is None:
        if comp is not None:
            from .fp_error import forward_error_bound
            fp = forward_error_bound(comp, orc.k, dtype=getattr(engine, "dtype", "fp64"))
        else:
            fp_available = False
    eps_ups = fp.eps_upsilon if (use_cs and fp is not None) else 0.0
    eps_x = fp.eps_unit(perms_per_unit) if (use_cs and fp is not None) else 0.0
    fp_dict = None if fp is None else dict(fp.as_dict(), eps_unit=eps_x, fp_available=fp_available)

    def _result(**kw):
        base = dict(kind=kind, n=orc.n, k=orc.k, n_players=N, alpha=alpha, eps=eps, top_k=top_k,
                    antithetic=antithetic, stratify=stratify, seed=seed, two_sided=two_sided,
                    bonferroni=bonferroni, sequential=sequential, alpha_player=alpha_player,
                    alpha_look=alpha_look, alpha_eff=alpha_eff, n_looks_max=n_looks_max,
                    max_perms=max_perms, batch_perms=batch_perms, perms_per_unit=int(perms_per_unit),
                    time_budget=time_budget, n_padded_rows=orc.n_padded_rows, graph_captures=orc.graph_captures,
                    pad_rows=orc.pad_rows, ci_method=ci, fp_error=fp_dict, fp_term=float(eps_x),
                    eps_upsilon=float(eps_ups), batch_schedule=batch_schedule,
                    n_solves=orc.n_solves, n_coalitions_requested=orc.n_requested,
                    n_engine_calls=orc.n_engine_calls, cache_hits=orc.cache_hits, engine_time=orc.engine_time)
        base.update(kw)
        return SampledShapleyResult(**base)


    rng_lo, rng_hi = -1.0, 1.0
    if value_range == "monotone":
        _t0_range = time.perf_counter()
        vv = orc.values(np.stack([orc.empty, orc.grand], axis=0))
        v_empty0, v_full0 = float(vv[0]), float(vv[1])
        ups = v_full0 - v_empty0
        if use_cs:
            if ups < -eps_ups:
                raise RuntimeError(f"value_range='monotone' but v(N) - v(empty) = {ups} < -eps_upsilon")
            if ups <= 2.0 * eps_ups:

                c = max(ups, 0.0) / 2.0
                est0 = np.full(N, c)
                hw0 = np.full(N, c + eps_ups)
                fired = eps is not None and bool(np.all(hw0 <= eps))
                cert = None
                if fired:
                    cert = {"kind": "eps", "m": 0, "n_perms": 0, "n_solves": orc.n_solves,
                            "n_engine_calls": orc.n_engine_calls, "wall_s": time.perf_counter() - _t0_range,
                            "engine_s": float(orc.engine_time), "look_index": 0, "terminal_look": True,
                            "terminal_reason": None, "estimates": est0.copy(), "half_widths": hw0.copy(),
                            "top_set": None, "eps": eps, "top_k": top_k, "alpha": alpha,
                            "alpha_player": alpha_player, "alpha_look": alpha_look, "alpha_eff": alpha_eff,
                            "n_looks_max": None, "two_sided": True, "bonferroni": True, "sequential": True,
                            "anytime_valid": True, "max_half_width": float(hw0.max()), "ci_method": ci,
                            "fp_term": float(eps_x), "eps_upsilon": float(eps_ups), "fp_error": fp_dict,
                            "note": "null path: v(N) - v(empty) <= 2 eps_upsilon; no sampling"}
                return _result(estimates=est0, half_widths=hw0, n_samples=0, n_perms=0,
                               wall_time=time.perf_counter() - _t0_range, status="eps" if fired else "null",
                               certified=fired, v_empty=v_empty0, v_full=v_full0, sample_var=np.zeros(N),
                               top_k_players=None, n_looks_used=0, look_m=[], certificate=cert,
                               budget_overshoot_s=0.0, n_batches=0, value_range=(0.0, max(ups, 0.0)),
                               cs_center=est0.copy(), cs_lower=np.zeros(N), cs_upper=np.full(N, max(ups, 0.0) + eps_ups),
                               batch_sizes=[])
            rng_lo, rng_hi = fp.statistical_range(ups, perms_per_unit) if fp is not None else (0.0, ups)
        else:
            if ups < -1e-12:
                raise RuntimeError(f"value_range='monotone' but v(N) - v(empty) = {ups} < 0")
            _t_endpoints = time.perf_counter()
            if ups <= 0.0:
                zero = np.zeros(N)
                cert = {"kind": "eps", "m": 0, "n_perms": 0, "n_solves": orc.n_solves,
                        "n_engine_calls": orc.n_engine_calls, "wall_s": 0.0, "engine_s": float(orc.engine_time),
                        "look_index": 0, "terminal_look": True, "terminal_reason": None,
                        "estimates": zero.copy(), "half_widths": zero.copy(), "top_set": None,
                        "eps": eps, "top_k": top_k, "alpha": alpha, "alpha_player": alpha_player,
                        "alpha_look": alpha_look, "alpha_eff": alpha_eff, "n_looks_max": int(n_looks_max),
                        "two_sided": two_sided, "bonferroni": bonferroni, "sequential": sequential,
                        "anytime_valid": True, "max_half_width": 0.0, "ci_method": ci,
                        "note": "null game: v(N) == v(empty), all marginals are exactly 0 by monotonicity"}
                return _result(estimates=zero, half_widths=zero.copy(), n_samples=0, n_perms=0,
                               wall_time=time.perf_counter() - _t0_range, status="eps", certified=True,
                               v_empty=v_empty0, v_full=v_full0, sample_var=zero.copy(), top_k_players=None,
                               n_looks_used=0, look_m=[], certificate=cert, budget_overshoot_s=0.0,
                               n_batches=0, value_range=(0.0, 0.0), batch_sizes=[])

            _tol0 = 1e-9 * max(1.0, ups)
            rng_lo, rng_hi = -_tol0, ups + _tol0
    elif value_range is not None:
        rng_lo, rng_hi = float(value_range[0]), float(value_range[1])
        if not rng_hi > rng_lo:
            raise ValueError("value_range must satisfy hi > lo")
    _range_tol = 0.0 if (value_range is None or use_cs) else 1e-12 * max(1.0, abs(rng_hi), abs(rng_lo))
    rng_span = rng_hi - rng_lo

    sum_x = np.zeros(N)
    sum_x2 = np.zeros(N)
    m = 0
    n_perms = 0
    n_batches = 0
    v_empty = v_full = math.nan
    status = None
    top_set = None
    certificate = None
    withheld = None
    n_looks_used = 0
    look_m: list = []
    batch_sizes: list = []
    m0 = None
    next_look_m = None
    est = np.zeros(N)
    var = np.full(N, np.nan)
    hw = np.full(N, np.inf)
    cs_L = np.full(N, -np.inf)
    cs_U = np.full(N, np.inf)
    cs = PrPlEBSequence(N, alpha_player) if use_cs else None
    t_start = time.perf_counter()
    j_batch = 0

    def snapshot(fired, terminal, terminal_reason):
        return {
            "kind": fired, "m": int(m), "n_perms": int(n_perms), "n_solves": int(orc.n_solves),
            "n_engine_calls": int(orc.n_engine_calls), "wall_s": time.perf_counter() - t_start,
            "engine_s": float(orc.engine_time), "look_index": int(n_looks_used),
            "terminal_look": bool(terminal), "terminal_reason": terminal_reason if terminal else None,
            "estimates": est.copy(), "half_widths": hw.copy(),
            "top_set": None if top_set is None else np.array(top_set, dtype=np.int64).copy(),
            "eps": eps, "top_k": top_k, "alpha": alpha, "alpha_player": alpha_player,
            "alpha_look": alpha_look, "alpha_eff": alpha_eff, "n_looks_max": n_looks_max,
            "two_sided": two_sided, "bonferroni": bonferroni, "sequential": sequential,
            "anytime_valid": bool(use_cs or (two_sided and bonferroni and sequential)),
            "max_half_width": float(np.max(hw)) if N else 0.0, "ci_method": ci,
            "cs_lower": cs_L.copy() if use_cs else None, "cs_upper": cs_U.copy() if use_cs else None,
            "fp_term": float(eps_x), "eps_upsilon": float(eps_ups), "fp_error": fp_dict,
            "batch_sizes": list(batch_sizes),
        }

    def rule_fires():

        nonlocal top_set
        if eps is not None and np.all(hw <= eps):
            return "eps"
        if top_k is not None:
            if use_cs:
                lo_x, hi_x = cs_L - eps_x, cs_U + eps_x
                order = np.argsort(-est, kind="stable")
                top, rest = order[:int(top_k)], order[int(top_k):]
                if np.all(np.isfinite(lo_x[top])) and np.min(lo_x[top]) > np.max(hi_x[rest]):
                    top_set = top
                    return "top_k"
            else:
                ok, ts = topk_certified(est, hw, int(top_k))
                if ok:
                    top_set = ts
                    return "top_k"
        return None

    def evaluate_rules(terminal: bool, terminal_reason: Optional[str] = None):

        nonlocal n_looks_used, certificate, withheld
        n_looks_used += 1
        look_m.append(int(m))
        fired = rule_fires()
        if fired is None:
            return None
        if (not use_cs) and terminal and terminal_reason == "time":

            withheld = dict(snapshot(fired, terminal, terminal_reason), withheld_reason="time_terminal")
            return None
        certificate = snapshot(fired, terminal, terminal_reason)
        return fired

    while True:

        if batch_schedule == "geometric":
            P_sched = min(batch_perms * (2 ** j_batch), batch_max)
            if eps is not None and m >= 2 and np.all(np.isfinite(hw)):
                hw_max = float(np.max(hw))
                if hw_max > eps:
                    m_pred = m * (hw_max / eps) ** 2
                    P_sched = min(P_sched, max(batch_perms, int(math.ceil(1.25 * (m_pred - m)))))
                else:
                    P_sched = batch_perms
        else:
            P_sched = batch_perms
        P = min(P_sched, -(-(max_perms - n_perms) // perms_per_unit))
        clock_truncated = False
        if m > 0 and time_budget is not None:
            elapsed = time.perf_counter() - t_start
            remaining = time_budget - elapsed
            sec_per_unit = elapsed / m
            fit = int(remaining // sec_per_unit) if (remaining > 0 and sec_per_unit > 0) else \
                (P if remaining > 0 else 0)
            if remaining <= 0 or fit < 1:
                if certifying and (not look_m or look_m[-1] != m):
                    status = evaluate_rules(terminal=True, terminal_reason="time")
                if status is None:
                    status = "time"
                break
            if fit < P:
                P, clock_truncated = fit, True

        base = np.stack([rng.permutation(N) for _ in range(P)], axis=0)
        group = [base]
        if do_strat:
            group = [np.roll(base, -c, axis=1) for c in range(N)]
        perms = np.concatenate(group, axis=0)
        if antithetic:
            perms = np.concatenate([perms, perms[:, ::-1]], axis=0)
        marg, ve, vf = _perm_marginals(orc, perms)
        v_empty, v_full = float(ve[0]), float(vf[0])
        units = marg.reshape(-1, P, N).mean(axis=0)
        if value_range is not None:
            for arr, what in ((marg, "marginal"), (units, "unit sample")):
                bad_lo = float(arr.min()); bad_hi = float(arr.max())
                if bad_lo < rng_lo - _range_tol or bad_hi > rng_hi + _range_tol:
                    raise RuntimeError(
                        f"{what} outside the declared value range [{rng_lo}, {rng_hi}]: observed "
                        f"[{bad_lo}, {bad_hi}] — the monotonicity/range assumption is violated")
        n_perms += perms.shape[0]
        n_batches += 1
        j_batch += 1
        batch_sizes.append(int(P))
        sum_x += units.sum(axis=0)
        sum_x2 += (units * units).sum(axis=0)
        m += P
        est = sum_x / m
        if m >= 2:
            var = np.maximum(sum_x2 - m * est * est, 0.0) / (m - 1)
        else:
            var = np.full(N, np.nan)
        if use_cs:
            Y = np.clip((units - rng_lo) / rng_span, 0.0, 1.0)
            L, U = cs.update(Y)
            cs_L = rng_lo + rng_span * L - 4.0 * _U64 * abs(rng_span)
            cs_U = rng_lo + rng_span * U + 4.0 * _U64 * abs(rng_span)
            hw = np.maximum(est - cs_L, cs_U - est) + eps_x
            if m < max(1, min_samples):
                hw = np.full(N, np.inf)
        else:
            hw = empirical_bernstein_half_width(var, m, alpha_eff, lo=rng_lo, hi=rng_hi, two_sided=two_sided) \
                if m >= max(2, min_samples) else np.full(N, np.inf)
        if callback is not None:
            callback(m, est, hw)

        terminal = n_perms >= max_perms
        if use_cs:
            if certifying:
                status = evaluate_rules(terminal=terminal, terminal_reason="max_perms" if terminal else None)
        else:
            if m0 is None:
                m0, next_look_m = m, m
            exact_look = (m == next_look_m) and not clock_truncated
            is_look = (not sequential) or exact_look or terminal
            if sequential:
                while next_look_m <= m:
                    next_look_m *= 2
            if certifying and is_look:
                status = evaluate_rules(terminal=terminal, terminal_reason="max_perms" if terminal else None)
        if status is None and terminal:
            status = "max_perms"
        if status is not None:
            break

    wall = time.perf_counter() - t_start
    overshoot = max(0.0, wall - time_budget) if time_budget is not None else 0.0
    return _result(
        estimates=est, half_widths=hw, n_samples=m, n_perms=n_perms, wall_time=wall, status=status,
        certified=status in ("eps", "top_k"), v_empty=v_empty, v_full=v_full, sample_var=var,
        top_k_players=top_set, n_looks_used=int(n_looks_used), look_m=look_m, certificate=certificate,
        budget_overshoot_s=overshoot, n_batches=n_batches,
        value_range=(rng_lo, rng_hi) if value_range is not None else None,
        cs_center=(None if cs is None else rng_lo + rng_span * cs.center), cs_lower=(cs_L if use_cs else None),
        cs_upper=(cs_U if use_cs else None), cs_anomaly=bool(cs.anomaly) if cs is not None else False,
        withheld_certificate=withheld, batch_sizes=batch_sizes)


def kendall_tau(a, b) -> float:

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    N = len(a)
    if N < 2:
        return 1.0
    iu = np.triu_indices(N, 1)
    sa = np.sign(a[:, None] - a[None, :])[iu]
    sb = np.sign(b[:, None] - b[None, :])[iu]
    return float(np.sum(sa * sb) / (N * (N - 1) / 2))
