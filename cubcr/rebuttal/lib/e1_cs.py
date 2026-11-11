from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, List, Optional, Sequence

import numpy as np

from cubcr.shapley_sampled import PrPlEBSequence, CoalitionOracle, _perm_marginals, _U64


class TracingPrPlEB(PrPlEBSequence):


    def __init__(self, N: int, alpha_p: float, lam_max: float = 0.5):
        super().__init__(N, alpha_p, lam_max)
        self.lams: List[np.ndarray] = []
        self.xhats: List[np.ndarray] = []
        self.xhats_excl: List[np.ndarray] = []
        self.Ps: List[int] = []

    def update(self, Y: np.ndarray):
        Y = np.asarray(Y, dtype=np.float64)
        P = Y.shape[0]

        i_glob = (self.m + np.arange(1, P + 1))[:, None].astype(np.float64)
        cum = np.cumsum(Y, axis=0)
        excl = cum - Y
        mu_prev = (0.5 + self.S1[None, :] + excl) / i_glob

        excl_strict = np.vstack([np.zeros((1, Y.shape[1])), cum[:-1]])
        mu_prev_strict = (0.5 + self.S1[None, :] + excl_strict) / i_glob
        L, U = super().update(Y)
        self.lams.append(np.array(self.lam_last, dtype=np.float64, copy=True))
        self.xhats.append(np.array(mu_prev, dtype=np.float64, copy=True))
        self.xhats_excl.append(np.array(mu_prev_strict, dtype=np.float64, copy=True))
        self.Ps.append(int(P))
        return L, U


@dataclass
class Look:
    m: int
    est: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    hw: np.ndarray
    stop: bool
    cs_L: Optional[np.ndarray] = None
    cs_U: Optional[np.ndarray] = None


def fp64_replay(units: np.ndarray, batch_sizes: Sequence[int], rng_lo: float, rng_hi: float, eps_x: float,
                alpha: float, eps: Optional[float] = None, min_samples: int = 2):

    units = np.asarray(units, dtype=np.float64)
    N = units.shape[1]
    rng_span = rng_hi - rng_lo
    cs = TracingPrPlEB(N, alpha / N)
    sum_x = np.zeros(N)
    m = 0
    looks: List[Look] = []
    for P in batch_sizes:
        u = units[m:m + P]
        if u.shape[0] != P:
            raise ValueError("not enough units for the batch schedule")
        sum_x += u.sum(axis=0)
        m += P
        est = sum_x / m

        Y = np.clip((u - rng_lo) / rng_span, 0.0, 1.0)
        L, U = cs.update(Y)
        cs_L = rng_lo + rng_span * L - 4.0 * _U64 * abs(rng_span)
        cs_U = rng_lo + rng_span * U + 4.0 * _U64 * abs(rng_span)
        hw = np.maximum(est - cs_L, cs_U - est) + eps_x
        if m < max(1, min_samples):
            hw = np.full(N, np.inf)

        stop = bool(eps is not None and np.all(hw <= eps))
        looks.append(Look(m=m, est=est.copy(), lower=cs_L - eps_x, upper=cs_U + eps_x, hw=hw, stop=stop,
                          cs_L=cs_L, cs_U=cs_U))
    return looks, cs


def _iv():
    from mpmath import iv
    return iv


def round_down(x) -> float:

    import mpmath
    f = float(x)
    if mpmath.mpf(f) > x:
        f = float(np.nextafter(f, -np.inf))
    return f


def round_up(x) -> float:
    import mpmath
    f = float(x)
    if mpmath.mpf(f) < x:
        f = float(np.nextafter(f, np.inf))
    return f


def fp_up(x: float) -> float:
    return float(np.nextafter(float(x), np.inf))


@dataclass
class ConservativeParams:
    a: float
    b: float
    beta: float
    eps: Optional[float]
    alpha: str = "0.05"
    dps: int = 40


def make_params(upsilon_fp: float, beta_decimal: str | float, eps_div: Optional[int], alpha: str = "0.05",
                dps: int = 40) -> ConservativeParams:


    iv = _iv()
    iv.dps = dps
    beta_iv = iv.mpf(str(beta_decimal)) if isinstance(beta_decimal, str) else iv.mpf(float(beta_decimal))
    beta_fp = round_up(beta_iv.b)
    ups = iv.mpf(float(upsilon_fp))
    a = round_down((-iv.mpf(beta_fp)).a)
    U = ups + 2 * iv.mpf(beta_fp)
    b = round_up((U + iv.mpf(beta_fp)).b)
    eps = None
    if eps_div is not None:
        e = (ups - 2 * iv.mpf(beta_fp)) / eps_div
        eps = round_down(e.a)
    return ConservativeParams(a=a, b=b, beta=beta_fp, eps=eps, alpha=alpha, dps=dps)


class ConservativeCS:


    def __init__(self, N: int, prm: ConservativeParams, a_paper: float, b_paper: float):
        iv = _iv()
        iv.dps = prm.dps
        self.iv = iv
        self.N = N
        self.prm = prm
        self.a_paper, self.b_paper = float(a_paper), float(b_paper)
        self.a = iv.mpf(prm.a)
        self.b = iv.mpf(prm.b)
        self.span = self.b - self.a
        self.inv_span = 1 / self.span

        self.log_term = iv.log(iv.mpf(2 * N) / iv.mpf(prm.alpha))
        self.SlamX = [iv.mpf(0) for _ in range(N)]
        self.Slam = [iv.mpf(0) for _ in range(N)]
        self.Spen = [iv.mpf(0) for _ in range(N)]
        self.Lrun = [None] * N
        self.Urun = [None] * N
        self.m = 0
        self.sum_x = np.zeros(N)

    def map_xhat(self, xhat_paper: np.ndarray) -> np.ndarray:

        z = self.a_paper + (self.b_paper - self.a_paper) * xhat_paper
        return np.clip((z - self.prm.a) / (self.prm.b - self.prm.a), 0.0, 1.0)

    def update(self, units_batch: np.ndarray, lam: np.ndarray, xhat_paper: np.ndarray):

        iv = self.iv
        u = np.asarray(units_batch, dtype=np.float64)
        P, N = u.shape
        if np.any(u < self.prm.a) or np.any(u > self.prm.b):
            raise RuntimeError("a unit lies outside the declared range [a, b]")
        if not (np.all(lam > 0) and np.all(lam < 1)):
            raise RuntimeError("lambda outside (0, 1)")
        xh = self.map_xhat(np.asarray(xhat_paper, dtype=np.float64))
        a, inv = self.a, self.inv_span
        uu = u.tolist()
        xx = xh.tolist()
        for p in range(N):
            sx = iv.mpf(0)
            sp = iv.mpf(0)
            for r in range(P):
                X = (iv.mpf(uu[r][p]) - a) * inv
                d = X - xx[r][p]
                sx += X
                sp += d * d
            lp = iv.mpf(float(lam[p]))
            h = -iv.log(1 - lp) - lp
            self.SlamX[p] += lp * sx
            self.Slam[p] += lp * P
            self.Spen[p] += h * sp
        self.sum_x += u.sum(axis=0)
        self.m += P

    def look(self):

        iv = self.iv
        import mpmath
        N = self.N
        est = self.sum_x / self.m
        lower = np.empty(N)
        upper = np.empty(N)
        H = np.empty(N)
        beta = iv.mpf(self.prm.beta)
        for p in range(N):
            c = self.SlamX[p] / self.Slam[p]
            q = (self.log_term + self.Spen[p]) / self.Slam[p]
            jlo = (self.a + self.span * (c - q)).a
            jhi = (self.a + self.span * (c + q)).b
            self.Lrun[p] = jlo if self.Lrun[p] is None else max(self.Lrun[p], jlo)
            self.Urun[p] = jhi if self.Urun[p] is None else min(self.Urun[p], jhi)
            lo = (iv.mpf(self.Lrun[p]) - beta).a
            hi = (iv.mpf(self.Urun[p]) + beta).b
            lower[p] = round_down(lo)
            upper[p] = round_up(hi)
            e = iv.mpf(float(est[p]))
            d1 = (e - iv.mpf(lower[p])).b
            d2 = (iv.mpf(upper[p]) - e).b
            H[p] = round_up(max(d1, d2))
        return est, lower, upper, H


def conservative_replay(units_provider: Callable[[int], np.ndarray], batch_sizes: Sequence[int], N: int,
                        prm: ConservativeParams, paper_lo: float, paper_hi: float, alpha_fp: float,
                        batch_perms: int, batch_max: int, max_units: int = 2_000_000, xhat_mode: str = "strict"):


    if xhat_mode not in ("strict", "paper"):
        raise ValueError("xhat_mode must be 'strict' or 'paper'")
    units = units_provider(int(sum(batch_sizes)))
    fpcs = TracingPrPlEB(N, alpha_fp / N)
    xstat = {"n_differ": 0, "n_total": 0, "max_abs": 0.0}
    paper_span = paper_hi - paper_lo
    ccs = ConservativeCS(N, prm, paper_lo, paper_hi)
    looks: List[Look] = []
    sizes: List[int] = []
    m = 0
    stopped = False
    stop_index = None

    def one_batch(P):
        nonlocal m, units
        if units.shape[0] < m + P:
            units = units_provider(m + P)
        u = units[m:m + P]
        Y = np.clip((u - paper_lo) / paper_span, 0.0, 1.0)
        fpcs.update(Y)
        xp, xs = fpcs.xhats[-1], fpcs.xhats_excl[-1]
        dif = np.abs(xp - xs)
        xstat["n_differ"] += int(np.count_nonzero(dif))
        xstat["n_total"] += int(dif.size)
        xstat["max_abs"] = max(xstat["max_abs"], float(dif.max()) if dif.size else 0.0)
        ccs.update(u, fpcs.lams[-1], xs if xhat_mode == "strict" else xp)
        m += P
        sizes.append(int(P))
        est, lo, hi, H = ccs.look()
        stop = bool(prm.eps is not None and np.all(H <= prm.eps))
        looks.append(Look(m=m, est=est, lower=lo, upper=hi, hw=H, stop=stop))
        return stop

    for P in batch_sizes:
        fired = one_batch(P)
        if fired and stop_index is None:
            stopped = True
            stop_index = len(looks) - 1
    continued = False
    if not stopped and prm.eps is not None:
        continued = True
        while m < max_units:
            j = len(sizes)
            P_sched = min(batch_perms * (2 ** j), batch_max)
            hw_max = float(np.max(looks[-1].hw))
            if hw_max > prm.eps:
                m_pred = m * (hw_max / prm.eps) ** 2
                P_sched = min(P_sched, max(batch_perms, int(math.ceil(1.25 * (m_pred - m)))))
            else:
                P_sched = batch_perms
            if one_batch(P_sched):
                stopped = True
                stop_index = len(looks) - 1
                break
    if prm.eps is None:
        stop_index = len(looks) - 1
    return {"looks": looks, "stop_index": stop_index,
            "m_stop": (looks[stop_index].m if stop_index is not None else None),
            "stopped": stopped, "n_batches": (stop_index + 1 if stop_index is not None else len(sizes)),
            "batch_sizes": sizes[:(stop_index + 1) if stop_index is not None else len(sizes)],
            "continued": continued, "units_used": m, "xhat_mode": xhat_mode,
            "xhat_n_differ": xstat["n_differ"], "xhat_n_total": xstat["n_total"], "xhat_max_abs_diff": xstat["max_abs"]}


def units_from_engine(engine, N: int, seed: int, m_total: int, kind: str = "agent", n: Optional[int] = None,
                      k: Optional[int] = None, chunk: int = 256) -> np.ndarray:


    rng = np.random.default_rng(seed)
    orc = CoalitionOracle(engine, N, kind, n=n if n is not None else (N if kind == "agent" else None), k=k,
                          use_cache=False)
    out = []
    done = 0
    while done < m_total:
        P = min(chunk, m_total - done)
        base = np.stack([rng.permutation(N) for _ in range(P)], axis=0)
        perms = np.concatenate([base, base[:, ::-1]], axis=0)
        marg, _, _ = _perm_marginals(orc, perms)
        out.append(marg.reshape(-1, P, N).mean(axis=0))
        done += P
    return np.concatenate(out, axis=0) if out else np.zeros((0, N))


def units_from_engine_tail(engine, N: int, seed: int, m_skip: int, m_tail: int, kind: str = "agent",
                           n: Optional[int] = None, k: Optional[int] = None, chunk: int = 256) -> np.ndarray:


    rng = np.random.default_rng(seed)
    for _ in range(int(m_skip)):
        rng.permutation(N)
    orc = CoalitionOracle(engine, N, kind, n=n if n is not None else (N if kind == "agent" else None), k=k,
                          use_cache=False)
    out = []
    done = 0
    while done < m_tail:
        P = min(chunk, m_tail - done)
        base = np.stack([rng.permutation(N) for _ in range(P)], axis=0)
        perms = np.concatenate([base, base[:, ::-1]], axis=0)
        marg, _, _ = _perm_marginals(orc, perms)
        out.append(marg.reshape(-1, P, N).mean(axis=0))
        done += P
    return np.concatenate(out, axis=0) if out else np.zeros((0, N))


class StepRecordingEngine:


    def __init__(self, engine):
        self.engine = engine
        self.compiled = getattr(engine, "compiled", None)
        self.dtype = getattr(engine, "dtype", "fp64")
        self.calls = []

    @property
    def stats(self):
        return getattr(self.engine, "stats", None)

    def solve(self, masks, k=None):
        v = self.engine.solve(masks, k)
        arr = np.asarray(masks, dtype=np.uint64)
        rows = arr.T.copy() if arr.ndim == 2 else arr.reshape(-1, 1).copy()
        self.calls.append((rows, np.array(v, dtype=np.float64, copy=True)))
        return v


class StepDictEngine:

    compiled = None
    dtype = "fp64"

    def __init__(self, calls):
        self.d = {}
        for rows, vals in calls:
            for r, v in zip(rows, vals.tolist()):
                self.d[np.ascontiguousarray(r, dtype=np.uint64).tobytes()] = float(v)

    def solve(self, masks, k=None):
        arr = np.asarray(masks, dtype=np.uint64)
        rows = arr.T if arr.ndim == 2 else arr.reshape(-1, 1)
        return np.array([self.d[np.ascontiguousarray(r).tobytes()] for r in rows], dtype=np.float64)


def replay_units_step(result, calls, n: int, k: int, seed: int, verify: bool = True):

    N = n * k
    eng = StepDictEngine(calls)
    orc = CoalitionOracle(eng, N, "step", n=n, k=k, use_cache=False)
    rng = np.random.default_rng(seed)
    units_l, bounds = [], [0]
    sum_x = np.zeros(N)
    m = 0
    for P in result.batch_sizes:
        base = np.stack([rng.permutation(N) for _ in range(P)], axis=0)
        perms = np.concatenate([base, base[:, ::-1]], axis=0)
        marg, _, _ = _perm_marginals(orc, perms)
        units = marg.reshape(-1, P, N).mean(axis=0)
        units_l.append(units)
        sum_x += units.sum(axis=0)
        m += P
        bounds.append(m)
    units = np.concatenate(units_l, axis=0)
    if verify:
        est = sum_x / m
        if m != result.n_samples or not np.array_equal(est, result.estimates):
            raise RuntimeError("step replay does not reproduce the sampler's estimates bitwise")
    return {"units": units, "batch_bounds": bounds}


def contained(phi_ref, lower, upper, tol: float = 0.0):
    phi_ref = np.asarray(phi_ref, dtype=np.float64)
    ok = (lower - tol <= phi_ref) & (phi_ref <= upper + tol)
    margin = np.minimum(phi_ref - lower, upper - phi_ref)
    return ok, margin
