from __future__ import annotations

import time
import warnings
from typing import Optional

import numpy as np

import rb

ARMS = ("paired_d", "paired_r", "kadd2", "leverage", "stratsvarm")
LABELS = {"paired_d": "Paired perm. (cuBCR), distinct-matched", "paired_r": "Paired perm. (cuBCR), requested-matched",
          "kadd2": "2-ADD-SHAP (shapiq)", "leverage": "Leverage SHAP", "stratsvarm": "Stratified SVARM"}
SHORT = {"paired_d": "Paired (dist.)", "paired_r": "Paired (req.)", "kadd2": "2-ADD-SHAP", "leverage": "Leverage SHAP",
         "stratsvarm": "Strat. SVARM"}
E4_NAME = {"paired_d": "cubcr_paired_perm", "kadd2": "shapiq_kADDSHAP", "leverage": "LeverageSHAP",
           "stratsvarm": "StratifiedSVARM"}


class TimedEngine:


    def __init__(self, engine, keep: bool = True):
        self.engine = engine
        self.compiled = getattr(engine, "compiled", None)
        self.dtype = getattr(engine, "dtype", "fp64")
        self.keep = keep
        self.calls: list = []
        self.masks: list = []
        self.values: list = []

    @property
    def stats(self):
        return getattr(self.engine, "stats", None)

    def solve(self, masks, k=None):
        t0 = time.perf_counter()
        v = self.engine.solve(masks, k)
        t1 = time.perf_counter()
        st = self.stats
        rec = {"t0": t0, "t1": t1, "wall_s": t1 - t0, "B": int(np.asarray(masks).size)}
        if isinstance(st, dict):
            for key in ("total_ms", "kernel_ms", "graph_captures", "waves", "batch"):
                rec[key] = st.get(key)
        self.calls.append(rec)
        if self.keep:
            self.masks.append(np.array(masks, dtype=np.uint64, copy=True).reshape(-1))
            self.values.append(np.array(v, dtype=np.float64, copy=True).reshape(-1))
        return v

    def all_masks(self) -> np.ndarray:
        return np.concatenate(self.masks) if self.masks else np.zeros(0, dtype=np.uint64)

    def all_values(self) -> np.ndarray:
        return np.concatenate(self.values) if self.values else np.zeros(0, dtype=np.float64)


def resolve_union(te: TimedEngine, k: int, n_rep: int = 2) -> list:


    allm = te.all_masks()
    allv = te.all_values()
    out = []
    eng = te.engine
    for _ in range(n_rep):
        t0 = time.perf_counter()
        v = eng.solve(allm, k)
        t1 = time.perf_counter()
        st = getattr(eng, "stats", None) or {}
        out.append({"B": int(allm.size), "wall_s": t1 - t0, "total_ms": st.get("total_ms"),
                    "kernel_ms": st.get("kernel_ms"), "graph_captures": st.get("graph_captures"),
                    "waves": st.get("waves"), "same_values": bool(np.array_equal(np.asarray(v, dtype=np.float64), allv))})
    return out


class ValueOracle:


    def __init__(self, engine, n: int, k: int, prefetch_endpoints: bool = True):
        self.engine = engine
        self.n = int(n)
        self.k = int(k)
        self.full = (1 << self.n) - 1
        self.prefetch = prefetch_endpoints
        self.cache_m = np.zeros(0, dtype=np.int64)
        self.cache_v = np.zeros(0, dtype=np.float64)
        self.n_requested = 0
        self.n_requests = 0
        self.n_engine = 0
        self.n_calls = 0
        self.t_values = 0.0
        self.t_engine = 0.0

    def _lookup(self, uniq):
        if self.cache_m.size == 0:
            return np.zeros(uniq.size, dtype=bool), np.zeros(uniq.size, dtype=np.int64)
        idx = np.searchsorted(self.cache_m, uniq)
        idx_c = np.minimum(idx, self.cache_m.size - 1)
        return self.cache_m[idx_c] == uniq, idx_c

    def values(self, masks) -> np.ndarray:
        t0 = time.perf_counter()
        m = np.asarray(masks, dtype=np.int64).reshape(-1)
        self.n_requested += int(m.size)
        self.n_requests += 1
        uniq, inv = np.unique(m, return_inverse=True)
        hit, idx = self._lookup(uniq)
        todo = uniq[~hit]
        if self.prefetch and self.n_calls == 0:
            ends = np.array([0, self.full], dtype=np.int64)
            ehit, _ = self._lookup(ends)
            todo = np.union1d(todo, ends[~ehit])
        vu = np.empty(uniq.size, dtype=np.float64)
        if np.any(hit):
            vu[hit] = self.cache_v[idx[hit]]
        if todo.size:
            te = time.perf_counter()
            vt = np.asarray(self.engine.solve(todo.astype(np.uint64), self.k), dtype=np.float64).reshape(-1)
            self.t_engine += time.perf_counter() - te
            self.n_calls += 1
            self.n_engine += int(todo.size)
            cm = np.concatenate([self.cache_m, todo]); cv = np.concatenate([self.cache_v, vt])
            o = np.argsort(cm, kind="stable")
            self.cache_m, self.cache_v = cm[o], cv[o]
            miss = ~hit
            if np.any(miss):
                vu[miss] = vt[np.searchsorted(todo, uniq[miss])]
        self.t_values += time.perf_counter() - t0
        return vu[inv]

    def cached(self, mask: int) -> float:
        hit, idx = self._lookup(np.array([int(mask)], dtype=np.int64))
        if not hit[0]:
            raise KeyError(mask)
        return float(self.cache_v[idx[0]])

    @property
    def v_empty(self) -> float:
        return self.cached(0)

    @property
    def v_full(self) -> float:
        return self.cached(self.full)


def _engine_game_class():
    import shapiq

    class EngineGame(shapiq.Game):


        def __init__(self, oracle: ValueOracle, d: int):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                super().__init__(n_players=d, normalize=True, normalization_value=0.0)
            self.oracle = oracle
            self.d = d
            self.w = (np.int64(1) << np.arange(d, dtype=np.int64))
            self.t_enter: list = []
            self.t_exit: list = []

        def value_function(self, coalitions):
            self.t_enter.append(time.perf_counter())
            C = np.asarray(coalitions).reshape(-1, self.d).astype(bool)
            m = (C.astype(np.int64) * self.w[None, :]).sum(axis=1)
            v = self.oracle.values(m)
            self.normalization_value = self.oracle.v_empty
            self.t_exit.append(time.perf_counter())
            return v
    return EngineGame


class LeverageOracleAdapter:


    def __init__(self, oracle: ValueOracle, d: int):
        self.oracle = oracle
        self.d = d
        self.w = (np.int64(1) << np.arange(d, dtype=np.int64))
        self.t_enter: list = []
        self.t_exit: list = []

    def value(self, S):
        self.t_enter.append(time.perf_counter())
        S = np.asarray(S).astype(bool).reshape(-1, self.d)
        v = self.oracle.values((S.astype(np.int64) * self.w[None, :]).sum(axis=1)) - self.oracle.v_empty
        self.t_exit.append(time.perf_counter())
        return v

    def edge_cases(self):
        v = self.oracle.values(np.array([0, (1 << self.d) - 1], dtype=np.int64)) - self.oracle.v_empty
        return float(v[0]), float(v[1])


def stratsvarm_plan(n: int, budget: int, seed: int, count_empty: bool = True) -> dict:


    from e4_stratified_svarm import paper_size_distribution, warmup_cost
    if n < 4:
        raise ValueError("Stratified SVARM needs n >= 4")
    T = int(budget) - (1 if count_empty else 0)
    wu = warmup_cost(n, count_empty=False)
    if T < wu:
        raise ValueError(f"budget {budget} below the Stratified SVARM warm-up cost {wu + int(count_empty)}")
    allm = (1 << n) - 1
    bit = (np.int64(1) << np.arange(n, dtype=np.int64))
    rng = np.random.default_rng(seed)
    choice = rng.choice
    players = list(range(n))

    def mask_of(A):
        return int(np.sum(bit[np.asarray(A, dtype=np.int64)])) if len(A) else 0

    def blocks(s):
        pi = choice(players, n, replace=False)
        out = []
        for kk in range(0, int(n / s)):
            out.append((list(pi[kk * s:(kk + 1) * s]), None))
        if n % s != 0:
            A = list(pi[n - (n % s):n])
            rest = [p for p in players if p not in A]
            B = list(choice(rest, s - (n % s), replace=False))
            out.append((A, B))
        return out

    wpos, wneg = [], []
    for s in range(2, n - 1):
        for A, B in blocks(s):
            S = A if B is None else list(set(list(A) + list(B)))
            wpos.append((s, [int(i) for i in A], mask_of(S)))
    for s in range(2, n - 1):
        for A, B in blocks(s):
            S = A if B is None else list(set(list(A) + list(B)))
            wneg.append((s, [int(i) for i in A], allm ^ mask_of(S)))
    M = T - wu
    probs = paper_size_distribution(n)
    if M > 0:
        sizes = rng.choice(n + 1, size=M, p=np.asarray(probs, dtype=np.float64)).astype(np.int64)
        keys = rng.random((M, n))
        ranks = np.argsort(np.argsort(keys, axis=1), axis=1)
        member = ranks < sizes[:, None]
        main = (member.astype(np.int64) * bit[None, :]).sum(axis=1)
    else:
        sizes = np.zeros(0, dtype=np.int64); main = np.zeros(0, dtype=np.int64)
    segs = []
    if count_empty:
        segs.append(np.zeros(1, dtype=np.int64))
    segs += [np.array([allm], dtype=np.int64), bit.copy(), allm ^ bit,
             np.array([w[2] for w in wpos], dtype=np.int64), np.array([w[2] for w in wneg], dtype=np.int64), main]
    masks = np.concatenate(segs)
    return {"n": n, "budget": int(budget), "seed": int(seed), "count_empty": count_empty, "masks": masks,
            "wpos": wpos, "wneg": wneg, "main": main, "sizes": sizes, "M": int(M), "warmup": int(wu + int(count_empty))}


def stratsvarm_aggregate(plan: dict, vals: np.ndarray) -> np.ndarray:


    from e4_stratified_svarm import _Strata
    n = plan["n"]
    vals = np.asarray(vals, dtype=np.float64)
    o = 1 if plan["count_empty"] else 0
    st = _Strata(n)
    for i in range(n):
        st.set_m(i, 0, 0.0)
    grand = float(vals[o]); o += 1
    for i in range(n):
        st.set_p(i, n - 1, grand)
    single = vals[o:o + n]; o += n
    compl = vals[o:o + n]; o += n
    for i in range(n):
        st.set_p(i, 0, float(single[i]))
        st.set_m(i, n - 1, float(compl[i]))
    ssum, csum = float(np.sum(single)), float(np.sum(compl))
    for j in range(n):
        st.sum_m[j, 1] = ssum - single[j]; st.cnt_m[j, 1] = n - 1
        st.sum_p[j, n - 2] = csum - compl[j]; st.cnt_p[j, n - 2] = n - 1
    for (s, A, _m) in plan["wpos"]:
        v = float(vals[o]); o += 1
        for i in A:
            st.set_p(int(i), s - 1, v)
    for (s, A, _m) in plan["wneg"]:
        v = float(vals[o]); o += 1
        for i in A:
            st.set_m(int(i), n - s, v)
    M = plan["M"]
    if M > 0:
        mv = vals[o:o + M]; o += M
        masks = plan["main"]; sizes = plan["sizes"]
        member = ((masks[:, None] >> np.arange(n, dtype=np.int64)[None, :]) & 1).astype(bool)
        rows, cols = np.nonzero(member)
        np.add.at(st.sum_p, (cols, sizes[rows] - 1), mv[rows])
        np.add.at(st.cnt_p, (cols, sizes[rows] - 1), 1.0)
        rows, cols = np.nonzero(~member)
        np.add.at(st.sum_m, (cols, sizes[rows]), mv[rows])
        np.add.at(st.cnt_m, (cols, sizes[rows]), 1.0)
    if o != vals.size:
        raise RuntimeError("value vector does not match the plan")
    return st.estimate()


def stratsvarm_distinct(n: int, budget: int, seed: int) -> int:
    return int(np.unique(stratsvarm_plan(n, budget, seed)["masks"]).size)


def stratsvarm_nominal_for_distinct(n: int, B: int, seeds, lo: Optional[int] = None, hi: Optional[int] = None) -> dict:


    from e4_stratified_svarm import min_budget
    lo = max(int(B), min_budget(n)) if lo is None else int(lo)
    hi = 4 * int(B) if hi is None else int(hi)

    def med(T):
        return float(np.median([stratsvarm_distinct(n, T, s) for s in seeds]))
    if med(lo) >= B:
        return {"T": lo, "median_distinct": med(lo)}
    while med(hi) < B:
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if med(mid) >= B:
            hi = mid
        else:
            lo = mid
    return {"T": hi, "median_distinct": med(hi)}


def paired_distinct_curve(n: int, seed: int, m_max: int) -> np.ndarray:


    rng = np.random.default_rng(seed)
    bit = (np.int64(1) << np.arange(n, dtype=np.int64))
    base = np.stack([rng.permutation(n) for _ in range(m_max)], axis=0)
    fw = np.cumsum(bit[base], axis=1)
    rv = np.cumsum(bit[base[:, ::-1]], axis=1)
    per_pair = np.concatenate([np.zeros((m_max, 1), dtype=np.int64), fw, rv], axis=1)
    flat = per_pair.reshape(-1)
    _, first = np.unique(flat, return_index=True)
    pair_of_first = first // per_pair.shape[1]
    counts = np.bincount(pair_of_first, minlength=m_max)
    return np.cumsum(counts)


def paired_m_for_distinct(n: int, B: int, seeds, m_max: Optional[int] = None) -> dict:

    m_max = m_max or max(64, int(4 * B / (n + 1)) + 64)
    while True:
        curves = np.stack([paired_distinct_curve(n, s, m_max) for s in seeds], axis=0)
        med = np.median(curves, axis=0)
        ok = np.nonzero(med >= B)[0]
        if ok.size:
            m = int(ok[0]) + 1
            return {"m": m, "median_distinct": float(med[m - 1]),
                    "per_seed_distinct": [int(c[m - 1]) for c in curves]}
        if m_max > 2_000_000:
            raise RuntimeError("cannot reach the distinct budget")
        m_max *= 2


def paired_m_requested(n: int, B: int) -> int:

    return (int(B) - 2) // (2 * (n + 1))


def _calls_summary(te: TimedEngine, n_calls0: int = 0, captures0: Optional[int] = None) -> dict:
    calls = te.calls[n_calls0:]
    wall = float(sum(c["wall_s"] for c in calls))
    tot = [c.get("total_ms") for c in calls]
    ker = [c.get("kernel_ms") for c in calls]
    caps = [c.get("graph_captures") for c in calls]
    out = {"engine_calls": len(calls), "engine_wall_s": wall, "engine_B": [c["B"] for c in calls],
           "engine_call_wall_s": [c["wall_s"] for c in calls],
           "gpu_total_ms": None if any(x is None for x in tot) else float(sum(tot)),
           "gpu_kernel_ms": None if any(x is None for x in ker) else float(sum(ker)),
           "engine_call_kernel_ms": ker, "engine_call_total_ms": tot,
           "engine_call_waves": [c.get("waves") for c in calls]}
    if captures0 is not None and calls and caps[-1] is not None:
        out["graph_captures_run"] = int(caps[-1]) - int(captures0)
    return out


def run_paired(te: TimedEngine, n: int, k: int, m: int, seed: int) -> dict:

    from cubcr.shapley_sampled import sample_shapley
    kw = dict(rb.PAPER_SAMPLER_KW, eps=None, seed=int(seed), n=int(n), k=int(k), max_perms=2 * int(m))
    t0 = time.perf_counter()
    r = sample_shapley(te, **kw)
    t1 = time.perf_counter()
    if r.n_samples != m:
        raise RuntimeError(f"paired sampler drew {r.n_samples} pairs, expected {m}")
    return {"wall_s": t1 - t0, "t_start": t0, "t_end": t1, "phi_hat": np.asarray(r.estimates, dtype=np.float64),
            "half_widths": np.asarray(r.half_widths, dtype=np.float64),
            "cs_lower": None if r.cs_lower is None else np.asarray(r.cs_lower, dtype=np.float64),
            "cs_upper": None if r.cs_upper is None else np.asarray(r.cs_upper, dtype=np.float64),
            "fp_term": float(r.fp_term), "cs_anomaly": bool(r.cs_anomaly), "status": r.status,
            "n_requested": int(r.n_coalitions_requested), "n_engine": int(r.n_solves), "n_units": int(r.n_samples),
            "n_perms": int(r.n_perms), "n_batches": int(r.n_batches), "batch_sizes": list(map(int, r.batch_sizes)),
            "sampler_engine_s": float(r.engine_time), "v_empty": float(r.v_empty), "v_full": float(r.v_full),
            "value_range": None if r.value_range is None else [float(x) for x in r.value_range],
            "clock_limited": bool(r.status == "time" or r.budget_overshoot_s > 0),
            "nominal": {"m": int(m), "max_perms": 2 * int(m)}, "host_split": None}


def run_kadd2(te: TimedEngine, n: int, k: int, T: int, seed: int) -> dict:
    import shapiq
    Game = _engine_game_class()
    orc = ValueOracle(te, n, k)
    game = Game(orc, n)
    t0 = time.perf_counter()
    ap = shapiq.approximator.kADDSHAP(n=int(n), random_state=int(seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iv = ap.approximate(budget=int(T), game=game)
    phi = np.array([iv[(p,)] for p in range(n)], dtype=np.float64)
    t1 = time.perf_counter()
    if orc.n_calls != 1 or len(game.t_enter) != 1:
        raise RuntimeError(f"kADD-SHAP made {len(game.t_enter)} requests / {orc.n_calls} engine calls (expected 1)")
    hs = {"coalition_generation_s": game.t_enter[0] - t0,
          "oracle_host_s": (game.t_exit[0] - game.t_enter[0]) - orc.t_engine,
          "estimator_arithmetic_s": t1 - game.t_exit[-1]}
    return _est_out(t0, t1, phi, orc, {"T": int(T)}, hs)


def run_leverage(te: TimedEngine, n: int, k: int, T: int, seed: int) -> dict:
    from e4_external import leverage_module
    mod = leverage_module()
    orc = ValueOracle(te, n, k)
    ad = LeverageOracleAdapter(orc, n)
    Te = int(T) - (int(T) % 2)
    t0 = time.perf_counter()
    est = mod.LeverageSHAP(int(n), ad, paired_sampling=True, random_state=int(seed))
    phi = np.asarray(est.shap_values(Te), dtype=np.float64).reshape(-1)
    t1 = time.perf_counter()
    if orc.n_calls != 1:
        raise RuntimeError(f"Leverage SHAP made {orc.n_calls} engine calls (expected 1)")
    hs = {"coalition_generation_s": ad.t_enter[0] - t0,
          "oracle_host_s": (ad.t_exit[0] - ad.t_enter[0]) - orc.t_engine,
          "estimator_arithmetic_s": t1 - ad.t_exit[-1]}
    return _est_out(t0, t1, phi, orc, {"T": Te}, hs)


def run_stratsvarm(te: TimedEngine, n: int, k: int, T: int, seed: int) -> dict:
    orc = ValueOracle(te, n, k)
    t0 = time.perf_counter()
    plan = stratsvarm_plan(int(n), int(T), int(seed))
    t1 = time.perf_counter()
    raw = orc.values(plan["masks"])
    t2 = time.perf_counter()
    phi = stratsvarm_aggregate(plan, raw - orc.v_empty)
    t3 = time.perf_counter()
    if orc.n_calls != 1:
        raise RuntimeError(f"Stratified SVARM made {orc.n_calls} engine calls (expected 1)")
    hs = {"coalition_generation_s": t1 - t0, "oracle_host_s": (t2 - t1) - orc.t_engine,
          "estimator_arithmetic_s": t3 - t2}
    return _est_out(t0, t3, phi, orc, {"T": int(T), "warmup": plan["warmup"], "M": plan["M"]}, hs)


def _est_out(t0, t1, phi, orc: ValueOracle, nominal: dict, host_split: dict) -> dict:
    return {"wall_s": t1 - t0, "t_start": t0, "t_end": t1, "phi_hat": phi, "half_widths": None, "cs_lower": None,
            "cs_upper": None, "fp_term": None, "cs_anomaly": None, "status": "OK" if np.all(np.isfinite(phi)) else "NONFINITE",
            "n_requested": int(orc.n_requested), "n_engine": int(orc.n_engine), "n_requests": int(orc.n_requests),
            "n_units": None, "n_perms": None, "n_batches": int(orc.n_calls), "batch_sizes": None,
            "oracle_engine_s": float(orc.t_engine), "v_empty": orc.v_empty, "v_full": orc.v_full,
            "value_range": None, "clock_limited": False, "nominal": nominal, "host_split": host_split}


def run_arm(arm: str, te: TimedEngine, n: int, k: int, nominal: dict, seed: int) -> dict:


    n_calls0 = len(te.calls)
    st = te.stats
    cap0 = st.get("graph_captures") if isinstance(st, dict) else None
    if arm in ("paired_d", "paired_r"):
        out = run_paired(te, n, k, int(nominal["m"]), seed)
    elif arm == "kadd2":
        out = run_kadd2(te, n, k, int(nominal["T"]), seed)
    elif arm == "leverage":
        out = run_leverage(te, n, k, int(nominal["T"]), seed)
    elif arm == "stratsvarm":
        out = run_stratsvarm(te, n, k, int(nominal["T"]), seed)
    else:
        raise KeyError(arm)
    out.update(_calls_summary(te, n_calls0, cap0))
    out["host_s"] = out["wall_s"] - out["engine_wall_s"]
    if out["engine_calls"]:
        out["first_call_offset_s"] = te.calls[n_calls0]["t0"] - out["t_start"]
    return out


def error_metrics(phi_hat, phi, upsilon: float) -> dict:
    from e4_estimators import metrics
    return metrics(phi_hat, phi, upsilon)


def interval_checks(out: dict, phi) -> dict:


    if out.get("half_widths") is None:
        return {"covers_paper": None, "covers_cs": None}
    phi = np.asarray(phi, dtype=np.float64)
    hw = out["half_widths"]
    cov = bool(np.all(np.abs(out["phi_hat"] - phi) <= hw + 1e-12))
    cs = None
    if out.get("cs_lower") is not None:
        lo = out["cs_lower"] - out["fp_term"]; hi = out["cs_upper"] + out["fp_term"]
        cs = bool(np.all((phi >= lo) & (phi <= hi)))
    return {"covers_paper": cov, "covers_cs": cs, "hw_max": float(np.max(hw)) if np.all(np.isfinite(hw)) else None}
