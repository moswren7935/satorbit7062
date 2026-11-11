import math
from itertools import product

import numpy as np
import pytest

import cubcr.shapley_sampled as ss
from cubcr.engine_cpu import CPUEngine
from cubcr.fp_error import FPErrorBound, forward_error_bound
from cubcr.shapley_sampled import PrPlEBSequence, sample_shapley
from tests.helpers import random_csg

SEED = 20260829
mp = pytest.importorskip("mpmath")
mp.mp.dps = 50


def reference_cs(Y_batches, alpha_p, lam_max=0.5):


    log_term = mp.log(mp.mpf(2) / mp.mpf(alpha_p))
    S1 = mp.mpf(0); Q = mp.mpf(0); SlamY = mp.mpf(0); Slam = mp.mpf(0); Spen = mp.mpf(0)
    m = 0
    Lr, Ur = mp.mpf("-inf"), mp.mpf("inf")
    out, lams = [], []
    for Yb in Y_batches:
        i0 = m + 1
        sig2 = (mp.mpf(1) / 4 + Q) / (m + 1)
        lam = min(mp.sqrt(2 * log_term / (sig2 * i0 * mp.log(1 + i0))), mp.mpf(lam_max))
        lams.append(lam)
        psi = (-mp.log(1 - lam) - lam) / 4
        for y in Yb:
            y = mp.mpf(y)
            i = m + 1
            mu_prev = (mp.mpf(1) / 2 + S1) / i
            v = 4 * (y - mu_prev) ** 2
            S1 += y
            mu_incl = (mp.mpf(1) / 2 + S1) / (i + 1)
            Q += (y - mu_incl) ** 2
            SlamY += lam * y; Slam += lam; Spen += v * psi
            m = i
        c = SlamY / Slam
        w = (log_term + Spen) / Slam
        Lr, Ur = max(Lr, c - w), min(Ur, c + w)
        out.append((Lr, Ur))
    return out, lams


def run_prod(Y_batches, alpha_p, N=1):
    cs = PrPlEBSequence(N, alpha_p)
    outs = []
    for Yb in Y_batches:
        L, U = cs.update(np.asarray(Yb, dtype=np.float64).reshape(-1, N))
        outs.append((L.copy(), U.copy()))
    return cs, outs


@pytest.mark.parametrize("seq_name", ["alternating", "constant0", "constant1", "random", "extremes"])
def test_theorem_trace_matches_high_precision_and_encloses_outward(seq_name):
    rng = np.random.default_rng(7)
    T = 37
    if seq_name == "alternating":
        Y = [float(i % 2) for i in range(T)]
    elif seq_name == "constant0":
        Y = [0.0] * T
    elif seq_name == "constant1":
        Y = [1.0] * T
    elif seq_name == "random":
        Y = rng.random(T).round(3).tolist()
    else:
        Y = ([0.0] * 5 + [1.0] * 5 + [0.0] * 5 + [1.0] * 5 + [0.5] * 17)
    batches = [Y[0:1], Y[1:4], Y[4:11], Y[11:12], Y[12:25], Y[25:37]]
    alpha_p = 0.05 / 3
    ref, lams_ref = reference_cs(batches, alpha_p)
    cs, prod = run_prod(batches, alpha_p)
    for (Lr, Ur), (Lp, Up), lam_ref, tr in zip(ref, prod, lams_ref, cs.trace):
        assert abs(float(tr["lambda"][0]) - float(lam_ref)) < 1e-14

        assert Lp[0] <= float(Lr) + 1e-15 and Up[0] >= float(Ur) - 1e-15
        assert abs(Lp[0] - float(Lr)) < 1e-11 and abs(Up[0] - float(Ur)) < 1e-11

    Ls = np.array([p[0][0] for p in prod]); Us = np.array([p[1][0] for p in prod])
    assert np.all(np.diff(Ls) >= 0) and np.all(np.diff(Us) <= 0)


def test_batch_lambda_is_predictable_and_prefix_is_exclusive():
    rng = np.random.default_rng(3)
    Y = rng.random(30)
    batches = [Y[:5], Y[5:12], Y[12:30]]
    cs1, out1 = run_prod(batches, 0.01)
    Y2 = Y.copy(); Y2[13] = 1.0 - Y2[13]
    batches2 = [Y2[:5], Y2[5:12], Y2[12:30]]
    cs2, out2 = run_prod(batches2, 0.01)
    for j in range(3):
        assert np.array_equal(cs1.trace[j]["lambda"], cs2.trace[j]["lambda"])
    assert out1[0] == out2[0] and np.array_equal(out1[1][0], out2[1][0])

    cs3, _ = run_prod([Y[:5], Y[5:12], Y[12:13]], 0.01)
    cs4, _ = run_prod([Y2[:5], Y2[5:12], Y2[12:13]], 0.01)
    assert np.array_equal(cs3.Spen, cs4.Spen)


def test_vectorised_players_equal_scalar_runs():
    rng = np.random.default_rng(11)
    Y = rng.random((40, 3))
    batches = [Y[:8], Y[8:9], Y[9:40]]
    _, outv = run_prod(batches, 0.02, N=3)
    for p in range(3):
        _, outs = run_prod([b[:, p] for b in batches], 0.02, N=1)
        for (Lv, Uv), (Ls, Us) in zip(outv, outs):
            assert abs(Lv[p] - Ls[0]) < 1e-13 and abs(Uv[p] - Us[0]) < 1e-13


@pytest.mark.parametrize("mu", [0.1, 0.3, 0.5, 0.8])
@pytest.mark.parametrize("alpha", [0.5, 0.2])
def test_exhaustive_bernoulli_anytime_coverage(mu, alpha):

    T = 11
    total_miss = 0.0
    for bits in product((0, 1), repeat=T):
        cs = PrPlEBSequence(1, alpha)
        missed = False
        for b in bits:
            L, U = cs.update(np.array([[float(b)]]))
            if not (L[0] <= mu <= U[0]):
                missed = True
                break
        if missed:
            k1 = sum(bits)
            total_miss += (mu ** k1) * ((1 - mu) ** (T - k1))
    assert total_miss <= alpha


class TableEngine:

    compiled = None
    dtype = "fp64"

    def __init__(self, table):
        self.table = np.asarray(table, dtype=np.float64)

    def solve(self, masks, k=None):
        return self.table[np.asarray(masks, dtype=np.uint64).astype(np.int64)]


def exact_shapley(table, N):
    phi = np.zeros(N)
    for S in range(1 << N):
        s = bin(S).count("1")
        for p in range(N):
            if not (S >> p) & 1:
                w = math.factorial(s) * math.factorial(N - s - 1) / math.factorial(N)
                phi[p] += w * (table[S | (1 << p)] - table[S])
    return phi


def random_monotone_table(rng, N, family):
    size = 1 << N
    pop = np.array([bin(S).count("1") for S in range(size)])
    if family == "mobius":
        c = rng.random(size) * (rng.random(size) < 0.3)
        v = np.zeros(size)
        for S in range(size):
            for T in range(size):
                if T & S == T:
                    v[S] += c[T]
    elif family == "concave":
        w = rng.random(N)
        tot = np.array([sum(w[p] for p in range(N) if (S >> p) & 1) for S in range(size)])
        v = np.sqrt(tot)
    elif family == "unanimity":
        R = int(rng.integers(1, size))
        v = np.array([1.0 if (S & R) == R else 0.0 for S in range(size)])
    else:
        w = rng.random(N)
        tot = np.array([sum(w[p] for p in range(N) if (S >> p) & 1) for S in range(size)])
        R = int(rng.integers(1, size))
        v = 0.2 * tot / tot.max() + 0.8 * np.array([1.0 if (S & R) == R else 0.0 for S in range(size)])
    v = v - v[0]
    v = v / max(v[-1], 1e-12) * float(rng.uniform(0.05, 1.0))
    return v


def _binomial_upper(n, p, q=0.999):

    from math import comb
    acc, c = 0.0, 0
    while True:
        acc += comb(n, c) * p ** c * (1 - p) ** (n - c)
        if acc >= q:
            return c
        c += 1


def test_mock_game_coverage_at_stop_and_anytime():
    rng = np.random.default_rng(5)
    n_games, fails_stop, fails_any, alpha = 160, 0, 0, 0.05
    for g in range(n_games):
        N = int(rng.integers(3, 8))
        fam = ["mobius", "concave", "unanimity", "mixture"][g % 4]
        table = random_monotone_table(rng, N, fam)
        while table[-1] - table[0] < 1e-3:
            table = random_monotone_table(rng, N, fam)
        phi = exact_shapley(table, N)
        ups = table[-1] - table[0]
        eng = TableEngine(table)
        seen = {"miss": False}

        def cb(m, est, hw, phi=phi, seen=seen):
            if np.isfinite(hw).all() and np.any(np.abs(est - phi) > hw):
                seen["miss"] = True

        res = sample_shapley(eng, N, kind="agent", seed=int(rng.integers(1 << 30)), eps=ups / 20, alpha=alpha,
                             value_range="monotone", ci="prpl_eb_cs", batch_schedule="geometric",
                             batch_perms=8, max_perms=200_000, callback=cb)
        assert res.certified and res.ci_method == "prpl_eb_cs" and not res.cs_anomaly
        assert abs(res.estimates.sum() - ups) < 1e-9
        if np.any(np.abs(res.estimates - phi) > res.half_widths):
            fails_stop += 1
        if seen["miss"]:
            fails_any += 1
    cap = _binomial_upper(n_games, alpha)
    assert fails_stop <= cap and fails_any <= cap


def test_geometric_schedule_and_certificate_fields():
    rng = np.random.default_rng(9)
    table = random_monotone_table(rng, 6, "mixture")
    res = sample_shapley(TableEngine(table), 6, kind="agent", seed=1, eps=(table[-1] - table[0]) / 20,
                         value_range="monotone", ci="prpl_eb_cs", batch_schedule="geometric", batch_perms=4,
                         max_perms=100_000)
    assert res.batch_sizes[0] == 4 and all(b <= 64 for b in res.batch_sizes) and res.n_batches == len(res.batch_sizes)
    assert res.certificate["ci_method"] == "prpl_eb_cs" and res.certificate["anytime_valid"]
    assert res.anytime_valid and res.n_looks_max is None and res.alpha_look == res.alpha_player
    assert np.all(res.cs_lower <= res.estimates + 1e-12) and np.all(res.cs_upper >= res.estimates - 1e-12)
    with pytest.raises(ValueError):
        sample_shapley(TableEngine(table), 6, ci="prpl_eb_cs")
    with pytest.raises(ValueError):
        sample_shapley(TableEngine(table), 6, ci="eb_looks", batch_schedule="geometric")
    with pytest.raises(ValueError):
        sample_shapley(TableEngine(table), 6, ci="prpl_eb_cs", value_range="monotone", bonferroni=False)


def test_fp_term_on_real_engine_and_null_path():
    csg = random_csg(n_agents=4, n_states=20, k=6, seed=SEED, max_active=3, max_actions=3)
    comp = csg.compile()
    eng = CPUEngine(comp, n_threads=2)
    fp = forward_error_bound(comp, 6)
    res = sample_shapley(eng, 4, kind="agent", seed=SEED, eps=0.02, value_range="monotone", ci="prpl_eb_cs",
                         batch_perms=8, max_perms=4000)
    assert res.fp_term == fp.eps_unit(res.perms_per_unit) and np.all(res.half_widths >= res.fp_term)
    assert res.value_range == fp.statistical_range(res.v_full - res.v_empty, res.perms_per_unit)
    assert res.fp_error["M"] == fp.M and res.fp_error["dtype"] == "fp64"

    bound = FPErrorBound(eta_v=1e-13, M=10, u=2.0 ** -53, gamma_M=1e-15, rho=1.0, k=1, max_nnz_row=1,
                         max_rows_per_state=1, dtype="fp64")
    null = TableEngine(np.full(16, 0.3))
    r0 = sample_shapley(null, 4, kind="agent", eps=1e-6, value_range="monotone", ci="prpl_eb_cs", fp_bound=bound)
    assert r0.status == "eps" and r0.certified and r0.n_perms == 0
    assert np.all(r0.half_widths == bound.eps_upsilon) and np.all(r0.estimates == 0.0)
    r1 = sample_shapley(null, 4, kind="agent", eps=1e-14, value_range="monotone", ci="prpl_eb_cs", fp_bound=bound)
    assert r1.status == "null" and not r1.certified and r1.certificate is None


class ClockEngine(TableEngine):
    def __init__(self, table, clock):
        super().__init__(table)
        self.clock = clock

    def solve(self, masks, k=None):
        self.clock[0] += 1.0
        return super().solve(masks, k)


def test_legacy_path_never_certifies_at_clock_dependent_looks(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(ss.time, "perf_counter", lambda: clock[0])
    N = 5
    table = np.array([bin(S).count("1") / N for S in range(1 << N)])
    eng = ClockEngine(table, clock)

    L_max = ss.look_schedule_size(10 ** 6, 4 * 2)
    alpha_eff = 0.05 / N / L_max
    Lc = math.log(4 / alpha_eff)
    hw = lambda m: 2 * 7 * Lc / (3 * (m - 1))
    eps = 0.5 * (hw(8) + hw(10))
    res = sample_shapley(eng, N, kind="agent", seed=1, batch_perms=4, eps=eps, max_perms=10 ** 6,
                         time_budget=2.5, use_cache=False, antithetic=True)
    assert res.status == "time" and not res.certified and res.certificate is None
    assert res.look_m == [4, 8, 10]
    assert res.withheld_certificate is not None and res.withheld_certificate["withheld_reason"] == "time_terminal"
    assert res.withheld_certificate["terminal_reason"] == "time"
