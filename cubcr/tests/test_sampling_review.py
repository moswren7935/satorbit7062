import math

import numpy as np
import pytest

from cubcr import vi_ref
from cubcr.engine_cpu import CPUEngine
from cubcr.shapley import shapley_from_table, shapley_by_permutations
from cubcr.shapley_sampled import (CoalitionOracle, SampledShapleyResult, empirical_bernstein_half_width,
                                   exact_player_table, sample_shapley, topk_certified)
from cubcr.screening import singleton_screening
from tests.helpers import random_csg

SEED = 20260829


def hand_mask(players, kind, n, k):

    if kind == "agent":
        m = 0
        for p in players:
            m |= 1 << int(p)
        return m
    cols = [0] * k
    for p in players:
        i, t = divmod(int(p), k)
        cols[t] |= 1 << i
    return cols


def castro_reference(csg, kind, n, k, perms, antithetic):

    N = n if kind == "agent" else n * k
    units = []
    for perm in perms:
        variants = [perm, perm[::-1]] if antithetic else [perm]
        acc = np.zeros(N)
        for pv in variants:
            prev = vi_ref.solve(csg, hand_mask([], kind, n, k), k)
            pre = []
            for p in pv:
                pre.append(int(p))
                cur = vi_ref.solve(csg, hand_mask(pre, kind, n, k), k)
                acc[int(p)] += cur - prev
                prev = cur
        units.append(acc / len(variants))
    return np.asarray(units)


def eb_by_hand(units, alpha):
    m = units.shape[0]
    y = (units + 1.0) / 2.0
    V = y.var(axis=0, ddof=1)
    L = math.log(2 / alpha)
    return 2.0 * (np.sqrt(2 * V * L / m) + 7 * L / (3 * (m - 1)))


@pytest.mark.parametrize("kind,n,k,seed", [("agent", 4, 3, 11), ("step", 2, 3, 5), ("step", 3, 2, 7)])
@pytest.mark.parametrize("antithetic", [True, False])
def test_sampler_equals_pure_python_castro_on_vi_ref(kind, n, k, seed, antithetic):

    csg = random_csg(n_agents=n, n_states=12, k=k, seed=seed, max_active=min(n, 3), max_actions=3)
    eng = CPUEngine(csg.compile(), n_threads=2)
    N = n if kind == "agent" else n * k
    P, nb = 5, 3

    rng = np.random.default_rng(SEED)
    perms = []
    for _ in range(nb):
        for _ in range(P):
            perms.append(rng.permutation(N))
    units = castro_reference(csg, kind, n, k, perms, antithetic)
    res = sample_shapley(eng, N, kind=kind, n=n, k=k, seed=SEED, batch_perms=P,
                         max_perms=P * nb * (2 if antithetic else 1), antithetic=antithetic, alpha=0.05)
    assert res.n_samples == P * nb and res.n_perms == P * nb * (2 if antithetic else 1)
    assert np.max(np.abs(res.estimates - units.mean(axis=0))) <= 1e-12
    assert np.max(np.abs(res.half_widths - eb_by_hand(units, 0.05))) <= 1e-10
    assert abs(res.v_empty - vi_ref.solve(csg, hand_mask([], kind, n, k), k)) <= 1e-12
    assert abs(res.v_full - vi_ref.solve(csg, hand_mask(range(N), kind, n, k), k)) <= 1e-12

    hm = res.heatmap()
    for p in range(N):
        i, t = divmod(p, k) if kind == "step" else (p, 0)
        assert hm[i, t] == res.estimates[p]


def test_stratified_fold_equals_hand_computation():

    n, k = 2, 3
    csg = random_csg(n_agents=n, n_states=12, k=k, seed=5, max_active=2, max_actions=3)
    eng = CPUEngine(csg.compile(), n_threads=2)
    N = n * k
    P, nb = 3, 2
    rng = np.random.default_rng(SEED)
    base = [rng.permutation(N) for _ in range(nb * P)]
    derived = []
    for b in base:
        shifts = [np.roll(b, -c) for c in range(N)]
        derived.append(shifts + [s[::-1] for s in shifts])
    units = np.stack([castro_reference(csg, "step", n, k, d, antithetic=False).mean(axis=0) for d in derived])
    res = sample_shapley(eng, N, kind="step", n=n, k=k, seed=SEED, batch_perms=P, stratify=True,
                         antithetic=True, max_perms=nb * P * 2 * N)
    assert res.n_samples == nb * P and res.n_perms == nb * P * 2 * N
    assert np.max(np.abs(res.estimates - units.mean(axis=0))) <= 1e-12
    assert np.max(np.abs(res.half_widths - eb_by_hand(units, 0.05))) <= 1e-10

    for d in derived:
        pos = np.zeros((N, N), dtype=int)
        for perm in d[:N]:
            for j, p in enumerate(perm):
                pos[p, j] += 1
        assert np.all(pos == 1)


def test_unbiased_over_many_seeds_all_variants():

    csg = random_csg(n_agents=4, n_states=14, k=3, seed=SEED, max_active=3, max_actions=3)
    eng = CPUEngine(csg.compile(), n_threads=2)
    phi = shapley_from_table(eng.solve(np.arange(16, dtype=np.uint64)), 4)
    for anti, strat in [(False, False), (True, False), (False, True), (True, True)]:
        ests = []
        for s in range(300):
            r = sample_shapley(eng, 4, kind="agent", seed=s, batch_perms=2, max_perms=2, antithetic=anti,
                               stratify=strat)
            ests.append(r.estimates)
        ests = np.asarray(ests)
        se = ests.std(axis=0, ddof=1) / math.sqrt(len(ests))
        assert np.all(np.abs(ests.mean(axis=0) - phi) <= 4 * se + 1e-12), (anti, strat, ests.mean(0), phi, se)


def test_exact_by_permutations_equals_table_and_sampler_limit():

    csg = random_csg(n_agents=4, n_states=14, k=3, seed=3, max_active=3, max_actions=3)
    eng = CPUEngine(csg.compile(), n_threads=2)
    table = eng.solve(np.arange(16, dtype=np.uint64))
    phi_t = shapley_from_table(table, 4)
    phi_p = shapley_by_permutations(lambda m: float(table[m]), 4)
    assert np.max(np.abs(phi_t - phi_p)) <= 1e-13
    assert abs(phi_t.sum() - (table[15] - table[0])) <= 1e-13


def test_eb_synthetic_coverage_and_constants():

    rng = np.random.default_rng(SEED)
    alpha, m, trials = 0.05, 40, 2000
    mean_true = 2 * 0.3 - 1.0
    hits = 0
    for _ in range(trials):
        x = 2 * rng.beta(0.6, 1.4, size=m) - 1.0
        hw = float(empirical_bernstein_half_width(np.var(x, ddof=1), m, alpha))
        hits += abs(x.mean() - mean_true) <= hw
    assert hits / trials >= 1 - alpha

    L = math.log(2 / alpha)
    assert float(empirical_bernstein_half_width(0.0, m, alpha)) == pytest.approx(2 * 7 * L / (3 * (m - 1)))

    assert np.isfinite(empirical_bernstein_half_width(-1e-18, m, alpha))
    assert np.isinf(empirical_bernstein_half_width(0.1, 1, alpha))


def test_screening_vs_hand_masks_on_vi_ref():
    n, k = 3, 3
    csg = random_csg(n_agents=n, n_states=14, k=k, seed=9, max_active=3, max_actions=3)
    eng = CPUEngine(csg.compile(), n_threads=2)
    sc = singleton_screening(eng, n, k, kind="step")
    N = n * k
    v0 = vi_ref.solve(csg, hand_mask([], "step", n, k), k)
    vN = vi_ref.solve(csg, hand_mask(range(N), "step", n, k), k)
    assert abs(sc.v_empty - v0) <= 1e-12 and abs(sc.v_full - vN) <= 1e-12
    for i in range(n):
        for t in range(k):
            p = i * k + t
            add = vi_ref.solve(csg, hand_mask([p], "step", n, k), k) - v0
            rem = vN - vi_ref.solve(csg, hand_mask([q for q in range(N) if q != p], "step", n, k), k)
            assert abs(sc.add[i, t] - add) <= 1e-12
            assert abs(sc.remove[i, t] - rem) <= 1e-12

    table = exact_player_table(eng, N, "step", n, k)
    assert abs(table[-1] - eng.solve(np.array([(1 << n) - 1], dtype=np.uint64))[0]) <= 1e-15
    assert abs(table[0] - eng.solve(np.array([0], dtype=np.uint64))[0]) <= 1e-15


def test_edge_cases_and_fail_loud():
    csg = random_csg(n_agents=3, n_states=12, k=3, seed=SEED, max_active=3, max_actions=3)
    eng = CPUEngine(csg.compile(), n_threads=2)
    with pytest.raises(ValueError):
        CoalitionOracle(eng, 3, "bogus")
    with pytest.raises(ValueError):
        sample_shapley(eng, 5, kind="agent", max_perms=2, batch_perms=1)
    with pytest.raises(ValueError):
        sample_shapley(eng, 8, kind="step", n=3, k=3, max_perms=2, batch_perms=1)
    with pytest.raises(ValueError):
        exact_player_table(eng, 25, "agent", 25, 3)
    with pytest.raises(Exception):
        sample_shapley(eng, 3, kind="agent", batch_perms=0, max_perms=4)

    a = sample_shapley(eng, kind="step", seed=1, max_perms=40, batch_perms=10)
    b = sample_shapley(eng, kind="step", seed=1, max_perms=40, batch_perms=10)
    c = sample_shapley(eng, kind="step", seed=2, max_perms=40, batch_perms=10)
    assert a.n_players == 9 and np.array_equal(a.estimates, b.estimates) and not np.array_equal(a.estimates, c.estimates)

    csg1 = random_csg(n_agents=1, n_states=10, k=3, seed=4, max_active=1, max_actions=3)
    e1 = CPUEngine(csg1.compile(), n_threads=1)
    r = sample_shapley(e1, 1, kind="agent", max_perms=4, batch_perms=2)
    v = e1.solve(np.array([0, 1], dtype=np.uint64))
    assert r.estimates[0] == pytest.approx(v[1] - v[0], abs=1e-15) and r.sample_var[0] == 0.0

    csg0 = random_csg(n_agents=3, n_states=12, k=0, seed=SEED, max_active=3, max_actions=3)
    e0 = CPUEngine(csg0.compile(), n_threads=1)
    r0 = sample_shapley(e0, 3, kind="agent", max_perms=4, batch_perms=2)
    assert np.all(r0.estimates == 0.0) and r0.v_full == 0.0 and r0.v_empty == 0.0

    ok, top = topk_certified(np.array([0.3, 0.2]), np.array([np.inf, np.inf]), 2)
    assert ok and set(top.tolist()) == {0, 1}
    r = sample_shapley(eng, 3, kind="agent", top_k=3, max_perms=100, batch_perms=4)
    assert r.status == "top_k" and r.n_perms == 8

    csga = random_csg(n_agents=3, n_states=14, k=3, seed=8, max_active=3, max_actions=3, n_avoid=2)
    ea = CPUEngine(csga.compile(), n_threads=2)
    ra = sample_shapley(ea, 3, kind="agent", max_perms=48, batch_perms=8, antithetic=False)
    N = 3
    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(N) for _ in range(48)]
    units = castro_reference(csga, "agent", 3, 3, perms, antithetic=False)
    assert np.max(np.abs(ra.estimates - units.mean(axis=0))) <= 1e-12
